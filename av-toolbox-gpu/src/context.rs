#[cfg(test)]
use std::sync::Arc;

use cubecl::prelude::{CubeCount, CubeDim};

use crate::Capabilities;
#[cfg(test)]
use crate::DeviceAllocator;
use crate::frame::{Frame, FrameFormat, FrameMeta, Mask, MaskFormat, PlaneBuffer};
use crate::memory::{BufferPool, Ring, RingSpec};

/// What a stage is given every time it does work.
///
/// It owns the pool a stage's frames come from and answers the work sizes a kernel launch needs.
/// It is not generic over the runtime, so a stage holds its own client from build time and takes
/// this per call.
pub struct FrameContext {
    pool: BufferPool,
    capabilities: Capabilities,
}

impl FrameContext {
    pub(crate) fn new(pool: BufferPool, capabilities: Capabilities) -> Self {
        Self { pool, capabilities }
    }

    /// A context over a fake allocator, for tests that need no device.
    #[cfg(test)]
    fn for_testing(allocator: Arc<dyn DeviceAllocator>, capabilities: Capabilities) -> Self {
        Self::new(BufferPool::new(allocator), capabilities)
    }

    /// A frame of `format`, carrying `meta`.
    ///
    /// Its planes are recycled memory holding an earlier frame's samples, so a stage that writes
    /// only some of them has to copy the rest from its input.
    pub fn alloc_frame(&self, format: &FrameFormat, meta: FrameMeta) -> Frame {
        self.pool.alloc_frame(format, meta)
    }

    /// A fresh frame in the same format as `frame`, carrying `meta`.
    ///
    /// The metadata is given rather than copied, because a stage that emits more or fewer frames
    /// than it consumes has no business inheriting its input's place in the stream. Its planes are
    /// recycled memory holding an earlier frame's samples, so a stage that writes only some of them
    /// has to copy the rest from its input.
    pub fn alloc_like(&self, frame: &Frame, meta: FrameMeta) -> Frame {
        self.pool.alloc_like(frame, meta)
    }

    /// A mask of `format`, carrying `meta`.
    ///
    /// Its buffer is recycled memory holding an earlier mask's samples, so a stage that writes only
    /// part of it has to fill the rest.
    pub fn alloc_mask(&self, format: &MaskFormat, meta: FrameMeta) -> Mask {
        self.pool.alloc_mask(format, meta)
    }

    /// Memory for one step's working set, returned to the pool when it drops.
    ///
    /// It arrives holding whatever was last written there, so read nothing from it that this step
    /// has not written.
    pub fn scratch(&self, bytes: usize) -> PlaneBuffer {
        self.pool.scratch(bytes)
    }

    pub fn ring(&self, spec: RingSpec) -> Ring {
        self.pool.ring(spec)
    }

    pub const fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// How many cubes cover `samples`, given [FrameContext::cube_dim](crate::FrameContext::cube_dim).
    ///
    /// A device refuses a launch that overruns any one axis of its cube count, so a count too
    /// large for the first axis spreads into the second and then the third, rounding up. Every axis
    /// is capped at what the device reports, so a count past all three caps multiplied together is
    /// truncated rather than covered.
    pub fn cube_count(&self, samples: usize) -> CubeCount {
        let units_per_cube = self.cube_dim().num_elems() as usize;
        let cubes = samples.div_ceil(units_per_cube);

        let (max_x, max_y, max_z) = self.capabilities.max_cube_count;

        let x = cubes.min(max_x as usize);
        let rows = cubes.div_ceil(x.max(1));
        let y = rows.min(max_y as usize).max(1);
        let layers = cubes.div_ceil((x * y).max(1));
        let z = layers.min(max_z as usize).max(1);

        CubeCount::Static(x as u32, y as u32, z as u32)
    }

    /// The cube shape kernels launch with, sized to what the device reports.
    ///
    /// The safe minimum plane width is squared into a two-dimensional cube, then both axes are
    /// clamped so the total never exceeds `max_units_per_cube`.
    pub fn cube_dim(&self) -> CubeDim {
        let plane_width = self.capabilities.plane_size.0;
        let max_units = self.capabilities.max_units_per_cube;

        let width = plane_width.min(max_units);
        let rows = (max_units / width).min(plane_width);

        CubeDim::new_2d(width, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::CountingAllocator;
    use crate::{Accelerator, ColourInfo, FrameLayout, SampleFormat, Subsampling, default_accelerators};

    fn any_accelerator() -> Accelerator {
        default_accelerators()[0]
    }

    fn capabilities() -> Capabilities {
        Capabilities::builder(any_accelerator())
            .plane_size(16, 16)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build()
    }

    /// A table whose cube count limits are small enough to bite, so a launch sized against it is
    /// spread rather than piled onto the first axis.
    ///
    /// The cube is 16x16, so every cube covers 256 samples and each axis holds at most four.
    fn tight_capabilities() -> Capabilities {
        Capabilities::builder(any_accelerator())
            .plane_size(16, 16)
            .max_units_per_cube(1024)
            .max_cube_count(4, 4, 4)
            .build()
    }

    /// The axes of a static count, which carries no `PartialEq` to compare whole.
    fn axes(count: CubeCount) -> (u32, u32, u32) {
        match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            CubeCount::Dynamic(_) => panic!("expected a static cube count"),
        }
    }

    fn frame_format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
        }
    }

    #[test]
    fn a_context_hands_out_frames_from_its_pool() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator.clone(), capabilities());

        let frame = context.alloc_frame(&frame_format(), FrameMeta::at(7));

        assert_eq!(frame.meta().index, 7);
        assert_eq!(allocator.count(), 3, "one per plane of a yuv frame");
    }

    /// The carried-forward defect: a stage that is not one frame in, one frame out must not
    /// inherit the index of the frame it was given.
    #[test]
    fn alloc_like_takes_the_metadata_it_is_given() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let source = context.alloc_frame(&frame_format(), FrameMeta::at(3));

        let derived = context.alloc_like(&source, FrameMeta::at(100));

        assert_eq!(derived.format(), source.format());
        assert_eq!(derived.meta().index, 100);
    }

    /// Whatever axes a launch is spread across, it has to cover every sample and stay inside every
    /// limit the device reported.
    #[test]
    fn cube_count_covers_every_sample() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, tight_capabilities());

        for samples in [1, 1000, 256 * 5, 256 * 17, 256 * 64] {
            let (x, y, z) = axes(context.cube_count(samples));

            assert!(x <= 4 && y <= 4 && z <= 4, "{samples} samples ran past a limit");
            assert!(
                (x * y * z) as usize * 256 >= samples,
                "{samples} samples were not covered by {x}x{y}x{z}",
            );
        }
    }

    /// A plane small enough to fit the first axis launches exactly as it always has.
    #[test]
    fn cube_count_stays_in_one_dimension_while_it_fits() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, tight_capabilities());

        assert_eq!(axes(context.cube_count(1000)), (4, 1, 1));
    }

    #[test]
    fn cube_count_spreads_into_the_second_dimension_once_the_first_is_full() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, tight_capabilities());

        assert_eq!(axes(context.cube_count(256 * 6)), (4, 2, 1));
    }

    #[test]
    fn cube_count_spreads_into_the_third_dimension_once_the_first_two_are_full() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, tight_capabilities());

        assert_eq!(axes(context.cube_count(256 * 20)), (4, 4, 2));
    }

    #[test]
    fn cube_dim_squares_the_plane_size() {
        let context = FrameContext::for_testing(Arc::new(CountingAllocator::new(64)), capabilities());

        assert_eq!(context.cube_dim(), CubeDim::new_2d(16, 16));
    }

    #[test]
    fn cube_dim_shrinks_to_fit_a_tighter_unit_cap() {
        let capabilities = Capabilities::builder(any_accelerator())
            .plane_size(16, 16)
            .max_units_per_cube(64)
            .max_cube_count(65535, 65535, 65535)
            .build();
        let context = FrameContext::for_testing(Arc::new(CountingAllocator::new(64)), capabilities);

        assert_eq!(context.cube_dim(), CubeDim::new_2d(16, 4));
    }

    /// A cap tighter than the plane width has to shrink the first axis too, or the cube still
    /// overruns it.
    #[test]
    fn cube_dim_clamps_both_axes_when_the_cap_is_tighter_than_the_plane_width() {
        let capabilities = Capabilities::builder(any_accelerator())
            .plane_size(16, 16)
            .max_units_per_cube(8)
            .max_cube_count(65535, 65535, 65535)
            .build();
        let context = FrameContext::for_testing(Arc::new(CountingAllocator::new(64)), capabilities);

        assert_eq!(context.cube_dim(), CubeDim::new_2d(8, 1));
    }
}
