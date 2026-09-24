use std::sync::Arc;

use cubecl::prelude::{CubeCount, CubeDim};

use crate::Capabilities;
#[cfg(test)]
use crate::DeviceAllocator;
use crate::batch::{PendingBatch, PlaneMap};
#[cfg(test)]
use crate::frame::Channel;
use crate::frame::{
    Frame,
    FrameFormat,
    FrameMeta,
    HostFrameMut,
    HostMaskMut,
    HostPlanes,
    HostStreamItem,
    Mask,
    MaskFormat,
    PlaneBuffer,
    StreamFormat,
    StreamItem,
};
use crate::memory::{BufferPool, FrameBudget, HostPool, Ring, RingSpec, RuntimeError};

/// One item to put on the device.
///
/// The format it loads as is the one its planes already carry, so a caller cannot pair a
/// `HostPlanes` with a format it disagrees with. The planes are borrowed, so a source running
/// frame after frame can refill one set rather than handing over a new one each time.
pub struct LoadItem<'a> {
    pub planes: &'a HostPlanes,
    pub meta: FrameMeta,
}

impl<'a> LoadItem<'a> {
    pub fn new(planes: &'a HostPlanes, meta: FrameMeta) -> Self {
        Self { planes, meta }
    }
}

/// What a stage is given every time it does work.
///
/// It owns the pool a stage's frames come from and answers the work sizes a kernel launch needs.
/// It is not generic over the runtime, so a stage holds its own client from build time and takes
/// this per call.
pub struct FrameContext {
    pool: BufferPool,
    host: HostPool,
    capabilities: Capabilities,
}

impl FrameContext {
    pub(crate) fn new(pool: BufferPool, capabilities: Capabilities) -> Self {
        Self {
            pool,
            host: HostPool::new(),
            capabilities,
        }
    }

    /// A context over a fake allocator, for tests that need no device.
    #[cfg(test)]
    fn for_testing(allocator: Arc<dyn DeviceAllocator>, capabilities: Capabilities) -> Self {
        Self::new(BufferPool::new(allocator), capabilities)
    }

    /// This context, charging every frame shaped allocation on either side to `budget`.
    pub fn with_budget(mut self, budget: Arc<FrameBudget>) -> Self {
        self.host = self.host.with_budget(budget.clone());
        self.pool = self.pool.with_budget(budget);
        self
    }

    /// The budget this context charges its allocations to, if it has one.
    pub fn budget(&self) -> Option<&Arc<FrameBudget>> {
        self.pool.budget()
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

    /// A writable host frame of `format`, carrying `meta`, charged to this context's budget.
    ///
    /// Its planes are recycled memory holding an earlier frame's samples, so a stage that writes
    /// only some of them has to fill the rest.
    pub fn alloc_host_frame(&self, format: &FrameFormat, meta: FrameMeta) -> HostFrameMut {
        self.host.alloc_frame(format, meta)
    }

    /// A writable host mask of `format`, carrying `meta`, charged to this context's budget.
    pub fn alloc_host_mask(&self, format: &MaskFormat, meta: FrameMeta) -> HostMaskMut {
        self.host.alloc_mask(format, meta)
    }

    /// Uploads every plane of every item and returns them as device items.
    ///
    /// A frame and a mask may sit in the same batch, because a mask is the same machinery with one
    /// plane.
    pub fn load(&self, items: &[LoadItem<'_>]) -> Result<Vec<StreamItem>, RuntimeError> {
        let mut loaded = Vec::with_capacity(items.len());
        for (item, entry) in items.iter().enumerate() {
            let stream_item = match entry.planes.format() {
                StreamFormat::Frames(format) => {
                    let frame = self.pool.load_frame(item, format, entry.planes, entry.meta)?;

                    StreamItem::from(frame)
                },
                StreamFormat::Mask(format) => {
                    let mask = self.pool.load_mask(item, format, entry.planes, entry.meta)?;

                    StreamItem::from(mask)
                },
            };

            loaded.push(stream_item);
        }

        Ok(loaded)
    }

    /// The batch of one: loads `item` and hands back the single result.
    ///
    /// This carries no upload logic of its own, so a fix to
    /// [FrameContext::load](crate::FrameContext::load) never has a second copy left behind here
    /// to drift from it.
    pub fn load_one(&self, item: &LoadItem<'_>) -> Result<StreamItem, RuntimeError> {
        let loaded = self.load(std::slice::from_ref(item))?;

        Ok(loaded
            .into_iter()
            .next()
            .expect("a batch of one item returns exactly one item"))
    }

    /// Starts reading every plane of every item back from the device.
    ///
    /// Nothing is converted on the way out. A frame already sits in the format its consumer
    /// negotiated for, so this is a straight readback. The map that will later cut the read apart
    /// is built first, and handles are pulled from `items` by walking that same map, so the two
    /// can never disagree on where a plane belongs. The items go to the batch, which holds them
    /// until the read lands, so nothing the read is about to copy from is recycled under it.
    pub fn unload(&self, items: Vec<StreamItem>) -> PendingBatch {
        let map = PlaneMap::for_items(&items);

        let handles = map
            .shapes()
            .iter()
            .map(|shape| {
                items[shape.item]
                    .plane(shape.channel)
                    .expect("the map only names channels the item's format carries")
                    .buffer()
                    .handle()
                    .clone()
            })
            .collect();
        let read = self.pool.read(handles);

        PendingBatch::new(read, map, items)
    }

    /// The batch of one: unloads `item` and hands back the single result.
    ///
    /// This carries no readback logic of its own, so a fix to
    /// [FrameContext::unload](crate::FrameContext::unload) never has a second copy left behind
    /// here to drift from it.
    pub fn unload_one(&self, item: StreamItem) -> PendingBatch {
        self.unload(vec![item])
    }

    /// Uploads a host item as a device item in the same place in the stream.
    pub fn load_host(&self, item: &HostStreamItem) -> Result<StreamItem, RuntimeError> {
        let meta = *item.meta();

        match item.format() {
            StreamFormat::Frames(format) => {
                let frame = self.pool.load_frame_from(0, &format, item, meta)?;

                Ok(StreamItem::Frame(frame))
            },
            StreamFormat::Mask(format) => {
                let mask = self.pool.load_mask_from(0, &format, item, meta)?;

                Ok(StreamItem::Mask(mask))
            },
        }
    }

    /// Reads a device item back as a host item in the same place in the stream, waiting for it to
    /// land.
    pub fn unload_host(&self, item: StreamItem) -> Result<HostStreamItem, RuntimeError> {
        let landed = self
            .unload_one(item)
            .wait()?
            .into_iter()
            .next()
            .expect("a batch of one item lands as exactly one item");

        let meta = *landed.meta();
        let (format, planes) = landed.into_planes().into_parts();

        Ok(HostPool::adopt_item(format, planes, meta))
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

/// A context has to move into whichever worker thread runs its stage.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<FrameContext>();
};

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::testing::CountingAllocator;
    use crate::{
        Accelerator,
        ColourInfo,
        FrameBudget,
        FrameLayout,
        HostStreamItem,
        MaskFormat,
        SampleFormat,
        SceneScore,
        Subsampling,
        default_accelerators,
    };

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

    fn frames_format() -> StreamFormat {
        StreamFormat::Frames(frame_format())
    }

    fn mask_format() -> MaskFormat {
        MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
        }
    }

    fn masks_format() -> StreamFormat {
        StreamFormat::Mask(mask_format())
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

    #[test]
    fn loading_a_batch_uploads_every_plane_of_every_item() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator.clone(), capabilities());
        let frame_planes = HostPlanes::zeroed(&frames_format());
        let mask_planes = HostPlanes::zeroed(&masks_format());
        let items = vec![
            LoadItem::new(&frame_planes, FrameMeta::at(0)),
            LoadItem::new(&mask_planes, FrameMeta::at(1)),
        ];

        let loaded = context.load(&items).expect("should load");

        assert_eq!(loaded.len(), 2);
        assert_eq!(
            allocator.count(),
            4,
            "three planes for the frame, one for the mask"
        );
    }

    #[test]
    fn load_one_is_the_batch_of_one() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator.clone(), capabilities());
        let planes = HostPlanes::zeroed(&frames_format());
        let item = LoadItem {
            planes: &planes,
            meta: FrameMeta::at(4),
        };

        let one = context.load_one(&item).expect("should load");

        assert_eq!(
            allocator.count(),
            3,
            "one reservation per plane, the same as the batch"
        );
        assert_eq!(one.as_frame().expect("a frame went in").meta().index, 4);
    }

    /// The chroma planes are the same size at 4:2:0, so only which bytes went where tells a
    /// correct load from one that swapped U and V.
    #[test]
    fn loading_a_frame_uploads_each_plane_of_its_own_bytes() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator.clone(), capabilities());
        let mut planes = HostPlanes::zeroed(&frames_format());
        planes.plane_mut(Channel::U).expect("carries chroma").fill(11);
        planes.plane_mut(Channel::V).expect("carries chroma").fill(22);
        let item = LoadItem::new(&planes, FrameMeta::at(0));

        context.load_one(&item).expect("should load");

        let uploads = allocator.uploads();
        assert!(
            uploads[1].iter().all(|&byte| byte == 11),
            "the U plane did not carry the bytes given for U"
        );
        assert!(
            uploads[2].iter().all(|&byte| byte == 22),
            "the V plane did not carry the bytes given for V"
        );
    }

    /// Which read position a plane came from, given every buffer is filled with its own position.
    fn marker_of(plane: &[u8]) -> u8 {
        let marker = plane[0];
        assert!(
            plane.iter().all(|byte| *byte == marker),
            "the plane holds bytes from more than one buffer"
        );

        marker
    }

    #[test]
    fn unloading_a_batch_returns_one_set_of_planes_per_item() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let items = vec![
            StreamItem::Frame(context.alloc_frame(&frame_format(), FrameMeta::at(0))),
            StreamItem::Mask(context.alloc_mask(&mask_format(), FrameMeta::at(1))),
        ];

        let landed = context.unload(items).wait().expect("the fake lands");

        assert_eq!(landed.len(), 2);
        assert_eq!(landed[0].len(), 3);
        assert_eq!(landed[1].len(), 1);
    }

    #[test]
    fn unload_one_is_the_batch_of_one() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let mask = context.alloc_mask(&mask_format(), FrameMeta::at(2));

        let landed = context
            .unload_one(StreamItem::Mask(mask))
            .wait()
            .expect("the fake lands");

        assert_eq!(landed.len(), 1);
        assert_eq!(landed[0].len(), 1);
    }

    /// The mask is sized to match the frame's chroma planes, so three of the four buffers this
    /// reads back are the same size. Only which bytes came back for which plane can tell a read
    /// built from the map's own order from one that flattened items by a second, independent walk
    /// and lost track of it.
    #[test]
    fn unloading_pulls_each_planes_handle_from_the_map_that_will_cut_it_apart() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let smaller_mask = MaskFormat {
            width: 32,
            height: 32,
            sample: SampleFormat::U8,
        };
        let items = vec![
            StreamItem::Frame(context.alloc_frame(&frame_format(), FrameMeta::at(0))),
            StreamItem::Mask(context.alloc_mask(&smaller_mask, FrameMeta::at(1))),
        ];

        let landed = context.unload(items).wait().expect("the fake lands");

        let luma = landed[0].plane(Channel::Y).expect("a yuv frame carries luma");
        let blue = landed[0].plane(Channel::U).expect("a yuv frame carries U");
        let red = landed[0].plane(Channel::V).expect("a yuv frame carries V");
        let mask_plane = landed[1].plane(Channel::Y).expect("a mask carries one plane");

        assert_eq!(marker_of(luma), 0);
        assert_eq!(marker_of(blue), 1);
        assert_eq!(marker_of(red), 2, "V holds the buffer read for U");
        assert_eq!(
            marker_of(mask_plane),
            3,
            "the mask holds a buffer read for the frame"
        );
    }

    /// The pool hands a released buffer straight to the next caller wanting that size, so a batch
    /// that let go of its items would have the next frame reading over the memory it is about to
    /// copy from.
    #[test]
    fn a_batch_in_flight_holds_on_to_the_memory_it_is_going_to_read() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator.clone(), capabilities());
        let frame = context.alloc_frame(&frame_format(), FrameMeta::at(0));

        let batch = context.unload_one(StreamItem::Frame(frame));
        let _next = context.alloc_frame(&frame_format(), FrameMeta::at(1));

        assert_eq!(
            allocator.count(),
            6,
            "the frame being read back was recycled into the next frame"
        );
        drop(batch);
    }

    /// A sink needs each frame's place in the stream to hand a muxer a timestamp, so the batch
    /// carries it rather than leaving the caller a list to keep in step.
    #[test]
    fn a_landed_batch_carries_the_metadata_of_every_item_it_read() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let stamped = FrameMeta {
            index: 12,
            chunk: 0,
            timestamp: Some(Duration::from_millis(500)),
            scene_score: Some(SceneScore::new(2.5)),
        };
        let items = vec![
            StreamItem::Frame(context.alloc_frame(&frame_format(), stamped)),
            StreamItem::Mask(context.alloc_mask(&mask_format(), FrameMeta::at(13))),
        ];

        let landed = context.unload(items).wait().expect("the fake lands");

        assert_eq!(landed[0].meta(), &stamped);
        assert_eq!(landed[1].meta().index, 13);
    }

    // A `HostPlanes` that disagrees with the format it loads as is no longer representable here:
    // `LoadItem` takes its format from the planes, so there is nothing left to disagree with.
    // `BufferPool::load_frame` and `load_mask` still take format and planes as separate
    // arguments, so the mismatch this used to cover is tested directly against them instead, in
    // `memory::pool::tests`.

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

    #[test]
    fn a_host_frame_from_a_budgeted_context_is_charged() {
        let budget = FrameBudget::new(1 << 20);
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities()).with_budget(budget.clone());

        let frame = context.alloc_host_frame(&frame_format(), FrameMeta::at(0));

        assert_eq!(budget.outstanding(), frame_format().total_bytes());
        drop(frame);
        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn a_host_frame_uploads_each_plane_in_channel_order_and_keeps_its_place() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator.clone(), capabilities());
        let stamped = FrameMeta {
            index: 12,
            chunk: 3,
            timestamp: Some(Duration::from_millis(40)),
            scene_score: Some(SceneScore::new(2.5)),
        };

        let mut frame = context.alloc_host_frame(&frame_format(), stamped);
        for (value, channel) in [Channel::Y, Channel::U, Channel::V].into_iter().enumerate() {
            frame.plane_mut(channel).expect("carried").fill(value as u8 + 1);
        }

        let loaded = context
            .load_host(&HostStreamItem::Frame(frame.freeze()))
            .expect("the fake uploads");

        assert_eq!(loaded.meta(), &stamped);
        assert_eq!(loaded.format(), StreamFormat::Frames(frame_format()));
        let uploads = allocator.uploads();
        assert_eq!(uploads.len(), 3);
        assert!(uploads[0].iter().all(|&byte| byte == 1));
        assert!(uploads[1].iter().all(|&byte| byte == 2));
        assert!(uploads[2].iter().all(|&byte| byte == 3));
    }

    #[test]
    fn a_host_mask_uploads_as_a_device_mask() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let mask = context.alloc_host_mask(&mask_format(), FrameMeta::at(5)).freeze();

        let loaded = context
            .load_host(&HostStreamItem::Mask(mask))
            .expect("the fake uploads");

        assert!(loaded.is_mask());
        assert_eq!(loaded.meta().index, 5);
    }

    #[test]
    fn an_unloaded_frame_lands_as_a_host_frame_in_the_same_place() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let stamped = FrameMeta::at(8).in_chunk(2);
        let frame = context.alloc_frame(&frame_format(), stamped);

        let landed = context
            .unload_host(StreamItem::Frame(frame))
            .expect("the fake lands");

        assert_eq!(landed.meta(), &stamped);
        assert_eq!(landed.format(), StreamFormat::Frames(frame_format()));
        let frame = landed.into_frame().expect("a frame");
        assert!(
            frame
                .plane(Channel::Y)
                .expect("luma")
                .bytes()
                .iter()
                .all(|&byte| byte == 0)
        );
        assert!(
            frame
                .plane(Channel::U)
                .expect("chroma")
                .bytes()
                .iter()
                .all(|&byte| byte == 1)
        );
        assert!(
            frame
                .plane(Channel::V)
                .expect("chroma")
                .bytes()
                .iter()
                .all(|&byte| byte == 2)
        );
    }

    #[test]
    fn unloaded_frames_free_their_memory_rather_than_filling_the_host_pool() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());

        for index in 0..100 {
            let frame = context.alloc_frame(&frame_format(), FrameMeta::at(index));
            let landed = context
                .unload_host(StreamItem::Frame(frame))
                .expect("the fake lands");
            drop(landed);
        }

        assert_eq!(context.host.free_buffers(), 0);
    }

    #[test]
    fn a_reserved_host_frame_is_still_recycled_after_unloads() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let frame = context.alloc_frame(&frame_format(), FrameMeta::at(0));
        let landed = context
            .unload_host(StreamItem::Frame(frame))
            .expect("the fake lands");
        drop(landed);

        let reserved = context.alloc_host_frame(&frame_format(), FrameMeta::at(1));
        let address = reserved.plane(Channel::Y).expect("luma").bytes().as_ptr();
        drop(reserved);

        let recycled = context.alloc_host_frame(&frame_format(), FrameMeta::at(2));
        let recycled_address = recycled.plane(Channel::Y).expect("luma").bytes().as_ptr();
        assert_eq!(recycled_address, address);
        assert_eq!(context.host.free_buffers(), 0);
    }

    #[test]
    fn an_unloaded_mask_lands_as_a_host_mask() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let context = FrameContext::for_testing(allocator, capabilities());
        let mask = context.alloc_mask(&mask_format(), FrameMeta::at(1));

        let landed = context
            .unload_host(StreamItem::Mask(mask))
            .expect("the fake lands");

        assert!(landed.is_mask());
        assert_eq!(landed.meta().index, 1);
    }
}
