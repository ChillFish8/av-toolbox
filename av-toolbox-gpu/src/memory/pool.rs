use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cubecl::server::Handle;

use crate::frame::gpu::Recycle;
use crate::frame::{Frame, FrameFormat, FrameMeta, Mask, MaskFormat, Plane, PlaneBuffer};
use crate::memory::allocator::DeviceAllocator;
use crate::memory::ring::{Ring, RingGeometry, RingSpec};

/// Device buffers kept for reuse, keyed by size.
///
/// CubeCL pools underneath this, so a reservation that misses here is not expensive. What the pool
/// adds is that a buffer released by one frame is handed straight to the next frame wanting that
/// size, which is what keeps a steady-state pipeline from asking at all.
///
/// A pool's buffers are reserved and released from the one thread driving a device's work. A
/// recycled handle carries the stream of whichever thread first allocated it, and CubeCL decides
/// cross-stream synchronisation from that.
#[derive(Debug, Clone)]
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

#[derive(Debug)]
struct PoolInner {
    allocator: Arc<dyn DeviceAllocator>,
    free: Mutex<HashMap<usize, Vec<Handle>>>,
}

impl BufferPool {
    pub fn new(allocator: Arc<dyn DeviceAllocator>) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                allocator,
                free: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// A buffer of exactly `bytes`, reused if one is free and reserved otherwise.
    ///
    /// A reused buffer still holds whatever was last written into it, so a caller that does not
    /// write every byte has to fill the rest itself.
    pub fn reserve(&self, bytes: usize) -> PlaneBuffer {
        let recycled = self
            .inner
            .free
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get_mut(&bytes)
            .and_then(Vec::pop);

        let handle = match recycled {
            Some(handle) => handle,
            None => self.inner.allocator.empty(bytes),
        };

        let recyclable: Arc<dyn Recycle> = self.inner.clone();
        let recycle = Arc::downgrade(&recyclable);

        PlaneBuffer::pooled(handle, bytes, recycle)
    }

    /// Transient memory for one stage's working set.
    ///
    /// It comes from the same free lists as everything else, so scratch released by one stage is
    /// what the next stage's output is built from. It arrives holding whatever that stage left in
    /// it, so read nothing from it that this stage has not written.
    pub fn scratch(&self, bytes: usize) -> PlaneBuffer {
        self.reserve(bytes)
    }

    /// The byte alignment every buffer view has to start on.
    pub fn alignment(&self) -> usize {
        self.inner.allocator.alignment()
    }

    /// A ring of `spec.slots` frames in one allocation.
    pub fn ring(&self, spec: RingSpec) -> Ring {
        let geometry = RingGeometry::new(&spec, self.alignment());
        let buffer = self.reserve(geometry.total_bytes());

        Ring::new(buffer, geometry, spec.format)
    }

    /// A frame of `format`, one buffer per channel.
    ///
    /// Its planes are recycled memory holding an earlier frame's samples, so a stage that writes
    /// only some of them has to copy the rest from its input.
    pub fn alloc_frame(&self, format: &FrameFormat, meta: FrameMeta) -> Frame {
        let planes = format
            .channels()
            .iter()
            .zip(format.plane_sizes())
            .map(|(channel, bytes)| {
                let (width, height) = format
                    .channel_dimensions(*channel)
                    .expect("the layout carries every channel it lists");
                let buffer = self.reserve(bytes);

                Plane::new(buffer, width, height)
            })
            .collect();

        Frame::new(*format, planes, meta)
    }

    /// A frame shaped like `frame`, carrying `meta`.
    ///
    /// Its planes are recycled memory holding an earlier frame's samples, so a stage that writes
    /// only some of them has to copy the rest from its input.
    pub fn alloc_like(&self, frame: &Frame, meta: FrameMeta) -> Frame {
        self.alloc_frame(frame.format(), meta)
    }

    /// A mask of `format`, which is one buffer.
    ///
    /// That buffer is recycled memory holding an earlier mask's samples, so a stage that writes
    /// only part of it has to fill the rest.
    pub fn alloc_mask(&self, format: &MaskFormat, meta: FrameMeta) -> Mask {
        let buffer = self.reserve(format.bytes());
        let plane = Plane::new(buffer, format.width, format.height);

        Mask::new(*format, plane, meta)
    }

    /// Releases every buffer being held for reuse.
    ///
    /// A pipeline that changes format stops asking for the old sizes, and nothing else would ever
    /// let go of them.
    pub fn clear(&self) {
        self.inner
            .free
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }
}

impl Recycle for PoolInner {
    fn recycle(&self, bytes: usize, handle: Handle) {
        self.free
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(bytes)
            .or_default()
            .push(handle);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::testing::CountingAllocator;
    use crate::{
        Channel,
        ColourInfo,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        MaskFormat,
        SampleFormat,
        Subsampling,
    };

    fn pool(allocator: Arc<CountingAllocator>) -> BufferPool {
        BufferPool::new(allocator)
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
    fn a_cold_pool_reserves_what_it_is_asked_for() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        let buffer = pool.reserve(128);

        assert_eq!(buffer.bytes(), 128);
        assert_eq!(allocator.requests(), vec![128]);
    }

    /// The whole point of the pool. A buffer that comes back is handed out again rather than
    /// reserved a second time.
    #[test]
    fn a_dropped_buffer_is_handed_out_again() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        drop(pool.reserve(128));
        let second = pool.reserve(128);

        assert_eq!(second.bytes(), 128);
        assert_eq!(allocator.count(), 1, "the pool reserved twice for one size");
    }

    #[test]
    fn buffers_of_different_sizes_do_not_satisfy_each_other() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        drop(pool.reserve(128));
        let _other = pool.reserve(256);

        assert_eq!(allocator.requests(), vec![128, 256]);
    }

    #[test]
    fn two_buffers_of_one_size_held_at_once_are_both_reserved() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        let _first = pool.reserve(128);
        let _second = pool.reserve(128);

        assert_eq!(allocator.count(), 2);
    }

    #[test]
    fn a_buffer_outliving_its_pool_does_not_panic() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let buffer = pool(allocator).reserve(128);

        drop(buffer);
    }

    #[test]
    fn scratch_comes_from_the_same_free_lists() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        drop(pool.scratch(512));
        let _reused = pool.reserve(512);

        assert_eq!(allocator.count(), 1);
    }

    #[test]
    fn a_frame_gets_one_buffer_per_channel_at_the_right_size() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        let frame = pool.alloc_frame(&frame_format(), FrameMeta::at(0));

        assert_eq!(allocator.requests(), vec![64 * 64, 32 * 32, 32 * 32]);
        assert_eq!(
            frame.plane(Channel::Y).expect("carries luma").dimensions(),
            (64, 64)
        );
        assert_eq!(
            frame.plane(Channel::U).expect("carries chroma").dimensions(),
            (32, 32)
        );
    }

    /// A pipeline running frame after frame is the case the pool exists for. Once the first frame
    /// is released, later frames cost no reservations at all.
    #[test]
    fn a_released_frame_pays_for_the_next_one() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        drop(pool.alloc_frame(&frame_format(), FrameMeta::at(0)));
        let _second = pool.alloc_frame(&frame_format(), FrameMeta::at(1));

        assert_eq!(
            allocator.count(),
            3,
            "the second frame reserved rather than reused"
        );
    }

    #[test]
    fn allocating_like_a_frame_matches_its_format() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator);

        let first = pool.alloc_frame(&frame_format(), FrameMeta::at(0));
        let second = pool.alloc_like(&first, FrameMeta::at(9));

        assert_eq!(second.format(), first.format());
        assert_eq!(second.meta().index, 9);
    }

    #[test]
    fn a_fresh_frame_is_writable_and_a_shared_one_is_not() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator);

        let frame = pool.alloc_frame(&frame_format(), FrameMeta::at(0));
        let shared = frame.clone();

        let returned = frame.try_into_mut().expect_err("a shared frame is not writable");
        drop(shared);

        assert!(returned.try_into_mut().is_ok());
    }

    #[test]
    fn a_mask_gets_one_buffer() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        let format = MaskFormat {
            width: 64,
            height: 32,
            sample: SampleFormat::U8,
        };
        let mask = pool.alloc_mask(&format, FrameMeta::at(0));

        assert_eq!(allocator.requests(), vec![64 * 32]);
        assert_eq!(mask.plane().dimensions(), (64, 32));
    }

    #[test]
    fn clearing_releases_what_was_held_for_reuse() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());

        drop(pool.reserve(128));
        pool.clear();
        let _fresh = pool.reserve(128);

        assert_eq!(
            allocator.count(),
            2,
            "a cleared pool handed back a released buffer"
        );
    }

    #[test]
    fn a_cloned_pool_shares_its_free_lists() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());
        let other = pool.clone();

        drop(pool.reserve(128));
        let _reused = other.reserve(128);

        assert_eq!(
            allocator.count(),
            1,
            "a cloned pool did not see the released buffer"
        );
    }
}
