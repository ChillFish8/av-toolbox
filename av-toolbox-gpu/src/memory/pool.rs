use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cubecl::server::Handle;

use crate::frame::gpu::Recycle;
use crate::frame::{
    Channel,
    Frame,
    FrameFormat,
    FrameMeta,
    HostPlanes,
    HostStreamItem,
    Mask,
    MaskFormat,
    Plane,
    PlaneBuffer,
};
use crate::memory::allocator::{DeviceAllocator, DeviceRead, RuntimeError};
use crate::memory::budget::{FrameBudget, Room};
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
    /// Charged by every frame shaped allocation: `alloc_frame`, `alloc_like`, `alloc_mask`,
    /// `load_frame` and `load_mask`. Rings and scratch never touch it.
    budget: Option<Arc<FrameBudget>>,
}

/// Anything host side a plane can be uploaded from.
pub(crate) trait HostPlaneSource {
    fn plane_bytes(&self, channel: Channel) -> Option<&[u8]>;
}

impl HostPlaneSource for HostPlanes {
    fn plane_bytes(&self, channel: Channel) -> Option<&[u8]> {
        self.plane(channel)
    }
}

impl HostPlaneSource for HostStreamItem {
    fn plane_bytes(&self, channel: Channel) -> Option<&[u8]> {
        HostStreamItem::plane_bytes(self, channel)
    }
}

/// One plane of one item on its way to the device.
struct PlaneUpload {
    item: usize,
    channel: Channel,
    width: u32,
    height: u32,
    expected: usize,
    /// Whether this plane's bytes are charged to the pool's budget.
    counted: bool,
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
            budget: None,
        }
    }

    /// This pool, charging every frame shaped allocation to `budget`.
    pub(crate) fn with_budget(mut self, budget: Arc<FrameBudget>) -> Self {
        self.budget = Some(budget);
        self
    }

    /// The budget this pool charges, if it has one.
    pub(crate) fn budget(&self) -> Option<&Arc<FrameBudget>> {
        self.budget.as_ref()
    }

    /// A buffer of exactly `bytes`, reused if one is free and reserved otherwise.
    ///
    /// A reused buffer still holds whatever was last written into it, so a caller that does not
    /// write every byte has to fill the rest itself.
    pub fn reserve(&self, bytes: usize) -> PlaneBuffer {
        self.reserve_impl(bytes, false)
    }

    /// Reserves `bytes`, charging it to the budget when `counted` is set.
    ///
    /// Every allocation in the pool passes through here, but only a frame, mask or like-frame
    /// allocation counts. A ring or scratch buffer is held for as long as the stage that owns it
    /// lives rather than returning between frames, so charging it here would never be credited
    /// back.
    ///
    /// This never waits on its own. A frame is several planes charged one call each, and a wait
    /// per plane would block the second plane on room only the first plane's own caller could
    /// ever free, since nothing else is holding it yet. Waiting once for a whole item's bytes,
    /// before any of its planes are reserved, is the caller's job: see [BufferPool::alloc_frame],
    /// [BufferPool::alloc_mask], [BufferPool::load_frame] and [BufferPool::load_mask].
    fn reserve_impl(&self, bytes: usize, counted: bool) -> PlaneBuffer {
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

        let budget = if counted { self.budget.clone() } else { None };
        if let Some(budget) = &budget {
            budget.charge(bytes);
        }

        self.recycle_handle(handle, bytes, budget)
    }

    /// Waits until `bytes` fits under this pool's budget, or returns immediately when it has
    /// none.
    ///
    /// Answers whether what follows should be charged. A released budget is one whose run is
    /// over, and one that gave up waiting is a run about to fail, so either way the allocation it
    /// just stopped waiting for goes through uncounted rather than charging a total nothing will
    /// ever credit back.
    fn wait_for_room(&self, bytes: usize) -> bool {
        let Some(budget) = &self.budget else {
            return false;
        };

        matches!(budget.wait_for_room(bytes), Room::Available)
    }

    /// Wraps `handle` so it goes back to this pool's free lists when its last holder drops it,
    /// crediting `budget` back at the same time when there is one.
    fn recycle_handle(&self, handle: Handle, bytes: usize, budget: Option<Arc<FrameBudget>>) -> PlaneBuffer {
        let recyclable: Arc<dyn Recycle> = self.inner.clone();
        let recycle = Arc::downgrade(&recyclable);
        let budget = budget.as_ref().map(Arc::downgrade);

        PlaneBuffer::pooled(handle, bytes, recycle, budget)
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

    /// Starts reading `handles` back from the device, in the order given.
    pub(crate) fn read(&self, handles: Vec<Handle>) -> DeviceRead {
        self.inner.allocator.read(handles)
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
    /// only some of them has to copy the rest from its input. Waits once for the whole frame's
    /// bytes before reserving any of its planes, rather than once per plane, so a budget smaller
    /// than one plane's share of a frame cannot leave an earlier plane of this same frame the
    /// only thing that could ever free the room a later one is waiting for.
    pub fn alloc_frame(&self, format: &FrameFormat, meta: FrameMeta) -> Frame {
        let counted = self.wait_for_room(format.total_bytes());

        let planes = format
            .channels()
            .iter()
            .zip(format.plane_sizes())
            .map(|(channel, bytes)| {
                let (width, height) = format
                    .channel_dimensions(*channel)
                    .expect("the layout carries every channel it lists");
                let buffer = self.reserve_impl(bytes, counted);

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
        let counted = self.wait_for_room(format.bytes());

        let buffer = self.reserve_impl(format.bytes(), counted);
        let plane = Plane::new(buffer, format.width, format.height);

        Mask::new(*format, plane, meta)
    }

    /// Uploads `planes` as a frame of `format`, carrying `meta`.
    pub fn load_frame(
        &self,
        item: usize,
        format: &FrameFormat,
        planes: &HostPlanes,
        meta: FrameMeta,
    ) -> Result<Frame, RuntimeError> {
        self.load_frame_from(item, format, planes, meta)
    }

    /// Uploads `planes` as a frame of `format`, carrying `meta`, from any host side source of
    /// plane bytes.
    ///
    /// Channels are walked in the format's own order, the one
    /// [FrameFormat::plane_sizes](crate::FrameFormat::plane_sizes) uses, so a plane lands against
    /// the channel it belongs to rather than wherever `planes` happened to carry it. Waits once
    /// for the whole frame's bytes first, for the same reason [BufferPool::alloc_frame] does.
    pub(crate) fn load_frame_from(
        &self,
        item: usize,
        format: &FrameFormat,
        planes: &dyn HostPlaneSource,
        meta: FrameMeta,
    ) -> Result<Frame, RuntimeError> {
        let counted = self.wait_for_room(format.total_bytes());

        let mut loaded = Vec::with_capacity(format.channels().len());
        for channel in format.channels() {
            let (width, height) = format
                .channel_dimensions(*channel)
                .expect("the layout carries every channel it lists");
            let expected = format
                .channel_bytes(*channel)
                .expect("the layout carries every channel it lists");

            let upload = PlaneUpload {
                item,
                channel: *channel,
                width,
                height,
                expected,
                counted,
            };

            let plane = self.load_plane(upload, planes)?;
            loaded.push(plane);
        }

        Ok(Frame::new(*format, loaded, meta))
    }

    /// Uploads `planes` as a mask of `format`, carrying `meta`.
    pub fn load_mask(
        &self,
        item: usize,
        format: &MaskFormat,
        planes: &HostPlanes,
        meta: FrameMeta,
    ) -> Result<Mask, RuntimeError> {
        self.load_mask_from(item, format, planes, meta)
    }

    /// Uploads `planes` as a mask of `format`, carrying `meta`, from any host side source of plane
    /// bytes.
    pub(crate) fn load_mask_from(
        &self,
        item: usize,
        format: &MaskFormat,
        planes: &dyn HostPlaneSource,
        meta: FrameMeta,
    ) -> Result<Mask, RuntimeError> {
        let counted = self.wait_for_room(format.bytes());

        let upload = PlaneUpload {
            item,
            channel: Channel::Y,
            width: format.width,
            height: format.height,
            expected: format.bytes(),
            counted,
        };

        let plane = self.load_plane(upload, planes)?;

        Ok(Mask::new(*format, plane, meta))
    }

    /// The plane `upload` names, uploaded from `planes`.
    ///
    /// `planes` is caller data, so a missing channel or a byte count that does not match the one
    /// expected is reported here rather than left to upload a short buffer, which would leave
    /// the plane holding whatever memory the device handed back.
    fn load_plane(&self, upload: PlaneUpload, planes: &dyn HostPlaneSource) -> Result<Plane, RuntimeError> {
        let PlaneUpload {
            item,
            channel,
            width,
            height,
            expected,
            counted,
        } = upload;

        let bytes = planes
            .plane_bytes(channel)
            .ok_or(RuntimeError::HostPlaneMissing {
                item,
                channel,
                width,
                height,
                expected,
            })?;

        if bytes.len() != expected {
            return Err(RuntimeError::HostPlaneSize {
                item,
                channel,
                width,
                height,
                expected,
                given: bytes.len(),
            });
        }

        let handle = self.inner.allocator.upload(bytes);
        let budget = if counted { self.budget.clone() } else { None };
        if let Some(budget) = &budget {
            budget.charge(expected);
        }

        let buffer = self.recycle_handle(handle, expected, budget);

        Ok(Plane::new(buffer, width, height))
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
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::thread;
    use std::time::Duration;

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
        StreamFormat,
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
            frame_rate: None,
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
    fn loading_a_frame_uploads_one_plane_per_channel() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator.clone());
        let format = frame_format();
        let planes = HostPlanes::zeroed(&StreamFormat::Frames(format));

        let frame = pool
            .load_frame(0, &format, &planes, FrameMeta::at(3))
            .expect("the planes match the format");

        assert_eq!(allocator.count(), 3, "one upload per plane of a yuv frame");
        assert_eq!(frame.meta().index, 3);
    }

    /// `load_frame` and `load_mask` still take their format as a plain argument, independent of
    /// what `planes` itself declares, unlike `LoadItem`, which takes its format from the planes
    /// alone. Calling the pool directly like this is what keeps `HostPlaneMissing` reachable.
    #[test]
    fn loading_a_masks_planes_as_a_frame_reports_the_missing_chroma() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator);
        let mask_format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
        };
        let planes = HostPlanes::zeroed(&StreamFormat::Mask(mask_format));

        let error = pool
            .load_frame(0, &frame_format(), &planes, FrameMeta::at(0))
            .expect_err("a mask has no chroma planes to load as a frame");

        assert!(error.to_string().contains('U'), "{error}");
    }

    /// Calling the pool directly like this, rather than through `FrameContext::load`, is what
    /// keeps `HostPlaneSize` reachable now that `LoadItem` can no longer pair a `HostPlanes` with
    /// a format it disagrees with.
    #[test]
    fn loading_a_short_host_plane_reports_a_mismatch_instead_of_uploading_it() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = pool(allocator);
        let short = MaskFormat {
            width: 8,
            height: 8,
            sample: SampleFormat::U8,
        };
        let target = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
        };
        let planes = HostPlanes::zeroed(&StreamFormat::Mask(short));

        let error = pool
            .load_mask(0, &target, &planes, FrameMeta::at(0))
            .expect_err("a short host plane should not load");

        assert!(error.to_string().contains("needs 4096 bytes"), "{error}");
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

    /// The wait belongs immediately before the charge, not merely alongside it, so an
    /// allocation that would take the budget past its limit genuinely blocks until something
    /// credits room back, rather than charging past the limit and moving on.
    #[test]
    fn an_allocation_past_the_limit_waits_for_a_dropped_frame_to_credit_room() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
        };
        let budget = FrameBudget::new(format.bytes());
        let pool = pool(allocator).with_budget(budget);

        let first = pool.alloc_mask(&format, FrameMeta::at(0));

        let (result_sender, result_receiver) = mpsc::channel();
        let waiting_pool = pool.clone();
        thread::spawn(move || {
            let second = waiting_pool.alloc_mask(&format, FrameMeta::at(1));
            result_sender.send(second).ok();
        });

        // A generous grace period for the spawned thread to run. A blocked allocation can
        // never answer inside it, since nothing has credited the budget back yet, so this
        // cannot flake a passing run. It only ever catches a wait that was skipped.
        assert!(
            matches!(
                result_receiver.recv_timeout(Duration::from_millis(200)),
                Err(RecvTimeoutError::Timeout)
            ),
            "the second allocation completed before the budget had room for it"
        );

        drop(first);

        result_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("the allocation should complete once the first frame is dropped");
    }

    /// A released budget is a run that is over, so the frame it just let through is not charged
    /// to a total nothing is ever going to credit back.
    #[test]
    fn a_released_budget_stops_charging_what_it_let_through() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let budget = FrameBudget::new(1 << 20);
        let pool = pool(allocator).with_budget(budget.clone());
        budget.release();

        let frame = pool.alloc_frame(&frame_format(), FrameMeta::at(0));

        assert_eq!(budget.outstanding(), 0);
        drop(frame);
    }

    /// A budget too small to ever fit a whole frame is a misconfiguration, not a reason to
    /// refuse the frame outright: `FrameBudget::wait_for_room` already always answers instantly
    /// when nothing is outstanding, so this proceeds one frame at a time rather than deadlocking
    /// or failing. Waiting once per frame is what makes that hold: a wait per plane would leave
    /// a frame's own first plane, charged and held by this same call, as the only thing that
    /// could ever free the room its second plane is waiting for.
    #[test]
    fn a_budget_smaller_than_one_frame_still_allocates_rather_than_hanging() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let budget = FrameBudget::new(1);
        let pool = pool(allocator).with_budget(budget);

        let (result_sender, result_receiver) = mpsc::channel();
        thread::spawn(move || {
            let first = pool.alloc_frame(&frame_format(), FrameMeta::at(0));
            drop(first);
            let second = pool.alloc_frame(&frame_format(), FrameMeta::at(1));
            result_sender.send(second).ok();
        });

        result_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("a budget smaller than one frame should still allocate rather than hang");
    }
}
