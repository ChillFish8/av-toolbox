use std::fmt;
use std::task::{Context, Poll, Waker};

use cubecl::bytes::Bytes;

use crate::frame::{Channel, HostItem, HostPlanes, StreamFormat, StreamItem};
use crate::memory::{DeviceRead, RuntimeError};

/// Where one plane of one item sits in a flattened batch.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct PlaneShape {
    pub(crate) item: usize,
    pub(crate) channel: Channel,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes: usize,
}

/// Every plane of every item of a batch, in the order they are read back.
///
/// A batch reads all of its planes in one call, so this order is the order the buffers come back
/// in and is what puts each one back with the item and channel it came from.
#[derive(Debug, Clone)]
pub(crate) struct PlaneMap {
    formats: Vec<StreamFormat>,
    shapes: Vec<PlaneShape>,
}

impl PlaneMap {
    pub(crate) fn for_items(items: &[StreamItem]) -> Self {
        let mut formats = Vec::with_capacity(items.len());
        let mut shapes = Vec::new();

        for (index, item) in items.iter().enumerate() {
            let format = item.format();
            push_shapes(index, &format, &mut shapes);

            formats.push(format);
        }

        Self { formats, shapes }
    }

    pub(crate) fn shapes(&self) -> &[PlaneShape] {
        &self.shapes
    }

    pub(crate) fn cut(
        &self,
        buffers: Vec<Bytes>,
        source: &[StreamItem],
    ) -> Result<Vec<HostItem>, RuntimeError> {
        let mut items = Vec::new();
        self.cut_into(buffers, source, &mut items)?;

        Ok(items)
    }

    /// Cuts `buffers` into one item per set of planes, reusing what `items` already holds.
    ///
    /// Each landed item takes its place in the stream from the device item it was read from, so
    /// `source` has to be the items this map was built for. The buffers are checked against the
    /// map before anything is written, so `items` is left as it was when a read comes back the
    /// wrong shape.
    pub(crate) fn cut_into(
        &self,
        buffers: Vec<Bytes>,
        source: &[StreamItem],
        items: &mut Vec<HostItem>,
    ) -> Result<(), RuntimeError> {
        self.check(&buffers)?;
        self.reshape(source, items);

        for (shape, buffer) in self.shapes.iter().zip(&buffers) {
            let plane = items[shape.item]
                .plane_mut(shape.channel)
                .expect("the map only names channels the item's format carries");

            plane.copy_from_slice(buffer);
        }

        Ok(())
    }

    /// Checks `buffers` is one buffer per plane, each the size the map gave it.
    fn check(&self, buffers: &[Bytes]) -> Result<(), RuntimeError> {
        if buffers.len() != self.shapes.len() {
            return Err(RuntimeError::ReadShape {
                expected: self.shapes.len(),
                returned: buffers.len(),
            });
        }

        for (shape, buffer) in self.shapes.iter().zip(buffers) {
            if buffer.len() != shape.bytes {
                return Err(RuntimeError::PlaneSize {
                    item: shape.item,
                    channel: shape.channel,
                    width: shape.width,
                    height: shape.height,
                    expected: shape.bytes,
                    returned: buffer.len(),
                });
            }
        }

        Ok(())
    }

    /// Sizes `items` to hold this batch, keeping the allocations that already fit.
    fn reshape(&self, source: &[StreamItem], items: &mut Vec<HostItem>) {
        assert_eq!(
            source.len(),
            self.formats.len(),
            "a map cuts a read of the items it was built for",
        );

        items.truncate(self.formats.len());

        let landing = self.formats.iter().zip(source);
        for (item, (format, device_item)) in items.iter_mut().zip(landing.clone()) {
            item.reshape(format, device_item.meta().clone());
        }

        for (format, device_item) in landing.skip(items.len()) {
            items.push(HostItem::new(
                HostPlanes::zeroed(format),
                device_item.meta().clone(),
            ));
        }
    }
}

fn push_shapes(item: usize, format: &StreamFormat, shapes: &mut Vec<PlaneShape>) {
    match format {
        StreamFormat::Frames(frame) => {
            for channel in frame.channels() {
                let (width, height) = frame
                    .channel_dimensions(*channel)
                    .expect("the layout carries every channel it lists");
                let bytes = frame
                    .channel_bytes(*channel)
                    .expect("the layout carries every channel it lists");

                shapes.push(PlaneShape {
                    item,
                    channel: *channel,
                    width,
                    height,
                    bytes,
                });
            }
        },
        StreamFormat::Mask(mask) => shapes.push(PlaneShape {
            item,
            channel: Channel::Y,
            width: mask.width,
            height: mask.height,
            bytes: mask.bytes(),
        }),
    }
}

/// A readback in flight, with the map that cuts it back into items.
///
/// Every plane of every item is read in one call, so the buffers come back flattened and the map
/// is what puts each one back with the item and channel it came from. The device items stay here
/// until the read lands, because their memory goes back to the pool the moment the last holder
/// drops it and the read has recorded nothing until its first poll.
pub struct PendingBatch {
    read: Option<DeviceRead>,
    map: PlaneMap,
    items: Vec<StreamItem>,
    polled: bool,
}

impl PendingBatch {
    /// A batch over `read`, which has to be a read of every plane `map` names, in map order.
    ///
    /// `items` are the items that read names, kept alive here for as long as it is in flight.
    pub(crate) fn new(read: DeviceRead, map: PlaneMap, items: Vec<StreamItem>) -> Self {
        Self {
            read: Some(read),
            map,
            items,
            polled: false,
        }
    }

    /// Blocks until the read lands, then cuts it into one item per set of planes.
    pub fn wait(self) -> Result<Vec<HostItem>, RuntimeError> {
        let mut items = Vec::new();
        self.wait_into(&mut items)?;

        Ok(items)
    }

    /// Waits, reusing `dst`'s allocations rather than making new ones.
    pub fn wait_into(mut self, dst: &mut Vec<HostItem>) -> Result<(), RuntimeError> {
        let read = self.read.take().expect("a batch holds its read until it lands");
        let buffers = cubecl::future::block_on(read)?;

        self.map.cut_into(buffers, &self.items, dst)
    }

    /// Polls the read once, handing the batch back when it has not landed.
    ///
    /// Nothing is registered to wake a caller, so a batch that comes back not ready has to be
    /// polled again by whoever holds it. That batch also owes the device a wait, because this
    /// first poll is what reserves the staging buffer it reads into. Dropping it blocks until the
    /// read lands rather than abandoning it, which on wgpu is the whole of the readback.
    pub fn try_wait(mut self) -> Result<TryWait, RuntimeError> {
        let mut context = Context::from_waker(Waker::noop());

        self.polled = true;
        let read = self.read.as_mut().expect("a batch holds its read until it lands");
        let polled = read.as_mut().poll(&mut context);

        let Poll::Ready(landed) = polled else {
            return Ok(TryWait::NotReady(self));
        };

        self.read = None;
        let buffers = landed?;
        let items = self.map.cut(buffers, &self.items)?;

        Ok(TryWait::Ready(items))
    }
}

impl Drop for PendingBatch {
    fn drop(&mut self) {
        // On wgpu the first poll reserves a staging buffer, records the copy and maps it, and the
        // buffer is unmapped only when the resolved bytes are dropped. CubeCL does all of that as
        // soon as a read is started rather than on a poll, and what holds it off until the first
        // one is the wrapper in `ClientAllocator::read`. That wrapper is the only reason `polled`
        // answers exactly whether a staging buffer is outstanding. Abandoning a polled but
        // unlanded read hands a still-mapped buffer back to the staging pool, and the next submit
        // touching it fails with "buffer is still mapped" on the device thread, taking every later
        // call on that device with it. So a polled read is waited out here. A read that was never
        // polled reserved nothing, so it has nothing to wait for.
        if !self.polled {
            return;
        }

        let Some(read) = self.read.take() else {
            return;
        };

        // Blocking while unwinding risks a double panic, which aborts the process outright, or an
        // unbounded wait in a run that is already failing. Leaking the read avoids both without
        // giving the staging buffer back. The reservation is held by the future itself, and the
        // unmap runs in the allocation controller that its completion builds, so a future that is
        // never dropped is a staging block the pool can never hand out again. That costs one block
        // and one poll guard, which leaves CubeCL's poll thread spinning, and it leaves the device
        // usable for whatever outlives the panic, such as a worker thread or a caught unwind.
        if std::thread::panicking() {
            std::mem::forget(read);
            return;
        }

        let _ = cubecl::future::block_on(read);
    }
}

impl fmt::Debug for PendingBatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PendingBatch")
            .field("items", &self.map.formats.len())
            .field("planes", &self.map.shapes.len())
            .field("polled", &self.polled)
            .finish_non_exhaustive()
    }
}

/// What a single poll of a [PendingBatch] found.
#[derive(Debug)]
pub enum TryWait {
    Ready(Vec<HostItem>),
    /// Not landed. This is the same batch, now committed to finishing, so dropping it blocks
    /// until the read lands.
    NotReady(PendingBatch),
}

#[cfg(test)]
mod tests {
    use std::panic;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::frame::{
        ColourInfo,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        MaskFormat,
        SampleFormat,
        Subsampling,
    };
    use crate::memory::BufferPool;
    use crate::testing::CountingAllocator;

    /// A read that stays pending until its `lands_after`th poll.
    ///
    /// It wakes itself whenever it returns pending, so a blocking wait on one keeps polling rather
    /// than parking forever.
    struct FakeRead {
        polls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        lands_after: usize,
        buffers: Vec<Bytes>,
    }

    impl Drop for FakeRead {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Future for FakeRead {
        type Output = Result<Vec<Bytes>, RuntimeError>;

        fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
            let polls = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
            if polls < self.lands_after {
                context.waker().wake_by_ref();
                return Poll::Pending;
            }

            let buffers = std::mem::take(&mut self.buffers);

            Poll::Ready(Ok(buffers))
        }
    }

    fn frames(width: u32, height: u32) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width,
            height,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
        })
    }

    fn mask(width: u32, height: u32) -> StreamFormat {
        StreamFormat::Mask(MaskFormat {
            width,
            height,
            sample: SampleFormat::U8,
        })
    }

    fn zeroed(bytes: usize) -> Bytes {
        Bytes::from_bytes_vec(vec![0; bytes])
    }

    /// A buffer filled with `position` repeated, the way the fake allocator marks its reads.
    fn marked(position: usize, bytes: usize) -> Bytes {
        Bytes::from_bytes_vec(vec![position as u8; bytes])
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

    fn pool() -> BufferPool {
        BufferPool::new(Arc::new(CountingAllocator::new(64)))
    }

    /// One item per format, over memory no device backs, each stamped with its own position.
    fn items_of(pool: &BufferPool, formats: &[StreamFormat]) -> Vec<StreamItem> {
        formats
            .iter()
            .enumerate()
            .map(|(index, format)| {
                let meta = FrameMeta::at(index as u64);

                match format {
                    StreamFormat::Frames(frame) => StreamItem::Frame(pool.alloc_frame(frame, meta)),
                    StreamFormat::Mask(mask) => StreamItem::Mask(pool.alloc_mask(mask, meta)),
                }
            })
            .collect()
    }

    /// A map over one item per format, with the items it was built for.
    fn map_of(formats: &[StreamFormat]) -> (PlaneMap, Vec<StreamItem>) {
        let items = items_of(&pool(), formats);
        let map = PlaneMap::for_items(&items);

        (map, items)
    }

    /// A batch whose read comes from the fake allocator, so every buffer is marked with its own
    /// position in the read.
    fn batch_over(formats: &[StreamFormat]) -> PendingBatch {
        let pool = pool();
        let items = items_of(&pool, formats);
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
        let read = pool.read(handles);

        PendingBatch::new(read, map, items)
    }

    fn fake_batch(polls: &Arc<AtomicUsize>, lands_after: usize, formats: &[StreamFormat]) -> PendingBatch {
        fake_batch_counting(polls, &Arc::new(AtomicUsize::new(0)), lands_after, formats)
    }

    fn fake_batch_counting(
        polls: &Arc<AtomicUsize>,
        drops: &Arc<AtomicUsize>,
        lands_after: usize,
        formats: &[StreamFormat],
    ) -> PendingBatch {
        let items = items_of(&pool(), formats);
        let map = PlaneMap::for_items(&items);
        let buffers = map.shapes().iter().map(|shape| zeroed(shape.bytes)).collect();

        let read = FakeRead {
            polls: Arc::clone(polls),
            drops: Arc::clone(drops),
            lands_after,
            buffers,
        };

        PendingBatch::new(Box::pin(read), map, items)
    }

    #[test]
    fn a_batch_cuts_its_bytes_back_into_one_set_of_planes_per_item() {
        let formats = vec![frames(64, 64), mask(64, 64)];
        let (map, read) = map_of(&formats);
        let bytes = map.shapes().iter().map(|shape| zeroed(shape.bytes)).collect();

        let planes = map.cut(bytes, &read).expect("the shapes came from the map");

        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0].len(), 3, "a yuv frame has three planes");
        assert_eq!(planes[1].len(), 1, "a mask has one");
    }

    #[test]
    fn a_batch_rejects_a_read_that_came_back_the_wrong_shape() {
        let (map, read) = map_of(&[frames(64, 64)]);

        let cut = map.cut(vec![zeroed(10)], &read);

        assert!(cut.is_err(), "three planes cannot be cut from one buffer");
    }

    #[test]
    fn a_batch_rejects_a_buffer_that_came_back_the_wrong_size() {
        let (map, read) = map_of(&[mask(8, 8)]);

        let cut = map.cut(vec![zeroed(63)], &read);

        assert!(cut.is_err(), "a 64 byte mask cannot be cut from 63 bytes");
    }

    #[test]
    fn a_read_that_came_back_wrong_leaves_the_planes_it_would_have_filled_alone() {
        let (map, read) = map_of(&[frames(64, 64)]);
        let held = HostPlanes::zeroed(&mask(8, 8));
        let mut items = vec![HostItem::new(held, FrameMeta::at(4))];
        items[0].plane_mut(Channel::Y).expect("a mask carries one plane")[0] = 7;

        let cut = map.cut_into(vec![zeroed(10)], &read, &mut items);

        assert!(cut.is_err(), "three planes cannot be cut from one buffer");
        assert_eq!(items.len(), 1);

        let plane = items[0].plane(Channel::Y).expect("a mask carries one plane");
        assert_eq!(
            plane.len(),
            64,
            "the planes were resized for a read that never landed"
        );
        assert_eq!(
            plane[0], 7,
            "the planes were written over by a read that never landed"
        );
    }

    #[test]
    fn a_rejected_read_names_the_plane_that_came_back_wrong() {
        let (map, read) = map_of(&[frames(64, 64)]);
        let buffers = vec![zeroed(4096), zeroed(1024), zeroed(99)];

        let error = map.cut(buffers, &read).expect_err("the V plane is short");

        assert!(error.to_string().contains("the V plane of item 0"), "{error}");
    }

    #[test]
    fn the_map_flattens_every_plane_of_every_item_in_read_order() {
        let formats = vec![frames(64, 64), mask(32, 32)];

        let (map, _read) = map_of(&formats);
        let shapes = map.shapes();

        assert_eq!(shapes.len(), 4);
        assert_eq!(
            (shapes[0].item, shapes[0].channel, shapes[0].bytes),
            (0, Channel::Y, 4096)
        );
        assert_eq!(
            (shapes[1].item, shapes[1].channel, shapes[1].bytes),
            (0, Channel::U, 1024)
        );
        assert_eq!(
            (shapes[2].item, shapes[2].channel, shapes[2].bytes),
            (0, Channel::V, 1024)
        );
        assert_eq!(
            (shapes[3].item, shapes[3].channel, shapes[3].bytes),
            (1, Channel::Y, 1024)
        );
        assert_eq!((shapes[0].width, shapes[0].height), (64, 64));
        assert_eq!((shapes[2].width, shapes[2].height), (32, 32));
    }

    /// The chroma planes and the mask are all the same size here, so only what a plane holds tells
    /// a correct cut from a swapped one.
    #[test]
    fn every_plane_holds_the_buffer_the_map_placed_at_its_position() {
        let formats = vec![frames(64, 64), mask(32, 32)];
        let (map, read) = map_of(&formats);

        let buffers = map
            .shapes()
            .iter()
            .enumerate()
            .map(|(position, shape)| marked(position, shape.bytes))
            .collect();
        let items = map.cut(buffers, &read).expect("the shapes came from the map");

        let luma = items[0].plane(Channel::Y).expect("a yuv frame carries luma");
        let blue = items[0].plane(Channel::U).expect("a yuv frame carries U");
        let red = items[0].plane(Channel::V).expect("a yuv frame carries V");
        let mask_plane = items[1].plane(Channel::Y).expect("a mask carries one plane");

        assert_eq!(marker_of(luma), 0);
        assert_eq!(marker_of(blue), 1);
        assert_eq!(marker_of(red), 2, "V holds the buffer read for U");
        assert_eq!(
            marker_of(mask_plane),
            3,
            "the mask holds a buffer read for the frame"
        );
    }

    #[test]
    fn waiting_on_a_batch_puts_every_buffer_the_device_returned_in_its_own_plane() {
        let batch = batch_over(&[frames(64, 64), mask(32, 32)]);

        let items = batch.wait().expect("the fake always lands");

        assert_eq!(items.len(), 2);
        assert_eq!(
            marker_of(items[0].plane(Channel::U).expect("a yuv frame carries U")),
            1
        );
        assert_eq!(
            marker_of(items[0].plane(Channel::V).expect("a yuv frame carries V")),
            2,
            "V holds the buffer read for U"
        );
        assert_eq!(
            marker_of(items[1].plane(Channel::Y).expect("a mask carries one plane")),
            3,
            "the mask holds a buffer read for the frame"
        );
    }

    #[test]
    fn waiting_into_planes_of_the_same_shape_keeps_their_allocations() {
        let formats = [frames(64, 64), mask(32, 32)];

        let mut items = Vec::new();
        let first = batch_over(&formats);
        first.wait_into(&mut items).expect("the fake always lands");
        let before = items[0]
            .plane(Channel::Y)
            .expect("a yuv frame carries luma")
            .as_ptr();

        let second = batch_over(&formats);
        second.wait_into(&mut items).expect("the fake always lands");
        let after = items[0]
            .plane(Channel::Y)
            .expect("a yuv frame carries luma")
            .as_ptr();

        assert_eq!(items.len(), 2);
        assert_eq!(before, after, "the luma allocation was replaced");
    }

    #[test]
    fn waiting_into_planes_of_another_shape_resizes_them() {
        let mut items = Vec::new();
        let first = batch_over(&[frames(64, 64), mask(32, 32)]);
        first.wait_into(&mut items).expect("the fake always lands");

        let second = batch_over(&[mask(8, 8)]);
        second.wait_into(&mut items).expect("the fake always lands");

        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0]
                .plane(Channel::Y)
                .expect("a mask carries one plane")
                .len(),
            64
        );
    }

    #[test]
    fn trying_to_wait_lands_a_read_that_has_already_resolved() {
        let batch = batch_over(&[frames(64, 64)]);

        let landed = batch.try_wait().expect("the fake always lands");

        let TryWait::Ready(items) = landed else {
            panic!("the fake resolves on its first poll");
        };
        assert_eq!(items.len(), 1);
        assert_eq!(
            marker_of(items[0].plane(Channel::V).expect("a yuv frame carries V")),
            2
        );
    }

    #[test]
    fn trying_to_wait_hands_back_the_same_batch_when_the_read_has_not_landed() {
        let polls = Arc::new(AtomicUsize::new(0));
        let batch = fake_batch(&polls, 2, &[mask(8, 8)]);

        let polled = batch.try_wait().expect("a pending read is not a failure");

        let TryWait::NotReady(batch) = polled else {
            panic!("the read does not land until its second poll");
        };
        assert_eq!(
            polls.load(Ordering::SeqCst),
            1,
            "the read was polled more than once"
        );

        let items = batch.wait().expect("the second poll lands");
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn dropping_a_batch_that_was_never_polled_leaves_its_read_alone() {
        let polls = Arc::new(AtomicUsize::new(0));
        let batch = fake_batch(&polls, 1, &[mask(8, 8)]);

        drop(batch);

        assert_eq!(
            polls.load(Ordering::SeqCst),
            0,
            "an unpolled read reserved nothing to wait on"
        );
    }

    #[test]
    fn dropping_a_batch_whose_read_has_landed_does_not_poll_it_again() {
        let polls = Arc::new(AtomicUsize::new(0));
        let batch = fake_batch(&polls, 1, &[mask(8, 8)]);

        let landed = batch.try_wait().expect("the read lands on its first poll");

        assert!(matches!(landed, TryWait::Ready(_)));
        assert_eq!(
            polls.load(Ordering::SeqCst),
            1,
            "a landed read was waited on again"
        );
    }

    /// Blocking while unwinding is what the panicking branch of the drop rule exists to avoid, so
    /// the read is neither waited on nor dropped here. Leaking it is what keeps the still-mapped
    /// staging buffer out of the pool, which is what a process surviving the panic needs.
    #[test]
    fn dropping_a_polled_batch_that_has_not_landed_leaks_it_while_panicking() {
        let polls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let counted_polls = Arc::clone(&polls);
        let counted_drops = Arc::clone(&drops);

        let unwound = panic::catch_unwind(move || {
            let batch = fake_batch_counting(&counted_polls, &counted_drops, 2, &[mask(8, 8)]);
            let polled = batch.try_wait().expect("a pending read is not a failure");

            let TryWait::NotReady(_held) = polled else {
                panic!("the read does not land until its second poll");
            };

            panic!("unwinding while the batch is still held");
        });

        assert!(unwound.is_err());
        assert_eq!(
            polls.load(Ordering::SeqCst),
            1,
            "the read was waited on while unwinding"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "the read was dropped while unwinding, handing its staging buffer back still mapped"
        );
    }

    /// The counterpart of the panicking drop. A read that lands is dropped as normal, so nothing
    /// is leaked on the path a caller actually takes.
    #[test]
    fn a_read_that_landed_is_dropped_rather_than_leaked() {
        let polls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let batch = fake_batch_counting(&polls, &drops, 1, &[mask(8, 8)]);

        batch.wait().expect("the read lands on its first poll");

        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    /// A landed batch has to say which frames it holds, or a sink is back to keeping a list of
    /// its own in step with the batch.
    #[test]
    fn every_landed_item_carries_the_metadata_of_the_item_it_was_read_from() {
        let batch = batch_over(&[frames(64, 64), mask(32, 32)]);

        let items = batch.wait().expect("the fake always lands");

        assert_eq!(items[0].meta().index, 0);
        assert_eq!(items[1].meta().index, 1);
    }

    #[test]
    fn dropping_a_polled_batch_that_has_not_landed_waits_for_it() {
        let polls = Arc::new(AtomicUsize::new(0));
        let batch = fake_batch(&polls, 2, &[mask(8, 8)]);
        let polled = batch.try_wait().expect("a pending read is not a failure");

        let TryWait::NotReady(batch) = polled else {
            panic!("the read does not land until its second poll");
        };
        drop(batch);

        assert_eq!(
            polls.load(Ordering::SeqCst),
            2,
            "the read was abandoned before it landed"
        );
    }
}
