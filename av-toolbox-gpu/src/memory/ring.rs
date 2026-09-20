use std::sync::Arc;

use cubecl::server::Handle;

use crate::frame::{Channel, Frame, FrameFormat, FrameMeta, Plane, PlaneBuffer};

/// How many frames a ring holds, and of what.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RingSpec {
    pub slots: usize,
    pub format: FrameFormat,
}

/// Where every plane of every slot sits inside a ring's one allocation.
///
/// Slots are evenly spaced and every plane starts on the device's alignment, so a kernel can take
/// the whole ring as one binding and index into it, or take a single plane as its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingGeometry {
    slots: usize,
    /// Offset of each plane from the start of a slot.
    plane_offsets: Vec<usize>,
    plane_sizes: Vec<usize>,
    /// Distance between the starts of two slots.
    slot_stride: usize,
}

impl RingGeometry {
    pub fn new(spec: &RingSpec, alignment: usize) -> Self {
        let plane_sizes = spec.format.plane_sizes();

        let mut plane_offsets = Vec::with_capacity(plane_sizes.len());
        let mut offset = 0;
        for bytes in &plane_sizes {
            plane_offsets.push(offset);
            offset = align_up(offset + bytes, alignment);
        }

        Self {
            slots: spec.slots,
            plane_offsets,
            plane_sizes,
            slot_stride: align_up(offset, alignment),
        }
    }

    pub const fn slots(&self) -> usize {
        self.slots
    }

    /// How many bytes the whole ring occupies.
    pub const fn total_bytes(&self) -> usize {
        self.slots * self.slot_stride
    }

    /// Where a slot starts, or `None` when the ring has no such slot.
    pub fn slot_offset(&self, slot: usize) -> Option<usize> {
        (slot < self.slots).then(|| slot * self.slot_stride)
    }

    /// Where one plane of one slot starts.
    pub fn plane_offset(&self, slot: usize, plane: usize) -> Option<usize> {
        let slot = self.slot_offset(slot)?;
        let plane = self.plane_offsets.get(plane)?;

        Some(slot + plane)
    }

    /// How many bytes one plane occupies.
    pub fn plane_bytes(&self, plane: usize) -> Option<usize> {
        self.plane_sizes.get(plane).copied()
    }
}

/// Rounds `value` up to the next multiple of `alignment`.
fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment > 0, "an alignment of zero has no next multiple");

    value.div_ceil(alignment) * alignment
}

/// Several frames in one allocation, so a temporal kernel can index across them.
#[derive(Debug)]
pub struct Ring {
    buffer: Arc<PlaneBuffer>,
    geometry: RingGeometry,
    format: FrameFormat,
}

impl Ring {
    pub(crate) fn new(buffer: PlaneBuffer, geometry: RingGeometry, format: FrameFormat) -> Self {
        Self {
            buffer: Arc::new(buffer),
            geometry,
            format,
        }
    }

    pub const fn slots(&self) -> usize {
        self.geometry.slots()
    }

    pub const fn geometry(&self) -> &RingGeometry {
        &self.geometry
    }

    pub const fn format(&self) -> &FrameFormat {
        &self.format
    }

    /// The whole ring, for a kernel that indexes across slots itself.
    pub fn whole(&self) -> &Handle {
        self.buffer.handle()
    }

    /// One plane of one slot, as a view into the ring's allocation.
    ///
    /// The view is not pooled. The ring is recycled as one piece, so dropping a view releases
    /// nothing.
    pub fn slot_plane(&self, slot: usize, channel: Channel) -> Option<Plane> {
        let index = self
            .format
            .channels()
            .iter()
            .position(|carried| *carried == channel)?;

        let offset = self.geometry.plane_offset(slot, index)?;
        let bytes = self.geometry.plane_bytes(index)?;
        let (width, height) = self.format.channel_dimensions(channel)?;

        let total = self.geometry.total_bytes();
        let tail = total - offset - bytes;

        let handle = self
            .buffer
            .handle()
            .clone()
            .offset_start(offset as u64)
            .offset_end(tail as u64);
        let buffer = PlaneBuffer::view(Some(self.buffer.clone()), handle, bytes);

        Some(Plane::new(buffer, width, height))
    }

    /// One slot as a whole frame, for a stage that wants to hand it downstream.
    pub fn slot_frame(&self, slot: usize, meta: FrameMeta) -> Option<Frame> {
        let mut planes = Vec::with_capacity(self.format.channels().len());
        for channel in self.format.channels() {
            planes.push(self.slot_plane(slot, *channel)?);
        }

        Some(Frame::new(self.format, planes, meta))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::memory::pool::BufferPool;
    use crate::testing::CountingAllocator;
    use crate::{Channel, ColourInfo, FrameLayout, SampleFormat, Subsampling};

    fn format(width: u32, height: u32) -> FrameFormat {
        FrameFormat {
            width,
            height,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
        }
    }

    fn geometry(slots: usize, alignment: usize) -> RingGeometry {
        let spec = RingSpec {
            slots,
            format: format(64, 64),
        };

        RingGeometry::new(&spec, alignment)
    }

    #[test]
    fn every_plane_starts_on_the_alignment() {
        let geometry = geometry(3, 64);

        for slot in 0..geometry.slots() {
            for plane in 0..3 {
                let offset = geometry.plane_offset(slot, plane).expect("the plane exists");
                assert_eq!(offset % 64, 0, "slot {slot} plane {plane} at {offset}");
            }
        }
    }

    #[test]
    fn slots_are_evenly_spaced() {
        let geometry = geometry(4, 64);
        let stride = geometry.slot_offset(1).expect("the slot exists");

        for slot in 0..geometry.slots() {
            let offset = geometry.slot_offset(slot).expect("the slot exists");
            assert_eq!(offset, slot * stride, "slot {slot}");
        }
    }

    /// A ring is one allocation, so its total has to cover the last plane of the last slot.
    #[test]
    fn the_total_covers_every_slot() {
        let geometry = geometry(3, 64);

        let last_plane = geometry.plane_offset(2, 2).expect("the plane exists");
        let last_bytes = geometry.plane_bytes(2).expect("the plane exists");

        assert!(
            geometry.total_bytes() >= last_plane + last_bytes,
            "total {} does not cover {} plus {}",
            geometry.total_bytes(),
            last_plane,
            last_bytes,
        );
    }

    #[test]
    fn planes_within_a_slot_do_not_overlap() {
        let geometry = geometry(2, 64);

        for plane in 0..2 {
            let start = geometry.plane_offset(0, plane).expect("the plane exists");
            let bytes = geometry.plane_bytes(plane).expect("the plane exists");
            let next = geometry.plane_offset(0, plane + 1).expect("the plane exists");

            assert!(start + bytes <= next, "plane {plane} runs into the next");
        }
    }

    #[test]
    fn an_alignment_of_one_wastes_nothing() {
        let geometry = geometry(2, 1);
        let format = format(64, 64);

        assert_eq!(geometry.total_bytes(), format.total_bytes() * 2);
    }

    #[test]
    fn a_slot_or_plane_that_does_not_exist_reports_nothing() {
        let geometry = geometry(2, 64);

        assert_eq!(geometry.slot_offset(2), None);
        assert_eq!(geometry.plane_offset(2, 0), None);
        assert_eq!(geometry.plane_offset(0, 3), None);
        assert_eq!(geometry.plane_bytes(3), None);
    }

    #[test]
    fn a_ring_of_one_slot_is_just_a_frame() {
        let geometry = geometry(1, 64);

        assert_eq!(geometry.slots(), 1);
        assert_eq!(geometry.slot_offset(0), Some(0));
    }

    /// Every other test uses a format whose planes are already aligned, so this is the only one
    /// that fails if the padding is dropped.
    #[test]
    fn a_plane_that_does_not_fill_its_alignment_is_padded() {
        let spec = RingSpec {
            slots: 2,
            format: format(10, 10),
        };

        let geometry = RingGeometry::new(&spec, 64);

        assert_eq!(geometry.plane_bytes(0), Some(100));
        assert_eq!(geometry.plane_bytes(1), Some(25));

        assert_eq!(geometry.plane_offset(0, 0), Some(0));
        assert_eq!(geometry.plane_offset(0, 1), Some(128));
        assert_eq!(geometry.plane_offset(0, 2), Some(192));

        assert_eq!(geometry.slot_offset(1), Some(256));
        assert_eq!(geometry.total_bytes(), 512);
    }

    fn ring(slots: usize) -> (Arc<CountingAllocator>, Ring) {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = BufferPool::new(allocator.clone());
        let spec = RingSpec {
            slots,
            format: format(64, 64),
        };

        let ring = pool.ring(spec);

        (allocator, ring)
    }

    /// A ring is one reservation however many slots it holds, which is what lets a temporal kernel
    /// take it as a single binding.
    #[test]
    fn a_ring_is_one_reservation() {
        let (allocator, ring) = ring(9);

        assert_eq!(allocator.count(), 1);
        assert_eq!(allocator.requests(), vec![ring.geometry().total_bytes()]);
    }

    #[test]
    fn every_slot_offers_a_plane_of_the_right_shape() {
        let (_allocator, ring) = ring(3);

        for slot in 0..ring.slots() {
            let luma = ring.slot_plane(slot, Channel::Y).expect("the ring carries luma");
            let chroma = ring
                .slot_plane(slot, Channel::U)
                .expect("the ring carries chroma");

            assert_eq!(luma.dimensions(), (64, 64));
            assert_eq!(chroma.dimensions(), (32, 32));
        }
    }

    #[test]
    fn a_slot_or_channel_the_ring_does_not_have_offers_nothing() {
        let (_allocator, ring) = ring(2);

        assert!(ring.slot_plane(2, Channel::Y).is_none());
        assert!(ring.slot_plane(0, Channel::R).is_none());
    }

    /// Slot views are windows onto the ring's one allocation, so dropping one must not hand any
    /// memory back to the pool while the ring still holds it.
    #[test]
    fn dropping_a_slot_view_recycles_nothing() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = BufferPool::new(allocator.clone());
        let ring = pool.ring(RingSpec {
            slots: 2,
            format: format(64, 64),
        });

        drop(ring.slot_plane(0, Channel::Y).expect("the ring carries luma"));

        let _other = pool.reserve(4096);
        assert_eq!(allocator.count(), 2, "a slot view was recycled into the pool");
    }

    /// A view is a window into the ring's one allocation, so the allocation must not go back to the
    /// pool while a window is still pointing into it.
    #[test]
    fn a_ring_dropped_while_a_view_lives_recycles_nothing() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = BufferPool::new(allocator.clone());
        let spec = RingSpec {
            slots: 2,
            format: format(64, 64),
        };

        let ring = pool.ring(spec);
        let total = ring.geometry().total_bytes();
        let view = ring.slot_plane(0, Channel::Y).expect("the ring carries luma");

        drop(ring);
        let _other = pool.reserve(total);
        assert_eq!(
            allocator.count(),
            2,
            "the ring was recycled while a view still pointed into it"
        );

        drop(view);
        let _reused = pool.reserve(total);
        assert_eq!(
            allocator.count(),
            2,
            "the ring was not recycled once its last view went"
        );
    }

    /// Dimensions come from the format and say nothing about where a view points, so the offset is
    /// the thing worth asserting.
    #[test]
    fn each_slot_view_points_at_its_own_slot() {
        let (_allocator, ring) = ring(3);

        for slot in 0..ring.slots() {
            let expected = ring.geometry().plane_offset(slot, 0).expect("the plane exists");
            let luma = ring.slot_plane(slot, Channel::Y).expect("the ring carries luma");

            assert_eq!(
                luma.buffer().handle().offset_start,
                Some(expected as u64),
                "slot {slot}"
            );
        }
    }

    /// A view has to stop at its own plane, not run to the end of the ring, or an overrun on one
    /// slot lands silently in the next.
    #[test]
    fn a_slot_view_is_bounded_at_its_far_end() {
        let (_allocator, ring) = ring(3);

        let luma = ring.slot_plane(0, Channel::Y).expect("the ring carries luma");

        assert_eq!(
            luma.buffer().handle().size_in_used(),
            luma.buffer().bytes() as u64,
            "the view reaches past its own plane",
        );
    }

    #[test]
    fn a_slot_can_be_taken_as_a_whole_frame() {
        let (_allocator, ring) = ring(2);

        let frame = ring
            .slot_frame(1, FrameMeta::at(4))
            .expect("the ring has this slot");

        assert_eq!(frame.format(), ring.format());
        assert_eq!(frame.meta().index, 4);
        assert_eq!(
            frame.plane(Channel::U).expect("carries chroma").dimensions(),
            (32, 32)
        );
        assert!(ring.slot_frame(2, FrameMeta::at(0)).is_none());
    }
}
