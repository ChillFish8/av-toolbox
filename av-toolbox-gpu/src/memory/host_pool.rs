use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, Weak};

use crate::frame::{
    FrameFormat,
    FrameMeta,
    HostFrame,
    HostFrameMut,
    HostMask,
    HostMaskMut,
    HostPlane,
    HostStreamItem,
    MaskFormat,
    StreamFormat,
};
use crate::memory::budget::{FrameBudget, Room};

/// Host memory for items on the host, recycled by size and charged to a budget the way device
/// memory is.
///
/// A host stage runs frame after frame at the same size, so handing a dropped buffer to the next
/// caller wanting that size means a steady run allocates nothing.
#[derive(Debug, Clone, Default)]
pub struct HostPool {
    free: Arc<HostFree>,
    /// Charged by every frame shaped allocation, the same as the device pool's.
    budget: Option<Arc<FrameBudget>>,
}

#[derive(Debug, Default)]
struct HostFree {
    lists: Mutex<HashMap<usize, Vec<Vec<u8>>>>,
}

impl HostFree {
    fn take(&self, bytes: usize) -> Option<Vec<u8>> {
        self.lists
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get_mut(&bytes)
            .and_then(Vec::pop)
    }

    fn put(&self, buffer: Vec<u8>) {
        self.lists
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(buffer.len())
            .or_default()
            .push(buffer);
    }
}

impl HostPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// This pool, charging every frame shaped allocation to `budget`.
    pub(crate) fn with_budget(mut self, budget: Arc<FrameBudget>) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Waits until `bytes` fits under this pool's budget, or returns immediately when it has none.
    ///
    /// Answers whether what follows should be charged, for the same reason the device pool does: a
    /// released or stalled budget is a run that is over or failing, and charging it would leave a
    /// total nothing ever credits back.
    pub(crate) fn wait_for_room(&self, bytes: usize) -> bool {
        let Some(budget) = &self.budget else {
            return false;
        };

        matches!(budget.wait_for_room(bytes), Room::Available)
    }

    /// A buffer of exactly `bytes`, recycled when one that size is free.
    ///
    /// A recycled buffer still holds whatever was last written into it, so a caller that does not
    /// write every byte has to fill the rest itself. This never waits: a caller allocating a whole
    /// frame waits once for the frame's bytes before reserving any plane, as the device pool does.
    pub(crate) fn reserve(&self, bytes: usize, counted: bool) -> HostBuffer {
        let buffer = self.free.take(bytes).unwrap_or_else(|| vec![0; bytes]);

        let budget = if counted { self.budget.clone() } else { None };
        if let Some(budget) = &budget {
            budget.charge(bytes);
        }

        HostBuffer {
            bytes: buffer,
            pool: Arc::downgrade(&self.free),
            budget: budget.as_ref().map(Arc::downgrade),
        }
    }

    /// Takes `bytes` over as a buffer that returns to this pool when it drops, without charging it.
    ///
    /// A readback lands in memory of its own, and moving it in here is what lets it be reused
    /// rather than freed once the host item it became is done with.
    pub(crate) fn adopt(&self, bytes: Vec<u8>) -> HostBuffer {
        HostBuffer {
            bytes,
            pool: Arc::downgrade(&self.free),
            budget: None,
        }
    }

    /// A host item made from `planes`, moved rather than copied, each joining this pool once the
    /// item is done with it.
    pub(crate) fn adopt_item(
        &self,
        format: StreamFormat,
        planes: Vec<Vec<u8>>,
        meta: FrameMeta,
    ) -> HostStreamItem {
        match format {
            StreamFormat::Frames(format) => {
                let planes = format
                    .channels()
                    .iter()
                    .zip(planes)
                    .map(|(channel, bytes)| {
                        let (width, height) = format
                            .channel_dimensions(*channel)
                            .expect("the layout carries every channel it lists");

                        HostPlane::new(self.adopt(bytes), width, height)
                    })
                    .collect();

                HostStreamItem::Frame(HostFrame::new(format, planes, meta))
            },
            StreamFormat::Mask(format) => {
                let bytes = planes
                    .into_iter()
                    .next()
                    .expect("a mask lands as exactly one plane");
                let plane = HostPlane::new(self.adopt(bytes), format.width, format.height);

                HostStreamItem::Mask(HostMask::new(format, plane, meta))
            },
        }
    }

    /// A writable frame of `format`, one buffer per channel.
    ///
    /// Waits once for the whole frame's bytes before reserving any plane, for the same reason the
    /// device pool does. Its planes are recycled memory holding an earlier frame's samples.
    pub fn alloc_frame(&self, format: &FrameFormat, meta: FrameMeta) -> HostFrameMut {
        let counted = self.wait_for_room(format.total_bytes());

        let planes = format
            .channels()
            .iter()
            .zip(format.plane_sizes())
            .map(|(channel, bytes)| {
                let (width, height) = format
                    .channel_dimensions(*channel)
                    .expect("the layout carries every channel it lists");

                HostPlane::new(self.reserve(bytes, counted), width, height)
            })
            .collect();

        HostFrame::new(*format, planes, meta)
            .try_into_mut()
            .expect("a frame fresh from the pool is held by nothing else")
    }

    /// A writable mask of `format`, which is one buffer.
    pub fn alloc_mask(&self, format: &MaskFormat, meta: FrameMeta) -> HostMaskMut {
        let counted = self.wait_for_room(format.bytes());
        let plane = HostPlane::new(self.reserve(format.bytes(), counted), format.width, format.height);

        HostMask::new(*format, plane, meta)
            .try_into_mut()
            .expect("a mask fresh from the pool is held by nothing else")
    }
}

/// The host memory behind one plane.
pub struct HostBuffer {
    bytes: Vec<u8>,
    pool: Weak<HostFree>,
    /// The budget these bytes were charged to, credited back on drop.
    budget: Option<Weak<FrameBudget>>,
}

impl HostBuffer {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl fmt::Debug for HostBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostBuffer")
            .field("len", &self.bytes.len())
            .finish()
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let bytes = std::mem::take(&mut self.bytes);

        if let Some(budget) = self.budget.as_ref().and_then(Weak::upgrade) {
            budget.credit(bytes.len());
        }

        if let Some(free) = self.pool.upgrade() {
            free.put(bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dropped_buffer_is_handed_to_the_next_caller_wanting_that_size() {
        let pool = HostPool::new();
        let first = pool.reserve(64, false);
        let address = first.bytes().as_ptr();
        drop(first);

        let second = pool.reserve(64, false);

        assert_eq!(second.bytes().as_ptr(), address);
    }

    #[test]
    fn a_buffer_of_another_size_is_never_handed_out() {
        let pool = HostPool::new();
        drop(pool.reserve(64, false));

        let other = pool.reserve(32, false);

        assert_eq!(other.len(), 32);
    }

    #[test]
    fn a_counted_buffer_is_charged_until_it_drops() {
        let budget = FrameBudget::new(1 << 20);
        let pool = HostPool::new().with_budget(budget.clone());

        let buffer = pool.reserve(100, true);
        assert_eq!(budget.outstanding(), 100);

        drop(buffer);
        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn an_uncounted_buffer_is_never_charged() {
        let budget = FrameBudget::new(1 << 20);
        let pool = HostPool::new().with_budget(budget.clone());

        let _buffer = pool.reserve(100, false);

        assert_eq!(budget.outstanding(), 0);
    }

    /// A readback lands in memory the pool never handed out, and it should still come back to be
    /// reused rather than being freed, without being charged to anything on the way.
    #[test]
    fn an_adopted_buffer_joins_the_pool_without_being_charged() {
        let budget = FrameBudget::new(1 << 20);
        let pool = HostPool::new().with_budget(budget.clone());

        let adopted = pool.adopt(vec![7; 16]);
        let address = adopted.bytes().as_ptr();
        assert_eq!(budget.outstanding(), 0);
        drop(adopted);

        let next = pool.reserve(16, false);
        assert_eq!(next.bytes().as_ptr(), address);
    }

    #[test]
    fn a_pool_without_a_budget_charges_nothing() {
        assert!(!HostPool::new().wait_for_room(10));
    }
}
