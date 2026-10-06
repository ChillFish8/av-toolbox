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

/// Host memory for items on the host, recycled by size and charged to a budget.
#[derive(Debug, Clone, Default)]
pub struct HostPool {
    free: Arc<HostFree>,
    budget: Option<Arc<FrameBudget>>,
    /// Whether an allocation waits for room on `budget` before it is charged.
    waits: bool,
}

#[derive(Debug, Default)]
struct HostFree {
    lists: Mutex<HashMap<usize, Vec<Vec<u8>>>>,
}

impl HostFree {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.lists
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .values()
            .map(Vec::len)
            .sum()
    }

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

    /// This pool, charging every frame shaped allocation to `budget` once it has room.
    pub(crate) fn with_budget(mut self, budget: Arc<FrameBudget>) -> Self {
        self.budget = Some(budget);
        self.waits = true;
        self
    }

    /// This pool, charging every frame shaped allocation to `budget` without waiting for room.
    pub(crate) fn with_charged_budget(mut self, budget: Arc<FrameBudget>) -> Self {
        self.budget = Some(budget);
        self.waits = false;
        self
    }

    /// How many buffers sit on the free list waiting to be recycled.
    #[cfg(test)]
    pub(crate) fn free_buffers(&self) -> usize {
        self.free.len()
    }

    /// Waits until `bytes` fits under this pool's budget, or returns immediately when it has none.
    ///
    /// A released or stalled budget belongs to a run that is over or failing, so charging it would
    /// leave a total nothing ever credits back. A pool that does not wait always charges.
    pub(crate) fn wait_for_room(&self, bytes: usize) -> bool {
        let Some(budget) = &self.budget else {
            return false;
        };

        if !self.waits {
            return true;
        }

        matches!(budget.wait_for_room(bytes), Room::Available)
    }

    /// A buffer of exactly `bytes`, recycled when one that size is free.
    ///
    /// A recycled buffer still holds whatever was last written into it, so a caller that does not
    /// write every byte has to fill the rest itself. This never waits.
    pub(crate) fn reserve(&self, bytes: usize, counted: bool) -> HostBuffer {
        let buffer = self.free.take(bytes).unwrap_or_else(|| vec![0; bytes]);

        let budget = if counted { self.budget.clone() } else { None };
        if let Some(budget) = &budget {
            budget.charge(bytes);
        }

        HostBuffer {
            bytes: buffer,
            pool: Some(Arc::downgrade(&self.free)),
            budget: budget.as_ref().map(Arc::downgrade),
        }
    }

    /// Takes `bytes` over as a buffer that is freed when it drops, without charging it.
    pub(crate) fn adopt(bytes: Vec<u8>) -> HostBuffer {
        HostBuffer {
            bytes,
            pool: None,
            budget: None,
        }
    }

    /// A host item made from `planes`, moved rather than copied, each freed once the item is done
    /// with it.
    pub(crate) fn adopt_item(format: StreamFormat, planes: Vec<Vec<u8>>, meta: FrameMeta) -> HostStreamItem {
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

                        let buffer = Self::adopt(bytes);

                        HostPlane::new(buffer, width, height)
                    })
                    .collect();

                HostStreamItem::Frame(HostFrame::new(format, planes, meta))
            },
            StreamFormat::Mask(format) => {
                let stream_format = StreamFormat::Mask(format);
                assert_eq!(
                    planes.len(),
                    stream_format.plane_sizes().len(),
                    "a mask lands as one plane per plane its format carries",
                );

                let mut planes = planes.into_iter();
                let full_bytes = planes.next().expect("a mask carries a full plane");
                let full_buffer = Self::adopt(full_bytes);
                let plane = HostPlane::new(full_buffer, format.width, format.height);

                let chroma = format.chroma_dimensions().map(|(width, height)| {
                    let chroma_bytes = planes.next().expect("the count was checked above");
                    let chroma_buffer = Self::adopt(chroma_bytes);

                    HostPlane::new(chroma_buffer, width, height)
                });

                let mask = HostMask::new(format, plane, meta, chroma);

                HostStreamItem::Mask(mask)
            },
        }
    }

    /// A writable frame of `format`, one buffer per channel.
    ///
    /// Waits once for the whole frame's bytes before reserving any plane. Its planes are recycled
    /// memory holding an earlier frame's samples.
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

                let buffer = self.reserve(bytes, counted);

                HostPlane::new(buffer, width, height)
            })
            .collect();

        HostFrame::new(*format, planes, meta)
            .try_into_mut()
            .expect("a frame fresh from the pool is held by nothing else")
    }

    /// A writable mask of `format`, which is one buffer for the full plane and one for the chroma
    /// plane when the format has one.
    ///
    /// Waits once for both planes' bytes before reserving either.
    pub fn alloc_mask(&self, format: &MaskFormat, meta: FrameMeta) -> HostMaskMut {
        let full_bytes = format.bytes();
        let chroma_bytes = format.chroma_bytes();
        let counted = self.wait_for_room(full_bytes + chroma_bytes);

        let buffer = self.reserve(full_bytes, counted);
        let plane = HostPlane::new(buffer, format.width, format.height);

        let chroma = format.chroma_dimensions().map(|(width, height)| {
            let chroma_buffer = self.reserve(chroma_bytes, counted);

            HostPlane::new(chroma_buffer, width, height)
        });

        HostMask::new(*format, plane, meta, chroma)
            .try_into_mut()
            .expect("a mask fresh from the pool is held by nothing else")
    }
}

/// The host memory behind one plane.
pub struct HostBuffer {
    bytes: Vec<u8>,
    /// The free list these bytes return to on drop, if they came from one.
    pool: Option<Weak<HostFree>>,
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

        if let Some(free) = self.pool.as_ref().and_then(Weak::upgrade) {
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

    #[test]
    fn an_adopted_buffer_keeps_the_memory_it_was_given() {
        let bytes = vec![7; 16];
        let address = bytes.as_ptr();

        let adopted = HostPool::adopt(bytes);

        assert_eq!(adopted.bytes().as_ptr(), address);
    }

    #[test]
    fn a_pool_without_a_budget_charges_nothing() {
        assert!(!HostPool::new().wait_for_room(10));
    }
}
