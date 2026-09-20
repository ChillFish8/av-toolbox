use std::sync::Mutex;

use cubecl::server::Handle;
use cubecl::stream_id::StreamId;

use crate::DeviceAllocator;

/// The wgpu-backed accelerator this build tests against.
///
/// Which one it is follows the enabled backend feature, so the same tests run against Vulkan on a
/// Linux machine and against Metal on a macOS one. The two features are mutually exclusive, so it
/// is never defined twice. A build with neither does not define it at all.
#[cfg(feature = "vulkan")]
pub(crate) const TEST_ACCELERATOR: crate::Accelerator = crate::Accelerator::Vulkan;

#[cfg(feature = "metal")]
pub(crate) const TEST_ACCELERATOR: crate::Accelerator = crate::Accelerator::Metal;

/// A [DeviceAllocator](crate::DeviceAllocator) that reserves nothing and remembers what it was
/// asked for.
///
/// A `Handle` can be built without a device, so everything above the seam can be tested by counting
/// the reservations it causes. A pool that recycles is one that stops asking.
#[derive(Debug)]
pub(crate) struct CountingAllocator {
    alignment: usize,
    requests: Mutex<Vec<usize>>,
}

impl CountingAllocator {
    pub(crate) fn new(alignment: usize) -> Self {
        Self {
            alignment,
            requests: Mutex::new(Vec::new()),
        }
    }

    /// The sizes asked for, in order.
    pub(crate) fn requests(&self) -> Vec<usize> {
        self.requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// How many reservations were asked for.
    pub(crate) fn count(&self) -> usize {
        self.requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len()
    }
}

impl DeviceAllocator for CountingAllocator {
    fn empty(&self, bytes: usize) -> Handle {
        self.requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(bytes);

        Handle::new(StreamId::current(), bytes as u64)
    }

    fn upload(&self, bytes: &[u8]) -> Handle {
        self.empty(bytes.len())
    }

    fn alignment(&self) -> usize {
        self.alignment
    }
}
