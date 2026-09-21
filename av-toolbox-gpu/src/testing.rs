use std::sync::Mutex;

use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use cubecl::stream_id::StreamId;

use crate::DeviceAllocator;
use crate::memory::DeviceRead;

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
    reads: Mutex<Vec<usize>>,
    uploads: Mutex<Vec<Vec<u8>>>,
}

impl CountingAllocator {
    pub(crate) fn new(alignment: usize) -> Self {
        Self {
            alignment,
            requests: Mutex::new(Vec::new()),
            reads: Mutex::new(Vec::new()),
            uploads: Mutex::new(Vec::new()),
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

    /// The sizes read back, in the order the handles were given.
    pub(crate) fn reads(&self) -> Vec<usize> {
        self.reads
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// The bytes uploaded, in order.
    pub(crate) fn uploads(&self) -> Vec<Vec<u8>> {
        self.uploads
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
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
        self.uploads
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(bytes.to_vec());

        self.empty(bytes.len())
    }

    fn alignment(&self) -> usize {
        self.alignment
    }

    /// Reads `handles` back, filling buffer `position` with the byte `position` repeated.
    ///
    /// Two handles the same size, such as a frame's U and V planes, would come back identical if
    /// the fake filled every buffer the same way. Marking each buffer by its position instead
    /// means a caller that swaps two same-sized handles gets back buffers a test can tell apart.
    fn read(&self, handles: Vec<Handle>) -> DeviceRead {
        let sizes: Vec<usize> = handles
            .iter()
            .map(|handle| handle.size_in_used() as usize)
            .collect();

        self.reads
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .extend(sizes.iter().copied());

        let buffers = sizes
            .into_iter()
            .enumerate()
            .map(|(position, size)| Bytes::from_bytes_vec(vec![position as u8; size]))
            .collect();

        Box::pin(async move { Ok(buffers) })
    }
}
