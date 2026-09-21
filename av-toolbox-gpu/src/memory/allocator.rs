use std::fmt;
use std::future::Future;
use std::pin::Pin;

use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::prelude::Runtime;
use cubecl::server::{Handle, ServerError};

use crate::frame::Channel;

/// A readback in flight.
///
/// It owns everything it needs, so a caller can hold one after the thing that started it is gone.
pub type DeviceRead = Pin<Box<dyn Future<Output = Result<Vec<Bytes>, RuntimeError>> + Send>>;

/// A device operation that could not complete.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(
        "reading data back from the device failed: {0}, the device is probably already broken \
         and the client that produced it should be replaced rather than retried"
    )]
    Readback(#[from] ServerError),
    #[error("the device returned {returned} buffers for a read of {expected} planes")]
    ReadShape { expected: usize, returned: usize },
    #[error(
        "the {channel} plane of item {item} is {width}x{height} and needs {expected} bytes, \
         but {returned} came back"
    )]
    PlaneSize {
        item: usize,
        channel: Channel,
        width: u32,
        height: u32,
        expected: usize,
        returned: usize,
    },
    #[error("item {item} has no {channel} plane, give it one of {width}x{height} at {expected} bytes")]
    HostPlaneMissing {
        item: usize,
        channel: Channel,
        width: u32,
        height: u32,
        expected: usize,
    },
    #[error(
        "the {channel} plane of item {item} is {width}x{height} and needs {expected} bytes, \
         but the host planes gave {given}"
    )]
    HostPlaneSize {
        item: usize,
        channel: Channel,
        width: u32,
        height: u32,
        expected: usize,
        given: usize,
    },
}

/// Reserves device memory.
///
/// Only this call needs to know which runtime is in use, so it is the single generic seam the rest
/// of the memory handling sits above.
pub trait DeviceAllocator: Send + Sync + fmt::Debug {
    /// Reserves `bytes` of uninitialised memory.
    ///
    /// An exhausted device panics rather than returning, because CubeCL offers no fallible
    /// allocation.
    fn empty(&self, bytes: usize) -> Handle;

    /// Reserves room for `bytes` and copies them in.
    ///
    /// It always reserves fresh memory rather than writing into a buffer already held, and takes
    /// one slice per call.
    fn upload(&self, bytes: &[u8]) -> Handle;

    /// The byte alignment every buffer view has to start on.
    fn alignment(&self) -> usize;

    /// Starts reading `handles` back from the device, in the order given.
    ///
    /// The future owns a clone of whatever it needs, so it is `'static`. Clients share their
    /// internals, so cloning one is cheap.
    fn read(&self, handles: Vec<Handle>) -> DeviceRead;
}

/// A [DeviceAllocator] backed by a CubeCL client.
pub struct ClientAllocator<R: Runtime> {
    client: ComputeClient<R>,
    alignment: usize,
}

impl<R: Runtime> ClientAllocator<R> {
    pub fn new(client: ComputeClient<R>) -> Self {
        let alignment = client.properties().memory.alignment as usize;

        Self { client, alignment }
    }
}

impl<R: Runtime> fmt::Debug for ClientAllocator<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientAllocator")
            .field("alignment", &self.alignment)
            .finish_non_exhaustive()
    }
}

impl<R: Runtime> DeviceAllocator for ClientAllocator<R> {
    fn empty(&self, bytes: usize) -> Handle {
        self.client.empty(bytes)
    }

    fn upload(&self, bytes: &[u8]) -> Handle {
        self.client.create_from_slice(bytes)
    }

    fn alignment(&self) -> usize {
        self.alignment
    }

    /// Starts reading `handles` back from the device.
    ///
    /// The `async` block is load-bearing and must not be flattened into an eager call.
    /// `read_async` reserves a staging buffer, records the copy and maps it the moment it is
    /// called, so wrapping it is what holds all of that off until the read is first polled.
    /// [PendingBatch](crate::PendingBatch) decides from that whether a drop has a mapped buffer to
    /// wait out. An eager call would give every batch one from the moment it is started, including
    /// the ones a drop abandons for never having been polled.
    fn read(&self, handles: Vec<Handle>) -> DeviceRead {
        let client = self.client.clone();

        Box::pin(async move {
            let buffers = client.read_async(handles).await?;

            Ok(buffers)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::CountingAllocator;

    #[test]
    fn a_fake_allocator_reports_what_it_was_asked_for() {
        let allocator = CountingAllocator::new(64);

        let first = allocator.empty(100);
        let second = allocator.empty(200);

        assert_eq!(allocator.requests(), vec![100, 200]);
        assert_eq!(allocator.count(), 2);
        assert_eq!(first.size(), 100);
        assert_eq!(second.size(), 200);
    }

    #[test]
    fn a_fake_allocator_reports_its_alignment() {
        assert_eq!(CountingAllocator::new(256).alignment(), 256);
    }

    #[test]
    fn uploading_reserves_room_for_the_bytes_given() {
        let allocator = CountingAllocator::new(64);
        allocator.upload(&[1, 2, 3, 4]);

        assert_eq!(allocator.requests(), vec![4]);
    }

    #[test]
    fn a_read_returns_one_buffer_per_handle_in_order() {
        let allocator = CountingAllocator::new(64);
        let first = allocator.empty(8);
        let second = allocator.empty(16);

        let read = allocator.read(vec![first, second]);
        let buffers = cubecl::future::block_on(read).expect("the fake always lands");

        assert_eq!(buffers.len(), 2);
        assert_eq!(buffers[0].len(), 8);
        assert_eq!(buffers[1].len(), 16);
    }

    #[test]
    fn a_read_records_the_sizes_it_was_asked_for_in_order() {
        let allocator = CountingAllocator::new(64);
        let first = allocator.empty(8);
        let second = allocator.empty(16);

        let read = allocator.read(vec![first, second]);
        cubecl::future::block_on(read).expect("the fake always lands");

        assert_eq!(allocator.reads(), vec![8, 16]);
    }

    /// Two same-sized handles, such as a frame's U and V planes, would come back identical if the
    /// fake filled every buffer the same way. This is what proves a swapped pair is still caught.
    #[test]
    fn same_sized_handles_still_come_back_as_distinguishable_buffers() {
        let allocator = CountingAllocator::new(64);
        let first = allocator.empty(4);
        let second = allocator.empty(4);

        let read = allocator.read(vec![first, second]);
        let buffers = cubecl::future::block_on(read).expect("the fake always lands");

        assert!(buffers[0].iter().all(|&byte| byte == 0));
        assert!(buffers[1].iter().all(|&byte| byte == 1));
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_real_client_reports_a_power_of_two_alignment() {
        use crate::Device;
        use crate::probe::{RuntimeTask, with_runtime};
        use crate::testing::TEST_ACCELERATOR;

        struct ReadAlignment;

        impl RuntimeTask for ReadAlignment {
            type Output = usize;

            fn run<R: Runtime>(self, client: &ComputeClient<R>) -> Self::Output {
                ClientAllocator::new(client.clone()).alignment()
            }
        }

        let alignment =
            with_runtime(TEST_ACCELERATOR, Device::Default, ReadAlignment).expect("the backend should open");

        assert!(alignment.is_power_of_two(), "{alignment}");
    }
}
