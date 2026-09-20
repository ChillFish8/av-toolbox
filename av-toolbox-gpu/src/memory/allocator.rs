use std::fmt;

use cubecl::client::ComputeClient;
use cubecl::prelude::Runtime;
use cubecl::server::Handle;

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
