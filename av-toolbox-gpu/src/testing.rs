use std::path::Path;
#[cfg(all(feature = "rocm", any(test, feature = "test-support")))]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};

use cubecl::bytes::Bytes;
use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl::device::DeviceId;
#[cfg(all(feature = "rocm", any(test, feature = "test-support")))]
use cubecl::prelude::Client;
use cubecl::server::Handle;
use cubecl_common::device::ServiceId;
use cubecl_environment::stream::StreamId;

use crate::memory::DeviceRead;
use crate::{
    Accelerator,
    BufferPool,
    Capabilities,
    Device,
    DeviceAllocator,
    FrameContext,
    StreamPolicy,
    default_accelerators,
};

/// The accelerator this build tests against.
///
/// Which one it is follows the enabled backend feature, so the same tests run against Vulkan on a
/// Linux machine and against Metal on a macOS one. A build with both tests Metal, since only an
/// Apple target can enable it. A build with neither does not define it at all.
#[cfg(all(test, feature = "vulkan", not(feature = "metal")))]
pub(crate) const TEST_ACCELERATOR: crate::Accelerator = crate::Accelerator::Vulkan;

#[cfg(all(test, feature = "metal"))]
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
    flushes: Arc<AtomicUsize>,
}

impl CountingAllocator {
    pub(crate) fn new(alignment: usize) -> Self {
        let flushes = AtomicUsize::new(0);

        Self {
            alignment,
            requests: Mutex::new(Vec::new()),
            reads: Mutex::new(Vec::new()),
            uploads: Mutex::new(Vec::new()),
            flushes: Arc::new(flushes),
        }
    }

    /// The sizes asked for, in order.
    #[cfg(test)]
    pub(crate) fn requests(&self) -> Vec<usize> {
        self.requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// How many reservations were asked for.
    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len()
    }

    /// The sizes read back, in the order the handles were given.
    #[cfg(test)]
    pub(crate) fn reads(&self) -> Vec<usize> {
        self.reads
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// The bytes uploaded, in order.
    #[cfg(test)]
    pub(crate) fn uploads(&self) -> Vec<Vec<u8>> {
        self.uploads
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    /// How many times queued work was submitted.
    #[cfg(test)]
    pub(crate) fn flushes(&self) -> usize {
        self.flushes.load(Ordering::SeqCst)
    }
}

impl DeviceAllocator for CountingAllocator {
    fn empty(&self, bytes: usize) -> Handle {
        self.requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(bytes);

        detached_handle(bytes)
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

    fn flush(&self) {
        self.flushes.fetch_add(1, Ordering::SeqCst);
    }
}

/// A handle of `bytes` with no device behind it.
///
/// Nothing may read or write the memory it names.
pub(crate) fn detached_handle(bytes: usize) -> Handle {
    let device_id = DeviceId::new(0, 0);
    let service = ServiceId::of::<CountingAllocator>(device_id);
    let stream = StreamId::current();

    Handle::new(service, stream, bytes as u64)
}

fn testing_capabilities(accelerator: Accelerator) -> Capabilities {
    Capabilities::builder(accelerator)
        .plane_size(16, 16)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

/// A [FrameContext] over a fake allocator, for tests with no device.
///
/// Allocations are counted rather than reserved, so anything above the allocator seam can be
/// exercised without an accelerator.
#[cfg(any(test, feature = "test-support"))]
pub fn testing_context() -> FrameContext {
    let allocator = Arc::new(CountingAllocator::new(64));
    let accelerator = default_accelerators()[0];
    let capabilities = testing_capabilities(accelerator);

    let policy = StreamPolicy::for_accelerator(accelerator);
    let pool = BufferPool::new(allocator, policy);

    FrameContext::new(pool, capabilities)
}

/// A [FrameContext] over a fake allocator under `policy`, with a count of how many times its
/// queued work was flushed.
#[cfg(any(test, feature = "test-support"))]
pub fn flush_counting_context(policy: StreamPolicy) -> (FrameContext, Arc<AtomicUsize>) {
    let counting = CountingAllocator::new(64);
    let allocator = Arc::new(counting);
    let flushes = allocator.flushes.clone();
    let accelerator = default_accelerators()[0];
    let capabilities = testing_capabilities(accelerator);

    let pool = BufferPool::new(allocator, policy);
    let context = FrameContext::new(pool, capabilities);

    (context, flushes)
}

/// The device GPU tests open, named by the `AVTB_DEVICE` environment variable.
///
/// Falls back to [Device::Default] when the variable is unset or empty. Panics when it is set to
/// something that is not a device selector.
///
/// The first call also keeps CubeCL's autotune results in the workspace's `target/test-autotune`
/// directory, unless CubeCL's config is already set.
#[cfg(any(test, feature = "test-support"))]
pub fn test_device() -> Device {
    static TUNE_CACHE: Once = Once::new();
    TUNE_CACHE.call_once(install_test_tune_cache);

    let value = std::env::var("AVTB_DEVICE").ok();
    let device = device_from_value(value.as_deref());

    #[cfg(feature = "rocm")]
    drain_at_exit(device);

    device
}

/// The ROCm test device's client, synced by [drain_exit_client] when the process exits.
#[cfg(all(feature = "rocm", any(test, feature = "test-support")))]
static EXIT_CLIENT: OnceLock<Client> = OnceLock::new();

#[cfg(all(feature = "rocm", any(test, feature = "test-support")))]
unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
}

/// Waits for the ROCm device thread to finish its queued tasks when the process exits.
///
/// The HIP runtime tears itself down at exit, and a device thread still running a test's queued
/// tasks then crashes inside it. This hook registers after the runtime has loaded, so it runs
/// before that teardown.
#[cfg(all(feature = "rocm", any(test, feature = "test-support")))]
fn drain_at_exit(device: Device) {
    if crate::default_accelerators().first() != Some(&Accelerator::Rocm) {
        return;
    }

    if EXIT_CLIENT.get().is_some() {
        return;
    }

    let Some(client) = crate::probe::probe_client(Accelerator::Rocm, device) else {
        return;
    };

    // Only the call that stores the client registers the hook, so it is registered once.
    if EXIT_CLIENT.set(client).is_err() {
        return;
    }

    // Safety: `drain_exit_client` takes no arguments, never unwinds, and stays valid for the
    // life of the process.
    let failed = unsafe { atexit(drain_exit_client) };
    if failed != 0 {
        tracing::warn!("cannot drain the test device at exit");
    }
}

/// Syncs the ROCm test device's client on a new thread.
///
/// The exiting thread may already have dropped its thread locals, which the client needs.
#[cfg(all(feature = "rocm", any(test, feature = "test-support")))]
extern "C" fn drain_exit_client() {
    let Some(client) = EXIT_CLIENT.get() else {
        return;
    };

    let name = String::from("test-device-exit");
    let spawned = std::thread::Builder::new().name(name).spawn(|| {
        let synced = client.sync();
        let _ = cubecl::future::block_on(synced);
    });

    if let Ok(handle) = spawned {
        let _ = handle.join();
    }
}

/// Points CubeCL's environment inside the workspace's target directory, leaving a config that is
/// already set alone.
#[cfg(any(test, feature = "test-support"))]
fn install_test_tune_cache() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let directory = manifest.join("../target/test-autotune");

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.environment.path = CacheConfig::Directory(directory);

    // A config already set means something else chose where results go, which is kept.
    let _ = CubeClRuntimeConfig::try_set(config);
}

#[cfg(any(test, feature = "test-support"))]
fn device_from_value(value: Option<&str>) -> Device {
    let Some(text) = value.filter(|text| !text.is_empty()) else {
        return Device::Default;
    };

    match text.parse() {
        Ok(device) => device,
        Err(error) => panic!("AVTB_DEVICE is set to {text:?}, which is not a device selector: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_from_value_falls_back_to_default_when_unset_or_empty() {
        assert_eq!(device_from_value(None), Device::Default);
        assert_eq!(device_from_value(Some("")), Device::Default);
    }

    #[test]
    fn device_from_value_parses_a_selector() {
        assert_eq!(
            device_from_value(Some("discrete:1")),
            Device::Discrete { index: 1 }
        );
        assert_eq!(device_from_value(Some("cpu")), Device::Cpu);
    }

    #[test]
    #[should_panic(expected = "AVTB_DEVICE is set to \"nonsense\"")]
    fn device_from_value_panics_on_a_bad_selector() {
        device_from_value(Some("nonsense"));
    }
}
