use std::sync::Arc;

use cubecl::prelude::Client;
use cubecl_environment::stream::StreamId;

use crate::memory::{BufferPool, ClientAllocator};
use crate::probe::{open_client, supported_device};
use crate::{Accelerator, Capabilities, Device, FrameContext, RuntimeError, StreamPolicy, Tuning, resolve};

/// The one stream every client of a [StreamPolicy::SharedStream] backend queues on.
const SHARED_STREAM: StreamId = StreamId { value: 0 };

/// What a stage is built against.
///
/// It carries the client the stage will launch kernels on, what that device can do, and the pool its
/// frames come from.
pub struct BuildContext {
    client: Client,
    cubecl_device: cubecl::Device,
    capabilities: Capabilities,
    pool: BufferPool,
    device: Device,
    tuning: Tuning,
}

impl BuildContext {
    fn new(accelerator: Accelerator, device: Device, cubecl_device: cubecl::Device, client: Client) -> Self {
        let policy = StreamPolicy::for_accelerator(accelerator);

        Self::with_policy(accelerator, device, cubecl_device, client, policy)
    }

    /// A context whose clients keep work across threads in order by `policy`.
    ///
    /// Under [StreamPolicy::SharedStream] the client is pinned to one stream before anything
    /// clones it, so every clone queues on that stream whichever thread uses it.
    pub(crate) fn with_policy(
        accelerator: Accelerator,
        device: Device,
        cubecl_device: cubecl::Device,
        mut client: Client,
        policy: StreamPolicy,
    ) -> Self {
        if policy == StreamPolicy::SharedStream {
            // Safety: CubeCL gives no contract beyond warning against this. Every clone of this
            // client is taken after this point, so all of them agree on the stream, and a stream
            // shared between threads is ordered by the order work reaches it.
            unsafe {
                client.set_stream(SHARED_STREAM);
            }
        }

        Self::assemble(accelerator, device, cubecl_device, client, policy)
    }

    /// A context that leaves each thread on its own stream and never flushes, so work handed
    /// between threads can run out of order.
    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn unordered(
        accelerator: Accelerator,
        device: Device,
        cubecl_device: cubecl::Device,
        client: Client,
    ) -> Self {
        // The pool reports `SharedStream` only so it never flushes. The client is deliberately
        // left unpinned, so each thread keeps its own stream.
        Self::assemble(
            accelerator,
            device,
            cubecl_device,
            client,
            StreamPolicy::SharedStream,
        )
    }

    fn assemble(
        accelerator: Accelerator,
        device: Device,
        cubecl_device: cubecl::Device,
        client: Client,
        policy: StreamPolicy,
    ) -> Self {
        let capabilities = Capabilities::from_properties(accelerator, client.properties());
        let pool_client = client.clone();
        let allocator = ClientAllocator::new(pool_client);
        let shared_allocator = Arc::new(allocator);
        let pool = BufferPool::new(shared_allocator, policy);

        Self {
            client,
            cubecl_device,
            capabilities,
            pool,
            device,
            tuning: Tuning::default(),
        }
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The CubeCL device the client was opened on.
    pub fn cubecl_device(&self) -> &cubecl::Device {
        &self.cubecl_device
    }

    pub const fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub const fn pool(&self) -> &BufferPool {
        &self.pool
    }

    /// The device the pipeline runs on, with [Device::Default] resolved where the backend can.
    pub const fn device(&self) -> Device {
        self.device
    }

    /// How stages built against this context pick between their kernels' candidate versions.
    pub const fn tuning(&self) -> Tuning {
        self.tuning
    }

    pub fn stream_policy(&self) -> StreamPolicy {
        self.pool.stream_policy()
    }

    /// The context this stage's work runs against.
    pub fn frame_context(&self) -> FrameContext {
        FrameContext::new(self.pool.clone(), self.capabilities.clone())
    }
}

/// Work to run against the context a backend opened.
pub trait BuildTask {
    type Output;

    fn run(self, context: &BuildContext) -> Self::Output;
}

/// Blocks until the work the calling thread queued on the device has finished.
pub type DeviceSync = Arc<dyn Fn() -> Result<(), RuntimeError> + Send + Sync>;

/// An opened accelerator, ready to build stages against.
pub struct Backend {
    accelerator: Accelerator,
    context: BuildContext,
}

/// The selector naming the device a client opened on `selected` runs on.
///
/// CUDA and ROCm number their devices from zero, so their default is the first one. A wgpu
/// default is asked of the adapters the backend lists.
fn resolved_selector(selector: Device, selected: &cubecl::Device, overridden: Option<Device>) -> Device {
    if selector != Device::Default {
        return selector;
    }

    match selected {
        cubecl::Device::Wgpu(wgpu_device) => resolve::wgpu_default(wgpu_device.backend, overridden),
        _ => Device::Discrete { index: 0 },
    }
}

impl Backend {
    /// Opens `accelerator` on `device`, or reports that it cannot run here.
    ///
    /// Unlike probing, the client is kept, because everything built against this backend launches
    /// its kernels on it.
    pub fn open(accelerator: Accelerator, device: Device) -> Option<Self> {
        // CubeCL reads the override as the client opens, and only for the default device.
        let wgpu_override = if device == Device::Default {
            resolve::wgpu_override()
        } else {
            None
        };

        let converted = device.to_cubecl(accelerator);
        let selected = supported_device(converted)?;
        let client = open_client(accelerator, device, &selected)?;
        let resolved = resolved_selector(device, &selected, wgpu_override);
        let context = BuildContext::new(accelerator, resolved, selected, client);

        Some(Self { accelerator, context })
    }

    /// Opens the first accelerator this build enables that starts on `device`.
    ///
    /// This is the same order [default_accelerators](crate::default_accelerators) gives, so a
    /// caller that wants a particular backend names it and one that does not takes what works.
    pub fn open_default(device: Device) -> Option<Self> {
        crate::default_accelerators()
            .into_iter()
            .find_map(|accelerator| Self::open(accelerator, device))
    }

    /// Has stages built against this backend pick their kernels by `tuning`.
    pub fn with_tuning(mut self, tuning: Tuning) -> Self {
        self.context.tuning = tuning;
        self
    }

    pub const fn accelerator(&self) -> Accelerator {
        self.accelerator
    }

    /// The context every stage built against this backend shares.
    pub const fn context(&self) -> &BuildContext {
        &self.context
    }

    /// The device this backend runs on, with [Device::Default] resolved to the one the backend picked.
    ///
    /// It stays [Device::Default] only when no selector can name the device picked.
    pub const fn device(&self) -> Device {
        self.context.device()
    }

    /// What this backend's device can do.
    pub const fn capabilities(&self) -> &Capabilities {
        self.context.capabilities()
    }

    pub fn stream_policy(&self) -> StreamPolicy {
        self.context.stream_policy()
    }

    /// The context stages built against this backend do their work through.
    pub fn frame_context(&self) -> FrameContext {
        self.context.frame_context()
    }

    /// A hook blocking until the work the calling thread queued on this device has finished.
    pub fn device_sync(&self) -> DeviceSync {
        let client = self.context.client().clone();

        Arc::new(move || {
            let pending = client.sync();
            let synced = cubecl::future::block_on(pending);

            synced.map_err(RuntimeError::Sync)
        })
    }

    /// Runs `task` against this backend's context.
    pub fn build<T: BuildTask>(&self, task: T) -> T::Output {
        task.run(&self.context)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use super::*;
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use crate::test_device;
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use crate::testing::TEST_ACCELERATOR;

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_device_sync_returns_on_an_idle_backend() {
        let backend = Backend::open(TEST_ACCELERATOR, test_device()).expect("should open");
        let sync = backend.device_sync();

        sync().expect("an idle device syncs");
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_backend_opens_and_runs_work_against_its_client() {
        let backend = Backend::open(TEST_ACCELERATOR, test_device()).expect("should open");

        assert_eq!(backend.accelerator(), TEST_ACCELERATOR);
        assert!(backend.capabilities().can_compute(crate::ElementType::F32));

        /// Work that reports what it was handed, which is all this needs to prove the dispatch
        /// reaches a real client.
        struct ReportAlignment;

        impl BuildTask for ReportAlignment {
            type Output = usize;

            fn run(self, context: &BuildContext) -> Self::Output {
                assert_eq!(
                    context.pool().alignment(),
                    context.capabilities().alignment as usize
                );

                context.pool().alignment()
            }
        }

        let alignment = backend.build(ReportAlignment);
        assert!(alignment.is_power_of_two(), "{alignment}");
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_device_the_backend_does_not_have_does_not_open() {
        assert!(Backend::open(TEST_ACCELERATOR, Device::Discrete { index: 99 }).is_none());
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn the_default_device_resolves_to_one_the_backend_lists() {
        // Opening the default device would leave the GPU the tests are pinned to.
        if test_device() != Device::Default {
            return;
        }

        let backend = Backend::open(TEST_ACCELERATOR, Device::Default).expect("should open");
        let reported = crate::enumerate_devices(&[TEST_ACCELERATOR]);

        let resolved = backend.device();

        assert_ne!(resolved, Device::Default);
        assert!(
            reported[0].devices.contains(&resolved),
            "{resolved} is not listed"
        );

        struct ReportDevice;

        impl BuildTask for ReportDevice {
            type Output = Device;

            fn run(self, context: &BuildContext) -> Self::Output {
                context.device()
            }
        }

        let context_device = backend.build(ReportDevice);

        assert_ne!(context_device, Device::Default);
        assert!(
            reported[0].devices.contains(&context_device),
            "{context_device} is not listed"
        );
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_backend_hands_its_tuning_to_the_context() {
        let backend = Backend::open(TEST_ACCELERATOR, test_device()).expect("should open");
        assert_eq!(backend.context().tuning(), Tuning::Cached);

        let tuned = backend.with_tuning(Tuning::Off);
        assert_eq!(tuned.context().tuning(), Tuning::Off);
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn one_backend_hands_out_one_pool() {
        let backend = Backend::open(TEST_ACCELERATOR, test_device()).expect("should open");

        struct ReserveTwice;

        impl BuildTask for ReserveTwice {
            type Output = usize;

            fn run(self, context: &BuildContext) -> Self::Output {
                drop(context.pool().reserve(1024));
                let reused = context.pool().reserve(1024);

                reused.bytes()
            }
        }

        assert_eq!(backend.build(ReserveTwice), 1024);
    }
}
