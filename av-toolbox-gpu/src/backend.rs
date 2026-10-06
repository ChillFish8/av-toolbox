use std::sync::{Arc, OnceLock};

use cubecl::client::ComputeClient;
use cubecl::prelude::Runtime;
use cubecl::stream_id::StreamId;

use crate::memory::{BufferPool, ClientAllocator};
use crate::probe::{open_client, supported_device};
#[cfg(any(feature = "vulkan", feature = "metal"))]
use crate::resolve;
use crate::{Accelerator, Capabilities, Device, FrameContext, RuntimeError, StreamPolicy};

/// The one stream every client of a [StreamPolicy::SharedStream] backend queues on.
const SHARED_STREAM: StreamId = StreamId { value: 0 };

/// What a stage is built against.
///
/// It carries the client the stage will launch kernels on, what that device can do, and the pool its
/// frames come from.
pub struct BuildContext<R: Runtime> {
    client: ComputeClient<R>,
    capabilities: Capabilities,
    pool: BufferPool,
}

impl<R: Runtime> BuildContext<R> {
    fn new(accelerator: Accelerator, client: ComputeClient<R>) -> Self {
        let policy = StreamPolicy::for_accelerator(accelerator);

        Self::with_policy(accelerator, client, policy)
    }

    /// A context whose clients keep work across threads in order by `policy`.
    ///
    /// Under [StreamPolicy::SharedStream] the client is pinned to one stream before anything
    /// clones it, so every clone queues on that stream whichever thread uses it.
    pub(crate) fn with_policy(
        accelerator: Accelerator,
        mut client: ComputeClient<R>,
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

        Self::assemble(accelerator, client, policy)
    }

    /// A context that leaves each thread on its own stream and never flushes, so work handed
    /// between threads can run out of order.
    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn unordered(accelerator: Accelerator, client: ComputeClient<R>) -> Self {
        // The pool reports `SharedStream` only so it never flushes. The client is deliberately
        // left unpinned, so each thread keeps its own stream.
        Self::assemble(accelerator, client, StreamPolicy::SharedStream)
    }

    fn assemble(accelerator: Accelerator, client: ComputeClient<R>, policy: StreamPolicy) -> Self {
        let capabilities = Capabilities::from_properties(accelerator, client.properties());
        let pool_client = client.clone();
        let allocator = ClientAllocator::new(pool_client);
        let shared_allocator = Arc::new(allocator);
        let pool = BufferPool::new(shared_allocator, policy);

        Self {
            client,
            capabilities,
            pool,
        }
    }

    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    pub const fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub const fn pool(&self) -> &BufferPool {
        &self.pool
    }

    pub fn stream_policy(&self) -> StreamPolicy {
        self.pool.stream_policy()
    }

    /// The context this stage's work runs against.
    pub fn frame_context(&self) -> FrameContext {
        FrameContext::new(self.pool.clone(), self.capabilities.clone())
    }
}

/// Work to run against whichever runtime a backend opened.
///
/// The runtime is only known once the backend has been matched, so the work is written once as a
/// generic method and [Backend::build] supplies the context it made.
pub trait BuildTask {
    type Output;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output;
}

/// Blocks until the work the calling thread queued on the device has finished.
pub type DeviceSync = Arc<dyn Fn() -> Result<(), RuntimeError> + Send + Sync>;

/// An opened accelerator, ready to build stages against.
pub struct Backend {
    accelerator: Accelerator,
    device: Device,
    /// The device `CUBECL_WGPU_DEFAULT_DEVICE` named when [Device::Default] was opened.
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    wgpu_override: Option<Device>,
    resolved: OnceLock<Device>,
    inner: Opened,
}

enum Opened {
    #[cfg(feature = "cuda")]
    Cuda(BuildContext<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(BuildContext<cubecl::hip::HipRuntime>),
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    Wgpu(BuildContext<cubecl::wgpu::WgpuRuntime>),
}

impl Backend {
    /// Opens `accelerator` on `device`, or reports that it cannot run here.
    ///
    /// Unlike probing, the client is kept, because everything built against this backend launches
    /// its kernels on it.
    pub fn open(accelerator: Accelerator, device: Device) -> Option<Self> {
        // CubeCL reads the override as the client opens, and only for the default device.
        #[cfg(any(feature = "vulkan", feature = "metal"))]
        let wgpu_override = if device == Device::Default {
            resolve::wgpu_override()
        } else {
            None
        };

        let inner = match accelerator {
            #[cfg(feature = "cuda")]
            Accelerator::Cuda => {
                let selector = device;
                let device = supported_device(device.to_cuda())?;
                let client = open_client::<cubecl::cuda::CudaRuntime>(accelerator, selector, &device)?;

                Opened::Cuda(BuildContext::new(accelerator, client))
            },
            #[cfg(feature = "rocm")]
            Accelerator::Rocm => {
                let selector = device;
                let device = supported_device(device.to_amd())?;
                let client = open_client::<cubecl::hip::HipRuntime>(accelerator, selector, &device)?;

                Opened::Rocm(BuildContext::new(accelerator, client))
            },
            #[cfg(feature = "vulkan")]
            Accelerator::Vulkan => {
                let selector = device;
                let device = supported_device(device.to_wgpu())?;
                let client = open_client::<cubecl::wgpu::WgpuRuntime>(accelerator, selector, &device)?;

                Opened::Wgpu(BuildContext::new(accelerator, client))
            },
            #[cfg(feature = "metal")]
            Accelerator::Metal => {
                let selector = device;
                let device = supported_device(device.to_wgpu())?;
                let client = open_client::<cubecl::wgpu::WgpuRuntime>(accelerator, selector, &device)?;

                Opened::Wgpu(BuildContext::new(accelerator, client))
            },
            // docs.rs widens the `Accelerator` variants behind `cfg(docsrs)` so they all appear in
            // the rendered enum, even when the matching backend feature is off. This arm keeps the
            // match exhaustive there and is never reached at runtime.
            #[cfg(docsrs)]
            _ => unreachable!(),
        };

        Some(Self {
            accelerator,
            device,
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            wgpu_override,
            resolved: OnceLock::new(),
            inner,
        })
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

    pub const fn accelerator(&self) -> Accelerator {
        self.accelerator
    }

    /// The device this backend runs on, with [Device::Default] resolved to the one the backend picked.
    ///
    /// It stays [Device::Default] only when no selector can name the device picked. Resolving
    /// asks the backend for its devices on the first call, and later calls reuse the answer.
    pub fn device(&self) -> Device {
        if self.device != Device::Default {
            return self.device;
        }

        *self.resolved.get_or_init(|| self.resolve_default())
    }

    fn resolve_default(&self) -> Device {
        match &self.inner {
            #[cfg(feature = "cuda")]
            Opened::Cuda(_) => Device::Discrete { index: 0 },
            #[cfg(feature = "rocm")]
            Opened::Rocm(_) => Device::Discrete { index: 0 },
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Opened::Wgpu(context) => resolve::wgpu_default(context.client(), self.wgpu_override),
        }
    }

    /// What this backend's device can do.
    pub fn capabilities(&self) -> &Capabilities {
        match &self.inner {
            #[cfg(feature = "cuda")]
            Opened::Cuda(context) => context.capabilities(),
            #[cfg(feature = "rocm")]
            Opened::Rocm(context) => context.capabilities(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Opened::Wgpu(context) => context.capabilities(),
        }
    }

    pub fn stream_policy(&self) -> StreamPolicy {
        match &self.inner {
            #[cfg(feature = "cuda")]
            Opened::Cuda(context) => context.stream_policy(),
            #[cfg(feature = "rocm")]
            Opened::Rocm(context) => context.stream_policy(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Opened::Wgpu(context) => context.stream_policy(),
        }
    }

    /// The context stages built against this backend do their work through.
    pub fn frame_context(&self) -> FrameContext {
        match &self.inner {
            #[cfg(feature = "cuda")]
            Opened::Cuda(context) => context.frame_context(),
            #[cfg(feature = "rocm")]
            Opened::Rocm(context) => context.frame_context(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Opened::Wgpu(context) => context.frame_context(),
        }
    }

    /// A hook blocking until the work the calling thread queued on this device has finished.
    pub fn device_sync(&self) -> DeviceSync {
        match &self.inner {
            #[cfg(feature = "cuda")]
            Opened::Cuda(context) => sync_hook(context.client()),
            #[cfg(feature = "rocm")]
            Opened::Rocm(context) => sync_hook(context.client()),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Opened::Wgpu(context) => sync_hook(context.client()),
        }
    }

    /// Runs `task` against this backend's context.
    pub fn build<T: BuildTask>(&self, task: T) -> T::Output {
        match &self.inner {
            #[cfg(feature = "cuda")]
            Opened::Cuda(context) => task.run(context),
            #[cfg(feature = "rocm")]
            Opened::Rocm(context) => task.run(context),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Opened::Wgpu(context) => task.run(context),
        }
    }
}

fn sync_hook<R: Runtime>(client: &ComputeClient<R>) -> DeviceSync {
    let client = client.clone();

    Arc::new(move || {
        let pending = client.sync();
        let synced = cubecl::future::block_on(pending);

        synced.map_err(RuntimeError::Sync)
    })
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

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
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
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn one_backend_hands_out_one_pool() {
        let backend = Backend::open(TEST_ACCELERATOR, test_device()).expect("should open");

        struct ReserveTwice;

        impl BuildTask for ReserveTwice {
            type Output = usize;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                drop(context.pool().reserve(1024));
                let reused = context.pool().reserve(1024);

                reused.bytes()
            }
        }

        assert_eq!(backend.build(ReserveTwice), 1024);
    }
}
