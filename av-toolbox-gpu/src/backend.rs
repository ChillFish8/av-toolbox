use std::sync::Arc;

use cubecl::client::ComputeClient;
use cubecl::prelude::Runtime;

use crate::memory::{BufferPool, ClientAllocator};
use crate::probe::{open_client, supported_device};
use crate::{Accelerator, Capabilities, Device, FrameContext};

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
        let capabilities = Capabilities::from_properties(accelerator, client.properties());
        let allocator = ClientAllocator::new(client.clone());

        Self {
            client,
            capabilities,
            pool: BufferPool::new(Arc::new(allocator)),
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

/// An opened accelerator, ready to build stages against.
pub struct Backend {
    accelerator: Accelerator,
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

        Some(Self { accelerator, inner })
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

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use super::*;
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use crate::testing::TEST_ACCELERATOR;

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_backend_opens_and_runs_work_against_its_client() {
        let backend = Backend::open(TEST_ACCELERATOR, Device::Default).expect("should open");

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
    fn one_backend_hands_out_one_pool() {
        let backend = Backend::open(TEST_ACCELERATOR, Device::Default).expect("should open");

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
