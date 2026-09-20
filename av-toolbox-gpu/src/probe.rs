use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use crate::{Accelerator, Device, DeviceError};

/// Backend/selector pairs already reported as unavailable, and the lock guarding the panic hook.
///
/// The hook is process wide, so two probes running at once would race to restore each other's.
/// Holding this for the length of a probe keeps them in single file, and the list inside it
/// keeps a pair from warning again every time it is probed.
static PROBED: Mutex<Vec<(Accelerator, Device)>> = Mutex::new(Vec::new());

/// Work to run against whichever CubeCL runtime backs an accelerator.
///
/// The runtime is a type, and which type it is only becomes known once the accelerator has
/// been matched. Writing the work as one generic method lets [with_runtime] supply the runtime
/// it picked, so the match that picks it is written once rather than once per caller.
pub(crate) trait RuntimeTask {
    type Output;

    fn run<R: Runtime>(self, client: &ComputeClient<R>) -> Self::Output;
}

/// Opens a client for `accelerator` on `device` and hands it to `task`.
///
/// Returns `None` when the backend cannot express `device`, and when its driver libraries are
/// missing so no client opens at all.
///
/// Each call opens a fresh client and synchronises it under a process-wide lock, which is what
/// makes it a probe. Code that dispatches per frame wants a client it holds onto instead.
pub(crate) fn with_runtime<T>(accelerator: Accelerator, selector: Device, task: T) -> Option<T::Output>
where
    T: RuntimeTask,
{
    match accelerator {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            let device = supported_device(selector.to_cuda())?;
            let client = open_client::<cubecl::cuda::CudaRuntime>(accelerator, selector, &device)?;
            Some(task.run(&client))
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            let device = supported_device(selector.to_amd())?;
            let client = open_client::<cubecl::hip::HipRuntime>(accelerator, selector, &device)?;
            Some(task.run(&client))
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            let device = supported_device(selector.to_wgpu())?;
            let client = open_client::<cubecl::wgpu::WgpuRuntime>(accelerator, selector, &device)?;
            Some(task.run(&client))
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            let device = supported_device(selector.to_wgpu())?;
            let client = open_client::<cubecl::wgpu::WgpuRuntime>(accelerator, selector, &device)?;
            Some(task.run(&client))
        },
        // docs.rs widens the `Accelerator` variants behind `cfg(docsrs)` so they all appear in
        // the rendered enum, even when the matching backend feature is off. This arm keeps the
        // match exhaustive there and is never reached at runtime.
        #[cfg(docsrs)]
        _ => unreachable!(),
    }
}

/// Unwraps a device conversion, logging why the selector does not fit this backend.
pub(crate) fn supported_device<T>(converted: Result<T, DeviceError>) -> Option<T> {
    match converted {
        Ok(device) => Some(device),
        Err(error) => {
            tracing::debug!("{error}");
            None
        },
    }
}

/// Opens a client for `accelerator` on `device`, or reports that the backend cannot run here.
///
/// A build can enable a backend whose driver libraries are not installed, and some backends do
/// not report that as an error. The CUDA runtime loads `libcuda` on its own worker thread and
/// panics there when the load fails, and that panic reaches the caller as a second panic when
/// CubeCL unwraps the dead worker's channel. Running the open under [quiet_panics] turns a
/// missing driver into "this backend is not available here", which is what makes one binary
/// with several backends enabled usable on a machine that has only one of them.
///
/// The client is synchronised before it is handed back. CubeCL kernels are fully asynchronous,
/// so a successful `sync` is what proves the backend works and no test kernel is needed.
pub(crate) fn open_client<R: Runtime>(
    accelerator: Accelerator,
    selector: Device,
    device: &R::Device,
) -> Option<ComputeClient<R>> {
    let mut probed = PROBED.lock().unwrap_or_else(|error| error.into_inner());

    let opened = quiet_panics(|| {
        let client = R::client(device);
        cubecl::future::block_on(client.sync()).map(|()| client)
    });

    match opened {
        Ok(Ok(client)) => Some(client),
        Ok(Err(error)) => {
            tracing::debug!(error = ?error, "could not use the {accelerator} runtime");
            None
        },
        Err(_) => {
            if !probed.contains(&(accelerator, selector)) {
                probed.push((accelerator, selector));

                if selector == Device::Default {
                    tracing::warn!(
                        "the {accelerator} backend is enabled but did not start, its driver \
                         libraries are probably missing"
                    );
                } else {
                    tracing::warn!(
                        "the {accelerator} backend did not start on {selector}, which may not exist"
                    );
                }
            }

            None
        },
    }
}

/// Runs `work`, turning a panic into an `Err` and sending the panic message to the debug log
/// rather than to stderr.
///
/// A failing backend prints its own panic from its worker thread before the caller ever sees
/// one, so the hook is quietened for as long as `work` runs and put back afterwards. The hook
/// is process wide, so callers hold [PROBED] across this.
pub(crate) fn quiet_panics<T>(work: impl FnOnce() -> T) -> std::thread::Result<T> {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|info| tracing::debug!("{info}")));
    let result = panic::catch_unwind(AssertUnwindSafe(work));
    panic::set_hook(previous);

    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn a_panic_inside_becomes_an_error() {
        let _probed = PROBED.lock().unwrap_or_else(|error| error.into_inner());

        assert!(quiet_panics(|| panic!("the backend fell over")).is_err());
        assert_eq!(quiet_panics(|| 7).expect("should not panic"), 7);
    }

    /// The hook has to come back however the probe ended, or every later panic in the process
    /// is reported at debug level. Holding `PROBED`, the way `open_client` does, keeps a probe
    /// on another thread from swapping the hook mid-test.
    #[test]
    fn the_panic_hook_is_restored() {
        let _probed = PROBED.lock().unwrap_or_else(|error| error.into_inner());

        let marker = Arc::new(AtomicBool::new(false));
        let flag = marker.clone();
        panic::set_hook(Box::new(move |_| flag.store(true, Ordering::SeqCst)));

        let _ = quiet_panics(|| panic!("swallowed by the quiet hook"));
        let swallowed = marker.load(Ordering::SeqCst);

        let _ = panic::catch_unwind(|| panic!("seen by the restored hook"));
        let restored = marker.load(Ordering::SeqCst);
        let _ = panic::take_hook();

        assert!(!swallowed, "the quiet hook did not replace the installed one");
        assert!(restored, "the probe left its own panic hook installed");
    }
}
