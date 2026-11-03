use std::sync::Once;

use av_toolbox_gpu::{
    Accelerator,
    Backend,
    CODEGEN_STACK_BYTES,
    COMPILATION_CACHE_ENV,
    CacheError,
    Device,
    Tuning,
    codegen_stack_is_sufficient,
    compilation_cache_dir,
    compilation_cache_disabled,
    default_accelerators,
    install_compilation_cache,
};
use av_toolbox_pipeline::OpenBackendError;

use crate::{SessionError, build_hash};

/// What a process without the kernel cache goes without, for the warning that says so.
const LOST_WITHOUT_CACHE: &str = "the tuning store, the tuning lock and parallel kernel compilation";

/// Which backend to open and how its stages pick their kernels.
#[derive(Debug, Clone)]
pub struct BackendOptions {
    /// The backends to try, in order of preference.
    pub accelerators: Vec<Accelerator>,
    pub device: Device,
    pub tuning: Tuning,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            accelerators: default_accelerators(),
            device: Device::Default,
            tuning: Tuning::Cached,
        }
    }
}

/// Installs the kernel cache and opens the first of the accelerators that starts on the chosen device.
///
/// The cache is installed once per process, so a later call reuses it. When something else in the process
/// configured CubeCL first, its settings are kept and a warning says what goes missing. A warning is logged
/// once when `RUST_MIN_STACK` is too small for kernel compilation.
///
/// With CUDA enabled and `CUDA_PATH` unset, the first call installs the bundled libdevice and sets
/// `CUDA_PATH` to it. Open the backend before spawning threads that read the environment.
pub fn open_backend(options: &BackendOptions) -> Result<Backend, SessionError> {
    install_cache();
    warn_if_stack_is_small();

    let opened = av_toolbox_pipeline::open_backend(&options.accelerators, options.device, options.tuning);
    let backend = opened.map_err(backend_error)?;

    let accelerator = backend.accelerator();
    tracing::info!(
        accelerator = %accelerator,
        device = %options.device,
        tuning = %options.tuning,
        "opened the backend"
    );

    Ok(backend)
}

/// Warns once per process when `RUST_MIN_STACK` is too small for kernel compilation.
fn warn_if_stack_is_small() {
    static WARN: Once = Once::new();

    WARN.call_once(warn_if_stack_is_small_now);
}

fn warn_if_stack_is_small_now() {
    if codegen_stack_is_sufficient() {
        return;
    }

    tracing::warn!(
        "RUST_MIN_STACK is below {CODEGEN_STACK_BYTES} bytes, kernel compilation may overflow its stack"
    );
}

/// Installs this build's kernel cache, once per process.
///
/// A cache that is off, or that could not be installed, is warned about.
pub(crate) fn install_cache() {
    static INSTALL: Once = Once::new();

    INSTALL.call_once(install_cache_now);
}

fn install_cache_now() {
    #[cfg(feature = "cuda")]
    install_libdevice();

    if compilation_cache_dir().is_some() {
        return;
    }

    let build = build_hash();
    let installed = install_compilation_cache(build);
    match installed {
        Ok(Some(path)) => tracing::info!(?path, "caching compiled kernels"),
        Ok(None) if compilation_cache_disabled() => warn_cache_off(),
        Ok(None) => warn_cache_not_installed(),
        Err(error) => warn_cache_kept_out(&error),
    }
}

/// Points `CUDA_PATH` at the bundled libdevice when it is unset, warning when the bundle cannot be installed.
#[cfg(feature = "cuda")]
fn install_libdevice() {
    // SAFETY: this runs once, inside the first `open_backend` or `Session::new`, and both document that they
    // must be called before the caller spawns threads that read the environment.
    let installed = unsafe { crate::libdevice::point_cuda_path_at_bundle() };
    if let Err(error) = installed {
        tracing::warn!(%error, "cannot install the bundled libdevice");
    }
}

fn warn_cache_off() {
    tracing::warn!(
        "kernel caching is off, so this process goes without {LOST_WITHOUT_CACHE}, unset \
         {COMPILATION_CACHE_ENV} to turn it back on"
    );
}

fn warn_cache_not_installed() {
    tracing::warn!(
        "the kernel cache could not be installed, so this process goes without {LOST_WITHOUT_CACHE}"
    );
}

fn warn_cache_kept_out(error: &CacheError) {
    tracing::warn!(
        %error,
        "the kernel cache could not be installed, so this process goes without {LOST_WITHOUT_CACHE}"
    );
}

fn backend_error(error: OpenBackendError) -> SessionError {
    let message = error.to_string();
    SessionError::message(message)
}
