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
    default_accelerators,
    install_compilation_cache,
};
use av_toolbox_pipeline::OpenBackendError;

use crate::SessionError;

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
/// configured CubeCL first, its settings are kept. A warning is logged when `RUST_MIN_STACK` is too small
/// for kernel compilation.
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

fn warn_if_stack_is_small() {
    if codegen_stack_is_sufficient() {
        return;
    }

    tracing::warn!(
        "RUST_MIN_STACK is below {CODEGEN_STACK_BYTES} bytes, kernel compilation may overflow its stack"
    );
}

fn install_cache() {
    if compilation_cache_dir().is_some() {
        return;
    }

    let installed = install_compilation_cache();
    match installed {
        Ok(Some(path)) => tracing::info!(?path, "caching compiled kernels"),
        Ok(None) => tracing::info!(
            "kernel caching is off and autotune results go to CubeCL's default place, unset \
             {COMPILATION_CACHE_ENV} to turn it back on"
        ),
        Err(CacheError::AlreadyInitialised) => tracing::info!(
            "CubeCL was configured before the kernel cache could be installed, keeping its settings"
        ),
    }
}

fn backend_error(error: OpenBackendError) -> SessionError {
    let message = error.to_string();
    SessionError::message(message)
}
