mod location;
mod prepare;
#[cfg(test)]
mod tests;
mod tidy;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Once, OnceLock};
use std::time::Duration;

use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

pub use self::location::default_cache_dir;
use self::location::{CacheLocation, resolve_cache_location};
use self::prepare::prepare_kernel_cache;
use self::tidy::{build_cache_dir, tidy_cache_root};

/// The variable that overrides where compiled kernels and autotune results are cached, or turns caching off.
///
/// `off`, `0`, `false` and `none` turn caching off, in any case. An empty value means the default. With
/// caching off, CubeCL keeps autotune results in its own default place: `target/environment` beside the
/// outermost `Cargo.toml` above the working directory, or the user cache directory when there is none.
pub const COMPILATION_CACHE_ENV: &str = "AVTB_COMPILATION_CACHE";

/// How long another build's directory can go unused before an install removes it.
pub(crate) const STALE_BUILD_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// This build's cache directory, once an install has succeeded.
static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Something went wrong installing the kernel cache.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// CubeCL's global config was set before the cache could be installed.
    #[error("CubeCL's global config is already set, install the cache before opening any client")]
    AlreadyInitialised,
    /// The build directory does not exist and could not be created.
    #[error("cannot create the kernel cache directory {path}", path = path.display())]
    CreateDir {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Installs the cache under [default_cache_dir], or the directory [COMPILATION_CACHE_ENV] names.
///
/// Returns the directory of `build` under that root, or `Ok(None)` when caching is off. A build directory
/// that cannot be created is logged and the install is skipped, because everything works without a cache.
///
/// This has to run before the first client is opened, because opening one locks CubeCL's global config.
pub fn install_compilation_cache(build: &str) -> Result<Option<PathBuf>, CacheError> {
    let env_override = std::env::var_os(COMPILATION_CACHE_ENV);
    let location = resolve_cache_location(env_override.as_deref(), default_cache_dir);

    let CacheLocation::Dir(root) = location else {
        return Ok(None);
    };

    install_or_skip(&root, build)
}

/// Installs the cache in the directory of `build` under `root`, ignoring [COMPILATION_CACHE_ENV].
///
/// Each build keeps its kernels in a directory of its own, and an install removes the directories of other
/// builds once they have gone unused for a week. Every process reads and writes the same cache file.
///
/// This has to run before the first client is opened, because opening one locks CubeCL's global config.
pub fn install_compilation_cache_at(root: &Path, build: &str) -> Result<PathBuf, CacheError> {
    let build_dir = build_cache_dir(root, build);
    if let Err(source) = std::fs::create_dir_all(&build_dir) {
        return Err(CacheError::CreateDir {
            path: build_dir,
            source,
        });
    }

    set_runtime_config(&build_dir)?;
    tidy_cache_root(root, build);

    let _ = CACHE_DIR.set(build_dir.clone());

    Ok(build_dir)
}

/// Installs the cache the first time it runs, and reports where it went.
///
/// This suits callers without a `main`, such as plugins. A failure is logged and ignored, since it means
/// something else configured CubeCL first, so this answers `None` rather than guessing where its cache is.
pub fn install_compilation_cache_once(build: &str) -> Option<&'static Path> {
    static ONCE: Once = Once::new();

    ONCE.call_once(|| log_install(build));

    compilation_cache_dir()
}

/// Whether [COMPILATION_CACHE_ENV] holds one of the words that turn caching off.
pub fn compilation_cache_disabled() -> bool {
    let env_override = std::env::var_os(COMPILATION_CACHE_ENV);
    let location = resolve_cache_location(env_override.as_deref(), PathBuf::new);
    location == CacheLocation::Disabled
}

/// This build's cache directory, or `None` when no install has succeeded.
pub fn compilation_cache_dir() -> Option<&'static Path> {
    CACHE_DIR.get().map(PathBuf::as_path)
}

/// Installs the cache in `root`, logging a build directory that cannot be created and answering `Ok(None)`.
pub(crate) fn install_or_skip(root: &Path, build: &str) -> Result<Option<PathBuf>, CacheError> {
    let installed = install_compilation_cache_at(root, build);
    match installed {
        Ok(build_dir) => Ok(Some(build_dir)),
        Err(CacheError::CreateDir { path, source }) => {
            static WARNED: Once = Once::new();

            WARNED.call_once(|| warn_create_dir_failed(&path, &source));
            Ok(None)
        },
        Err(error) => Err(error),
    }
}

fn warn_create_dir_failed(path: &Path, source: &io::Error) {
    tracing::warn!(?path, %source, "cannot create the kernel cache directory, continuing without a cache");
}

fn log_install(build: &str) {
    let installed = install_compilation_cache(build);
    match installed {
        Ok(Some(path)) => tracing::info!(?path, "caching compiled kernels"),
        Ok(None) => tracing::info!("kernel caching is off, every run recompiles"),
        Err(error) => tracing::warn!(%error, "leaving the kernel cache CubeCL already has alone"),
    }
}

/// Points CubeCL's global config at the cache in `build_dir`, then opens the cache once before CubeCL does.
fn set_runtime_config(build_dir: &Path) -> Result<(), CacheError> {
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = true;
    config.environment.path = CacheConfig::Directory(build_dir.to_path_buf());

    let installed = CubeClRuntimeConfig::try_set(config);
    if !installed {
        return Err(CacheError::AlreadyInitialised);
    }

    let database = cubecl::environment::path();
    prepare_kernel_cache(&database);

    Ok(())
}
