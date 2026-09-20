use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use cubecl::config::cache::CacheConfig;
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use etcetera::base_strategy::{BaseStrategy, choose_base_strategy};

/// The variable that overrides where compiled kernels are cached.
pub const COMPILATION_CACHE_ENV: &str = "AVTB_COMPILATION_CACHE";

/// Where the cache ended up, once something has installed it.
static INSTALLED: OnceLock<Option<PathBuf>> = OnceLock::new();

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("CubeCL's global config is already set, install the cache before opening any client")]
    AlreadyInitialised,
}

/// Where kernels are cached when nothing says otherwise.
pub fn default_cache_dir() -> PathBuf {
    let platform_cache = choose_base_strategy().ok().map(|strategy| strategy.cache_dir());
    match platform_cache {
        Some(cache) => cache.join("av-toolbox"),
        None => {
            tracing::warn!("no platform cache directory available, using the temporary directory");
            std::env::temp_dir().join("av-toolbox")
        },
    }
}

/// Points CubeCL at the cache directory, honouring [COMPILATION_CACHE_ENV].
///
/// Returns where the cache went, or `None` when the variable turned caching off.
///
/// This has to run before the first client is opened, because opening one locks CubeCL's global
/// config.
pub fn install_compilation_cache() -> Result<Option<PathBuf>, CacheError> {
    let raw = std::env::var_os(COMPILATION_CACHE_ENV);
    let default = default_cache_dir();

    let Some(path) = resolve_cache_dir(raw.as_deref(), default) else {
        return Ok(None);
    };

    set_runtime_config(&path)?;
    let _ = INSTALLED.set(Some(path.clone()));

    Ok(Some(path))
}

/// Points CubeCL at `dir`, ignoring [COMPILATION_CACHE_ENV].
///
/// This is for a library consumer that wants to choose the location itself.
///
/// This has to run before the first client is opened, because opening one locks CubeCL's global
/// config.
pub fn install_compilation_cache_at(dir: &Path) -> Result<(), CacheError> {
    set_runtime_config(dir)?;
    let _ = INSTALLED.set(Some(dir.to_path_buf()));

    Ok(())
}

/// Installs the cache, leaving CubeCL alone if something else configured it first.
///
/// Returns where the cache went, or `None` when caching is off or somebody else got there first.
///
/// Call this once, before other threads exist, since two concurrent first calls can race and leave
/// [compilation_cache_dir] reporting `None` for a cache that is actually configured.
pub fn install_compilation_cache_once() -> Option<&'static Path> {
    if INSTALLED.get().is_none()
        && let Err(error) = install_compilation_cache()
    {
        tracing::debug!("{error}, leaving its kernel cache alone");
    }

    compilation_cache_dir()
}

/// Where the cache was installed, or `None` if it was not.
pub fn compilation_cache_dir() -> Option<&'static Path> {
    INSTALLED.get()?.as_deref()
}

/// Reads the override variable, falling back to `default`.
///
/// `off` in any casing, with any surrounding space, disables caching. An empty value reads as unset.
fn resolve_cache_dir(raw: Option<&OsStr>, default: PathBuf) -> Option<PathBuf> {
    let Some(raw) = raw else {
        return Some(default);
    };

    // A value that is not UTF-8 is still a usable path, so only a value that reads as text can be
    // the empty string or `off`.
    let Some(text) = raw.to_str().map(str::trim) else {
        return Some(PathBuf::from(raw));
    };

    match text {
        "" => Some(default),
        value if value.eq_ignore_ascii_case("off") => None,
        _ => Some(PathBuf::from(raw)),
    }
}

/// Points CubeCL's global config at `path`.
fn set_runtime_config(path: &Path) -> Result<(), CacheError> {
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = Some(CacheConfig::File(path.to_path_buf()));
    config.autotune.cache = CacheConfig::File(path.to_path_buf());

    // `RuntimeConfig::set` panics when the singleton is already set, and CubeCL exposes no fallible
    // version, so catching the panic is the only way to turn that into an error a caller can act
    // on. The hook is quietened the same way probing does it, so a handled panic here does not
    // still print to stderr.
    let outcome = crate::probe::quiet_panics(|| {
        CubeClRuntimeConfig::set(config);
    });

    outcome.map_err(|_| CacheError::AlreadyInitialised)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    #[test]
    fn an_unset_variable_falls_back_to_the_default() {
        let resolved = resolve_cache_dir(None, PathBuf::from("/default"));
        assert_eq!(resolved, Some(PathBuf::from("/default")));
    }

    #[test]
    fn a_path_in_the_variable_wins() {
        let raw = OsString::from("/mnt/cache");
        let resolved = resolve_cache_dir(Some(&raw), PathBuf::from("/default"));

        assert_eq!(resolved, Some(PathBuf::from("/mnt/cache")));
    }

    /// Benchmarks turn the cache off, because a warm cache hides the compilation cost a first run
    /// pays.
    #[test]
    fn the_off_value_disables_caching() {
        let raw = OsString::from("off");
        assert_eq!(resolve_cache_dir(Some(&raw), PathBuf::from("/default")), None);
    }

    #[test]
    fn the_off_value_is_case_insensitive_and_ignores_surrounding_space() {
        for spelling in ["OFF", " off ", "Off"] {
            let raw = OsString::from(spelling);
            assert_eq!(
                resolve_cache_dir(Some(&raw), PathBuf::from("/default")),
                None,
                "{spelling}",
            );
        }
    }

    #[test]
    fn an_empty_variable_falls_back_to_the_default() {
        let raw = OsString::from("");
        let resolved = resolve_cache_dir(Some(&raw), PathBuf::from("/default"));

        assert_eq!(resolved, Some(PathBuf::from("/default")));
    }

    /// A path the platform accepts but Rust cannot read as text is still a path, not an unset
    /// variable.
    #[cfg(unix)]
    #[test]
    fn a_path_that_is_not_utf8_is_still_used() {
        use std::os::unix::ffi::OsStringExt;

        let raw = OsString::from_vec(vec![b'/', b'm', b'n', b't', b'/', 0xFF]);
        let resolved = resolve_cache_dir(Some(&raw), PathBuf::from("/default"));

        assert_eq!(resolved, Some(PathBuf::from(raw)));
    }
}
