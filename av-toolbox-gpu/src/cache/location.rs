use std::ffi::OsStr;
use std::path::PathBuf;

use etcetera::base_strategy::{BaseStrategy, choose_base_strategy};

/// The directory under the platform cache that holds every build's kernels.
const CACHE_DIR_NAME: &str = "av-toolbox";

/// The values of [COMPILATION_CACHE_ENV](crate::COMPILATION_CACHE_ENV) that turn caching off, compared
/// without regard to case.
///
/// `off` is the documented spelling. The others stop a reasonable guess from silently creating a directory
/// named `0`.
const DISABLE_WORDS: [&str; 4] = ["off", "0", "false", "none"];

/// Where compiled kernels go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CacheLocation {
    Disabled,
    Dir(PathBuf),
}

/// The directory compiled kernels are cached in when nothing overrides it.
pub fn default_cache_dir() -> PathBuf {
    let platform_cache = choose_base_strategy().ok().map(|strategy| strategy.cache_dir());
    if platform_cache.is_none() {
        tracing::warn!("no platform cache directory available, falling back to the temporary directory");
    }

    let temp_dir = std::env::temp_dir();
    resolve_default_dir(platform_cache, temp_dir)
}

pub(crate) fn resolve_default_dir(platform_cache: Option<PathBuf>, temp_dir: PathBuf) -> PathBuf {
    platform_cache.unwrap_or(temp_dir).join(CACHE_DIR_NAME)
}

/// Decides where compiled kernels go from the environment override, falling back to `default`.
///
/// An empty or blank override reads as unset. Any of the disable words, in any case and with padding
/// trimmed, turns caching off.
pub(crate) fn resolve_cache_location(
    env: Option<&OsStr>,
    default: impl FnOnce() -> PathBuf,
) -> CacheLocation {
    let Some(raw) = env else {
        let dir = default();
        return CacheLocation::Dir(dir);
    };

    let text = raw.to_string_lossy();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        let dir = default();
        return CacheLocation::Dir(dir);
    }

    let disabled = DISABLE_WORDS
        .iter()
        .any(|word| trimmed.eq_ignore_ascii_case(word));
    if disabled {
        return CacheLocation::Disabled;
    }

    // A value that is not UTF-8 cannot be trimmed portably, so it is used unchanged.
    let dir = match raw.to_str() {
        Some(utf8) => PathBuf::from(utf8.trim()),
        None => PathBuf::from(raw),
    };
    CacheLocation::Dir(dir)
}
