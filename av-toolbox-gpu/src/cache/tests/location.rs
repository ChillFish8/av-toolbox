use std::ffi::OsString;
use std::path::PathBuf;

use crate::cache::location::{CacheLocation, resolve_cache_location, resolve_default_dir};

const DEFAULT: &str = "/home/u/.cache/av-toolbox";

fn resolve(env: Option<&str>) -> CacheLocation {
    let env = env.map(OsString::from);
    let default = PathBuf::from(DEFAULT);
    resolve_cache_location(env.as_deref(), || default)
}

fn default_location() -> CacheLocation {
    let default = PathBuf::from(DEFAULT);
    CacheLocation::Dir(default)
}

#[test]
fn with_nothing_set_the_default_wins() {
    let location = resolve(None);

    assert_eq!(location, default_location());
}

#[test]
fn an_explicit_path_overrides_the_default() {
    let location = resolve(Some("/mnt/cache"));

    assert_eq!(location, CacheLocation::Dir(PathBuf::from("/mnt/cache")));
}

#[test]
fn the_disable_words_turn_caching_off() {
    for word in ["off", "0", "false", "none"] {
        let upper = word.to_ascii_uppercase();
        let padded = format!("  {upper} ");

        for spelling in [word, upper.as_str(), padded.as_str()] {
            let location = resolve(Some(spelling));
            assert_eq!(
                location,
                CacheLocation::Disabled,
                "{spelling:?} should disable caching"
            );
        }
    }
}

#[test]
fn a_path_containing_a_disable_word_is_still_a_path() {
    let location = resolve(Some("/tmp/offsite"));

    assert_eq!(location, CacheLocation::Dir(PathBuf::from("/tmp/offsite")));
}

#[test]
fn an_empty_variable_takes_the_default() {
    let empty = resolve(Some(""));
    let blank = resolve(Some("   "));

    assert_eq!(empty, default_location());
    assert_eq!(blank, default_location());
}

#[test]
fn a_padded_explicit_path_is_trimmed() {
    let location = resolve(Some(" /mnt/cache "));

    assert_eq!(location, CacheLocation::Dir(PathBuf::from("/mnt/cache")));
}

#[cfg(unix)]
#[test]
fn a_non_utf8_override_is_carried_through_untrimmed() {
    use std::os::unix::ffi::OsStringExt;

    // `0xFF` is not valid UTF-8 in any position, so `bytes` never decodes to a `&str`.
    let bytes = vec![b'/', b'm', b'n', b't', b'/', 0xFF, b'x'];
    let env = OsString::from_vec(bytes.clone());
    let expected_os = OsString::from_vec(bytes);
    let expected = PathBuf::from(expected_os);

    let location = resolve_cache_location(Some(&env), || PathBuf::from(DEFAULT));

    assert_eq!(location, CacheLocation::Dir(expected));
}

#[test]
fn resolve_default_dir_joins_the_platform_cache_directory() {
    let platform_cache = Some(PathBuf::from("/home/u/.cache"));
    let temp_dir = PathBuf::from("/tmp");

    let resolved = resolve_default_dir(platform_cache, temp_dir);

    assert_eq!(resolved, PathBuf::from("/home/u/.cache/av-toolbox"));
}

#[test]
fn resolve_default_dir_falls_back_to_the_temporary_directory() {
    let temp_dir = PathBuf::from("/tmp");

    let resolved = resolve_default_dir(None, temp_dir);

    assert_eq!(resolved, PathBuf::from("/tmp/av-toolbox"));
}
