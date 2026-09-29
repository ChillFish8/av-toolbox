use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};

use etcetera::base_strategy::{BaseStrategy, choose_base_strategy};
use semver::Version;

/// The variable that overrides where models are stored.
pub const MODELS_ENV: &str = "AVTB_MODELS";

/// The release tag of this build.
pub const TAG: &str = concat!("v", env!("CARGO_PKG_VERSION"));

/// Where models are stored when nothing says otherwise.
pub fn default_root() -> PathBuf {
    let platform_data = choose_base_strategy().ok().map(|strategy| strategy.data_dir());
    match platform_data {
        Some(data) => data.join("av-toolbox").join("models"),
        None => {
            tracing::warn!("no platform data directory available, storing models in the temporary directory");
            std::env::temp_dir().join("av-toolbox").join("models")
        },
    }
}

/// Where models are stored, honouring [MODELS_ENV].
pub fn models_root() -> PathBuf {
    let raw = std::env::var_os(MODELS_ENV);
    let default = default_root();

    resolve_root(raw.as_deref(), default)
}

/// Reads the override variable, falling back to `default` when it is unset or blank.
fn resolve_root(raw: Option<&OsStr>, default: PathBuf) -> PathBuf {
    let Some(raw) = raw else {
        return default;
    };

    let blank = raw.to_str().is_some_and(|text| text.trim().is_empty());
    if blank {
        return default;
    }

    PathBuf::from(raw)
}

/// The release tags installed under `root`, highest version first.
///
/// Only directories named `v` and a semantic version count. A missing root has none.
pub fn installed_tags(root: &Path) -> io::Result<Vec<String>> {
    let listing = std::fs::read_dir(root);
    let entries = match listing {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };

    let mut tags = Vec::new();
    for entry in entries {
        let entry = entry?;
        let is_directory = entry.file_type()?.is_dir();
        let file_name = entry.file_name();

        let Some(name) = file_name.to_str() else {
            continue;
        };

        let Some(version) = parse_tag(name) else {
            continue;
        };

        if is_directory {
            tags.push((version, name.to_string()));
        }
    }

    tags.sort_by(|left, right| right.0.cmp(&left.0));
    let names = tags.into_iter().map(|(_, name)| name).collect();

    Ok(names)
}

fn parse_tag(name: &str) -> Option<Version> {
    let stripped = name.strip_prefix('v')?;
    Version::parse(stripped).ok()
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    fn default() -> PathBuf {
        PathBuf::from("/default")
    }

    #[test]
    fn an_unset_or_blank_variable_falls_back_to_the_default() {
        let empty = OsString::from("");
        let spaces = OsString::from("   ");

        let unset = resolve_root(None, default());
        let empty_root = resolve_root(Some(&empty), default());
        let spaces_root = resolve_root(Some(&spaces), default());

        assert_eq!(unset, default());
        assert_eq!(empty_root, default());
        assert_eq!(spaces_root, default());
    }

    #[test]
    fn a_path_in_the_variable_wins() {
        let raw = OsString::from("/mnt/models");

        let resolved = resolve_root(Some(&raw), default());

        assert_eq!(resolved, PathBuf::from("/mnt/models"));
    }

    #[test]
    fn the_default_root_ends_in_av_toolbox_models() {
        let root = default_root();

        assert!(root.ends_with("av-toolbox/models"), "{}", root.display());
    }

    #[test]
    fn the_tag_is_v_and_the_crate_version() {
        let expected = format!("v{}", env!("CARGO_PKG_VERSION"));

        assert_eq!(TAG, expected);
    }

    #[test]
    fn installed_tags_are_highest_version_first_and_skip_other_names() {
        let root = tempfile::tempdir().expect("a temp dir");
        let base = root.path();
        for name in [
            "v0.9.0",
            "v0.10.0",
            "v0.10.0-rc.1",
            ".staging-abc",
            "latest",
            "v1.x",
        ] {
            let directory = base.join(name);
            std::fs::create_dir(&directory).expect("created");
        }
        let plain_file = base.join("v2.0.0");
        std::fs::write(&plain_file, b"not a directory").expect("written");

        let tags = installed_tags(base).expect("listed");

        assert_eq!(tags, ["v0.10.0", "v0.10.0-rc.1", "v0.9.0"]);
    }

    #[test]
    fn a_missing_root_has_no_tags() {
        let root = tempfile::tempdir().expect("a temp dir");
        let missing = root.path().join("never-created");

        let tags = installed_tags(&missing).expect("an empty listing");

        assert!(tags.is_empty());
    }
}
