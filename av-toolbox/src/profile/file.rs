use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use av_toolbox_gpu::{Accelerator, Device};
use av_toolbox_graph::StageSpec;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the profile command writes the file in a later task")
)]
const HEADER: &str = "# Worker counts measured by `av-toolbox profile`.\n";

/// The worker counts one profile run settled on, and where it ran.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct ProfileFile {
    pub(crate) pipeline: String,
    pub(crate) device: String,
    pub(crate) frames: u64,
    pub(crate) workers: BTreeMap<String, usize>,
}

#[derive(Debug, Error)]
pub(crate) enum FileError {
    #[error("cannot read the profile {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("cannot parse the profile {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the profile command writes the file in a later task")
    )]
    #[error("cannot write the profile {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("cannot write the profile: {0}")]
    Serialise(#[from] toml::ser::Error),
}

pub(crate) fn read(path: &Path) -> Result<ProfileFile, FileError> {
    let shown = path.display().to_string();

    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(source) => return Err(FileError::Read { path: shown, source }),
    };

    match toml::from_str(&text) {
        Ok(profile) => Ok(profile),
        Err(source) => Err(FileError::Parse { path: shown, source }),
    }
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the profile command writes the file in a later task")
)]
pub(crate) fn write(path: &Path, profile: &ProfileFile) -> Result<(), FileError> {
    let body = toml::to_string(profile)?;
    let text = format!("{HEADER}{body}");

    let written = std::fs::write(path, text);
    written.map_err(|source| write_error(path, source))
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the profile command writes the file in a later task")
)]
fn write_error(path: &Path, source: io::Error) -> FileError {
    let shown = path.display().to_string();
    FileError::Write { path: shown, source }
}

/// Sets `workers` on every stage that declared none, returning the keys that name no stage.
pub(crate) fn apply(workers: &BTreeMap<String, usize>, specs: &mut [StageSpec]) -> Vec<String> {
    let mut unknown = Vec::new();

    for (key, count) in workers {
        let found = specs.iter_mut().find(|spec| stage_key(spec) == *key);
        let Some(spec) = found else {
            unknown.push(key.clone());
            continue;
        };

        if spec.workers.is_none() {
            spec.workers = Some(*count);
        }
    }

    unknown
}

/// How a profile names a stage, `<kind>.<name>`.
pub(crate) fn stage_key(spec: &StageSpec) -> String {
    format!("{}.{}", spec.kind, spec.name.value)
}

/// The device a profile ran on, such as `vulkan discrete:1`.
pub(crate) fn device_label(accelerator: Accelerator, device: Device) -> String {
    format!("{accelerator} {device}")
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use av_toolbox_graph::{ArgSpans, Span, Spanned, StageSpec};
    use serde_json::json;

    use super::*;

    fn spec(kind: StageKind, name: &str, workers: Option<usize>) -> StageSpec {
        StageSpec {
            name: Spanned::unknown(name.to_string()),
            kind,
            stage_type: Spanned::unknown("test/stage".to_string()),
            inputs: Vec::new(),
            args: Spanned::unknown(json!({})),
            arg_spans: ArgSpans::default(),
            workers,
            span: Span::unknown(),
        }
    }

    fn profile() -> ProfileFile {
        let mut workers = BTreeMap::new();
        workers.insert("filter.based".to_string(), 1);
        workers.insert("transform.scenes.detect".to_string(), 3);

        ProfileFile {
            pipeline: "examples/anti-aliasing.toml".to_string(),
            device: "vulkan discrete:1".to_string(),
            frames: 300,
            workers,
        }
    }

    #[test]
    fn a_written_profile_reads_back_the_same() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = folder.path().join("clip.profile.toml");

        write(&path, &profile()).expect("writes");
        let read_back = read(&path).expect("reads");

        assert_eq!(read_back, profile());
    }

    #[test]
    fn a_written_profile_starts_with_its_header_and_quotes_its_keys() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = folder.path().join("clip.profile.toml");

        write(&path, &profile()).expect("writes");
        let text = std::fs::read_to_string(&path).expect("reads");

        assert!(text.starts_with(HEADER), "{text}");
        assert!(text.contains("\"transform.scenes.detect\" = 3"), "{text}");
    }

    #[test]
    fn a_missing_profile_names_its_path() {
        let error = read(Path::new("/nowhere/clip.profile.toml")).expect_err("missing");

        assert!(
            error.to_string().contains("/nowhere/clip.profile.toml"),
            "{error}"
        );
    }

    #[test]
    fn applying_sets_undeclared_stages_and_keeps_declared_ones() {
        let mut specs = vec![
            spec(StageKind::Filter, "based", Some(2)),
            spec(StageKind::Transform, "scenes.detect", None),
        ];

        let unknown = apply(&profile().workers, &mut specs);

        assert!(unknown.is_empty());
        assert_eq!(specs[0].workers, Some(2));
        assert_eq!(specs[1].workers, Some(3));
    }

    #[test]
    fn applying_returns_keys_naming_no_stage() {
        let mut specs = vec![spec(StageKind::Filter, "based", None)];

        let unknown = apply(&profile().workers, &mut specs);

        assert_eq!(unknown, vec!["transform.scenes.detect".to_string()]);
    }
}
