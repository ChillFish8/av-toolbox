use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use av_toolbox_core::{
    ArtefactData,
    ArtefactSink,
    ArtefactSinkDefinition,
    InputSignature,
    KeyFilter,
    MergedArtefact,
    Pattern,
    StageDefinition,
    StageError,
};
use schemars::JsonSchema;
use serde::Deserialize;

const STAGE: &str = "artefacts/files";

/// Arguments for `artefacts/files`.
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct FilesArgs {
    /// The directory to write into, created when there is something to write.
    pub(crate) directory: PathBuf,
    /// Artefacts to keep, such as `filter.grain.*`, where `*` matches any run of characters. Every artefact is
    /// kept when unset.
    pub(crate) include: Option<Vec<Pattern>>,
    /// Artefacts to drop, which wins over `include`.
    pub(crate) exclude: Option<Vec<Pattern>>,
}

pub(crate) struct Files;

impl StageDefinition for Files {
    type Args = FilesArgs;

    fn package() -> &'static str {
        "artefacts"
    }

    fn identifier() -> &'static str {
        "files"
    }

    fn description() -> &'static str {
        "Writes each artefact the pipeline's stages produce to its own file in a directory."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::ARTEFACTS
    }
}

impl ArtefactSinkDefinition for Files {
    fn selection(args: &Self::Args) -> KeyFilter {
        let include = args.include.clone();
        let exclude = args.exclude.clone();

        KeyFilter::new(include, exclude)
    }

    fn build(args: Self::Args) -> Result<Box<dyn ArtefactSink>, StageError> {
        refuse_non_directory(&args.directory)?;

        let sink = FilesSink {
            directory: args.directory,
        };

        Ok(Box::new(sink))
    }
}

/// Writes each artefact to `<directory>/<key>.<extension>`.
struct FilesSink {
    directory: PathBuf,
}

impl FilesSink {
    fn write_one(&self, artefact: &MergedArtefact) -> Result<(), StageError> {
        let key = artefact.key();
        let extension = artefact.kind.extension();
        let file_name = format!("{key}.{extension}");
        let path = self.directory.join(file_name);

        let file = File::create(&path).map_err(|error| write_error(&path, error))?;
        let mut writer = BufWriter::new(file);

        match &artefact.data {
            ArtefactData::Structured(value) => {
                let written = serde_json::to_writer_pretty(&mut writer, value);
                written.map_err(|error| write_error(&path, error))?;
            },
            ArtefactData::Opaque(bytes) => {
                let written = writer.write_all(bytes);
                written.map_err(|error| write_error(&path, error))?;
            },
        }

        writer.flush().map_err(|error| write_error(&path, error))
    }
}

impl ArtefactSink for FilesSink {
    fn write(&mut self, artefacts: &[&MergedArtefact]) -> Result<usize, StageError> {
        if artefacts.is_empty() {
            return Ok(0);
        }

        let created = std::fs::create_dir_all(&self.directory);
        created.map_err(|error| create_error(&self.directory, error))?;

        for artefact in artefacts {
            self.write_one(artefact)?;
        }

        Ok(artefacts.len())
    }
}

/// Refuses `directory` when it exists and is not a directory.
fn refuse_non_directory(directory: &Path) -> Result<(), StageError> {
    if !directory.exists() || directory.is_dir() {
        return Ok(());
    }

    let message = format!(
        "{STAGE} cannot write into {}, which is not a directory",
        directory.display()
    );

    Err(StageError::msg(message))
}

fn create_error(directory: &Path, error: std::io::Error) -> StageError {
    let message = format!("{STAGE} could not create {}", directory.display());

    StageError::with_source(message, error)
}

fn write_error<E>(path: &Path, error: E) -> StageError
where
    E: std::error::Error + Send + Sync + 'static,
{
    let message = format!("{STAGE} could not write {}", path.display());

    StageError::with_source(message, error)
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::{ArtefactData, ArtefactKind, ArtefactSinkDefinition, MergedArtefact};
    use serde_json::json;

    use super::*;

    fn structured() -> MergedArtefact {
        MergedArtefact {
            stage: String::from("filter.grain"),
            name: "cuts",
            kind: ArtefactKind::Structured,
            data: ArtefactData::Structured(json!({ "cuts": [0, 24] })),
        }
    }

    fn opaque() -> MergedArtefact {
        MergedArtefact {
            stage: String::from("filter.grain"),
            name: "table",
            kind: ArtefactKind::Opaque { extension: "tbl" },
            data: ArtefactData::Opaque(b"filmgrn1\n".to_vec()),
        }
    }

    fn args(directory: &std::path::Path) -> FilesArgs {
        let value = json!({ "directory": directory });

        serde_json::from_value(value).expect("valid args")
    }

    #[test]
    fn a_structured_artefact_is_written_as_json_under_its_key() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let mut sink = Files::build(args(folder.path())).expect("builds");
        let artefact = structured();

        let written = sink.write(&[&artefact]).expect("writes");
        let path = folder.path().join("filter.grain.cuts.json");
        let text = std::fs::read_to_string(path).expect("the file exists");
        let read_back: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");

        assert_eq!(written, 1);
        assert_eq!(read_back, json!({ "cuts": [0, 24] }));
    }

    #[test]
    fn an_opaque_artefact_is_written_as_its_bytes_with_its_extension() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let mut sink = Files::build(args(folder.path())).expect("builds");
        let artefact = opaque();

        sink.write(&[&artefact]).expect("writes");
        let path = folder.path().join("filter.grain.table.tbl");
        let bytes = std::fs::read(path).expect("the file exists");

        assert_eq!(bytes, b"filmgrn1\n");
    }

    #[test]
    fn building_creates_no_directory() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let nested = folder.path().join("reports").join("run");

        let built = Files::build(args(&nested));

        assert!(built.is_ok());
        assert!(!nested.exists());
    }

    #[test]
    fn writing_creates_a_missing_directory() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let nested = folder.path().join("reports").join("run");
        let mut sink = Files::build(args(&nested)).expect("builds");
        let artefact = opaque();

        sink.write(&[&artefact]).expect("writes");
        let path = nested.join("filter.grain.table.tbl");
        let bytes = std::fs::read(path).expect("the file exists");

        assert_eq!(bytes, b"filmgrn1\n");
    }

    #[test]
    fn a_directory_that_became_a_file_fails_the_write() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let taken = folder.path().join("taken");
        let mut sink = Files::build(args(&taken)).expect("builds");
        std::fs::write(&taken, b"").expect("writes a file");
        let artefact = opaque();

        let error = sink.write(&[&artefact]).expect_err("refused");

        assert!(error.to_string().contains("artefacts/files"));
    }

    #[test]
    fn writing_nothing_creates_no_directory() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let nested = folder.path().join("reports");
        let mut sink = Files::build(args(&nested)).expect("builds");

        let written = sink.write(&[]).expect("writes");

        assert_eq!(written, 0);
        assert!(!nested.exists());
    }

    #[test]
    fn a_directory_that_is_a_file_is_refused() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let file = folder.path().join("taken");
        std::fs::write(&file, b"").expect("writes a file");

        let error = Files::build(args(&file)).err().expect("refused");

        assert!(error.to_string().contains("artefacts/files"));
    }

    #[test]
    fn an_existing_file_is_overwritten() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = folder.path().join("filter.grain.table.tbl");
        std::fs::write(&path, b"an older, much longer table").expect("writes a file");
        let mut sink = Files::build(args(folder.path())).expect("builds");
        let artefact = opaque();

        sink.write(&[&artefact]).expect("writes");
        let bytes = std::fs::read(path).expect("the file exists");

        assert_eq!(bytes, b"filmgrn1\n");
    }

    #[test]
    fn the_selection_reads_include_and_exclude() {
        let value = json!({ "directory": "out", "include": ["filter.*"], "exclude": ["*.table"] });
        let parsed: FilesArgs = serde_json::from_value(value).expect("valid args");

        let selection = Files::selection(&parsed);

        assert!(selection.keeps("filter.grain.cuts"));
        assert!(!selection.keeps("filter.grain.table"));
        assert!(!selection.keeps("mask.edges.map"));
    }
}
