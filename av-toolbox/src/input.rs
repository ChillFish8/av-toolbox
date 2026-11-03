use std::io;
use std::path::{Path, PathBuf};

use av_toolbox_graph::StageSpec;
use av_toolbox_pipeline::{DiskFiles, Files, LoadFailure, Supplied};

use crate::SessionError;
use crate::report::PipelineFiles;

/// Where a session reads its pipeline from.
#[derive(Debug, Clone)]
pub enum PipelineInput {
    /// A TOML file on disk, which may call other files beside it.
    File(PathBuf),
    /// TOML text held in memory, named `name` in diagnostics.
    ///
    /// It cannot call other files, since there is no directory to find them in.
    Text { name: String, text: String },
}

/// A pipeline's files and the stages they define.
pub(crate) struct Loaded {
    pub(crate) files: PipelineFiles,
    pub(crate) specs: Vec<StageSpec>,
}

/// One pipeline file held in memory, refusing to read any other.
struct TextFiles<'a> {
    path: &'a Path,
    text: &'a str,
}

impl Files for TextFiles<'_> {
    fn read(&self, path: &Path) -> io::Result<String> {
        if path != self.path {
            return Err(refused_read());
        }

        Ok(self.text.to_owned())
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        if path != self.path {
            return Err(refused_read());
        }

        Ok(path.to_path_buf())
    }
}

fn refused_read() -> io::Error {
    io::Error::other("a pipeline given as text cannot read other files")
}

/// Reads the pipeline and every file it calls.
pub(crate) fn load(input: &PipelineInput, supplied: &Supplied) -> Result<Loaded, SessionError> {
    let loaded = match input {
        PipelineInput::File(path) => av_toolbox_pipeline::load(path, &DiskFiles, supplied),
        PipelineInput::Text { name, text } => {
            let path = Path::new(name);
            let files = TextFiles { path, text };
            av_toolbox_pipeline::load(path, &files, supplied)
        },
    };

    match loaded {
        Ok(loaded) => {
            let files = PipelineFiles::new(loaded.sources);
            Ok(Loaded {
                files,
                specs: loaded.specs,
            })
        },
        Err(failure) => Err(load_failure(failure)),
    }
}

fn load_failure(failure: LoadFailure) -> SessionError {
    let files = PipelineFiles::new(failure.sources);
    let found = failure.diagnostics.into_inner();

    SessionError::pipeline(files.into(), found)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_input(text: &str) -> PipelineInput {
        PipelineInput::Text {
            name: "pipeline.toml".to_owned(),
            text: text.to_owned(),
        }
    }

    #[test]
    fn a_text_pipeline_loads_its_stages() {
        let input = text_input("[source.input]\ntype = \"channel/frames\"\n");
        let supplied = Supplied::new();

        let loaded = load(&input, &supplied).expect("the text loads");

        assert_eq!(loaded.specs.len(), 1);
        assert_eq!(loaded.specs[0].name.value, "input");
    }

    #[test]
    fn a_text_pipeline_cannot_call_other_files() {
        let input = text_input("[call.part]\npath = \"part.toml\"\n");
        let supplied = Supplied::new();

        let error = load(&input, &supplied).err().expect("the call is refused");

        let found = error.diagnostics();
        assert_eq!(found.len(), 1);
        let message = found[0].message();
        assert!(message.contains("part.toml"), "{message}");
        assert!(message.contains("cannot read other files"), "{message}");
    }
}
