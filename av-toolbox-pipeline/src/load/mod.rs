mod expand;
mod files;
mod resolve;
#[cfg(test)]
mod tests;

use std::path::Path;

use av_toolbox_graph::{Diagnostics, StageSpec};

use self::expand::Loader;
pub use self::files::{DiskFiles, Files, SourceFile, SourceMap};
use crate::variables::Supplied;

/// A pipeline read from its files, ready to build.
#[derive(Debug)]
pub struct Loaded {
    pub specs: Vec<StageSpec>,
    pub sources: SourceMap,
}

/// Why a pipeline could not be read, with the files read so far for pointing at.
#[derive(Debug)]
pub struct LoadFailure {
    pub diagnostics: Diagnostics,
    pub sources: SourceMap,
}

/// Reads the pipeline at `root` and every file it calls into one list of stages.
///
/// Variables in the root file are filled from `supplied`, then from their defaults. Every
/// problem across every file is reported, not only the first.
pub fn load(root: &Path, files: &dyn Files, supplied: &Supplied) -> Result<Loaded, LoadFailure> {
    let mut loader = Loader::new(files, supplied);
    loader.load_root(root);

    loader.finish()
}
