mod calls;
mod references;
mod variables;

use std::path::Path;

use av_toolbox_graph::{Diagnostic, DiagnosticCode, StageSpec};

use super::{Loaded, load};
use crate::testing::MemoryFiles;
use crate::variables::Supplied;

fn loaded(entries: &[(&str, &str)], root: &str, supplied: &Supplied) -> Loaded {
    let files = MemoryFiles::new(entries);
    let path = Path::new(root);

    let result = load(path, &files, supplied);
    let Ok(loaded) = result else {
        panic!("the pipeline should load: {:?}", result.err());
    };

    loaded
}

fn failed(entries: &[(&str, &str)], root: &str, supplied: &Supplied) -> Vec<Diagnostic> {
    let files = MemoryFiles::new(entries);
    let path = Path::new(root);

    let result = load(path, &files, supplied);
    let Err(failure) = result else {
        panic!("the pipeline should fail to load");
    };

    failure.diagnostics.into_inner()
}

fn codes(diagnostics: &[Diagnostic]) -> Vec<DiagnosticCode> {
    diagnostics.iter().map(|diagnostic| diagnostic.code()).collect()
}

fn stage<'a>(specs: &'a [StageSpec], name: &str) -> &'a StageSpec {
    let found = specs.iter().find(|spec| spec.name.value == name);
    found.unwrap_or_else(|| panic!("no stage named {name}"))
}
