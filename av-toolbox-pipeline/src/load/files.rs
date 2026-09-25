use std::io;
use std::path::{Path, PathBuf};

use av_toolbox_graph::{Diagnostic, FileId, Span};

/// Where the loader reads pipeline files from.
pub trait Files {
    fn read(&self, path: &Path) -> io::Result<String>;
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf>;
}

/// Reads pipeline files from disk.
#[derive(Debug, Copy, Clone, Default)]
pub struct DiskFiles;

impl Files for DiskFiles {
    fn read(&self, path: &Path) -> io::Result<String> {
        std::fs::read_to_string(path)
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        std::fs::canonicalize(path)
    }
}

/// One file a pipeline was loaded from.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub text: String,
    /// The header of each call that led to this file, innermost first.
    pub called_from: Vec<Span>,
}

/// Every file a pipeline was loaded from, by [FileId](av_toolbox_graph::FileId).
#[derive(Debug, Clone, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, path: PathBuf, text: String, called_from: Vec<Span>) -> FileId {
        let index = self.files.len();
        let index = u32::try_from(index).expect("fewer than 2^32 pipeline files");
        let source = SourceFile {
            path,
            text,
            called_from,
        };

        self.files.push(source);
        FileId::new(index)
    }

    pub fn get(&self, file: FileId) -> Option<&SourceFile> {
        let index = file.index();
        self.files.get(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = (FileId, &SourceFile)> {
        self.files.iter().enumerate().map(|(index, source)| {
            let index = index as u32;
            (FileId::new(index), source)
        })
    }

    /// Adds a "called from here" note at each call that led to the diagnostic's file.
    pub fn with_call_notes(&self, diagnostic: Diagnostic) -> Diagnostic {
        let span = diagnostic.span();
        if span.is_unknown() {
            return diagnostic;
        }

        let Some(source) = self.get(span.file) else {
            return diagnostic;
        };

        let mut noted = diagnostic;
        for call in &source.called_from {
            noted = noted.with_note("called from here", *call);
        }

        noted
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_graph::{DiagnosticCode, FileId};

    use super::*;

    #[test]
    fn a_diagnostic_in_a_called_file_gets_a_note_per_call() {
        let mut sources = SourceMap::new();
        let root = sources.push("main.toml".into(), String::new(), Vec::new());
        let outer_call = Span::in_file(root, 1, 5);
        let middle = sources.push("a.toml".into(), String::new(), vec![outer_call]);
        let inner_call = Span::in_file(middle, 2, 6);
        let inner = sources.push("b.toml".into(), String::new(), vec![inner_call, outer_call]);
        let span = Span::in_file(inner, 3, 7);
        let diagnostic = Diagnostic::error(DiagnosticCode::UnknownStage, "", span, "x");

        let noted = sources.with_call_notes(diagnostic);
        let spans: Vec<Span> = noted.notes().iter().map(|note| note.span).collect();

        assert_eq!(spans, [inner_call, outer_call]);
        assert_eq!(root, FileId::ROOT);
    }

    #[test]
    fn a_diagnostic_in_the_root_file_is_unchanged() {
        let mut sources = SourceMap::new();
        sources.push("main.toml".into(), String::new(), Vec::new());
        let span = Span::new(1, 2);
        let diagnostic = Diagnostic::error(DiagnosticCode::UnknownStage, "", span, "x");

        let noted = sources.with_call_notes(diagnostic);

        assert!(noted.notes().is_empty());
    }
}
