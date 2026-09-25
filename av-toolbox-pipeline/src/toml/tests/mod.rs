mod calls;
mod errors;
mod shapes;
mod spans;
mod variables;

use av_toolbox_graph::{Diagnostic, FileId, Span};

use super::{RawFile, parse};

fn parsed(text: &str) -> RawFile {
    parse(text, FileId::ROOT).expect("the file should parse")
}

fn reported(text: &str) -> Vec<Diagnostic> {
    let diagnostics = parse(text, FileId::ROOT).expect_err("the file should fail");
    diagnostics.into_inner()
}

/// The span of the first `needle` in `text`.
fn locate(text: &str, needle: &str) -> Span {
    let start = text.find(needle).expect("the needle is in the text");
    Span::new(start, start + needle.len())
}
