mod errors;
mod shapes;
mod spans;

use av_toolbox_graph::{Diagnostic, Span, StageSpec};

use super::parse;

fn parsed(text: &str) -> Vec<StageSpec> {
    parse(text).expect("the file should parse")
}

fn reported(text: &str) -> Vec<Diagnostic> {
    let diagnostics = parse(text).expect_err("the file should fail");
    diagnostics.into_inner()
}

/// The span of the first `needle` in `text`.
fn locate(text: &str, needle: &str) -> Span {
    let start = text.find(needle).expect("the needle is in the text");
    Span::new(start, start + needle.len())
}
