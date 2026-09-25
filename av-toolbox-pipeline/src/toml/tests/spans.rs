use av_toolbox_graph::Span;

use super::{locate, parsed};

const PIPELINE: &str = r#"
[source.clip]
type = "ffms2/file"
args.path = "in.mkv"
args.limits = [40, 80]

[filter.denoise]
type = "denoise/nl4d"
inputs = ["clip", "spare"]
args = { sigma = 1 }

[mask.join]
type = "merge/masked"
inputs = { base = "clip", overlay = "denoise" }
"#;

#[test]
fn the_stage_and_name_span_the_header() {
    let file = parsed(PIPELINE);
    let header = locate(PIPELINE, "[filter.denoise]");

    assert_eq!(file.stages[1].span, header);
    assert_eq!(file.stages[1].name.span, header);
}

#[test]
fn the_type_spans_its_value() {
    let file = parsed(PIPELINE);

    assert_eq!(
        file.stages[1].stage_type.span,
        locate(PIPELINE, "\"denoise/nl4d\"")
    );
}

#[test]
fn each_positional_input_spans_its_element() {
    let file = parsed(PIPELINE);

    let array = locate(PIPELINE, "[\"clip\", \"spare\"]");
    let first = Span::new(array.start + 1, array.start + 7);

    assert_eq!(file.stages[1].inputs[0].span, first);
    assert_eq!(file.stages[1].inputs[1].span, locate(PIPELINE, "\"spare\""));
}

#[test]
fn each_named_input_spans_its_entry() {
    let file = parsed(PIPELINE);

    assert_eq!(file.stages[2].inputs[0].span, locate(PIPELINE, "base = \"clip\""));
    assert_eq!(
        file.stages[2].inputs[1].span,
        locate(PIPELINE, "overlay = \"denoise\"")
    );
}

#[test]
fn the_args_span_is_the_args_table() {
    let file = parsed(PIPELINE);

    assert_eq!(file.stages[1].args.span, locate(PIPELINE, "{ sigma = 1 }"));
}

#[test]
fn arg_spans_hold_every_value() {
    let file = parsed(PIPELINE);
    let spans = &file.stages[0].arg_spans;

    assert_eq!(spans.narrowest("/path"), Some(locate(PIPELINE, "\"in.mkv\"")));
    assert_eq!(spans.narrowest("/limits/1"), Some(locate(PIPELINE, "80")));
    assert_eq!(spans.narrowest("/limits"), Some(locate(PIPELINE, "[40, 80]")));
}

#[test]
fn a_dotted_stage_spans_its_name_key() {
    let text = "source.blur.type = \"test/x\"\n";

    let file = parsed(text);

    assert_eq!(file.stages[0].span, locate(text, "blur"));
}

#[test]
fn an_arg_key_needing_escape_still_locates_its_value() {
    let text = "[filter.a]\ntype = \"a/b\"\nargs = { \"a/b\" = 1, \"c~d\" = 2 }\n";

    let file = parsed(text);
    let spans = &file.stages[0].arg_spans;

    assert_eq!(spans.narrowest("/a~1b"), Some(locate(text, "1")));
    assert_eq!(spans.narrowest("/c~0d"), Some(locate(text, "2")));
}
