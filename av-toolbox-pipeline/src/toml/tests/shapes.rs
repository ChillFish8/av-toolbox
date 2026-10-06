use av_toolbox_core::StageKind;
use av_toolbox_graph::FileId;
use serde_json::json;

use super::parsed;
use crate::toml::{RawWorkers, parse};

const PASSTHROUGH: &str = r#"
[source.input]
type = "y4m/stdin"

[sink.output]
type = "y4m/stdout"
inputs = ["input"]
"#;

fn names(text: &str) -> Vec<String> {
    let file = parsed(text);
    file.stages.into_iter().map(|stage| stage.name.value).collect()
}

#[test]
fn a_minimal_pipeline_parses() {
    let file = parsed(PASSTHROUGH);

    assert_eq!(file.stages.len(), 2);
    assert_eq!(file.stages[0].kind, StageKind::Source);
    assert_eq!(file.stages[0].stage_type.value, "y4m/stdin");
    assert!(file.stages[0].inputs.is_empty());
    assert_eq!(file.stages[1].kind, StageKind::Sink);
    assert_eq!(file.stages[1].inputs[0].value.target, "input");
    assert_eq!(file.stages[1].inputs[0].value.name, None);
}

#[test]
fn stages_keep_file_order_across_kinds() {
    let text = r#"
[source.first]
type = "a/b"

[sink.second]
type = "a/b"
inputs = ["first"]

[source.third]
type = "a/b"
"#;

    assert_eq!(names(text), ["first", "second", "third"]);
}

#[test]
fn every_kind_table_is_read() {
    let text = r#"
[source.a]
type = "a/b"
[filter.b]
type = "a/b"
[mask.c]
type = "a/b"
[sink.e]
type = "a/b"
"#;

    let file = parsed(text);
    let kinds: Vec<StageKind> = file.stages.iter().map(|stage| stage.kind).collect();

    assert_eq!(
        kinds,
        [
            StageKind::Source,
            StageKind::Filter,
            StageKind::Mask,
            StageKind::Sink
        ]
    );
}

#[test]
fn named_inputs_carry_their_slot() {
    let text = r#"
[mask.join]
type = "merge/masked"
inputs = { base = "clip", overlay = "denoise" }
"#;

    let file = parsed(text);
    let inputs: Vec<(Option<&str>, &str)> = file.stages[0]
        .inputs
        .iter()
        .map(|input| (input.value.name.as_deref(), input.value.target.as_str()))
        .collect();

    assert_eq!(inputs, [(Some("base"), "clip"), (Some("overlay"), "denoise")]);
}

#[test]
fn workers_are_read() {
    let text = "[filter.a]\ntype = \"a/b\"\nworkers = 3\n";
    let file = parsed(text);

    let workers = file.stages[0]
        .workers
        .as_ref()
        .map(|workers| workers.value.clone());

    assert_eq!(workers, Some(RawWorkers::Count(3)));
}

#[test]
fn missing_args_are_an_empty_table() {
    let file = parsed(PASSTHROUGH);

    assert_eq!(file.stages[0].args.value, json!({}));
    assert!(file.stages[0].arg_spans.is_empty());
}

#[test]
fn args_convert_to_json() {
    let text = r#"
[filter.a]
type = "a/b"
args = { sigma = 0.5, radius = 3, label = "x", on = true, list = [1, 2], nested = { deep = "y" } }
"#;

    let expected = json!({
        "sigma": 0.5,
        "radius": 3,
        "label": "x",
        "on": true,
        "list": [1, 2],
        "nested": { "deep": "y" },
    });

    let file = parsed(text);

    assert_eq!(file.stages[0].args.value, expected);
}

#[test]
fn dotted_args_convert_to_json() {
    let text = "[source.a]\ntype = \"ffms2/file\"\nargs.path = \"in.mkv\"\nargs.track = 1\n";
    let file = parsed(text);

    assert_eq!(file.stages[0].args.value, json!({ "path": "in.mkv", "track": 1 }));
}

#[test]
fn a_stage_written_with_dotted_keys_parses() {
    let text = "source.blur.type = \"test/x\"\n";

    let file = parsed(text);

    assert_eq!(file.stages[0].name.value, "blur");
    assert_eq!(file.stages[0].stage_type.value, "test/x");
}

#[test]
fn an_inline_stage_table_parses() {
    let text = "sink.out = { type = \"y4m/stdout\", inputs = [\"a\"] }\n";

    let file = parsed(text);

    assert_eq!(file.stages[0].stage_type.value, "y4m/stdout");
    assert_eq!(file.stages[0].inputs[0].value.target, "a");
}

#[test]
fn templates_in_args_are_kept_as_written() {
    let file = parsed("[source.a]\ntype = \"a/b\"\nargs.path = \"{{ dir }}/in.y4m\"\n");

    assert_eq!(file.stages[0].args.value["path"], "{{ dir }}/in.y4m");
}

#[test]
fn templated_workers_are_kept_as_written() {
    let file = parsed("[filter.a]\ntype = \"a/b\"\nworkers = \"{{ count }}\"\n");
    let workers = file.stages[0]
        .workers
        .as_ref()
        .map(|workers| workers.value.clone());

    assert_eq!(workers, Some(RawWorkers::Template("{{ count }}".into())));
}

#[test]
fn spans_carry_the_file_they_were_parsed_from() {
    let file_id = FileId::new(2);

    let file = parse("[source.a]\ntype = \"a/b\"\n", file_id).expect("parses");

    assert_eq!(file.stages[0].span.file, file_id);
    assert_eq!(file.stages[0].stage_type.span.file, file_id);
}

#[test]
fn variable_call_input_and_output_spans_carry_their_file() {
    let text = r#"
[variable.a]
type = "string"
default = "x"

[call.b]
path = "b.toml"
inputs = ["c"]
variables.d = 1

[input.c]

[output.e]
inputs = ["c"]
"#;
    let file_id = FileId::new(2);

    let file = parse(text, file_id).expect("parses");
    let variable = &file.variables[0];
    let call = &file.calls[0];
    let output = &file.outputs[0];
    let default_file = variable.default.as_ref().map(|default| default.span.file);

    assert_eq!(variable.name.span.file, file_id);
    assert_eq!(default_file, Some(file_id));
    assert_eq!(call.span.file, file_id);
    assert_eq!(call.path.span.file, file_id);
    assert_eq!(call.inputs[0].span.file, file_id);
    assert_eq!(call.variables[0].value.span.file, file_id);
    assert_eq!(file.inputs[0].name.span.file, file_id);
    assert_eq!(output.name.span.file, file_id);
    assert_eq!(output.source.span.file, file_id);
}
