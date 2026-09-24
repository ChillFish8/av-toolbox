use av_toolbox_core::StageKind;
use serde_json::json;

use super::parsed;

const PASSTHROUGH: &str = r#"
[source.input]
type = "y4m/stdin"

[sink.output]
type = "y4m/stdout"
inputs = ["input"]
"#;

fn names(text: &str) -> Vec<String> {
    let specs = parsed(text);
    specs.into_iter().map(|spec| spec.name.value).collect()
}

#[test]
fn a_minimal_pipeline_parses() {
    let specs = parsed(PASSTHROUGH);

    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].kind, StageKind::Source);
    assert_eq!(specs[0].stage_type.value, "y4m/stdin");
    assert!(specs[0].inputs.is_empty());
    assert_eq!(specs[1].kind, StageKind::Sink);
    assert_eq!(specs[1].inputs[0].value.target, "input");
    assert_eq!(specs[1].inputs[0].value.name, None);
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
[transform.d]
type = "a/b"
[sink.e]
type = "a/b"
"#;

    let kinds: Vec<StageKind> = parsed(text).iter().map(|spec| spec.kind).collect();

    assert_eq!(
        kinds,
        [
            StageKind::Source,
            StageKind::Filter,
            StageKind::Mask,
            StageKind::Transform,
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

    let specs = parsed(text);
    let inputs: Vec<(Option<&str>, &str)> = specs[0]
        .inputs
        .iter()
        .map(|input| (input.value.name.as_deref(), input.value.target.as_str()))
        .collect();

    assert_eq!(inputs, [(Some("base"), "clip"), (Some("overlay"), "denoise")]);
}

#[test]
fn workers_are_read() {
    let text = "[filter.a]\ntype = \"a/b\"\nworkers = 3\n";

    assert_eq!(parsed(text)[0].workers, Some(3));
}

#[test]
fn missing_args_are_an_empty_table() {
    let specs = parsed(PASSTHROUGH);

    assert_eq!(specs[0].args.value, json!({}));
    assert!(specs[0].arg_spans.is_empty());
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

    assert_eq!(parsed(text)[0].args.value, expected);
}

#[test]
fn dotted_args_convert_to_json() {
    let text = "[source.a]\ntype = \"ffms2/file\"\nargs.path = \"in.mkv\"\nargs.track = 1\n";

    assert_eq!(
        parsed(text)[0].args.value,
        json!({ "path": "in.mkv", "track": 1 })
    );
}

#[test]
fn a_stage_written_with_dotted_keys_parses() {
    let text = "source.blur.type = \"test/x\"\n";

    let specs = parsed(text);

    assert_eq!(specs[0].name.value, "blur");
    assert_eq!(specs[0].stage_type.value, "test/x");
}

#[test]
fn an_inline_stage_table_parses() {
    let text = "sink.out = { type = \"y4m/stdout\", inputs = [\"a\"] }\n";

    let specs = parsed(text);

    assert_eq!(specs[0].stage_type.value, "y4m/stdout");
    assert_eq!(specs[0].inputs[0].value.target, "a");
}
