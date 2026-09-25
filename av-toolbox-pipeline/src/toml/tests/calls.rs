use av_toolbox_graph::DiagnosticCode;
use serde_json::json;

use super::{locate, parsed, reported};

const CALLER: &str = r#"
[call.denoised]
path = "parts/denoise.toml"
inputs = { clip = "src" }
variables.strength = 1.2
"#;

#[test]
fn a_call_reads_its_path_inputs_and_variables() {
    let file = parsed(CALLER);
    let call = &file.calls[0];

    assert_eq!(call.name.value, "denoised");
    assert_eq!(call.span, locate(CALLER, "[call.denoised]"));
    assert_eq!(call.path.value, "parts/denoise.toml");
    assert_eq!(call.path.span, locate(CALLER, "\"parts/denoise.toml\""));
    assert_eq!(call.inputs[0].value.name.as_deref(), Some("clip"));
    assert_eq!(call.inputs[0].value.target, "src");
    assert_eq!(call.inputs_span, Some(locate(CALLER, "{ clip = \"src\" }")));
    assert_eq!(call.variables[0].name.value, "strength");
    assert_eq!(call.variables[0].value.value, json!(1.2));
}

#[test]
fn a_call_path_may_hold_a_template() {
    let file = parsed("[call.a]\npath = \"{{ dir }}/part.toml\"\n");

    assert_eq!(file.calls[0].path.value, "{{ dir }}/part.toml");
}

#[test]
fn a_call_without_a_path_is_refused_on_its_header() {
    let text = "[call.a]\ninputs = [\"x\"]\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
    assert_eq!(diagnostics[0].span(), locate(text, "[call.a]"));
}

#[test]
fn a_template_in_a_call_variable_name_is_refused() {
    let text = "[call.a]\npath = \"x.toml\"\nvariables = { \"{{ y }}\" = 1 }\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
    assert_eq!(diagnostics[0].span(), locate(text, "\"{{ y }}\""));
}

#[test]
fn a_call_refuses_stage_settings() {
    let diagnostics = reported("[call.a]\npath = \"x.toml\"\nworkers = 2\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
}

#[test]
fn inputs_and_outputs_are_read() {
    let text = r#"
[input.clip]
description = "The clip"

[output.clean]
inputs = ["denoise"]
"#;

    let file = parsed(text);

    assert_eq!(file.inputs[0].name.value, "clip");
    assert_eq!(file.inputs[0].description.as_deref(), Some("The clip"));
    assert_eq!(file.outputs[0].name.value, "clean");
    assert_eq!(file.outputs[0].source.value, "denoise");
    assert_eq!(file.outputs[0].source.span, locate(text, "\"denoise\""));
}

#[test]
fn an_output_reading_two_stages_is_refused() {
    let diagnostics = reported("[output.a]\ninputs = [\"x\", \"y\"]\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
}

#[test]
fn an_output_without_inputs_is_refused() {
    let diagnostics = reported("[output.a]\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
}

#[test]
fn an_input_refuses_other_keys() {
    let diagnostics = reported("[input.a]\ntype = \"x\"\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
}
