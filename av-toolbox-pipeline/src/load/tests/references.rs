use av_toolbox_graph::{DeferredArg, Diagnostic, DiagnosticCode, Expr, Literal, Piece, Scope, StageSpec};
use serde_json::json;

use super::{codes, failed, loaded, stage};
use crate::variables::Supplied;

const MAIN: &str = r#"
[variable.floor]
type = "integer"
default = 10

[source.distorted]
type = "y4m/stdin"

[filter.resize]
type = "test/resize"
inputs = ["distorted"]
args.width = "{{ distorted.width }}"

[sink.out]
type = "y4m/stdout"
inputs = ["resize"]
"#;

const PART: &str = r#"
[variable.width]
type = "integer"
default = 64
min = 16

[input.clip]

[filter.scale]
type = "test/scale"
inputs = ["clip"]
args.width = "{{ clip.width }}"

[output.clean]
inputs = ["scale"]
"#;

const CALLER: &str = r#"
[source.src]
type = "y4m/stdin"

[call.scaled]
path = "part.toml"
inputs = ["src"]

[sink.out]
type = "y4m/stdout"
inputs = ["scaled"]
"#;

const RESIZE_CLEAN: &str = r#"
[filter.resize]
type = "test/resize"
inputs = ["src"]
args.width = "{{ scaled.clean.width }}"
"#;

const REFERENCED_PART: &str = r#"
[input.clip]

[filter.probe]
type = "test/probe"
inputs = ["clip"]

[filter.scale]
type = "test/scale"
inputs = ["clip"]
args.width = "{{ probe.width }}"

[output.raw]
inputs = ["probe"]

[output.clean]
inputs = ["scale"]
"#;

/// Answers only `input.bit_depth`.
struct DepthScope;

impl Scope for DepthScope {
    fn variable(&self, _name: &str) -> Option<Literal> {
        None
    }

    fn field(&self, stage: &str, field: &str) -> Result<Literal, String> {
        match (stage, field) {
            ("input", "bit_depth") => Ok(Literal::Integer(8)),
            _ => {
                let message = format!("no field {stage}.{field}");
                Err(message)
            },
        }
    }
}

fn only_deferred(spec: &StageSpec) -> &DeferredArg {
    let [deferred] = spec.deferred.as_slice() else {
        panic!(
            "{} should defer one arg, it defers {:?}",
            spec.name.value, spec.deferred
        );
    };

    deferred
}

fn main_only(main: &str) -> Vec<StageSpec> {
    let loaded = loaded(&[("/work/main.toml", main)], "/work/main.toml", &Supplied::new());

    loaded.specs
}

fn with_part(main: &str, part: &str) -> Vec<StageSpec> {
    let entries = [("/work/main.toml", main), ("/work/part.toml", part)];
    let loaded = loaded(&entries, "/work/main.toml", &Supplied::new());

    loaded.specs
}

fn failed_with_part(main: &str, part: &str) -> Vec<Diagnostic> {
    let entries = [("/work/main.toml", main), ("/work/part.toml", part)];

    failed(&entries, "/work/main.toml", &Supplied::new())
}

#[test]
fn a_stage_reference_is_deferred() {
    let specs = main_only(MAIN);
    let resize = stage(&specs, "resize");
    let deferred = only_deferred(resize);

    assert_eq!(resize.args.value["width"], json!("{{ distorted.width }}"));
    assert_eq!(deferred.pointer, "/width");
    assert_eq!(deferred.template.stages(), ["distorted"]);
}

#[test]
fn a_template_without_references_is_not_deferred() {
    let specs = main_only(MAIN);
    let distorted = stage(&specs, "distorted");

    assert!(distorted.deferred.is_empty());
}

#[test]
fn variables_inside_a_deferred_template_are_inlined() {
    let main = MAIN
        .replace("[source.distorted]", "[source.input]")
        .replace("\"distorted\"", "\"input\"")
        .replace("{{ distorted.width }}", "{{ max(floor, input.bit_depth) }}");

    let specs = main_only(&main);
    let resize = stage(&specs, "resize");
    let deferred = only_deferred(resize);
    let filled = deferred.template.fill(&DepthScope);

    assert_eq!(filled, Ok(json!(10)));
}

#[test]
fn a_reference_in_a_called_file_resolves_to_the_callers_stage() {
    let specs = with_part(CALLER, PART);
    let scale = stage(&specs, "scaled.scale");
    let deferred = only_deferred(scale);

    assert_eq!(deferred.template.stages(), ["src"]);
}

#[test]
fn a_reference_two_calls_deep_resolves_to_the_root_stage() {
    let middle = r#"
[input.clip]

[call.inner]
path = "inner.toml"
inputs = ["clip"]

[output.done]
inputs = ["inner"]
"#;
    let main = CALLER.replace("part.toml", "middle.toml");
    let entries = [
        ("/work/main.toml", main.as_str()),
        ("/work/middle.toml", middle),
        ("/work/inner.toml", PART),
    ];

    let loaded = loaded(&entries, "/work/main.toml", &Supplied::new());
    let scale = stage(&loaded.specs, "scaled.inner.scale");
    let deferred = only_deferred(scale);

    assert_eq!(deferred.template.stages(), ["src"]);
}

#[test]
fn a_call_output_can_be_referenced() {
    let main = format!("{CALLER}{RESIZE_CLEAN}");

    let specs = with_part(&main, PART);
    let resize = stage(&specs, "resize");
    let deferred = only_deferred(resize);

    assert_eq!(deferred.template.stages(), ["scaled.scale"]);
}

#[test]
fn a_call_variable_can_carry_a_reference() {
    let part = PART.replace("{{ clip.width }}", "{{ width }}");
    let main = CALLER.replace(
        "inputs = [\"src\"]\n",
        "inputs = [\"src\"]\nvariables.width = \"{{ src.width }}\"\n",
    );

    let specs = with_part(&main, &part);
    let scale = stage(&specs, "scaled.scale");
    let deferred = only_deferred(scale);
    let [Piece::Expr(only)] = deferred.template.pieces() else {
        panic!("the template should be one expression");
    };

    assert!(matches!(only.expr, Expr::Checked { .. }));
    assert_eq!(deferred.template.stages(), ["src"]);
}

#[test]
fn a_reference_in_workers_is_refused() {
    let main = MAIN.replace(
        "inputs = [\"distorted\"]\n",
        "inputs = [\"distorted\"]\nworkers = \"{{ distorted.width }}\"\n",
    );

    let diagnostics = failed(
        &[("/work/main.toml", main.as_str())],
        "/work/main.toml",
        &Supplied::new(),
    );

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(
        diagnostics[0].message(),
        "stage references can only be used in args"
    );
}

#[test]
fn a_deferred_variable_in_workers_is_refused() {
    let part = PART.replace(
        "inputs = [\"clip\"]\n",
        "inputs = [\"clip\"]\nworkers = \"{{ width }}\"\n",
    );
    let main = CALLER.replace(
        "inputs = [\"src\"]\n",
        "inputs = [\"src\"]\nvariables.width = \"{{ src.width }}\"\n",
    );

    let diagnostics = failed_with_part(&main, &part);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(
        diagnostics[0].message(),
        "stage references can only be used in args"
    );
}

#[test]
fn a_reference_in_a_call_path_is_refused() {
    let main = CALLER.replace("\"part.toml\"", "\"{{ src.layout }}.toml\"");

    let diagnostics = failed_with_part(&main, PART);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(
        diagnostics[0].message(),
        "stage references can only be used in args"
    );
}

#[test]
fn a_reference_to_a_path_variable_is_refused() {
    let part = format!("[variable.dir]\ntype = \"path\"\n{PART}");
    let main = CALLER.replace(
        "inputs = [\"src\"]\n",
        "inputs = [\"src\"]\nvariables.dir = \"{{ src.width }}\"\n",
    );

    let diagnostics = failed_with_part(&main, &part);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidVariable]);
    assert_eq!(
        diagnostics[0].message(),
        "a path variable cannot come from a stage reference"
    );
}

#[test]
fn a_reference_with_too_many_parts_is_refused() {
    let main = MAIN.replace("{{ distorted.width }}", "{{ a.b.c.width }}");

    let diagnostics = failed(
        &[("/work/main.toml", main.as_str())],
        "/work/main.toml",
        &Supplied::new(),
    );

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("has too many parts"));
}

#[test]
fn a_call_variable_mixing_text_and_a_reference_is_refused() {
    let part = PART.replace("{{ clip.width }}", "{{ width }}");
    let main = CALLER.replace(
        "inputs = [\"src\"]\n",
        "inputs = [\"src\"]\nvariables.width = \"x{{ src.width }}\"\n",
    );

    let diagnostics = failed_with_part(&main, &part);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("must hold only the template"));
}

#[test]
fn a_stage_read_only_through_a_reference_is_kept() {
    let main = CALLER.replace("inputs = [\"scaled\"]", "inputs = [\"scaled.clean\"]");

    let specs = with_part(&main, REFERENCED_PART);
    let scale = stage(&specs, "scaled.scale");
    let deferred = only_deferred(scale);

    assert_eq!(deferred.template.stages(), ["scaled.probe"]);
    assert!(specs.iter().any(|spec| spec.name.value == "scaled.probe"));
}
