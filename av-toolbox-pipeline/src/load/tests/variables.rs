use std::path::Path;

use av_toolbox_graph::{DiagnosticCode, Span};
use serde_json::json;

use super::{codes, failed, loaded, stage};
use crate::load;
use crate::variables::{Supplied, VariableType, VariableValue};

const PIPELINE: &str = r#"
[variable.sigma]
type = "float"
description = "How strong the filter is"
min = 0.0

[variable.dir]
type = "path"
default = "out"

[source.src]
type = "y4m/pipe"
args.path = "{{ dir }}/in.y4m"
args.sigma = "{{ sigma }}"

[sink.out]
type = "y4m/stdout"
inputs = ["src"]
"#;

fn with_flag(name: &str, value: &str) -> Supplied {
    let mut supplied = Supplied::new();
    supplied.set_flag(name, value);

    supplied
}

#[test]
fn a_flag_fills_a_lone_template_with_its_type() {
    let supplied = with_flag("sigma", "1.5");

    let loaded = loaded(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["sigma"], json!(1.5));
}

#[test]
fn a_path_default_resolves_against_its_file() {
    let supplied = with_flag("sigma", "1");

    let loaded = loaded(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["path"], json!("/work/out/in.y4m"));
}

#[test]
fn a_path_flag_resolves_against_the_working_directory() {
    let mut supplied = with_flag("sigma", "1");
    supplied.set_flag("dir", "elsewhere");

    let loaded = loaded(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);
    let source = stage(&loaded.specs, "src");
    let expected = std::path::absolute("elsewhere/in.y4m").expect("absolute");

    assert_eq!(source.args.value["path"], json!(expected.to_string_lossy()));
}

#[test]
fn a_bare_root_file_name_resolves_against_the_working_directory() {
    let supplied = with_flag("sigma", "1");

    let loaded = loaded(&[("main.toml", PIPELINE)], "main.toml", &supplied);
    let source = stage(&loaded.specs, "src");
    let expected = std::path::absolute("out/in.y4m").expect("absolute");

    assert_eq!(source.args.value["path"], json!(expected.to_string_lossy()));
}

#[test]
fn a_root_path_spelt_with_dot_dot_resolves_the_same() {
    let supplied = with_flag("sigma", "1");

    let loaded = loaded(
        &[("/work/main.toml", PIPELINE)],
        "/work/sub/../main.toml",
        &supplied,
    );
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["path"], json!("/work/out/in.y4m"));
}

#[test]
fn the_environment_fills_a_variable_without_a_flag() {
    let variables = [("AVTB_VAR_SIGMA".to_string(), "2".to_string())];
    let supplied = Supplied::from_environment(variables);

    let loaded = loaded(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["sigma"], json!(2.0));
}

#[test]
fn a_missing_required_variable_names_its_description() {
    let diagnostics = failed(
        &[("/work/main.toml", PIPELINE)],
        "/work/main.toml",
        &Supplied::new(),
    );

    assert_eq!(codes(&diagnostics), [DiagnosticCode::MissingVariable]);

    let notes: Vec<&str> = diagnostics[0]
        .notes()
        .iter()
        .map(|note| note.message.as_str())
        .collect();

    assert!(notes.iter().any(|note| note.contains("How strong the filter is")));
    assert!(notes.iter().any(|note| note.contains("--var sigma=")));
    assert!(notes.iter().any(|note| note.contains("AVTB_VAR_SIGMA=")));
}

#[test]
fn a_flag_breaking_a_rule_is_an_invalid_variable() {
    let supplied = with_flag("sigma", "-1");

    let diagnostics = failed(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidVariable]);
    assert!(diagnostics[0].message().contains("--var sigma"));
}

#[test]
fn an_unknown_flag_suggests_the_closest_variable() {
    let mut supplied = with_flag("sigma", "1");
    supplied.set_flag("sigam", "1");

    let diagnostics = failed(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownVariable]);
    assert!(
        diagnostics[0].notes()[0]
            .message
            .contains("did you mean `sigma`?")
    );
}

#[test]
fn an_upper_case_flag_suggests_the_lower_case_name() {
    let mut supplied = with_flag("sigma", "1");
    supplied.set_flag("DIR", "elsewhere");

    let diagnostics = failed(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);
    let note = &diagnostics[0].notes()[0].message;

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownVariable]);
    assert_eq!(note, "variable names are lower case, did you mean `dir`?");
}

#[test]
fn an_unknown_template_suggests_the_first_declared_of_tied_names() {
    let pipeline = r#"
[variable.sigmb]
type = "float"
default = 1.0

[variable.sigmc]
type = "float"
default = 1.0

[source.src]
type = "y4m/pipe"
args.sigma = "{{ sigma }}"
"#;

    let diagnostics = failed(
        &[("/work/main.toml", pipeline)],
        "/work/main.toml",
        &Supplied::new(),
    );
    let note = &diagnostics[0].notes()[0].message;

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownVariable]);
    assert!(note.contains("did you mean `sigmb`?"));
}

#[test]
fn an_unknown_environment_variable_is_ignored() {
    let variables = [
        ("AVTB_VAR_SIGMA".to_string(), "1".to_string()),
        ("AVTB_VAR_OTHER".to_string(), "x".to_string()),
    ];
    let supplied = Supplied::from_environment(variables);

    let loaded = loaded(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);

    assert_eq!(loaded.specs.len(), 2);
}

#[test]
fn an_invalid_default_is_reported_on_the_default() {
    let text = "[variable.a]\ntype = \"integer\"\ndefault = \"x\"\n\n[sink.o]\ntype = \"y4m/stdout\"\n";

    let diagnostics = failed(&[("/work/main.toml", text)], "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidVariable]);
    let default_start = text.find("\"x\"").expect("present");
    assert_eq!(diagnostics[0].span().start, default_start);
}

#[test]
fn an_unknown_template_name_is_reported_on_its_string() {
    let text = "[source.a]\ntype = \"y4m/pipe\"\nargs.path = \"{{ nothing }}\"\n";

    let diagnostics = failed(&[("/work/main.toml", text)], "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownVariable]);
    let string_start = text.find("\"{{ nothing }}\"").expect("present");
    assert_eq!(diagnostics[0].span().start, string_start);
}

#[test]
fn templated_workers_resolve_to_a_count() {
    let text = "[variable.n]\ntype = \"integer\"\ndefault = 3\n\n[source.a]\ntype = \"y4m/stdin\"\nworkers = \"{{ n }}\"\n";

    let loaded = loaded(&[("/work/main.toml", text)], "/work/main.toml", &Supplied::new());

    assert_eq!(loaded.specs[0].workers, Some(3));
}

#[test]
fn templated_workers_of_zero_are_refused() {
    let text = "[variable.n]\ntype = \"integer\"\ndefault = 0\n\n[source.a]\ntype = \"y4m/stdin\"\nworkers = \"{{ n }}\"\n";

    let diagnostics = failed(&[("/work/main.toml", text)], "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
}

#[test]
fn inputs_in_the_root_file_are_refused() {
    let text = "[input.clip]\n\n[output.o]\ninputs = [\"clip\"]\n";

    let diagnostics = failed(&[("/work/main.toml", text)], "/work/main.toml", &Supplied::new());

    assert_eq!(
        codes(&diagnostics),
        [
            DiagnosticCode::InvalidDefinition,
            DiagnosticCode::InvalidDefinition
        ]
    );
}

#[test]
fn an_unreadable_root_file_is_reported() {
    let diagnostics = failed(&[], "/work/missing.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnreadableFile]);
    assert!(
        diagnostics[0]
            .message()
            .contains("cannot read /work/missing.toml")
    );
}

#[test]
fn the_root_file_is_in_the_source_map() {
    let supplied = with_flag("sigma", "1");

    let loaded = loaded(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);
    let root = loaded
        .sources
        .get(av_toolbox_graph::FileId::ROOT)
        .expect("the root file");

    assert_eq!(root.path, Path::new("/work/main.toml"));
    assert_eq!(root.text, PIPELINE);
}

#[test]
fn the_root_variables_are_listed_in_declaration_order() {
    let supplied = with_flag("sigma", "1.5");

    let loaded = loaded(&[("/work/main.toml", PIPELINE)], "/work/main.toml", &supplied);
    let names: Vec<&str> = loaded
        .variables
        .iter()
        .map(|variable| variable.name.as_str())
        .collect();

    assert_eq!(names, ["sigma", "dir"]);
    assert_eq!(loaded.variables[0].variable_type, VariableType::Float);
    assert_eq!(loaded.variables[0].value, Some(VariableValue::Float(1.5)));
    assert_eq!(
        loaded.variables[1].value,
        Some(VariableValue::Path("/work/out".to_string()))
    );
}

const EXPRESSION_PIPELINE: &str = r#"
[variable.sigma]
type = "float"

[source.src]
type = "y4m/pipe"
args.sigma = "{{ sigma * 2 }}"
"#;

#[test]
fn an_expression_fills_from_variables() {
    let supplied = with_flag("sigma", "1.5");

    let loaded = loaded(
        &[("/work/main.toml", EXPRESSION_PIPELINE)],
        "/work/main.toml",
        &supplied,
    );
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["sigma"], json!(3.0));
}

#[test]
fn an_unknown_variable_in_an_expression_is_reported() {
    let pipeline = EXPRESSION_PIPELINE.replace("{{ sigma * 2 }}", "{{ sigmaa + 1 }}");
    let supplied = with_flag("sigma", "1.5");

    let diagnostics = failed(
        &[("/work/main.toml", pipeline.as_str())],
        "/work/main.toml",
        &supplied,
    );
    let note = &diagnostics[0].notes()[0].message;

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownVariable]);
    assert!(note.contains("did you mean `sigma`?"));
}

#[test]
fn an_error_inside_an_expression_is_narrowed_to_it() {
    let pipeline = EXPRESSION_PIPELINE.replace("{{ sigma * 2 }}", "x {{ sigma + true }}");
    let supplied = with_flag("sigma", "1.5");

    let diagnostics = failed(
        &[("/work/main.toml", pipeline.as_str())],
        "/work/main.toml",
        &supplied,
    );
    let start = pipeline.find("{{ sigma").expect("template start");
    let end = start + "{{ sigma + true }}".len();

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(diagnostics[0].span(), Span::new(start, end));
}

#[test]
fn a_template_without_spaces_fills() {
    let pipeline = EXPRESSION_PIPELINE.replace("{{ sigma * 2 }}", "{{sigma}}");
    let supplied = with_flag("sigma", "1.5");

    let loaded = loaded(
        &[("/work/main.toml", pipeline.as_str())],
        "/work/main.toml",
        &supplied,
    );
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["sigma"], json!(1.5));
}

#[test]
fn an_empty_template_is_refused() {
    let pipeline = EXPRESSION_PIPELINE.replace("{{ sigma * 2 }}", "{{ }}");
    let supplied = with_flag("sigma", "1.5");

    let diagnostics = failed(
        &[("/work/main.toml", pipeline.as_str())],
        "/work/main.toml",
        &supplied,
    );

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(
        diagnostics[0]
            .message()
            .contains("a template must hold an expression")
    );
}

fn optional_pipeline(extra: &str, arg: &str) -> String {
    format!(
        "[variable.x]\ntype = \"string\"\noptional = true\n{extra}\n\n[source.src]\ntype = \"y4m/pipe\"\n{arg}\n"
    )
}

#[test]
fn an_unset_optional_variable_fills_to_none() {
    let text = optional_pipeline("", "args.flag = \"{{ x is none }}\"");

    let loaded = loaded(&[("/work/main.toml", &text)], "/work/main.toml", &Supplied::new());
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["flag"], json!(true));
}

#[test]
fn a_set_optional_variable_is_used() {
    let text = optional_pipeline("", "args.value = \"{{ x }}\"");
    let supplied = with_flag("x", "hello");

    let loaded = loaded(&[("/work/main.toml", &text)], "/work/main.toml", &supplied);
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["value"], json!("hello"));
}

#[test]
fn an_invalid_optional_flag_is_reported_once() {
    let text = r#"
[variable.count]
type = "integer"
optional = true

[source.src]
type = "y4m/pipe"
args.n = "{{ count + 1 }}"
"#;
    let supplied = with_flag("count", "abc");

    let diagnostics = failed(&[("/work/main.toml", text)], "/work/main.toml", &supplied);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidVariable]);
}

#[test]
fn choices_are_not_checked_when_unset() {
    let text = optional_pipeline("choices = [\"a\"]", "args.flag = \"{{ x is none }}\"");

    let loaded = loaded(&[("/work/main.toml", &text)], "/work/main.toml", &Supplied::new());
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["flag"], json!(true));
}

#[test]
fn a_required_variable_without_a_default_is_still_missing() {
    let text =
        "[variable.x]\ntype = \"string\"\n\n[source.src]\ntype = \"y4m/pipe\"\nargs.value = \"{{ x }}\"\n";

    let diagnostics = failed(&[("/work/main.toml", text)], "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::MissingVariable]);
}

#[test]
fn an_unset_root_variable_resolves_to_no_value() {
    let text = optional_pipeline("", "args.flag = \"{{ x is none }}\"");

    let loaded = loaded(&[("/work/main.toml", &text)], "/work/main.toml", &Supplied::new());

    assert_eq!(loaded.variables[0].name, "x");
    assert_eq!(loaded.variables[0].value, None);
}

#[test]
fn a_variable_named_none_fails_the_load_at_its_header() {
    for keyword in ["none", "is"] {
        let text = format!("[variable.{keyword}]\ntype = \"string\"\noptional = true\n");
        let start = text.find(&format!("[variable.{keyword}]")).expect("header");
        let end = start + format!("[variable.{keyword}]").len();

        let diagnostics = failed(&[("/work/main.toml", &text)], "/work/main.toml", &Supplied::new());

        assert_eq!(
            codes(&diagnostics),
            [DiagnosticCode::InvalidDefinition],
            "{keyword}"
        );
        assert!(diagnostics[0].message().contains("template keyword"));
        let span = diagnostics[0].span();
        assert_eq!((span.start, span.end), (start, end), "{keyword}");
    }
}

const DISPLAY_PIPELINE: &str = r#"
[variable.display_config]
type = "path"
optional = true

[source.src]
type = "y4m/pipe"
args.display = "cvvdp"
args.display_config = "{{ if display_config is not none then read(display_config) else none }}"
"#;

const DISPLAY_JSON: &str = "{\"name\": \"custom\"}";

#[test]
fn an_unset_display_config_leaves_the_arg_out() {
    let entries = [("/work/main.toml", DISPLAY_PIPELINE)];

    let loaded = loaded(&entries, "/work/main.toml", &Supplied::new());
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value, json!({ "display": "cvvdp" }));
}

#[test]
fn a_set_path_is_read_into_the_arg() {
    let entries = [
        ("/work/main.toml", DISPLAY_PIPELINE),
        ("/work/display.json", DISPLAY_JSON),
    ];
    let supplied = with_flag("display_config", "/work/display.json");

    let loaded = loaded(&entries, "/work/main.toml", &supplied);
    let source = stage(&loaded.specs, "src");

    assert_eq!(source.args.value["display_config"], json!(DISPLAY_JSON));
}

#[test]
fn read_resolves_against_the_template_file() {
    let main = r#"
[call.inner]
path = "sub/inner.toml"

[sink.out]
type = "y4m/stdout"
inputs = ["inner"]
"#;
    let inner = r#"
[source.src]
type = "y4m/pipe"
args.config = '{{ read("d.json") }}'

[output.done]
inputs = ["src"]
"#;
    let entries = [
        ("/work/main.toml", main),
        ("/work/sub/inner.toml", inner),
        ("/work/sub/d.json", DISPLAY_JSON),
    ];

    let loaded = loaded(&entries, "/work/main.toml", &Supplied::new());
    let source = stage(&loaded.specs, "inner.src");

    assert_eq!(source.args.value["config"], json!(DISPLAY_JSON));
}

#[test]
fn read_of_a_missing_file_is_reported() {
    let supplied = with_flag("display_config", "/work/missing.json");

    let diagnostics = failed(
        &[("/work/main.toml", DISPLAY_PIPELINE)],
        "/work/main.toml",
        &supplied,
    );

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("cannot read"));
}

#[test]
fn read_of_a_non_utf8_file_is_reported() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let main_path = directory.path().join("main.toml");
    let pipeline = r#"
[source.src]
type = "y4m/pipe"
args.config = '{{ read("bad.json") }}'
"#;
    std::fs::write(&main_path, pipeline).expect("the pipeline is written");

    let bad_path = directory.path().join("bad.json");
    std::fs::write(&bad_path, [0xff, 0xfe]).expect("the bad file is written");

    let result = load::load(&main_path, &load::DiskFiles, &Supplied::new());
    let Err(failure) = result else {
        panic!("the pipeline should fail to load");
    };
    let diagnostics = failure.diagnostics.into_inner();
    let template_start = pipeline.find("'{{").expect("the template");

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("cannot read"));
    assert_eq!(diagnostics[0].span().start, template_start);
}

const STAGE_READING: &str = r#"
[source.src]
type = "y4m/pipe"

[filter.next]
type = "test/next"
inputs = ["src"]
args.value = 'TEMPLATE'
"#;

#[test]
fn read_with_a_stage_reference_is_refused() {
    let templates = [
        "{{ read(src.width) }}",
        "{{ if src.width > 1 then read(\"a\") else none }}",
    ];

    for template in templates {
        let text = STAGE_READING.replace("TEMPLATE", template);

        let diagnostics = failed(&[("/work/main.toml", &text)], "/work/main.toml", &Supplied::new());

        assert_eq!(
            codes(&diagnostics),
            [DiagnosticCode::InvalidDefinition],
            "{template}"
        );
        assert!(
            diagnostics[0].message().contains("`read`"),
            "{template}: {}",
            diagnostics[0].message()
        );
    }
}

#[test]
fn none_nested_in_an_array_is_refused() {
    let text = optional_pipeline("", "args.sizes = [\"{{ x }}\"]");

    let diagnostics = failed(&[("/work/main.toml", &text)], "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("top-level arg"));
}

#[test]
fn none_nested_in_an_object_is_refused() {
    let text = optional_pipeline("", "args.options = { name = \"{{ x }}\" }");

    let diagnostics = failed(&[("/work/main.toml", &text)], "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("top-level arg"));
}

#[test]
fn none_in_workers_is_refused() {
    let text = optional_pipeline("", "workers = \"{{ x }}\"");

    let diagnostics = failed(&[("/work/main.toml", &text)], "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("`none`"));
}
