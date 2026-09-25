use std::path::Path;

use av_toolbox_graph::DiagnosticCode;
use serde_json::json;

use super::{codes, failed, loaded, stage};
use crate::variables::Supplied;

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
