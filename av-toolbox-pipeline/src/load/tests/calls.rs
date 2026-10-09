use std::path::Path;

use av_toolbox_graph::{DiagnosticCode, FileId, Span};

use super::{codes, failed, loaded, stage};
use crate::load;
use crate::testing::MemoryFiles;
use crate::variables::Supplied;

const PART: &str = r#"
[variable.strength]
type = "float"
default = 0.8

[input.clip]

[filter.denoise]
type = "test/denoise"
inputs = ["clip"]
args.sigma = "{{ strength }}"

[output.clean]
inputs = ["denoise"]
"#;

const MAIN: &str = r#"
[source.src]
type = "y4m/stdin"

[call.denoised]
path = "parts/denoise.toml"
inputs = { clip = "src" }
variables.strength = 1.2

[sink.out]
type = "y4m/stdout"
inputs = ["denoised"]
"#;

fn files<'a>(main: &'a str, part: &'a str) -> [(&'static str, &'a str); 2] {
    [("/work/main.toml", main), ("/work/parts/denoise.toml", part)]
}

#[test]
fn a_called_stage_is_prefixed_and_reads_the_callers_stage() {
    let loaded = loaded(&files(MAIN, PART), "/work/main.toml", &Supplied::new());
    let denoise = stage(&loaded.specs, "denoised.denoise");

    assert_eq!(denoise.inputs[0].value.target, "src");
    assert_eq!(denoise.args.value["sigma"], serde_json::json!(1.2));
}

#[test]
fn a_single_output_call_is_read_by_its_name() {
    let loaded = loaded(&files(MAIN, PART), "/work/main.toml", &Supplied::new());
    let out = stage(&loaded.specs, "out");

    assert_eq!(out.inputs[0].value.target, "denoised.denoise");
}

#[test]
fn an_output_is_read_by_call_and_output_name() {
    let main = MAIN.replace("inputs = [\"denoised\"]", "inputs = [\"denoised.clean\"]");

    let loaded = loaded(&files(&main, PART), "/work/main.toml", &Supplied::new());
    let out = stage(&loaded.specs, "out");

    assert_eq!(out.inputs[0].value.target, "denoised.denoise");
}

#[test]
fn a_called_stage_keeps_its_own_file_and_span() {
    let loaded = loaded(&files(MAIN, PART), "/work/main.toml", &Supplied::new());
    let denoise = stage(&loaded.specs, "denoised.denoise");
    let header_start = PART.find("[filter.denoise]").expect("present");

    assert_ne!(denoise.span.file, FileId::ROOT);
    assert_eq!(denoise.span.start, header_start);
}

#[test]
fn a_call_with_several_outputs_needs_an_output_name() {
    let part = format!("{PART}\n[output.raw]\ninputs = [\"clip\"]\n");

    let diagnostics = failed(&files(MAIN, &part), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownCallOutput]);
    assert!(diagnostics[0].notes()[0].message.contains("clean, raw"));
}

#[test]
fn an_unknown_output_is_reported() {
    let main = MAIN.replace("inputs = [\"denoised\"]", "inputs = [\"denoised.dirty\"]");

    let diagnostics = failed(&files(&main, PART), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownCallOutput]);
}

#[test]
fn a_missing_call_input_is_reported() {
    let main = MAIN.replace("inputs = { clip = \"src\" }\n", "");

    let diagnostics = failed(&files(&main, PART), "/work/main.toml", &Supplied::new());

    assert!(codes(&diagnostics).contains(&DiagnosticCode::InvalidInputs));
}

#[test]
fn an_unknown_call_input_is_reported() {
    let main = MAIN.replace("{ clip = \"src\" }", "{ clip = \"src\", extra = \"src\" }");

    let diagnostics = failed(&files(&main, PART), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidInputs]);
}

#[test]
fn a_single_input_may_be_given_as_a_list() {
    let main = MAIN.replace("{ clip = \"src\" }", "[\"src\"]");

    let loaded = loaded(&files(&main, PART), "/work/main.toml", &Supplied::new());
    let denoise = stage(&loaded.specs, "denoised.denoise");

    assert_eq!(denoise.inputs[0].value.target, "src");
}

#[test]
fn an_unknown_call_variable_is_reported() {
    let main = MAIN.replace("variables.strength", "variables.power");

    let diagnostics = failed(&files(&main, PART), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownVariable]);
}

#[test]
fn a_call_variable_may_use_the_callers_templates() {
    let main = format!(
        "[variable.power]\ntype = \"float\"\ndefault = 2.5\n{}",
        MAIN.replace("variables.strength = 1.2", "variables.strength = \"{{ power }}\"")
    );

    let loaded = loaded(&files(&main, PART), "/work/main.toml", &Supplied::new());
    let denoise = stage(&loaded.specs, "denoised.denoise");

    assert_eq!(denoise.args.value["sigma"], serde_json::json!(2.5));
}

#[test]
fn flags_do_not_reach_a_called_file() {
    let mut supplied = Supplied::new();
    supplied.set_flag("strength", "3.0");

    let diagnostics = failed(&files(MAIN, PART), "/work/main.toml", &supplied);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownVariable]);
}

#[test]
fn nested_calls_are_prefixed_and_resolve_relative_to_each_file() {
    let outer = r#"
[input.clip]

[call.inner]
path = "denoise.toml"
inputs = ["clip"]

[output.done]
inputs = ["inner"]
"#;
    let main = MAIN
        .replace("parts/denoise.toml", "parts/outer.toml")
        .replace("variables.strength = 1.2\n", "");
    let entries = [
        ("/work/main.toml", main.as_str()),
        ("/work/parts/outer.toml", outer),
        ("/work/parts/denoise.toml", PART),
    ];

    let loaded = loaded(&entries, "/work/main.toml", &Supplied::new());
    let denoise = stage(&loaded.specs, "denoised.inner.denoise");
    let out = stage(&loaded.specs, "out");

    assert_eq!(denoise.inputs[0].value.target, "src");
    assert_eq!(out.inputs[0].value.target, "denoised.inner.denoise");
}

#[test]
fn an_output_may_forward_an_input() {
    let part = "[input.clip]\n\n[output.same]\ninputs = [\"clip\"]\n";
    let main = MAIN.replace("variables.strength = 1.2\n", "");

    let loaded = loaded(&files(&main, part), "/work/main.toml", &Supplied::new());
    let out = stage(&loaded.specs, "out");

    assert_eq!(out.inputs[0].value.target, "src");
}

#[test]
fn calling_the_same_file_twice_gives_two_copies() {
    let main = r#"
[source.src]
type = "y4m/stdin"

[call.first]
path = "parts/denoise.toml"
inputs = ["src"]

[call.second]
path = "parts/denoise.toml"
inputs = ["first"]

[sink.out]
type = "y4m/stdout"
inputs = ["second"]
"#;

    let loaded = loaded(&files(main, PART), "/work/main.toml", &Supplied::new());
    let first = stage(&loaded.specs, "first.denoise");
    let second = stage(&loaded.specs, "second.denoise");

    assert_eq!(second.inputs[0].value.target, "first.denoise");
    assert_ne!(first.span.file, second.span.file);
}

#[test]
fn a_file_calling_itself_is_a_call_cycle() {
    let looping = r#"
[input.clip]

[call.again]
path = "./denoise.toml"
inputs = ["clip"]

[output.o]
inputs = ["again"]
"#;

    let diagnostics = failed(&files(MAIN, looping), "/work/main.toml", &Supplied::new());
    let cycle = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code() == DiagnosticCode::CallCycle)
        .expect("a call cycle");

    assert!(cycle.message().contains("calling ./denoise.toml here"));
}

#[test]
fn two_files_calling_each_other_is_a_call_cycle_with_notes() {
    let first = r#"
[input.clip]

[call.b]
path = "b.toml"
inputs = ["clip"]

[output.o]
inputs = ["b"]
"#;
    let second = r#"
[input.clip]

[call.a]
path = "denoise.toml"
inputs = ["clip"]

[output.o]
inputs = ["a"]
"#;
    let entries = [
        ("/work/main.toml", MAIN),
        ("/work/parts/denoise.toml", first),
        ("/work/parts/b.toml", second),
    ];
    let header_start = first.find("[call.b]").expect("present");
    let header_end = header_start + "[call.b]".len();
    let call_b = Span::in_file(FileId::new(1), header_start, header_end);

    let diagnostics = failed(&entries, "/work/main.toml", &Supplied::new());
    let cycle = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code() == DiagnosticCode::CallCycle)
        .expect("a call cycle");
    let spans: Vec<Span> = cycle.notes().iter().map(|note| note.span).collect();

    assert_eq!(spans, [call_b]);
}

#[test]
fn a_call_reading_its_own_output_is_a_cycle() {
    let part = "[input.clip]\n\n[output.same]\ninputs = [\"clip\"]\n";
    let main = r#"
[call.looped]
path = "parts/denoise.toml"
inputs = ["looped"]

[sink.out]
type = "y4m/stdout"
inputs = ["looped"]
"#;

    let diagnostics = failed(&files(main, part), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::Cycle]);
}

#[test]
fn an_unreadable_called_file_is_reported_on_its_path() {
    let diagnostics = failed(&[("/work/main.toml", MAIN)], "/work/main.toml", &Supplied::new());
    let path_start = MAIN.find("\"parts/denoise.toml\"").expect("present");

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnreadableFile]);
    assert_eq!(diagnostics[0].span().start, path_start);
}

#[test]
fn a_broken_called_file_does_not_report_its_outputs_as_unknown() {
    let broken = "[input.clip\n";

    let diagnostics = failed(&files(MAIN, broken), "/work/main.toml", &Supplied::new());

    assert!(!codes(&diagnostics).contains(&DiagnosticCode::UnknownCallOutput));
    assert!(codes(&diagnostics).contains(&DiagnosticCode::InvalidSyntax));
}

#[test]
fn an_error_in_a_called_file_points_into_that_file() {
    let broken = PART.replace("[\"denoise\"]", "[\"denoise\", \"extra\"]");

    let files = MemoryFiles::new(&files(MAIN, &broken));
    let path = Path::new("/work/main.toml");
    let failure = load(path, &files, &Supplied::new()).expect_err("should fail");
    let diagnostic = failure.diagnostics.iter().next().expect("one diagnostic");
    let noted = failure.sources.with_call_notes(diagnostic.clone());

    assert_ne!(diagnostic.span().file, FileId::ROOT);
    assert_eq!(
        noted.notes().last().map(|note| note.span.file),
        Some(FileId::ROOT)
    );
}

#[test]
fn a_duplicate_name_across_a_call_and_a_stage_is_reported() {
    let main = format!("{MAIN}\n[filter.denoised]\ntype = \"a/b\"\ninputs = [\"src\"]\n");

    let diagnostics = failed(&files(&main, PART), "/work/main.toml", &Supplied::new());

    assert!(codes(&diagnostics).contains(&DiagnosticCode::DuplicateStageName));
}

#[test]
fn an_unread_output_drops_its_stages() {
    let extra = r#"
[filter.spare]
type = "test/denoise"
inputs = ["clip"]

[output.extra]
inputs = ["spare"]
"#;
    let part = format!("{PART}{extra}");
    let main = MAIN.replace("inputs = [\"denoised\"]", "inputs = [\"denoised.clean\"]");

    let loaded = loaded(&files(&main, &part), "/work/main.toml", &Supplied::new());
    let names: Vec<&str> = loaded.specs.iter().map(|spec| spec.name.value.as_str()).collect();

    assert!(!names.contains(&"denoised.spare"));
    assert!(names.contains(&"denoised.denoise"));
}

#[test]
fn an_unread_output_keeps_its_stages_beside_an_artefact_sink() {
    let extra = r#"
[filter.spare]
type = "test/denoise"
inputs = ["clip"]

[output.extra]
inputs = ["spare"]
"#;
    let reports = r#"
[sink.reports]
type = "artefacts/files"
args.directory = "reports"
"#;
    let part = format!("{PART}{extra}");
    let read_clean = MAIN.replace("inputs = [\"denoised\"]", "inputs = [\"denoised.clean\"]");
    let main = format!("{read_clean}{reports}");

    let loaded = loaded(&files(&main, &part), "/work/main.toml", &Supplied::new());
    let names: Vec<&str> = loaded.specs.iter().map(|spec| spec.name.value.as_str()).collect();

    assert!(names.contains(&"denoised.spare"));
    assert!(names.contains(&"denoised.denoise"));
}

#[test]
fn a_called_stage_reaching_nothing_is_kept() {
    let part = format!("{PART}\n[filter.lost]\ntype = \"test/denoise\"\ninputs = [\"clip\"]\n");

    let loaded = loaded(&files(MAIN, &part), "/work/main.toml", &Supplied::new());
    let names: Vec<&str> = loaded.specs.iter().map(|spec| spec.name.value.as_str()).collect();

    assert!(names.contains(&"denoised.lost"));
}

#[test]
fn a_stage_feeding_an_unread_output_and_a_kept_stage_is_kept() {
    let extra = r#"
[filter.shared]
type = "test/denoise"
inputs = ["clip"]

[filter.lost]
type = "test/denoise"
inputs = ["shared"]

[output.extra]
inputs = ["shared"]
"#;
    let part = format!("{PART}{extra}");
    let main = MAIN.replace("inputs = [\"denoised\"]", "inputs = [\"denoised.clean\"]");

    let loaded = loaded(&files(&main, &part), "/work/main.toml", &Supplied::new());
    let shared = stage(&loaded.specs, "denoised.shared");
    let lost = stage(&loaded.specs, "denoised.lost");

    assert_eq!(shared.inputs[0].value.target, "src");
    assert_eq!(lost.inputs[0].value.target, "denoised.shared");
}

#[test]
fn a_list_of_several_inputs_is_one_error_on_the_list() {
    let main = MAIN.replace("{ clip = \"src\" }", "[\"src\", \"src\"]");
    let list_start = main.find("[\"src\", \"src\"]").expect("present");

    let diagnostics = failed(&files(&main, PART), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidInputs]);
    assert_eq!(diagnostics[0].span().start, list_start);
}

#[test]
fn a_single_listed_input_for_several_inputs_is_one_error_on_the_list() {
    let part = format!("[input.other]\n{PART}");
    let main = MAIN.replace("{ clip = \"src\" }", "[\"src\"]");
    let list_start = main.find("[\"src\"]").expect("present");

    let diagnostics = failed(&files(&main, &part), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidInputs]);
    assert_eq!(diagnostics[0].span().start, list_start);
}

#[test]
fn a_call_input_typo_is_reported_on_the_callers_input() {
    let main = MAIN.replace("{ clip = \"src\" }", "{ clip = \"scr\" }");
    let input_start = main.find("clip = \"scr\"").expect("present");

    let diagnostics = failed(&files(&main, PART), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownStageReference]);
    assert_eq!(diagnostics[0].span().file, FileId::ROOT);
    assert_eq!(diagnostics[0].span().start, input_start);
    assert!(diagnostics[0].notes()[0].message.contains("did you mean `src`?"));
}

#[test]
fn an_unread_output_with_a_typo_is_reported_in_the_called_file() {
    let part = format!("{PART}\n[output.extra]\ninputs = [\"denoies\"]\n");
    let main = MAIN.replace("inputs = [\"denoised\"]", "inputs = [\"denoised.clean\"]");
    let source_start = part.find("\"denoies\"").expect("present");

    let diagnostics = failed(&files(&main, &part), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownStageReference]);
    assert_ne!(diagnostics[0].span().file, FileId::ROOT);
    assert_eq!(diagnostics[0].span().start, source_start);
}

#[test]
fn a_failed_call_variable_is_reported_once() {
    let part = PART.replace("default = 0.8\n", "");
    let main = MAIN.replace(
        "variables.strength = 1.2",
        "variables.strength = \"{{ strenght }}\"",
    );

    let diagnostics = failed(&files(&main, &part), "/work/main.toml", &Supplied::new());

    assert_eq!(codes(&diagnostics), [DiagnosticCode::UnknownVariable]);
}

#[test]
fn a_path_call_value_resolves_against_the_calling_file() {
    let part = r#"
[variable.clip]
type = "path"

[source.read]
type = "y4m/pipe"
args.path = "{{ clip }}"

[output.done]
inputs = ["read"]
"#;
    let main = r#"
[call.part]
path = "parts/p.toml"
variables.clip = "media/x.y4m"

[sink.out]
type = "y4m/stdout"
inputs = ["part"]
"#;
    let entries = [("/work/main.toml", main), ("/work/parts/p.toml", part)];

    let loaded = loaded(&entries, "/work/main.toml", &Supplied::new());
    let read = stage(&loaded.specs, "part.read");

    assert_eq!(read.args.value["path"], serde_json::json!("/work/media/x.y4m"));
}
