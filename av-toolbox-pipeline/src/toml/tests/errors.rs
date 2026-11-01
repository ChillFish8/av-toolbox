use av_toolbox_graph::{Diagnostic, DiagnosticCode};

use super::{locate, reported};

fn codes(diagnostics: &[Diagnostic]) -> Vec<DiagnosticCode> {
    diagnostics.iter().map(|diagnostic| diagnostic.code()).collect()
}

#[test]
fn bad_syntax_is_invalid_syntax() {
    let diagnostics = reported("[source.a\ntype = \"x\"\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidSyntax);
}

#[test]
fn every_syntax_error_is_reported() {
    let diagnostics = reported("[source.a]\ntype = \n[x\n");

    assert!(diagnostics.len() >= 2);
}

#[test]
fn an_unknown_top_level_table_is_refused() {
    let text = "[stage.a]\ntype = \"a/b\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(diagnostics[0].path(), "stage");
    assert_eq!(diagnostics[0].span(), locate(text, "stage"));
}

#[test]
fn a_removed_transform_notes_how_to_log_scenes() {
    let text = "[transform.x]\ntype = \"scene/detect\"\n";

    let diagnostics = reported(text);

    let notes: Vec<&str> = diagnostics[0]
        .notes()
        .iter()
        .map(|note| note.message.as_str())
        .collect();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("requested by stages"), "{notes:?}");
    assert!(notes[0].contains("`metadata/ndjson`"), "{notes:?}");
    assert!(notes[0].contains("`args.scenes`"), "{notes:?}");
}

#[test]
fn a_transform_table_is_refused_as_removed() {
    let text = "[transform.x]\ntype = \"scene/detect\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::RemovedTransforms]);
    assert_eq!(diagnostics[0].path(), "transform");
    assert_eq!(diagnostics[0].span(), locate(text, "transform"));
    assert!(
        diagnostics[0]
            .message()
            .contains("transforms were removed, stages now request scene breaks themselves")
    );
}

#[test]
fn a_plural_kind_suggests_the_singular() {
    let diagnostics = reported("[sources.a]\ntype = \"a/b\"\n");

    assert!(
        diagnostics[0].notes()[0]
            .message
            .contains("did you mean `source`?")
    );
}

#[test]
fn a_kind_that_is_not_a_table_is_refused() {
    let diagnostics = reported("source = 3\n");

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
}

#[test]
fn a_stage_that_is_not_a_table_is_refused() {
    let diagnostics = reported("[source]\na = 3\n");

    assert_eq!(diagnostics[0].path(), "source.a");
}

#[test]
fn an_unknown_stage_key_lists_the_allowed_keys() {
    let text = "[filter.a]\ntype = \"a/b\"\ninput = [\"x\"]\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].path(), "filter.a.input");
    assert_eq!(diagnostics[0].span(), locate(text, "input"));
    assert!(
        diagnostics[0].notes()[0]
            .message
            .contains("type, inputs, workers and args")
    );
}

#[test]
fn a_missing_type_is_reported_on_the_header() {
    let text = "[filter.a]\ninputs = [\"x\"]\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].path(), "filter.a.type");
    assert_eq!(diagnostics[0].span(), locate(text, "[filter.a]"));
}

#[test]
fn a_type_that_is_not_a_string_is_refused() {
    let text = "[filter.a]\ntype = 4\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].path(), "filter.a.type");
    assert_eq!(diagnostics[0].span(), locate(text, "4"));
    assert_eq!(
        diagnostics[0].message(),
        "type must be a string, such as \"y4m/pipe\""
    );
}

#[test]
fn inputs_of_the_wrong_shape_are_refused() {
    let text = "[filter.a]\ntype = \"a/b\"\ninputs = \"clip\"\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].path(), "filter.a.inputs");
    assert_eq!(diagnostics[0].span(), locate(text, "\"clip\""));
}

#[test]
fn an_input_that_is_not_a_name_is_refused() {
    let text = "[filter.a]\ntype = \"a/b\"\ninputs = [3]\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].span(), locate(text, "3"));
}

#[test]
fn zero_workers_are_refused() {
    let diagnostics = reported("[filter.a]\ntype = \"a/b\"\nworkers = 0\n");

    assert_eq!(diagnostics[0].path(), "filter.a.workers");
}

#[test]
fn workers_written_as_text_without_a_template_are_refused() {
    let text = "[filter.a]\ntype = \"a/b\"\nworkers = \"8\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(
        diagnostics[0].message(),
        "workers must be a whole number of at least 1"
    );
    assert_eq!(diagnostics[0].span(), locate(text, "\"8\""));
}

#[test]
fn negative_workers_are_refused() {
    let diagnostics = reported("[filter.a]\ntype = \"a/b\"\nworkers = -2\n");

    assert_eq!(diagnostics[0].path(), "filter.a.workers");
}

#[test]
fn a_name_with_other_characters_is_refused() {
    let text = "[source.\"my stage\"]\ntype = \"a/b\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(diagnostics[0].span(), locate(text, "[source.\"my stage\"]"));
}

#[test]
fn a_datetime_argument_is_refused() {
    let text = "[source.a]\ntype = \"a/b\"\nargs.when = 1979-05-27\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].path(), "source.a.args.when");
    assert_eq!(diagnostics[0].span(), locate(text, "1979-05-27"));
}

#[test]
fn a_non_finite_float_argument_is_refused() {
    let text = "[source.a]\ntype = \"a/b\"\nargs.level = nan\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].path(), "source.a.args.level");
}

#[test]
fn an_integer_past_64_bits_is_refused() {
    let text = "[source.a]\ntype = \"a/b\"\nargs.size = 9223372036854775808\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].path(), "source.a.args.size");
}

#[test]
fn every_broken_stage_is_reported() {
    let text = "[filter.a]\ntype = 1\n\n[filter.b]\ntype = 2\n";

    assert_eq!(reported(text).len(), 2);
}

#[test]
fn a_valid_stage_alongside_a_broken_one_still_fails_the_whole_file() {
    let text = "[filter.good]\ntype = \"a/b\"\n\n[filter.bad]\ntype = 4\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].path(), "filter.bad.type");
    assert_eq!(diagnostics[0].span(), locate(text, "4"));
}

#[test]
fn a_plural_variable_table_suggests_the_singular() {
    let diagnostics = reported("[variables.a]\ntype = \"string\"\n");

    assert!(
        diagnostics[0].notes()[0]
            .message
            .contains("did you mean `variable`?")
    );
}

#[test]
fn a_template_in_type_is_refused() {
    let text = "[source.a]\ntype = \"{{ kind }}\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(diagnostics[0].span(), locate(text, "\"{{ kind }}\""));
}

#[test]
fn a_template_in_inputs_is_refused() {
    let text = "[sink.a]\ntype = \"a/b\"\ninputs = [\"{{ from }}\"]\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
}

#[test]
fn a_template_in_an_argument_name_is_refused() {
    let text = "[source.a]\ntype = \"a/b\"\nargs = { \"{{ key }}\" = 1 }\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
}

#[test]
fn an_unclosed_template_in_args_is_refused_at_its_string() {
    let text = "[source.a]\ntype = \"a/b\"\nargs.path = \"{{ dir\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(diagnostics[0].span(), locate(text, "\"{{ dir\""));
}

#[test]
fn a_template_in_a_named_input_slot_is_refused() {
    let text = "[sink.a]\ntype = \"a/b\"\ninputs = { \"{{ x }}\" = \"a\" }\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(diagnostics[0].span(), locate(text, "\"{{ x }}\""));
}

#[test]
fn an_output_with_inputs_that_are_not_names_is_one_error() {
    let listed = reported("[output.o]\ninputs = [1]\n");
    let text = reported("[output.o]\ninputs = \"x\"\n");

    assert_eq!(codes(&listed), [DiagnosticCode::InvalidDefinition]);
    assert_eq!(codes(&text), [DiagnosticCode::InvalidDefinition]);
}

#[test]
fn a_hyphenated_stage_name_suggests_an_underscore() {
    let text = "[filter.my-clip]\ntype = \"a/b\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    let notes: Vec<&str> = diagnostics[0]
        .notes()
        .iter()
        .map(|note| note.message.as_str())
        .collect();
    assert_eq!(notes, ["did you mean `my_clip`?"]);
}

#[test]
fn a_stage_name_starting_with_a_digit_asks_for_a_rename() {
    let text = "[source.4k]\ntype = \"a/b\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    let message = diagnostics[0].message();
    assert!(message.contains("starts with a digit"), "{message}");
    assert!(message.contains("rename it"), "{message}");
}

#[test]
fn a_stage_name_that_is_a_template_keyword_asks_for_a_rename() {
    let text = "[filter.then]\ntype = \"a/b\"\n";

    let diagnostics = reported(text);

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    let message = diagnostics[0].message();
    assert!(message.contains("template keyword"), "{message}");
    assert!(message.contains("rename it"), "{message}");
}

#[test]
fn a_call_name_that_is_a_template_keyword_asks_for_a_rename() {
    let text = "[call.not]\npath = \"part.toml\"\n";

    let diagnostics = reported(text);

    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("template keyword")),
        "{diagnostics:?}"
    );
}

#[test]
fn a_stage_named_is_is_refused() {
    let diagnostics = reported("[filter.is]\ntype = \"a/b\"\n");

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("template keyword"));
}

#[test]
fn a_stage_named_none_is_refused() {
    let diagnostics = reported("[filter.none]\ntype = \"a/b\"\n");

    assert_eq!(codes(&diagnostics), [DiagnosticCode::InvalidDefinition]);
    assert!(diagnostics[0].message().contains("template keyword"));
}
