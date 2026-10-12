use std::io::IsTerminal;

use av_toolbox_core::{ArtefactDecl, ArtefactKind, InputSignature, StageInfo, StreamType};

use super::arguments::{self, Argument};
use crate::list_stages::tag_name;

/// The width text wraps to when stdout is not a terminal.
const PIPED_WIDTH: usize = 100;

/// How far a description sits from the left edge, under the name it describes.
const DESCRIPTION_INDENT: usize = 6;

/// The narrowest a wrapped line may get, however small the terminal.
const MIN_LINE_WIDTH: usize = 40;

/// The terminal's width, or [PIPED_WIDTH](crate::describe::text::PIPED_WIDTH) when stdout is not a terminal.
pub(super) fn output_width() -> usize {
    let stdout = std::io::stdout();
    if !stdout.is_terminal() {
        return PIPED_WIDTH;
    }

    let terminal = console::Term::stdout();
    let size = terminal.size_checked();
    match size {
        Some((_rows, columns)) => usize::from(columns),
        None => PIPED_WIDTH,
    }
}

/// Describes each stage in turn, separated by a blank line.
pub(super) fn render(stages: &[StageInfo<'_>], width: usize) -> String {
    let blocks: Vec<String> = stages.iter().map(|info| render_stage(info, width)).collect();

    blocks.join("\n")
}

fn render_stage(info: &StageInfo<'_>, width: usize) -> String {
    let header = header(info);
    let summary_line = info.summary();
    let summary = wrap(summary_line, width, 0);
    let mut drawn = format!("{header}\n\n{summary}");

    if let Some(details) = info.details() {
        let wrapped = wrap(details, width, 0);
        drawn.push('\n');
        drawn.push_str(&wrapped);
    }

    let signature = info.inputs();
    let inputs = render_inputs(signature);
    drawn.push_str("\nINPUTS\n");
    drawn.push_str(&inputs);

    let stage_artefacts = info.artefacts();
    let declarations = stage_artefacts.declared();
    let declared = render_artefacts(declarations, width);
    drawn.push_str("\nARTEFACTS\n");
    drawn.push_str(&declared);

    let schema = info.schema();
    let read = arguments::read(schema);
    let rendered = render_arguments(&read, width);
    drawn.push_str("\nARGUMENTS\n");
    drawn.push_str(&rendered);

    drawn
}

/// The stage's type, kind and tags on one line.
fn header(info: &StageInfo<'_>) -> String {
    let identifier = info.full_identifier();
    let kind = info.kind();
    let tags = info.tags();
    if tags.is_empty() {
        return format!("{identifier}  {kind}");
    }

    let names: Vec<&str> = tags.iter().map(tag_name).collect();
    let joined = names.join(", ");
    format!("{identifier}  {kind}  {joined}")
}

fn render_inputs(signature: &InputSignature) -> String {
    if signature.is_empty() {
        return "  none\n".to_string();
    }

    let mut lines = String::new();
    for param in signature.params() {
        let accepts = param.accepts();
        let stream = stream_name(accepts);
        let need = if param.is_required() {
            "required"
        } else {
            "optional"
        };
        let name = param.name();
        let line = format!("  {name}  {stream}  {need}\n");

        lines.push_str(&line);
    }

    if signature.is_variadic() {
        let accepts = signature.variadic_accepts();
        let stream = stream_name(accepts);
        let line = format!("  ...  {stream}  any number\n");

        lines.push_str(&line);
    }

    lines
}

fn stream_name(accepts: Option<StreamType>) -> String {
    match accepts {
        Some(stream) => stream.to_string(),
        None => "any".to_string(),
    }
}

fn render_artefacts(declared: &[ArtefactDecl], width: usize) -> String {
    if declared.is_empty() {
        return "  none\n".to_string();
    }

    let mut lines = String::new();
    for decl in declared {
        let kind = artefact_kind_name(&decl.kind);
        let extension = decl.kind.extension();
        let line = format!("  {}  {kind}  .{extension}\n", decl.name);
        let description = wrap(decl.description, width, DESCRIPTION_INDENT);

        lines.push_str(&line);
        lines.push_str(&description);
    }

    lines
}

pub(super) fn artefact_kind_name(kind: &ArtefactKind) -> &'static str {
    match kind {
        ArtefactKind::Structured => "structured",
        ArtefactKind::Opaque { .. } => "opaque",
    }
}

fn render_arguments(arguments: &[Argument], width: usize) -> String {
    if arguments.is_empty() {
        return "  none\n".to_string();
    }

    let name_width = arguments
        .iter()
        .map(|argument| argument.name.chars().count())
        .max()
        .unwrap_or(0);
    let shape_width = arguments
        .iter()
        .map(|argument| argument.shape.chars().count())
        .max()
        .unwrap_or(0);

    let mut drawn = String::new();
    for argument in arguments {
        let name = &argument.name;
        let shape = &argument.shape;
        let note = argument_note(argument);
        let row = format!("  {name:<name_width$}  {shape:<shape_width$}  {note}");
        let trimmed = row.trim_end();
        drawn.push_str(trimmed);
        drawn.push('\n');

        let Some(description) = &argument.description else {
            continue;
        };

        let wrapped = wrap(description, width, DESCRIPTION_INDENT);
        drawn.push_str(&wrapped);
    }

    drawn
}

/// `required`, the default value, or nothing for an argument that may be left out.
fn argument_note(argument: &Argument) -> String {
    if argument.required {
        return "required".to_string();
    }

    match &argument.default {
        Some(default) => format!("default {default}"),
        None => String::new(),
    }
}

/// Wraps `text` to `width` columns with every line indented by `indent`.
///
/// Blank lines in `text` stay as blank lines between paragraphs.
fn wrap(text: &str, width: usize, indent: usize) -> String {
    let available = width.saturating_sub(indent).max(MIN_LINE_WIDTH);
    let margin = " ".repeat(indent);
    let paragraphs: Vec<String> = text
        .split("\n\n")
        .map(|paragraph| wrap_paragraph(paragraph, available, &margin))
        .collect();

    paragraphs.join("\n")
}

fn wrap_paragraph(paragraph: &str, available: usize, margin: &str) -> String {
    let mut wrapped = String::new();
    let mut line = String::new();

    for word in paragraph.split_whitespace() {
        let fits = line.is_empty() || line.chars().count() + 1 + word.chars().count() <= available;
        if !fits {
            wrapped.push_str(margin);
            wrapped.push_str(&line);
            wrapped.push('\n');
            line.clear();
        }

        if !line.is_empty() {
            line.push(' ');
        }

        line.push_str(word);
    }

    if !line.is_empty() {
        wrapped.push_str(margin);
        wrapped.push_str(&line);
        wrapped.push('\n');
    }

    wrapped
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::ArtefactDecl;

    use super::*;

    fn argument(
        name: &str,
        shape: &str,
        required: bool,
        default: Option<&str>,
        description: Option<&str>,
    ) -> Argument {
        Argument {
            name: name.to_string(),
            shape: shape.to_string(),
            required,
            default: default.map(str::to_string),
            description: description.map(str::to_string),
        }
    }

    #[test]
    fn arguments_line_up_with_their_notes_and_descriptions() {
        let arguments = [
            argument("depth", "integer", true, None, Some("The output bit depth.")),
            argument(
                "preset",
                "fast | slow",
                false,
                Some("fast"),
                Some("How much quality is traded for speed."),
            ),
            argument("lambda", "float, optional", false, None, None),
        ];

        let drawn = render_arguments(&arguments, 100);
        let expected = concat!(
            "  depth   integer          required\n",
            "      The output bit depth.\n",
            "  preset  fast | slow      default fast\n",
            "      How much quality is traded for speed.\n",
            "  lambda  float, optional\n",
        );

        assert_eq!(drawn, expected);
    }

    #[test]
    fn a_stage_with_no_arguments_lists_none() {
        assert_eq!(render_arguments(&[], 100), "  none\n");
    }

    #[test]
    fn long_text_wraps_at_the_width() {
        let text = "abcd abcd abcd abcd abcd abcd abcd abcd abcd";

        let wrapped = wrap(text, 40, 0);

        assert_eq!(wrapped, "abcd abcd abcd abcd abcd abcd abcd abcd\nabcd\n");
    }

    #[test]
    fn paragraphs_keep_a_blank_line_between_them() {
        assert_eq!(wrap("first\n\nsecond", 100, 2), "  first\n\n  second\n");
    }

    #[test]
    fn a_tiny_width_still_wraps_at_the_minimum() {
        let wrapped = wrap("one two", 1, DESCRIPTION_INDENT);

        assert_eq!(wrapped, "      one two\n");
    }

    #[test]
    fn declared_artefacts_are_listed_with_their_kind_and_description() {
        let declared = [
            ArtefactDecl::opaque("table", "tbl", "A grain table."),
            ArtefactDecl::structured("cuts", "Where each scene starts."),
        ];

        let described = render_artefacts(&declared, 100);
        let expected = concat!(
            "  table  opaque  .tbl\n",
            "      A grain table.\n",
            "  cuts  structured  .json\n",
            "      Where each scene starts.\n",
        );

        assert_eq!(described, expected);
    }

    #[test]
    fn a_stage_with_no_artefacts_lists_none() {
        assert_eq!(render_artefacts(&[], 100), "  none\n");
    }

    #[test]
    fn a_built_in_stage_renders_every_section_in_order() {
        let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
        let stages = crate::describe::find(&registry, "y4m/stdout").expect("registered");

        let drawn = render(&stages, 100);

        assert!(
            drawn.starts_with("y4m/stdout  sink\n\nWrites the stream as y4m to standard output.\n\nINPUTS\n"),
            "{drawn}"
        );

        let artefacts = drawn.find("\nARTEFACTS\n").expect("an artefacts section");
        let arguments = drawn.find("\nARGUMENTS\n").expect("an arguments section");
        assert!(artefacts < arguments, "{drawn}");
        assert!(drawn.contains("  frame_rate  "), "{drawn}");
    }
}
