use std::process::ExitCode;

use av_toolbox_core::{
    ArtefactDecl,
    ArtefactKind,
    InputSignature,
    PipelineRegistry,
    StageInfo,
    StageKind,
    StreamType,
    Tag,
};

use crate::cli::ListStagesArgs;
use crate::load;

pub(crate) fn run(list: &ListStagesArgs) -> ExitCode {
    let Some(registry) = load::registry() else {
        return ExitCode::FAILURE;
    };

    let Some(stage_type) = &list.stage_type else {
        let drawn = table(&registry);
        print!("{drawn}");

        return ExitCode::SUCCESS;
    };

    let described = describe(&registry, stage_type);
    match described {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        },
        Err(message) => {
            tracing::error!("{message}");
            ExitCode::FAILURE
        },
    }
}

/// What heads the providers, which the TOML cannot name.
const PROVIDERS_HEADING: &str =
    "PROVIDERS, placed by the graph where stages demand them and never named in a pipeline";

/// Every registered stage, one row each, sorted by type and then kind.
///
/// Providers follow under their own heading.
fn table(registry: &PipelineRegistry) -> String {
    let mut stages: Vec<StageInfo<'_>> = registry.stages().collect();
    stages.sort_by_key(|info| {
        let kind = info.kind().to_string();
        (info.full_identifier(), kind)
    });

    let width = stages
        .iter()
        .map(|info| info.full_identifier().len())
        .chain(std::iter::once("TYPE".len()))
        .max()
        .unwrap_or(0);

    let (providers, named): (Vec<StageInfo<'_>>, Vec<StageInfo<'_>>) = stages
        .into_iter()
        .partition(|info| info.kind() == StageKind::Provider);

    let mut drawn = format!("{:<width$}  {:<9}  TAGS\n", "TYPE", "KIND");
    let named_rows = rows(&named, width);
    drawn.push_str(&named_rows);

    if providers.is_empty() {
        return drawn;
    }

    drawn.push('\n');
    drawn.push_str(PROVIDERS_HEADING);
    drawn.push('\n');

    let provider_rows = provider_rows(registry, &providers, width);
    drawn.push_str(&provider_rows);

    drawn
}

/// One row per stage, its type padded to `width`.
fn rows(stages: &[StageInfo<'_>], width: usize) -> String {
    let mut drawn = String::new();

    for info in stages {
        let kind = info.kind().to_string();
        let tags_list = info.tags();
        let tags = tag_names(tags_list);
        let stage_type = info.full_identifier();
        let row = format!("{stage_type:<width$}  {kind:<9}  {tags}\n");

        drawn.push_str(&row);
    }

    drawn
}

/// One row per provider, its type padded to `width`, then the demand kind it serves and its side.
fn provider_rows(registry: &PipelineRegistry, providers: &[StageInfo<'_>], width: usize) -> String {
    let mut drawn = String::new();

    for info in providers {
        let stage_type = info.full_identifier();
        let served = registry
            .providers()
            .find(|(identifier, _, _)| *identifier == stage_type);
        let Some((_, kind, residency)) = served else {
            continue;
        };

        let kind = kind.name();
        let side = residency.provider_tag();
        let row = format!("{stage_type:<width$}  {kind:<11}  {side}\n");

        drawn.push_str(&row);
    }

    drawn
}

/// Describes every stage registered as `stage_type`, or says which type was probably meant.
fn describe(registry: &PipelineRegistry, stage_type: &str) -> Result<String, String> {
    let matching: Vec<StageInfo<'_>> = registry
        .stages()
        .filter(|info| info.full_identifier() == stage_type)
        .collect();

    if matching.is_empty() {
        let closest = closest_type(registry, stage_type);
        let message = match closest {
            Some(closest) => format!("no stage is registered as {stage_type}, did you mean {closest}?"),
            None => format!("no stage is registered as {stage_type}"),
        };

        return Err(message);
    }

    let blocks: Vec<String> = matching.iter().map(describe_stage).collect();
    Ok(blocks.join("\n"))
}

fn describe_stage(info: &StageInfo<'_>) -> String {
    let signature = info.inputs();
    let inputs = describe_inputs(signature);

    let schema_value = info.schema();
    let schema = serde_json::to_string_pretty(schema_value).expect("a schema is plain JSON");

    let stage_type = info.full_identifier();
    let kind = info.kind();

    let artefacts = info.artefacts();
    let declared = describe_artefacts(artefacts.declared());

    format!("{stage_type} ({kind})\n\ninputs:\n{inputs}\nartefacts:\n{declared}\narguments:\n{schema}\n")
}

fn describe_artefacts(declared: &[ArtefactDecl]) -> String {
    if declared.is_empty() {
        return "  none\n".to_string();
    }

    let mut lines = String::new();
    for decl in declared {
        let kind = match decl.kind {
            ArtefactKind::Structured => "structured",
            ArtefactKind::Opaque { .. } => "opaque",
        };
        let extension = decl.kind.extension();
        let line = format!("  {}  {kind}  .{extension}\n", decl.name);
        let description = format!("    {}\n", decl.description);

        lines.push_str(&line);
        lines.push_str(&description);
    }

    lines
}

fn describe_inputs(signature: &InputSignature) -> String {
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

fn tag_names(tags: &[Tag]) -> String {
    if tags.is_empty() {
        return "-".to_string();
    }

    let names: Vec<&str> = tags.iter().map(tag_name).collect();
    names.join(", ")
}

fn tag_name(tag: &Tag) -> &'static str {
    match tag {
        Tag::Denoise => "denoise",
        Tag::Deband => "deband",
        Tag::SuperResolution => "super-resolution",
        Tag::Deinterlace => "deinterlace",
        Tag::Color => "color",
        Tag::Conversion => "conversion",
        Tag::Evaluation => "evaluation",
        Tag::Other(name) => name,
    }
}

/// The registered type nearest `wanted`, when one is within three edits.
fn closest_type<'a>(registry: &'a PipelineRegistry, wanted: &str) -> Option<&'a str> {
    let mut best: Option<(usize, &'a str)> = None;

    for info in registry.stages() {
        let candidate = info.full_identifier();
        let distance = edit_distance(wanted, candidate);
        let closer = best.is_none_or(|(shortest, _)| distance < shortest);
        if distance <= 3 && closer {
            best = Some((distance, candidate));
        }
    }

    best.map(|(_, name)| name)
}

fn edit_distance(left: &str, right: &str) -> usize {
    let right: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right.len()).collect();

    for (index, left_char) in left.chars().enumerate() {
        let mut current = vec![index + 1; right.len() + 1];
        for (position, right_char) in right.iter().enumerate() {
            let substitution = previous[position] + usize::from(left_char != *right_char);
            let insertion = current[position] + 1;
            let deletion = previous[position + 1] + 1;

            current[position + 1] = substitution.min(insertion).min(deletion);
        }

        previous = current;
    }

    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built_in() -> PipelineRegistry {
        av_toolbox_pipeline::registry().expect("the built-in stages register")
    }

    /// The rows of the stage table, before the providers' heading.
    fn stage_rows(drawn: &str) -> Vec<&str> {
        drawn
            .lines()
            .skip(1)
            .take_while(|line| !line.is_empty())
            .collect()
    }

    #[test]
    fn the_table_lists_every_stage_sorted_by_type() {
        let drawn = table(&built_in());
        let lines: Vec<&str> = drawn.lines().collect();

        assert!(lines[0].starts_with("TYPE"));
        assert!(drawn.contains("y4m/stdout"));

        let types: Vec<&str> = stage_rows(&drawn)
            .iter()
            .map(|line| line.split_whitespace().next().expect("a row has a type"))
            .collect();
        let mut sorted = types.clone();
        sorted.sort();
        assert_eq!(types, sorted);
    }

    #[test]
    fn providers_are_listed_under_their_own_heading() {
        let drawn = table(&built_in());

        let in_stages = stage_rows(&drawn)
            .iter()
            .any(|line| line.starts_with("provider/scene"));
        assert!(!in_stages, "{drawn}");

        let heading = drawn
            .lines()
            .position(|line| line == PROVIDERS_HEADING)
            .expect("a providers heading");
        let rows: Vec<&str> = drawn.lines().skip(heading + 1).collect();
        assert!(rows[0].starts_with("provider/scene-score"), "{drawn}");
        assert!(rows[1].starts_with("provider/scene-split"), "{drawn}");
    }

    #[test]
    fn a_stage_is_described_with_its_inputs_and_arguments() {
        let described = describe(&built_in(), "y4m/stdout").expect("registered");

        assert!(described.starts_with("y4m/stdout (sink)"));
        assert!(described.contains("input"));
        assert!(described.contains("\"frame_rate\""));
    }

    #[test]
    fn declared_artefacts_are_listed_with_their_kind_and_description() {
        let declared = [
            ArtefactDecl::opaque("table", "tbl", "A grain table."),
            ArtefactDecl::structured("cuts", "Where each scene starts."),
        ];

        let described = describe_artefacts(&declared);
        let expected = concat!(
            "  table  opaque  .tbl\n",
            "    A grain table.\n",
            "  cuts  structured  .json\n",
            "    Where each scene starts.\n",
        );

        assert_eq!(described, expected);
    }

    #[test]
    fn a_stage_with_no_artefacts_lists_none() {
        assert_eq!(describe_artefacts(&[]), "  none\n");
    }

    #[test]
    fn a_type_registered_twice_describes_both() {
        let described = describe(&built_in(), "y4m/pipe").expect("registered");

        assert!(described.contains("y4m/pipe (source)"));
        assert!(described.contains("y4m/pipe (sink)"));
    }

    #[test]
    fn an_unknown_type_suggests_the_closest() {
        let error = describe(&built_in(), "y4m/stdot").expect_err("not registered");

        assert!(error.contains("did you mean y4m/stdout?"));
    }

    #[test]
    fn a_type_far_from_any_other_has_no_suggestion() {
        let error = describe(&built_in(), "zzzzzzzzzzzz").expect_err("not registered");

        assert!(!error.contains("did you mean"));
    }
}
