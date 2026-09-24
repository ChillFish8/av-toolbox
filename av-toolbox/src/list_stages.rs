use std::process::ExitCode;

use av_toolbox_core::{InputSignature, PipelineRegistry, StageInfo, StreamType, Tag};

use crate::cli::{Args, ListStagesArgs};
use crate::{load, report};

pub(crate) fn run(args: &Args, list: &ListStagesArgs) -> ExitCode {
    let mut stderr = report::stderr(args.color);
    let Some(registry) = load::registry(&mut stderr) else {
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
            report::error_line(&mut stderr, &message);
            ExitCode::FAILURE
        },
    }
}

/// Every registered stage, one row each, sorted by type and then kind.
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

    let mut drawn = format!("{:<width$}  {:<9}  TAGS\n", "TYPE", "KIND");
    for info in &stages {
        let kind = info.kind().to_string();
        let tags_list = info.tags();
        let tags = tag_names(tags_list);
        let stage_type = info.full_identifier();
        let row = format!("{stage_type:<width$}  {kind:<9}  {tags}\n");

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

    format!("{stage_type} ({kind})\n\ninputs:\n{inputs}\narguments:\n{schema}\n")
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

    #[test]
    fn the_table_lists_every_stage_sorted_by_type() {
        let drawn = table(&built_in());
        let lines: Vec<&str> = drawn.lines().collect();

        assert!(lines[0].starts_with("TYPE"));
        assert!(drawn.contains("y4m/stdout"));

        let types: Vec<&str> = lines[1..]
            .iter()
            .map(|line| line.split_whitespace().next().expect("a row has a type"))
            .collect();
        let mut sorted = types.clone();
        sorted.sort();
        assert_eq!(types, sorted);
    }

    #[test]
    fn a_stage_is_described_with_its_inputs_and_arguments() {
        let described = describe(&built_in(), "y4m/stdout").expect("registered");

        assert!(described.starts_with("y4m/stdout (sink)"));
        assert!(described.contains("input"));
        assert!(described.contains("\"frame_rate\""));
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
