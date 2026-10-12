mod arguments;
mod text;

use std::process::ExitCode;

use av_toolbox_core::{PipelineRegistry, StageInfo};

use crate::cli::DescribeArgs;
use crate::load;

pub(crate) fn run(describe: &DescribeArgs) -> ExitCode {
    let Some(registry) = load::registry() else {
        return ExitCode::FAILURE;
    };

    let found = find(&registry, &describe.stage_type);
    let stages = match found {
        Ok(stages) => stages,
        Err(message) => {
            tracing::error!("{message}");
            return ExitCode::FAILURE;
        },
    };

    let width = text::output_width();
    let rendered = text::render(&stages, width);
    print!("{rendered}");

    ExitCode::SUCCESS
}

/// Every stage registered as `stage_type` sorted by kind, or which type was probably meant.
pub(crate) fn find<'a>(
    registry: &'a PipelineRegistry,
    stage_type: &str,
) -> Result<Vec<StageInfo<'a>>, String> {
    let mut matching: Vec<StageInfo<'a>> = registry
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

    matching.sort_by_key(|info| info.kind().to_string());
    Ok(matching)
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
    fn a_type_registered_twice_finds_both_sorted_by_kind() {
        let registry = built_in();
        let found = find(&registry, "y4m/pipe").expect("registered");

        let kinds: Vec<String> = found.iter().map(|info| info.kind().to_string()).collect();

        assert_eq!(kinds, ["sink", "source"]);
    }

    #[test]
    fn an_unknown_type_suggests_the_closest() {
        let error = find(&built_in(), "y4m/stdot").expect_err("not registered");

        assert!(error.contains("did you mean y4m/stdout?"));
    }

    #[test]
    fn a_type_far_from_any_other_has_no_suggestion() {
        let error = find(&built_in(), "zzzzzzzzzzzz").expect_err("not registered");

        assert!(!error.contains("did you mean"));
    }
}
