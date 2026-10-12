use std::process::ExitCode;

use av_toolbox_core::{PipelineRegistry, StageInfo, StageKind, Tag};

use crate::cli::ListStagesArgs;
use crate::load;

pub(crate) fn run(_list: &ListStagesArgs) -> ExitCode {
    let Some(registry) = load::registry() else {
        return ExitCode::FAILURE;
    };

    let drawn = table(&registry);
    print!("{drawn}");

    ExitCode::SUCCESS
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

fn tag_names(tags: &[Tag]) -> String {
    if tags.is_empty() {
        return "-".to_string();
    }

    let names: Vec<&str> = tags.iter().map(tag_name).collect();
    names.join(", ")
}

pub(crate) fn tag_name(tag: &Tag) -> &'static str {
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
}
