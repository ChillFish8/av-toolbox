use std::process::ExitCode;

use av_toolbox_core::{PipelineRegistry, StageInfo, StageKind, Tag};

use crate::cli::ListStagesArgs;
use crate::load;

pub(crate) fn run(list: &ListStagesArgs) -> ExitCode {
    let Some(registry) = load::registry() else {
        return ExitCode::FAILURE;
    };

    let kinds: Vec<StageKind> = list.kinds.iter().map(|kind| kind.stage_kind()).collect();
    let drawn = table(&registry, &kinds);
    print!("{drawn}");

    ExitCode::SUCCESS
}

/// What heads the providers, which the TOML cannot name.
const PROVIDERS_HEADING: &str =
    "PROVIDERS, placed by the graph where stages demand them and never named in a pipeline";

/// Every registered stage of `kinds`, one row each, sorted by type and then kind.
///
/// An empty `kinds` lists every kind. Providers follow under their own heading.
fn table(registry: &PipelineRegistry, kinds: &[StageKind]) -> String {
    let mut stages: Vec<StageInfo<'_>> = registry
        .stages()
        .filter(|info| {
            let kind = info.kind();
            wanted(kinds, kind)
        })
        .collect();
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

    let only_providers = !kinds.is_empty() && kinds.iter().all(|kind| *kind == StageKind::Provider);
    let mut drawn = String::new();

    if !only_providers {
        let named_rows = rows(&named, width);
        drawn.push_str(&named_rows);
    }

    if providers.is_empty() {
        return drawn;
    }

    if !drawn.is_empty() {
        drawn.push('\n');
    }

    drawn.push_str(PROVIDERS_HEADING);
    drawn.push('\n');

    let provider_rows = provider_rows(registry, &providers, width);
    drawn.push_str(&provider_rows);

    drawn
}

fn wanted(kinds: &[StageKind], kind: StageKind) -> bool {
    kinds.is_empty() || kinds.contains(&kind)
}

/// A header, then one row per stage with its type padded to `width`.
fn rows(stages: &[StageInfo<'_>], width: usize) -> String {
    let tags: Vec<String> = stages
        .iter()
        .map(|info| {
            let stage_tags = info.tags();
            tag_names(stage_tags)
        })
        .collect();
    let tags_width = tags
        .iter()
        .map(String::len)
        .chain(std::iter::once("TAGS".len()))
        .max()
        .unwrap_or(0);

    let mut drawn = format!(
        "{:<width$}  {:<9}  {:<tags_width$}  SUMMARY\n",
        "TYPE", "KIND", "TAGS"
    );

    for (info, tags) in stages.iter().zip(&tags) {
        let kind = info.kind().to_string();
        let stage_type = info.full_identifier();
        let summary = info.summary();
        let row = format!("{stage_type:<width$}  {kind:<9}  {tags:<tags_width$}  {summary}\n");

        drawn.push_str(&row);
    }

    drawn
}

/// One row per provider, its type padded to `width`, then the demand kind it serves, its side and summary.
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
        let summary = info.summary();
        let row = format!("{stage_type:<width$}  {kind:<11}  {side:<6}  {summary}\n");

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
        let drawn = table(&built_in(), &[]);
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
        let drawn = table(&built_in(), &[]);

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
    fn the_table_shows_each_stages_summary() {
        let drawn = table(&built_in(), &[]);

        assert!(
            drawn.lines().next().expect("a header").ends_with("SUMMARY"),
            "{drawn}"
        );

        let row = drawn
            .lines()
            .find(|line| line.starts_with("y4m/stdout "))
            .expect("y4m/stdout is listed");
        assert!(
            row.ends_with("Writes the stream as y4m to standard output."),
            "{row}"
        );
    }

    #[test]
    fn filtering_by_kind_lists_only_that_kind() {
        let drawn = table(&built_in(), &[StageKind::Sink]);

        let kinds: Vec<&str> = stage_rows(&drawn)
            .iter()
            .map(|line| line.split_whitespace().nth(1).expect("a row has a kind"))
            .collect();

        assert!(!kinds.is_empty(), "{drawn}");
        assert!(kinds.iter().all(|kind| *kind == "sink"), "{drawn}");
        assert!(!drawn.contains(PROVIDERS_HEADING), "{drawn}");
    }

    #[test]
    fn filtering_by_provider_lists_only_the_providers() {
        let drawn = table(&built_in(), &[StageKind::Provider]);

        assert!(drawn.starts_with(PROVIDERS_HEADING), "{drawn}");
        assert!(!drawn.contains("TYPE"), "{drawn}");
    }

    #[test]
    fn filtering_by_two_kinds_lists_both() {
        let drawn = table(&built_in(), &[StageKind::Sink, StageKind::Provider]);

        assert!(drawn.contains("y4m/stdout"), "{drawn}");
        assert!(drawn.contains(PROVIDERS_HEADING), "{drawn}");
    }
}
