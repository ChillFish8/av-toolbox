use av_toolbox_core::{PipelineRegistry, StageKind};
use av_toolbox_graph::{ArgSpans, Spanned, StageSpec};
use serde_json::json;

use super::file::stage_key;

/// The sink every pass writes into, so nothing leaves the pipeline.
const DISCARD: &str = "null/discard";

/// Drops every artefact sink, since a profiling pass writes nothing out of the pipeline.
pub(super) fn drop_artefact_sinks(specs: &mut Vec<StageSpec>, registry: &PipelineRegistry) {
    specs.retain(|spec| !registry.is_artefact_sink(&spec.stage_type.value));
}

/// Replaces every sink with `null/discard`, keeping its inputs, and returns the replaced sinks' keys.
pub(super) fn discard_sinks(specs: &mut [StageSpec]) -> Vec<String> {
    let mut replaced = Vec::new();

    for spec in specs.iter_mut().filter(|spec| spec.kind == StageKind::Sink) {
        let key = stage_key(spec);
        replaced.push(key);

        spec.stage_type = Spanned::new(DISCARD.to_string(), spec.stage_type.span);
        spec.args = Spanned::new(json!({}), spec.args.span);
        spec.arg_spans = ArgSpans::default();
    }

    replaced
}

/// A copy of `specs` with every declared width cleared, so every stage runs one worker wide.
pub(super) fn undeclared(specs: &[StageSpec]) -> Vec<StageSpec> {
    let mut cleared = specs.to_vec();

    for spec in &mut cleared {
        spec.workers = None;
    }

    cleared
}

/// How many items reach the sinks for every frame, one per sink input.
pub(super) fn sink_inputs(specs: &[StageSpec]) -> u64 {
    let inputs: usize = specs
        .iter()
        .filter(|spec| spec.kind == StageKind::Sink)
        .map(|spec| spec.inputs.len())
        .sum();

    inputs.max(1) as u64
}

/// How many frames reached the sinks after the warmup, or `None` when the warmup was never passed.
pub(super) fn counted_frames(report_frames: u64, warmup_items: u64, sink_inputs: u64) -> Option<u64> {
    if report_frames < warmup_items {
        return None;
    }

    let counted_items = report_frames - warmup_items;
    Some(counted_items / sink_inputs)
}

#[cfg(test)]
mod tests {
    use av_toolbox_graph::{InputRef, Span};

    use super::*;

    fn spec(kind: StageKind, name: &str, inputs: &[&str]) -> StageSpec {
        let inputs = inputs
            .iter()
            .map(|target| {
                Spanned::unknown(InputRef {
                    name: None,
                    target: target.to_string(),
                })
            })
            .collect();

        StageSpec {
            name: Spanned::unknown(name.to_string()),
            kind,
            stage_type: Spanned::unknown("y4m/stdout".to_string()),
            inputs,
            args: Spanned::unknown(json!({ "frame_rate": "25" })),
            arg_spans: ArgSpans::default(),
            workers: Some(2),
            span: Span::unknown(),
        }
    }

    #[test]
    fn sinks_become_discards_and_keep_their_inputs() {
        let mut specs = vec![
            spec(StageKind::Filter, "blur", &["clip"]),
            spec(StageKind::Sink, "output", &["blur"]),
        ];

        let replaced = discard_sinks(&mut specs);

        assert_eq!(replaced, vec!["sink.output".to_string()]);
        assert_eq!(specs[1].stage_type.value, "null/discard");
        assert_eq!(specs[1].args.value, json!({}));
        assert_eq!(specs[1].inputs.len(), 1);
        assert_eq!(specs[0].stage_type.value, "y4m/stdout");
    }

    #[test]
    fn artefact_sinks_are_dropped_before_profiling() {
        let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
        let mut files = spec(StageKind::Sink, "reports", &[]);
        files.stage_type = Spanned::unknown(String::from("artefacts/files"));
        let mut specs = vec![spec(StageKind::Sink, "output", &["blur"]), files];

        drop_artefact_sinks(&mut specs, &registry);
        let names: Vec<&str> = specs.iter().map(|spec| spec.name.value.as_str()).collect();

        assert_eq!(names, vec!["output"]);
    }

    #[test]
    fn undeclared_clears_every_width() {
        let specs = vec![spec(StageKind::Filter, "blur", &["clip"])];

        let cleared = undeclared(&specs);

        assert_eq!(cleared[0].workers, None);
        assert_eq!(specs[0].workers, Some(2));
    }

    #[test]
    fn sink_inputs_count_every_sink_input() {
        let specs = vec![
            spec(StageKind::Sink, "a", &["x", "y"]),
            spec(StageKind::Sink, "b", &["z"]),
        ];

        assert_eq!(sink_inputs(&specs), 3);
    }

    #[test]
    fn counted_frames_are_none_before_the_warmup_is_passed() {
        assert_eq!(counted_frames(20, 30, 1), None);
        assert_eq!(counted_frames(330, 30, 1), Some(300));
        assert_eq!(counted_frames(660, 60, 2), Some(300));
    }
}
