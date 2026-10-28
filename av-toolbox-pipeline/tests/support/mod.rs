use av_toolbox_core::StageKind;
use av_toolbox_graph::{ArgSpans, InputRef, Spanned, StageSpec};
use serde_json::json;

/// Puts a `resample/depth` to `bits` without dither in front of every y4m sink in `specs`, so float clips
/// round to their nearest codes.
pub(crate) fn written_at(specs: &mut Vec<StageSpec>, bits: u32) {
    let mut conversions = Vec::new();

    let sinks = specs
        .iter_mut()
        .filter(|spec| spec.kind == StageKind::Sink)
        .filter(|spec| spec.stage_type.value.starts_with("y4m/"));
    for sink in sinks {
        let name = format!("{}_depth", sink.name.value);
        let mut inputs = Vec::new();
        for input in &mut sink.inputs {
            let target = std::mem::replace(&mut input.value.target, name.clone());
            let reference = InputRef { name: None, target };
            inputs.push(Spanned::new(reference, input.span));
        }

        conversions.push(StageSpec {
            name: Spanned::new(name, sink.name.span),
            kind: StageKind::Filter,
            stage_type: Spanned::new(String::from("resample/depth"), sink.stage_type.span),
            inputs,
            args: Spanned::new(json!({ "depth": bits, "dither": "none" }), sink.args.span),
            arg_spans: ArgSpans::new(),
            workers: None,
            deferred: Vec::new(),
            span: sink.span,
        });
    }

    specs.extend(conversions);
}
