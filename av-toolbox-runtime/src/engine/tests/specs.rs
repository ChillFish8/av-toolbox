use av_toolbox_core::StageKind;
use av_toolbox_graph::{ArgSpans, InputRef, Span, Spanned, StageSpec};
use serde_json::json;

fn spec(kind: StageKind, name: &str, stage_type: &str) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: Vec::new(),
        args: Spanned::unknown(json!({})),
        arg_spans: ArgSpans::default(),
        workers: None,
        span: Span::unknown(),
    }
}

fn positional(target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: None,
        target: target.to_string(),
    })
}

pub(crate) fn source_spec(name: &str, chunks: &[u64], id: u32) -> StageSpec {
    let mut source = spec(StageKind::Source, name, "test/listed");
    source.args = Spanned::unknown(json!({ "chunks": chunks, "id": id }));
    source
}

/// A source declaring its frames arrive at `rate`, as numerator and denominator.
pub(crate) fn rated_source_spec(name: &str, chunks: &[u64], id: u32, rate: [u32; 2]) -> StageSpec {
    let mut source = spec(StageKind::Source, name, "test/listed");
    source.args = Spanned::unknown(json!({ "chunks": chunks, "id": id, "frame_rate": rate }));
    source
}

pub(crate) fn host_source_spec(name: &str, chunks: &[u64], id: u32) -> StageSpec {
    let mut source = spec(StageKind::Source, name, "test/host-listed");
    source.args = Spanned::unknown(json!({ "chunks": chunks, "id": id }));
    source
}

/// A source whose frame indices start at `index_base` rather than zero.
pub(crate) fn offset_source_spec(name: &str, chunks: &[u64], id: u32, index_base: u64) -> StageSpec {
    let mut source = spec(StageKind::Source, name, "test/listed");
    source.args = Spanned::unknown(json!({ "chunks": chunks, "id": id, "index_base": index_base }));
    source
}

/// A source declared to run wider than one worker, which this engine has to refuse.
pub(crate) fn widened_source_spec(name: &str, chunks: &[u64], id: u32, workers: usize) -> StageSpec {
    let mut source = source_spec(name, chunks, id);
    source.workers = Some(workers);
    source
}

pub(crate) fn filter_spec(name: &str, input: &str, workers: Option<usize>) -> StageSpec {
    let mut filter = spec(StageKind::Filter, name, "test/pass");
    filter.inputs = vec![positional(input)];
    filter.workers = workers;
    filter
}

pub(crate) fn host_pass_spec(name: &str, input: &str, workers: Option<usize>) -> StageSpec {
    let mut filter = spec(StageKind::Filter, name, "test/host-pass");
    filter.inputs = vec![positional(input)];
    filter.workers = workers;
    filter
}

/// A filter emitting one frame of its own from `end_of_stream`, at `workers` of them.
pub(crate) fn flushing_spec(name: &str, input: &str, workers: usize, per_chunk: bool) -> StageSpec {
    let mut flushing = spec(StageKind::Filter, name, "test/flushing");
    flushing.inputs = vec![positional(input)];
    flushing.args = Spanned::unknown(json!({ "per_chunk": per_chunk }));
    flushing.workers = Some(workers);
    flushing
}

/// A filter holding `lag[n]` frames through chunk `n`, the last entry repeating.
pub(crate) fn delay_spec(
    name: &str,
    input: &str,
    lag: &[usize],
    workers: usize,
    per_frame: bool,
) -> StageSpec {
    let mut delay = spec(StageKind::Filter, name, "test/delay");
    delay.inputs = vec![positional(input)];
    delay.args = Spanned::unknown(json!({ "lag": lag, "per_frame": per_frame }));
    delay.workers = Some(workers);
    delay
}

/// A per frame `test/delay` holding frames without declaring that it does.
pub(crate) fn undeclared_delay_spec(name: &str, input: &str, lag: &[usize], workers: usize) -> StageSpec {
    let mut delay = delay_spec(name, input, lag, workers, true);
    delay.args = Spanned::unknown(json!({ "lag": lag, "per_frame": true, "undeclared": true }));
    delay
}

/// A filter emitting two frames for every input.
pub(crate) fn double_spec(name: &str, input: &str, workers: Option<usize>) -> StageSpec {
    let mut double = spec(StageKind::Filter, name, "test/double");
    double.inputs = vec![positional(input)];
    double.workers = workers;
    double
}

/// A filter that allocates its own output rather than passing its input through.
pub(crate) fn allocating_spec(name: &str, input: &str) -> StageSpec {
    let mut filter = spec(StageKind::Filter, name, "test/allocating");
    filter.inputs = vec![positional(input)];
    filter
}

pub(crate) fn masking_spec(name: &str, input: &str) -> StageSpec {
    let mut mask = spec(StageKind::Mask, name, "test/masking");
    mask.inputs = vec![positional(input)];
    mask
}

/// A filter emitting three frames, then none, then one, cycling with each input's timestamp.
pub(crate) fn multiplying_spec(name: &str, input: &str, workers: Option<usize>) -> StageSpec {
    let mut filter = spec(StageKind::Filter, name, "test/multiplying");
    filter.inputs = vec![positional(input)];
    filter.workers = workers;
    filter
}

pub(crate) fn join_spec(name: &str, base: &str, overlay: &str) -> StageSpec {
    let mut join = spec(StageKind::Filter, name, "test/join");
    join.inputs = vec![positional(base), positional(overlay)];
    join
}

/// A stage whose chunk authority is an optional input this never binds.
pub(crate) fn referring_spec(name: &str, input: &str, workers: usize, per_chunk: bool) -> StageSpec {
    let mut referring = spec(StageKind::Filter, name, "test/referring");
    referring.inputs = vec![positional(input)];
    referring.args = Spanned::unknown(json!({ "per_chunk": per_chunk }));
    referring.workers = Some(workers);
    referring
}

pub(crate) fn slow_spec(name: &str, input: &str, workers: usize) -> StageSpec {
    let mut slow = spec(StageKind::Filter, name, "test/slow");
    slow.inputs = vec![positional(input)];
    slow.workers = Some(workers);
    slow
}

pub(crate) fn host_slow_spec(name: &str, input: &str, workers: usize) -> StageSpec {
    let mut slow = spec(StageKind::Filter, name, "test/host-slow");
    slow.inputs = vec![positional(input)];
    slow.workers = Some(workers);
    slow
}

/// A filter recording which of its instances processed each frame, at `workers` of them.
pub(crate) fn tracked_spec(name: &str, input: &str, workers: usize, per_chunk: bool) -> StageSpec {
    let mut tracked = spec(StageKind::Filter, name, "test/tracked");
    tracked.inputs = vec![positional(input)];
    tracked.args = Spanned::unknown(json!({ "per_chunk": per_chunk }));
    tracked.workers = Some(workers);
    tracked
}

/// A frame mode filter of two inputs, recording the pair each bundle handed it.
pub(crate) fn matching_spec(name: &str, base: &str, overlay: &str, workers: usize) -> StageSpec {
    let mut matching = spec(StageKind::Filter, name, "test/matching");
    matching.inputs = vec![positional(base), positional(overlay)];
    matching.workers = Some(workers);
    matching
}

pub(crate) fn failing_spec(
    name: &str,
    input: &str,
    id: usize,
    fail_at: u32,
    workers: Option<usize>,
) -> StageSpec {
    let mut failing = spec(StageKind::Filter, name, "test/failing");
    failing.inputs = vec![positional(input)];
    failing.args = Spanned::unknown(json!({ "id": id, "fail_at": fail_at }));
    failing.workers = workers;
    failing
}

/// A widened failing stage where only `failing_worker` ever fails, on its own first call.
pub(crate) fn failing_worker_spec(
    name: &str,
    input: &str,
    id: usize,
    failing_worker: usize,
    workers: usize,
) -> StageSpec {
    let mut failing = spec(StageKind::Filter, name, "test/failing");
    failing.inputs = vec![positional(input)];
    failing.args = Spanned::unknown(json!({ "id": id, "fail_at": 1, "failing_worker": failing_worker }));
    failing.workers = Some(workers);
    failing
}

/// The same as [failing_worker_spec], declared per chunk so its region routes on a scene layer.
pub(crate) fn chunked_failing_worker_spec(
    name: &str,
    input: &str,
    id: usize,
    failing_worker: usize,
    workers: usize,
) -> StageSpec {
    let mut failing = failing_worker_spec(name, input, id, failing_worker, workers);
    let args = json!({ "id": id, "fail_at": 1, "failing_worker": failing_worker, "per_chunk": true });
    failing.args = Spanned::unknown(args);
    failing
}

pub(crate) fn panicking_spec(name: &str, input: &str, workers: usize) -> StageSpec {
    let mut panicking = spec(StageKind::Filter, name, "test/panicking");
    panicking.inputs = vec![positional(input)];
    panicking.workers = Some(workers);
    panicking
}

pub(crate) fn hoarding_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/hoarding");
    sink.inputs = vec![positional(input)];
    sink
}

/// A sink reading two inputs, so one step of it consumes two items.
pub(crate) fn pairing_spec(name: &str, base: &str, overlay: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/pairing");
    sink.inputs = vec![positional(base), positional(overlay)];
    sink
}

/// A filter handing over its frame count as `frames`, or misbehaving as `mode` says.
pub(crate) fn artefacting_spec(
    name: &str,
    input: &str,
    workers: usize,
    per_chunk: bool,
    mode: &str,
) -> StageSpec {
    let mut artefacting = spec(StageKind::Filter, name, "test/artefacting");
    artefacting.inputs = vec![positional(input)];
    artefacting.args = Spanned::unknown(json!({ "mode": mode, "per_chunk": per_chunk }));
    artefacting.workers = Some(workers);
    artefacting
}

/// An artefact sink recording what it is handed, keeping only what `include` matches.
pub(crate) fn collecting_spec(name: &str, include: &[&str]) -> StageSpec {
    let mut collecting = spec(StageKind::Sink, name, "test/collecting");
    collecting.args = Spanned::unknown(json!({ "include": include }));
    collecting
}

/// A filter recording a `mark` for every frame and a `chunk` as each stream ends, at `workers` of them.
pub(crate) fn marking_spec(name: &str, input: &str, workers: usize, per_chunk: bool) -> StageSpec {
    let mut marking = spec(StageKind::Filter, name, "test/marking");
    marking.inputs = vec![positional(input)];
    marking.args = Spanned::unknown(json!({ "per_chunk": per_chunk }));
    marking.workers = Some(workers);
    marking
}

/// A metadata-only sink writing every frame and record it is handed into `FLOWS`.
pub(crate) fn flow_recording_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/flow-recording");
    sink.inputs = vec![positional(input)];
    sink
}

/// The same as [flow_recording_spec], declared frame by frame so it can share a widened region.
pub(crate) fn per_frame_flow_recording_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = flow_recording_spec(name, input);
    sink.args = Spanned::unknown(json!({ "per_frame": true }));
    sink
}

pub(crate) fn sink_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/recording");
    sink.inputs = vec![positional(input)];
    sink
}

pub(crate) fn host_sink_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/host-recording");
    sink.inputs = vec![positional(input)];
    sink
}

pub(crate) fn gated_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/gated");
    sink.inputs = vec![positional(input)];
    sink
}

pub(crate) fn meta_sink_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/meta-recording");
    sink.inputs = vec![positional(input)];
    sink
}

/// A metadata-only sink demanding tags, which the graph serves with a metadata provider.
pub(crate) fn tag_recording_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/tag-recording");
    sink.inputs = vec![positional(input)];
    sink
}

/// The same as [tag_recording_spec], with the tags provider reading host cuts from a host provider.
pub(crate) fn host_cut_tag_recording_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/tag-recording");
    sink.inputs = vec![positional(input)];
    sink.args = Spanned::unknown(json!({ "host_cuts": true }));
    sink
}
