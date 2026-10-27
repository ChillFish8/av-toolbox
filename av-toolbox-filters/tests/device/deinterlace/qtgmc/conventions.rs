use std::time::Duration;

use av_toolbox_gpu::{FieldOrder, FrameFormat, FrameMeta, LayerId};
use av_toolbox_providers::SceneChunk;
use serde_json::json;

use super::{
    Clip,
    assert_matches,
    assert_motion,
    assert_paired_outputs,
    cubic,
    deinterlaced,
    expected,
    inputs,
    interlaced,
    reference_defaults,
    run_chunks,
};
use crate::stream::{Input, Output, Stream};
use crate::support::{backend, built_filter};

fn cubic_args() -> serde_json::Value {
    json!({ "spatial": { "kind": "cubic" } })
}

/// The inputs of `clip`, numbered from `first`, in chunk 3.
fn chunk_three_inputs(clip: &Clip, format: &FrameFormat, first: u64) -> Vec<Input> {
    let mut chunk = inputs(clip, format, first);
    for input in &mut chunk {
        input.meta.provided.set(LayerId(0), SceneChunk(3));
    }

    chunk
}

fn bwdif_outputs(input: FrameFormat, output: FrameFormat, chunk: Vec<Input>) -> Vec<Output> {
    let backend = backend();
    let filter = built_filter(&backend, "deinterlace/bwdif", input, output, json!({}));
    let stream = Stream {
        filter,
        input,
        output,
        chunks: vec![chunk],
    };

    backend.build(stream)
}

fn metas(outputs: &[Output]) -> Vec<FrameMeta> {
    outputs.iter().map(|output| output.meta.clone()).collect()
}

#[test]
fn single_rate_emits_the_first_field_of_each_frame() {
    let clip = Clip::named("field_objects_8");
    let input = interlaced(&clip, FieldOrder::TopFirst);
    let output = deinterlaced(input, false);
    let chunk = chunk_three_inputs(&clip, &input, 10);
    let input_metas: Vec<FrameMeta> = chunk.iter().map(|input| input.meta.clone()).collect();

    let args = json!({ "rate": "single", "spatial": { "kind": "cubic" } });
    let mut runs = run_chunks(args, input, output, vec![chunk]);
    let run = runs.pop().expect("one chunk was run");

    assert_eq!(metas(&run.outputs), input_metas);

    let settings = reference_defaults();
    let wanted = expected(&clip, &settings, &cubic);
    assert_motion(&run.motion, &wanted, "single rate");

    let pairs = run
        .outputs
        .iter()
        .enumerate()
        .map(|(frame, output)| (2 * frame, output, &wanted.fields[2 * frame]));
    assert_paired_outputs(pairs, &clip, wanted.input, "single rate");
}

#[test]
fn fields_are_numbered_and_timed_as_bwdif_numbers_them() {
    let clip = Clip::named("field_pan_8").cut(0..4);
    let input = interlaced(&clip, FieldOrder::TopFirst);
    let output = deinterlaced(input, true);
    let chunk = chunk_three_inputs(&clip, &input, 10);

    let bwdif = bwdif_outputs(input, output, chunk.clone());
    let mut runs = run_chunks(cubic_args(), input, output, vec![chunk]);
    let run = runs.pop().expect("one chunk was run");

    let found = metas(&run.outputs);
    assert_eq!(found, metas(&bwdif));

    let indices: Vec<u64> = found.iter().map(|meta| meta.index).collect();
    let timestamps: Vec<Option<Duration>> = found.iter().map(|meta| meta.timestamp).collect();
    let expected_timestamps: Vec<Option<Duration>> = (20..28)
        .map(|field| Some(Duration::from_millis(20 * field)))
        .collect();
    assert_eq!(indices, (20..28).collect::<Vec<u64>>());
    assert_eq!(timestamps, expected_timestamps);
}

#[test]
fn an_unknown_rate_leaves_the_second_field_untimed() {
    let clip = Clip::named("field_pan_8").cut(0..2);
    let input = FrameFormat {
        frame_rate: None,
        ..interlaced(&clip, FieldOrder::TopFirst)
    };
    let output = deinterlaced(input, true);
    let chunk = inputs(&clip, &input, 0);

    let mut runs = run_chunks(cubic_args(), input, output, vec![chunk]);
    let run = runs.pop().expect("one chunk was run");

    let timestamps: Vec<Option<Duration>> = run.outputs.iter().map(|output| output.meta.timestamp).collect();
    let forty_milliseconds = Duration::from_millis(40);
    assert_eq!(
        timestamps,
        vec![Some(Duration::ZERO), None, Some(forty_milliseconds), None]
    );
}

#[test]
fn each_chunk_restarts_and_matches_its_own_reference() {
    let clip = Clip::named("field_objects_8");
    let opening = clip.cut(0..4);
    let closing = clip.cut(4..8);
    let input = interlaced(&clip, FieldOrder::TopFirst);
    let output = deinterlaced(input, true);
    let chunks = vec![inputs(&opening, &input, 0), inputs(&closing, &input, 4)];

    let runs = run_chunks(cubic_args(), input, output, chunks);
    assert_eq!(runs.len(), 2);

    let settings = reference_defaults();
    for (run, part) in runs.iter().zip([&opening, &closing]) {
        let wanted = expected(part, &settings, &cubic);
        assert_matches(run, &wanted, part, &part.name);
    }

    let first_index = runs[1].outputs[0].meta.index;
    assert_eq!(first_index, 8, "the second chunk keeps its own numbering");
}

#[test]
fn a_one_frame_chunk_yields_both_fields() {
    let clip = Clip::named("field_objects_8").cut(0..1);
    let input = interlaced(&clip, FieldOrder::TopFirst);
    let output = deinterlaced(input, true);
    let chunk = inputs(&clip, &input, 0);

    let mut runs = run_chunks(cubic_args(), input, output, vec![chunk]);
    let run = runs.pop().expect("one chunk was run");
    assert_eq!(run.outputs.len(), 2);

    let settings = reference_defaults();
    let wanted = expected(&clip, &settings, &cubic);
    assert_matches(&run, &wanted, &clip, &clip.name);
}

#[test]
fn a_stream_shorter_than_the_lag_flushes_every_field() {
    let clip = Clip::named("field_objects_8");
    let input = interlaced(&clip, FieldOrder::TopFirst);
    let output = deinterlaced(input, true);
    let parts: Vec<Clip> = (1..=3).map(|count| clip.cut(0..count)).collect();
    let chunks = parts.iter().map(|part| inputs(part, &input, 0)).collect();

    let runs = run_chunks(cubic_args(), input, output, chunks);

    let settings = reference_defaults();
    for (run, part) in runs.iter().zip(&parts) {
        assert_eq!(run.outputs.len(), part.field_count(), "{}", part.name);

        let wanted = expected(part, &settings, &cubic);
        assert_matches(run, &wanted, part, &part.name);
    }
}
