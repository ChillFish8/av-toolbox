use std::time::Duration;

use av_toolbox_cubek::deinterlace::BwdifField;
use av_toolbox_cubek::testing::{
    BwdifReferencePlanes,
    DecimateDropFixture,
    SyntheticClip,
    VfmSettings,
    decimate_drop_fixtures,
    reference_bwdif,
    reference_match_weave,
    reference_vfm_metrics,
    synthetic_clips,
    synthetic_film_frame,
    vfm_decisions,
    vfm_matched_frames,
};
use av_toolbox_filters::testing::VfmDecision;
use av_toolbox_gpu::{
    ChromaLocation,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
    MetaValue,
    SampleFormat,
    Subsampling,
};
use serde_json::json;

use crate::stream::{Input, Output, Stream};
use crate::support::{
    assert_within_one,
    backend,
    built_filter,
    element_of,
    frame,
    from_bytes,
    ten_bit,
    to_bytes,
};

const STAGE: &str = "ivtc/match";
const DECIMATE: &str = "ivtc/decimate";
const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
const FILM_FRAMES: usize = 10;
/// The video frames of one 3:2 cadence cycle.
const CYCLE: usize = 5;
const MATCH_NAMES: [&str; 5] = ["p", "c", "n", "b", "u"];

fn clip_named(name: &str) -> SyntheticClip {
    synthetic_clips()
        .into_iter()
        .find(|clip| clip.name == name)
        .unwrap_or_else(|| panic!("there is no synthetic clip {name}"))
}

/// The format of `clip`'s frames, 4:2:0 unless `grey`, at its bit depth and field order.
fn clip_format(clip: &SyntheticClip, grey: bool) -> FrameFormat {
    let layout = if grey {
        FrameLayout::Grey
    } else {
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        }
    };
    let sample = if clip.bits == 8 {
        SampleFormat::U8
    } else {
        ten_bit()
    };
    let field_order = if clip.order == 1 {
        FieldOrder::TopFirst
    } else {
        FieldOrder::BottomFirst
    };

    FrameFormat {
        frame_rate: FrameRate::new(30000, 1001),
        field_order: Some(field_order),
        ..frame(WIDTH, HEIGHT, layout, sample, ChromaLocation::Left)
    }
}

fn progressive(input: FrameFormat) -> FrameFormat {
    FrameFormat {
        field_order: Some(FieldOrder::Progressive),
        ..input
    }
}

/// Code values as the bytes of a plane of `format`.
fn encoded_plane(format: &FrameFormat, values: &[i32]) -> Vec<u8> {
    let element = element_of(format.sample);
    let samples: Vec<u16> = values.iter().map(|value| *value as u16).collect();

    to_bytes(&samples, element)
}

fn decoded_plane(format: &FrameFormat, bytes: &[u8]) -> Vec<i32> {
    let element = element_of(format.sample);
    let samples = from_bytes(bytes, element);

    samples.into_iter().map(i32::from).collect()
}

/// Every frame of `clip` as inputs of `format`, which keeps luma alone when grey.
fn clip_inputs(clip: &SyntheticClip, format: &FrameFormat) -> Vec<Input> {
    let plane_count = format.channels().len();

    clip.frames
        .iter()
        .enumerate()
        .map(|(index, planes)| Input {
            meta: FrameMeta::at(index as u64),
            planes: planes[..plane_count]
                .iter()
                .map(|plane| encoded_plane(format, plane))
                .collect(),
        })
        .collect()
}

fn run(args: serde_json::Value, input: FrameFormat, chunks: Vec<Vec<Input>>) -> Vec<Output> {
    let backend = backend();
    let output = progressive(input);
    let filter = built_filter(&backend, STAGE, input, output, args);
    let stream = Stream {
        filter,
        input,
        output,
        chunks,
    };

    backend.build(stream)
}

/// The value `ivtc/match` attached to `output` under `key`.
fn attached(output: &Output, key: &str) -> MetaValue {
    let mut attachments = output.meta.attachments.clone();
    attachments.stamp(STAGE);

    let stage_values = attachments.stages().find(|(path, _)| *path == STAGE);
    let Some((_, values)) = stage_values else {
        panic!("frame {} carries nothing from {STAGE}", output.meta.index);
    };

    let value = values.iter().find(|(found, _)| found == key);
    let Some((_, value)) = value else {
        panic!("frame {} carries no {key}", output.meta.index);
    };

    value.clone()
}

/// The match, mic, combed flag and scene change flag `output` carries.
fn decision_of(output: &Output) -> (MetaValue, MetaValue, MetaValue, MetaValue) {
    let matched = attached(output, "match");
    let mic = attached(output, "mic");
    let combed = attached(output, "combed");
    let scene_change = attached(output, "scene_change");

    (matched, mic, combed, scene_change)
}

fn match_value(matched: usize) -> MetaValue {
    let name = MATCH_NAMES[matched].to_owned();

    MetaValue::String(name)
}

/// The default decision for `clip`, with VFM's field against the clip's order when `opposite`.
fn default_decision(clip: &SyntheticClip, opposite: bool) -> VfmDecision {
    VfmDecision {
        mode: 1,
        micmatch: 1,
        mi: 80,
        shift: clip.bits - 8,
        opposite,
        scthresh: 12.0 / 100.0,
    }
}

/// VFM's resolved field for `clip`, its order unless `opposite`.
fn resolved_field(clip: &SyntheticClip, opposite: bool) -> usize {
    let order = clip.order as usize;
    if opposite { 1 - order } else { order }
}

/// The match and its mic the CPU pipeline picks for frame `frame` with the default arguments.
fn reference_choice(clip: &SyntheticClip, frame: usize, with_chroma: bool) -> (usize, u32) {
    let decision = default_decision(clip, false);
    let settings = VfmSettings {
        field: resolved_field(clip, false),
        pairs: decision.pairs(),
        exclude_from: 16,
        exclude_to: 16,
        cthresh: 9,
        chroma: with_chroma,
        mchroma: with_chroma,
    };

    let metrics = reference_vfm_metrics(clip, frame, &settings);
    let scene_change = decision.scene_change(metrics.current_scene, metrics.next_scene);
    let matched = decision.choose(metrics.sums, metrics.mics, scene_change);

    (matched, metrics.mics[matched])
}

/// Plane `plane_index` of `matched`'s weave of frame `frame` of `clip`, with VFM's `field`.
fn reference_plane(
    clip: &SyntheticClip,
    frame: usize,
    plane_index: usize,
    matched: usize,
    field: usize,
) -> Vec<i32> {
    let last = clip.frames.len() - 1;
    let previous = frame.saturating_sub(1);
    let next = usize::min(frame + 1, last);
    let frames = [
        clip.frames[previous][plane_index].as_slice(),
        clip.frames[frame][plane_index].as_slice(),
        clip.frames[next][plane_index].as_slice(),
    ];
    let shift = u32::from(plane_index > 0);
    let width = (WIDTH >> shift) as usize;
    let height = (HEIGHT >> shift) as usize;

    reference_match_weave(frames, width, height, matched, field)
}

fn assert_matches_the_reference(
    clip: &SyntheticClip,
    format: &FrameFormat,
    outputs: &[Output],
    with_chroma: bool,
) {
    assert_eq!(outputs.len(), clip.frames.len(), "{}", clip.name);

    let field = resolved_field(clip, false);
    for (frame, output) in outputs.iter().enumerate() {
        let (matched, _) = reference_choice(clip, frame, with_chroma);
        let label = format!("{} frame {frame}", clip.name);

        assert_eq!(attached(output, "match"), match_value(matched), "{label}");

        for (plane_index, bytes) in output.planes.iter().enumerate() {
            let found = decoded_plane(format, bytes);
            let expected = reference_plane(clip, frame, plane_index, matched, field);

            assert_eq!(found, expected, "{label} plane {plane_index}");
        }
    }
}

/// How many different frames `outputs` holds.
fn distinct_frames(format: &FrameFormat, outputs: &[Output]) -> usize {
    let mut distinct: Vec<Vec<Vec<i32>>> = Vec::new();
    for output in outputs {
        let planes: Vec<Vec<i32>> = output
            .planes
            .iter()
            .map(|bytes| decoded_plane(format, bytes))
            .collect();
        if !distinct.contains(&planes) {
            distinct.push(planes);
        }
    }

    distinct.len()
}

#[test]
fn a_telecined_clip_matches_back_to_its_film_frames() {
    for name in ["tff_8", "bff_8"] {
        let clip = clip_named(name);
        let format = clip_format(&clip, false);
        let inputs = clip_inputs(&clip, &format);

        let outputs = run(json!({}), format, vec![inputs]);

        assert_matches_the_reference(&clip, &format, &outputs, true);
        for output in &outputs {
            assert_eq!(attached(output, "combed"), MetaValue::Bool(false), "{name}");
        }

        for film in 0..FILM_FRAMES {
            let film_frame = synthetic_film_frame(name, film);
            let shown = outputs.iter().any(|output| {
                output
                    .planes
                    .iter()
                    .zip(&film_frame)
                    .all(|(bytes, values)| decoded_plane(&format, bytes) == *values)
            });

            assert!(shown, "{name} film frame {film} never comes out");
        }

        let (cycles, _) = outputs.as_chunks::<CYCLE>();
        for (cycle, outputs) in cycles.iter().enumerate() {
            let shown = distinct_frames(&format, outputs);

            assert_eq!(
                shown,
                CYCLE - 1,
                "{name} cycle {cycle} shows {shown} distinct frames"
            );
        }
    }
}

/// Checks every output of `clip` through the stage with the default arguments carries what `vfmProcess`
/// decided for that frame.
fn assert_agrees_with_the_plugin(name: &str) {
    let clip = clip_named(name);
    let format = clip_format(&clip, false);
    let inputs = clip_inputs(&clip, &format);

    let outputs = run(json!({}), format, vec![inputs]);

    let fixtures: Vec<_> = vfm_decisions()
        .into_iter()
        .filter(|fixture| {
            fixture.clip == name && fixture.mode == 1 && fixture.micmatch == 1 && !fixture.opposite
        })
        .collect();
    assert_eq!(fixtures.len(), outputs.len(), "{name}");

    for fixture in fixtures {
        let output = &outputs[fixture.frame];
        let mic = fixture.mics[fixture.matched];
        let expected = (
            match_value(fixture.matched),
            MetaValue::Int(i64::from(mic)),
            MetaValue::Bool(fixture.combed),
            MetaValue::Bool(fixture.scene_change),
        );

        assert_eq!(decision_of(output), expected, "{name} frame {}", fixture.frame);
    }
}

#[test]
fn the_plugin_and_the_stage_agree() {
    for name in ["tff_8", "bff_8", "cut_8", "orphan_8"] {
        assert_agrees_with_the_plugin(name);
    }
}

#[test]
fn a_ten_bit_clip_agrees_with_the_plugin() {
    for name in ["tff_10", "orphan_10"] {
        assert_agrees_with_the_plugin(name);
    }
}

#[test]
fn a_broken_cadence_is_rebuilt_by_bwdif() {
    let clip = clip_named("orphan_8");
    let format = clip_format(&clip, false);
    let inputs = clip_inputs(&clip, &format);

    let outputs = run(json!({}), format, vec![inputs]);

    let output = &outputs[7];
    assert_eq!(attached(output, "combed"), MetaValue::Bool(true));

    let field = BwdifField {
        rebuilt: resolved_field(&clip, false) as u32,
        first: true,
    };
    for (plane_index, bytes) in output.planes.iter().enumerate() {
        let widened = |frame: usize| -> Vec<f32> {
            clip.frames[frame][plane_index]
                .iter()
                .map(|value| *value as f32)
                .collect()
        };
        let previous = widened(6);
        let current = widened(7);
        let next = widened(8);
        let planes = BwdifReferencePlanes {
            previous: &previous,
            current: &current,
            next: &next,
            spatial: None,
        };
        let shift = u32::from(plane_index > 0);

        let values = reference_bwdif(&planes, WIDTH >> shift, HEIGHT >> shift, field);
        let expected: Vec<u8> = values
            .iter()
            .map(|value| value.round().clamp(0.0, 255.0) as u8)
            .collect();

        assert_within_one(bytes, &expected, &format!("plane {plane_index}"));
    }
}

#[test]
fn the_fallback_keeps_the_matching_field() {
    let clip = clip_named("tff_8");
    let format = clip_format(&clip, false);
    let inputs = clip_inputs(&clip, &format);

    let outputs = run(
        json!({ "field": "opposite", "mi": 0 }),
        format,
        vec![inputs.clone()],
    );

    assert_eq!(outputs.len(), inputs.len());
    for (output, input) in outputs.iter().zip(&inputs) {
        let frame = output.meta.index;
        assert_eq!(attached(output, "combed"), MetaValue::Bool(true), "frame {frame}");

        for (plane_index, (found, given)) in output.planes.iter().zip(&input.planes).enumerate() {
            let width = (WIDTH >> u32::from(plane_index > 0)) as usize;
            let found_rows = found.chunks(width).skip(1).step_by(2);
            let given_rows = given.chunks(width).skip(1).step_by(2);
            for (row, (found_row, given_row)) in found_rows.zip(given_rows).enumerate() {
                let label = format!("frame {frame} plane {plane_index} odd row {row}");

                assert_eq!(found_row, given_row, "{label}");
            }
        }
    }
}

#[test]
fn without_a_fallback_the_combed_weave_comes_out() {
    let clip = clip_named("orphan_8");
    let format = clip_format(&clip, false);
    let inputs = clip_inputs(&clip, &format);

    let outputs = run(json!({ "fallback": { "kind": "none" } }), format, vec![inputs]);

    assert_eq!(attached(&outputs[7], "combed"), MetaValue::Bool(true));
    assert_matches_the_reference(&clip, &format, &outputs, true);
}

#[test]
fn each_chunk_is_matched_on_its_own() {
    let clip = clip_named("tff_8");
    let format = clip_format(&clip, false);
    let mut inputs = clip_inputs(&clip, &format);
    for input in &mut inputs[5..] {
        input.meta.chunk = 1;
    }

    let second_chunk = inputs.split_off(5);

    let split = run(json!({}), format, vec![inputs, second_chunk.clone()]);
    let alone = run(json!({}), format, vec![second_chunk]);

    assert_eq!(split.len(), 12);
    assert_eq!(split[5..], alone[..]);
}

#[test]
fn a_grey_stream_matches_on_luma_alone() {
    let clip = clip_named("tff_8");
    let format = clip_format(&clip, true);
    let inputs = clip_inputs(&clip, &format);

    let outputs = run(json!({}), format, vec![inputs]);

    assert_matches_the_reference(&clip, &format, &outputs, false);
}

/// The time of frame `k` at `numerator / denominator` frames a second, truncated to whole nanoseconds.
fn frame_time(k: u64, numerator: u32, denominator: u32) -> Duration {
    let nanos = u128::from(k) * u128::from(denominator) * 1_000_000_000 / u128::from(numerator);
    let nanos = u64::try_from(nanos).expect("the time fits in nanoseconds");

    Duration::from_nanos(nanos)
}

/// `format` with one frame in five dropped, so 30000/1001 becomes 24000/1001.
fn decimated(format: FrameFormat) -> FrameFormat {
    let frame_rate = format
        .frame_rate
        .and_then(|rate| FrameRate::new(rate.numerator() * 4, rate.denominator() * 5));

    FrameFormat { frame_rate, ..format }
}

fn run_decimate(args: serde_json::Value, input: FrameFormat, inputs: Vec<Input>) -> Vec<Output> {
    let backend = backend();
    let output = decimated(input);
    let filter = built_filter(&backend, DECIMATE, input, output, args);
    let stream = Stream {
        filter,
        input,
        output,
        chunks: vec![inputs],
    };

    backend.build(stream)
}

/// The outputs of one stage as the inputs of the next.
fn chained(outputs: Vec<Output>) -> Vec<Input> {
    outputs
        .into_iter()
        .map(|output| Input {
            meta: output.meta,
            planes: output.planes,
        })
        .collect()
}

/// Each of `frames` as an input of `format`, frame `k` stamped at `timestamp(k)`.
fn timed_inputs(
    format: &FrameFormat,
    frames: &[[Vec<i32>; 3]],
    timestamp: impl Fn(u64) -> Duration,
) -> Vec<Input> {
    let plane_count = format.channels().len();

    frames
        .iter()
        .enumerate()
        .map(|(index, planes)| {
            let index = index as u64;
            let meta = FrameMeta {
                timestamp: Some(timestamp(index)),
                ..FrameMeta::at(index)
            };

            Input {
                meta,
                planes: planes[..plane_count]
                    .iter()
                    .map(|plane| encoded_plane(format, plane))
                    .collect(),
            }
        })
        .collect()
}

/// Every input frame `ivtc/decimate` emits for `clip`, in order, by the harness's cycle choices.
fn harness_outputs(fixtures: &[&DecimateDropFixture]) -> Vec<usize> {
    let mut cycles = fixtures.to_vec();
    cycles.sort_by_key(|fixture| fixture.cycle);

    cycles
        .iter()
        .flat_map(|fixture| fixture.outputs.iter().copied())
        .collect()
}

/// Whether `first` and `second` are drop fixtures of the same clip, source and settings.
fn same_run(first: &DecimateDropFixture, second: &DecimateDropFixture) -> bool {
    first.clip == second.clip
        && first.matched == second.matched
        && first.block_x == second.block_x
        && first.block_y == second.block_y
        && first.chroma == second.chroma
        && first.dupthresh == second.dupthresh
        && first.scthresh == second.scthresh
}

/// The harness's cycle choices for `clip` under VDecimate's defaults, with or without chroma.
fn default_drops(fixtures: &[DecimateDropFixture], clip: &str, chroma: bool) -> Vec<usize> {
    let defaults: Vec<&DecimateDropFixture> = fixtures
        .iter()
        .filter(|fixture| fixture.clip == clip && fixture.matched && fixture.chroma == chroma)
        .filter(|fixture| fixture.block_x == 32 && fixture.block_y == 32)
        .filter(|fixture| fixture.dupthresh == 1.1 && fixture.scthresh == 15.0)
        .collect();
    assert!(
        !defaults.is_empty(),
        "the harness has no default drops for {clip}"
    );

    harness_outputs(&defaults)
}

/// Which input each of `outputs` is, from its timestamp of whole seconds.
fn source_frames(outputs: &[Output]) -> Vec<usize> {
    outputs
        .iter()
        .map(|output| {
            let timestamp = output.meta.timestamp.expect("the output keeps its timestamp");
            timestamp.as_secs() as usize
        })
        .collect()
}

fn assert_shows(format: &FrameFormat, output: &Output, expected: &[Vec<i32>; 3], label: &str) {
    for (plane_index, bytes) in output.planes.iter().enumerate() {
        let found = decoded_plane(format, bytes);

        assert_eq!(found, expected[plane_index], "{label} plane {plane_index}");
    }
}

/// Runs the matched frames of `clip` through `ivtc/decimate` with `args`, each stamped with its own index
/// in seconds at an unknown rate, so an output's timestamp names its input.
fn decimate_matched(clip: &SyntheticClip, grey: bool, args: serde_json::Value) -> (FrameFormat, Vec<Output>) {
    let format = FrameFormat {
        frame_rate: None,
        ..progressive(clip_format(clip, grey))
    };
    let matched_frames = vfm_matched_frames(clip);
    let inputs = timed_inputs(&format, &matched_frames, Duration::from_secs);

    let outputs = run_decimate(args, format, inputs);

    (format, outputs)
}

#[test]
fn a_matched_clip_decimates_to_its_film_frames() {
    let fixtures = decimate_drop_fixtures();
    let first = Duration::from_secs(3);

    for name in ["tff_8", "bff_8"] {
        let clip = clip_named(name);
        let format = clip_format(&clip, false);
        let frames = &clip.frames;
        let inputs = timed_inputs(&format, frames, |k| first + frame_time(k, 30000, 1001));

        let matched = run(json!({}), format, vec![inputs]);
        let tail = matched[10..].to_vec();
        let progressive_format = progressive(format);
        let outputs = run_decimate(json!({}), progressive_format, chained(matched));

        assert_eq!(
            decimated(progressive_format).frame_rate,
            FrameRate::new(24000, 1001)
        );
        assert_eq!(outputs.len(), 9, "{name}");

        for (film, output) in outputs[..8].iter().enumerate() {
            let film_frame = synthetic_film_frame(name, film);
            assert_shows(&format, output, &film_frame, &format!("{name} film frame {film}"));
        }

        let expected_drops = default_drops(&fixtures, name, true);
        let kept_tail = expected_drops[8] - 10;
        assert_eq!(outputs[8].planes, tail[kept_tail].planes, "{name} tail");

        for (k, output) in outputs.iter().enumerate() {
            let expected = first + frame_time(k as u64, 24000, 1001);

            assert_eq!(output.meta.timestamp, Some(expected), "{name} output {k}");
        }
    }
}

#[test]
fn the_device_drops_the_harness_drops() {
    let fixtures = decimate_drop_fixtures();

    let mut checked = 0;
    for clip in synthetic_clips() {
        let raw_format = FrameFormat {
            frame_rate: None,
            ..clip_format(&clip, false)
        };
        let raw_inputs = timed_inputs(&raw_format, &clip.frames, Duration::from_secs);
        let matched_frames = vfm_matched_frames(&clip);

        let heads = fixtures
            .iter()
            .filter(|fixture| fixture.clip == clip.name && fixture.cycle == 0);
        for head in heads {
            let run_fixtures: Vec<&DecimateDropFixture> = fixtures
                .iter()
                .filter(|fixture| same_run(fixture, head))
                .collect();
            let expected = harness_outputs(&run_fixtures);
            let args = json!({
                "chroma": head.chroma,
                "dupthresh": head.dupthresh,
                "scthresh": head.scthresh,
                "blockx": head.block_x,
                "blocky": head.block_y,
            });
            let label = format!(
                "{} matched {} blocks {}x{} chroma {} dupthresh {} scthresh {}",
                clip.name,
                head.matched,
                head.block_x,
                head.block_y,
                head.chroma,
                head.dupthresh,
                head.scthresh,
            );

            let (format, outputs) = if head.matched {
                decimate_matched(&clip, false, args)
            } else {
                let outputs = run_decimate(args, raw_format, raw_inputs.clone());
                (raw_format, outputs)
            };

            let sources = if head.matched {
                &matched_frames
            } else {
                &clip.frames
            };

            assert_eq!(source_frames(&outputs), expected, "{label}");
            for (output, source) in outputs.iter().zip(&expected) {
                assert_shows(&format, output, &sources[*source], &label);
            }

            checked += run_fixtures.len();
        }
    }

    assert_eq!(checked, fixtures.len());
}

#[test]
fn a_short_last_cycle_drops_one_frame() {
    for name in ["tff_8", "bff_8"] {
        let clip = clip_named(name);

        let (_, outputs) = decimate_matched(&clip, false, json!({}));

        assert_eq!(outputs.len(), 4 + 4 + 1, "{name}");
    }
}

#[test]
fn a_one_frame_tail_is_dropped() {
    let mut clip = clip_named("tff_8");
    clip.frames.truncate(11);

    let (_, outputs) = decimate_matched(&clip, false, json!({}));

    assert_eq!(outputs.len(), 4 + 4);
    assert!(source_frames(&outputs).iter().all(|source| *source < 10));
}

#[test]
fn an_unknown_rate_keeps_input_timestamps() {
    let clip = clip_named("tff_8");
    let format = FrameFormat {
        frame_rate: None,
        ..progressive(clip_format(&clip, false))
    };
    let matched_frames = vfm_matched_frames(&clip);
    let timestamp = |k: u64| Duration::from_millis(500 + 7 * k * k);
    let inputs = timed_inputs(&format, &matched_frames, timestamp);

    let outputs = run_decimate(json!({}), format, inputs);

    let fixtures = decimate_drop_fixtures();
    let expected = default_drops(&fixtures, "tff_8", true);
    assert_eq!(outputs.len(), expected.len());
    for (output, source) in outputs.iter().zip(expected) {
        let expected_timestamp = timestamp(source as u64);

        assert_eq!(output.meta.timestamp, Some(expected_timestamp), "input {source}");
    }
}

#[test]
fn a_ten_bit_clip_decimates_like_the_harness() {
    let clip = clip_named("tff_10");
    let format = clip_format(&clip, false);
    let inputs = timed_inputs(&format, &clip.frames, |k| frame_time(k, 30000, 1001));

    let matched = run(json!({}), format, vec![inputs]);
    let outputs = run_decimate(json!({}), progressive(format), chained(matched));

    let fixtures = decimate_drop_fixtures();
    let expected = default_drops(&fixtures, "tff_10", true);
    let matched_frames = vfm_matched_frames(&clip);
    assert_eq!(outputs.len(), expected.len());
    for (k, (output, source)) in outputs.iter().zip(expected).enumerate() {
        let label = format!("output {k} from input {source}");
        let expected_timestamp = frame_time(k as u64, 24000, 1001);

        assert_shows(&format, output, &matched_frames[source], &label);
        assert_eq!(output.meta.timestamp, Some(expected_timestamp), "{label}");
    }
}

#[test]
fn a_grey_stream_counts_luma_alone() {
    let clip = clip_named("tff_8");

    let (format, outputs) = decimate_matched(&clip, true, json!({}));

    let fixtures = decimate_drop_fixtures();
    let expected = default_drops(&fixtures, "tff_8", false);
    let matched_frames = vfm_matched_frames(&clip);
    assert_eq!(source_frames(&outputs), expected);
    for (output, source) in outputs.iter().zip(&expected) {
        assert_shows(
            &format,
            output,
            &matched_frames[*source],
            &format!("input {source}"),
        );
    }
}
