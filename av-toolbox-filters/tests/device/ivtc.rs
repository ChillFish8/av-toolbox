use av_toolbox_cubek::deinterlace::BwdifField;
use av_toolbox_cubek::testing::{
    BwdifReferencePlanes,
    SyntheticClip,
    VfmSettings,
    reference_bwdif,
    reference_match_weave,
    reference_vfm_metrics,
    synthetic_clips,
    synthetic_film_frame,
    vfm_decisions,
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
const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
const FILM_FRAMES: usize = 10;
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
