use std::time::Duration;

use av_toolbox_cubek::motion::{BlockGrid, MotionSettings};
use av_toolbox_cubek::testing::{
    QtgmcReferenceSettings,
    ReferenceInput,
    badly_clips,
    reference_scene_changed,
    reference_shimmer,
};
use av_toolbox_gpu::{FieldOrder, FrameFormat, FrameMeta, Subsampling};
use serde_json::json;

use super::formats::{at_16_bit, at_422};
use super::{
    ChunkRun,
    Clip,
    Closeness,
    Eedi3Reference,
    Expected,
    assert_matches,
    cubic,
    deinterlaced,
    expected,
    inputs,
    interlaced,
    reference_defaults,
    run_chunks,
};
use crate::support::{element_of, from_bytes};

/// Every progressive mode, with the reference's input for it.
const MODES: [(&str, ReferenceInput); 3] = [
    ("progressive", ReferenceInput::Progressive),
    ("repair", ReferenceInput::Repair),
    ("repair_reversed", ReferenceInput::RepairReversed),
];

impl Clip {
    /// The badly deinterlaced clip called `name`.
    fn badly(name: &str) -> Self {
        let found = badly_clips()
            .into_iter()
            .find(|clip| clip.name == name)
            .expect("the badly deinterlaced clip exists");

        Self {
            name: name.to_string(),
            bits: found.bits,
            width: found.width,
            height: found.height,
            subsampling: Subsampling::Yuv420,
            frames: found.frames,
        }
    }
}

/// The 25 frames per second format of `clip`, flagged progressive.
fn progressive_format(clip: &Clip) -> FrameFormat {
    let base = interlaced(clip, FieldOrder::TopFirst);

    FrameFormat {
        field_order: Some(FieldOrder::Progressive),
        ..base
    }
}

/// The arguments of `mode` with the cubic interpolation, top field first for the repair modes.
fn mode_args(mode: &str) -> serde_json::Value {
    match mode {
        "progressive" => json!({ "input": mode }),
        _ => json!({ "input": mode, "order": "top", "spatial": { "kind": "cubic" } }),
    }
}

fn mode_settings(input: ReferenceInput) -> QtgmcReferenceSettings {
    QtgmcReferenceSettings {
        input,
        ..reference_defaults()
    }
}

/// Runs each of `parts` as its own chunk of progressive frames, numbered from `first`.
fn run_parts(args: serde_json::Value, parts: &[&Clip], first: u64) -> Vec<ChunkRun> {
    let input = progressive_format(parts[0]);
    let output = deinterlaced(input, false);
    let chunks = parts.iter().map(|part| inputs(part, &input, first)).collect();

    run_chunks(args, input, output, chunks)
}

fn run_progressive(args: serde_json::Value, clip: &Clip) -> ChunkRun {
    let mut runs = run_parts(args, &[clip], 0);

    runs.pop().expect("one chunk was run")
}

/// Asserts the repair mask flags the stage recorded equal the reference's.
///
/// Each entry with a next entry has one, flagged by MVTools' default rule on its search to that entry.
fn assert_repair_flags(run: &ChunkRun, wanted: &Expected, clip: &Clip, label: &str) {
    let grid = BlockGrid::new(&MotionSettings::default(), clip.width as u32, clip.height as u32);
    let rule = QtgmcReferenceSettings::MASK_SCENE_CHANGE;

    let mut expected_flags = Vec::new();
    for (field, searches) in wanted.motion.iter().enumerate() {
        let Some(next) = searches.iter().find(|search| search.delta == 1) else {
            continue;
        };

        let flagged = reference_scene_changed(&next.field.sads, &grid, rule);
        expected_flags.push((field, flagged));
    }

    let found: Vec<(usize, bool)> = run
        .repairs
        .iter()
        .map(|repair| (repair.field, repair.flagged))
        .collect();
    assert_eq!(found, expected_flags, "{label}: the repair mask flags differ");
}

/// Asserts `run` of `clip` in the mode of `settings` matches the reference, with its repair mask flags.
fn assert_mode_matches(
    run: &ChunkRun,
    clip: &Clip,
    settings: &QtgmcReferenceSettings,
    label: &str,
) -> Closeness {
    let wanted = expected(clip, settings, &cubic);
    let closeness = assert_matches(run, &wanted, clip, label);

    if settings.input == ReferenceInput::Progressive {
        assert!(
            run.repairs.is_empty(),
            "{label}: progressive input repairs nothing"
        );
        return closeness;
    }

    assert_repair_flags(run, &wanted, clip, label);
    closeness
}

/// Runs `clip` in `mode` and asserts it matches the reference.
fn assert_clip_matches(clip: &Clip, mode: &str, input: ReferenceInput) -> Closeness {
    let run = run_progressive(mode_args(mode), clip);
    let label = format!("{mode} {}", clip.name);
    let settings = mode_settings(input);

    assert_mode_matches(&run, clip, &settings, &label)
}

fn metas(outputs: &[super::Output]) -> Vec<FrameMeta> {
    outputs.iter().map(|output| output.meta.clone()).collect()
}

#[test]
fn progressive_matches_the_reference() {
    let clips = [
        Clip::badly("badly_pan_8"),
        Clip::badly("badly_objects_8"),
        Clip::named("field_noisy_10"),
    ];

    for clip in &clips {
        assert_clip_matches(clip, "progressive", ReferenceInput::Progressive);
    }
}

// The chroma keeps its 32 rows, which an odd luma height of 63 still has at 4:2:0.
#[test]
fn progressive_matches_the_reference_at_an_odd_height() {
    let clip = Clip::badly("badly_objects_8");
    let height = clip.height - 1;
    let frames = clip
        .frames
        .iter()
        .map(|planes| {
            let luma = planes[0][..clip.width * height].to_vec();
            [luma, planes[1].clone(), planes[2].clone()]
        })
        .collect();
    let odd = Clip {
        name: format!("{} at {height} rows", clip.name),
        height,
        frames,
        ..clip.clone()
    };

    assert_clip_matches(&odd, "progressive", ReferenceInput::Progressive);
}

#[test]
fn repair_matches_the_reference() {
    let clips = [Clip::badly("badly_pan_8"), Clip::badly("badly_objects_8")];

    for clip in &clips {
        assert_clip_matches(clip, "repair", ReferenceInput::Repair);
    }
}

#[test]
fn repair_reversed_matches_the_reference() {
    let clips = [
        Clip::badly("badly_pan_8_reversed"),
        Clip::badly("badly_static_8_reversed"),
    ];

    for clip in &clips {
        assert_clip_matches(clip, "repair_reversed", ReferenceInput::RepairReversed);
    }
}

// Bottom field first keeps the other parity of every frame, as reversing the dominance does.
#[test]
fn bottom_field_first_repair_matches_the_reversed_reference() {
    let clip = Clip::badly("badly_pan_8_reversed");
    let args = json!({ "input": "repair", "order": "bottom", "spatial": { "kind": "cubic" } });
    let run = run_progressive(args, &clip);

    let settings = mode_settings(ReferenceInput::RepairReversed);
    assert_mode_matches(&run, &clip, &settings, "repair bottom first");
}

#[test]
fn every_mode_matches_the_reference_at_every_radius_extreme() {
    let clip = Clip::badly("badly_objects_8");
    let radii = [
        ("tr0", 0),
        ("tr0", 2),
        ("tr1", 0),
        ("tr1", 2),
        ("tr2", 0),
        ("tr2", 3),
    ];

    for (mode, input) in MODES {
        let mut worst = 0.0f64;
        for (name, radius) in radii {
            let mut args = mode_args(mode);
            args[name] = json!(radius);
            let run = run_progressive(args, &clip);

            let mut settings = mode_settings(input);
            match name {
                "tr0" => settings.tr0 = radius,
                "tr1" => settings.tr1 = radius,
                _ => settings.tr2 = radius,
            }

            let label = format!("{mode} {name} {radius}");
            let closeness = assert_mode_matches(&run, &clip, &settings, &label);
            worst = worst.max(closeness.share);
        }

        println!(
            "{mode}: the worst share over the radius extremes is {:.3} %",
            worst * 100.0
        );
    }
}

#[test]
fn repair_matches_the_reference_at_422_10_bit_and_16_bit() {
    let clip = Clip::named("field_noisy_10");
    let subsampled = at_422(&clip);
    let lifted = at_16_bit(&clip);

    for formatted in [&subsampled, &lifted] {
        assert_clip_matches(formatted, "repair", ReferenceInput::Repair);
    }
}

#[test]
fn repair_s_default_eedi3_path_matches_the_reference() {
    let clip = Clip::badly("badly_objects_8").cut(0..3);
    let run = run_progressive(json!({ "input": "repair", "order": "top" }), &clip);

    let eedi3 = Eedi3Reference::new(clip.bits);
    let interpolate = |plane: &[f32], width, height, parity| eedi3.interpolate(plane, width, height, parity);
    let settings = mode_settings(ReferenceInput::Repair);
    let wanted = expected(&clip, &settings, &interpolate);
    assert_matches(&run, &wanted, &clip, "repair eedi3");
    assert_repair_flags(&run, &wanted, &clip, "repair eedi3");
}

#[test]
fn each_chunk_restarts_and_matches_its_own_reference() {
    let clip = Clip::badly("badly_objects_8");
    let opening = clip.cut(0..3);
    let closing = clip.cut(3..8);
    let input = progressive_format(&clip);
    let output = deinterlaced(input, false);
    let chunks = vec![inputs(&opening, &input, 0), inputs(&closing, &input, 3)];

    let runs = run_chunks(mode_args("repair"), input, output, chunks);
    assert_eq!(runs.len(), 2);

    for (run, (part, first_frame)) in runs.iter().zip([(&opening, 0), (&closing, 3)]) {
        let settings = QtgmcReferenceSettings {
            first_frame,
            ..mode_settings(ReferenceInput::Repair)
        };
        assert_mode_matches(run, part, &settings, &part.name);
    }

    let first_index = runs[1].outputs[0].meta.index;
    assert_eq!(first_index, 3, "the second chunk keeps its own numbering");
}

// With every radius and the sharpening off, an output is Rep2 of its `edi` against itself. Woven frames
// comb where they move, so their rebuild differs from them.
#[test]
fn the_last_repaired_frame_of_a_chunk_is_the_original_luma() {
    let clip = Clip::named("field_pan_8").cut(0..3);
    let mut args = mode_args("repair");
    args["tr0"] = json!(0);
    args["tr1"] = json!(0);
    args["tr2"] = json!(0);
    args["sharpness"] = json!(0.0);
    let run = run_progressive(args, &clip);
    assert_eq!(run.outputs.len(), 3);

    let repaired: Vec<usize> = run.repairs.iter().map(|repair| repair.field).collect();
    assert_eq!(repaired, [0, 1], "only frames with a next frame build a mask");

    let format = progressive_format(&clip);
    let element = element_of(format.sample);
    let luma: Vec<Vec<f32>> = clip
        .frames
        .iter()
        .map(|planes| planes[0].iter().map(|sample| *sample as f32).collect())
        .collect();
    let kept: Vec<Vec<u16>> = luma
        .iter()
        .map(|plane| {
            let shimmered = reference_shimmer(plane, plane, clip.width, clip.height, clip.bits);
            shimmered.iter().map(|sample| sample.round() as u16).collect()
        })
        .collect();

    let last = from_bytes(&run.outputs[2].planes[0], element);
    assert_eq!(last, kept[2], "the last frame keeps its own luma");

    let first = from_bytes(&run.outputs[0].planes[0], element);
    assert_ne!(first, kept[0], "an earlier frame rebuilds where it moves");
}

// With every radius and the sharpening off, a frame's output reads only itself and the next frame, so it
// is the same in the chunk as in the whole clip once both keep the same field.
#[test]
fn parity_follows_the_stream_index_across_chunks() {
    let clip = Clip::named("field_pan_8");
    let closing = clip.cut(3..8);
    let input = progressive_format(&clip);
    let output = deinterlaced(input, false);
    let chunks = vec![inputs(&clip, &input, 0), inputs(&closing, &input, 3)];

    let mut args = mode_args("repair");
    args["tr0"] = json!(0);
    args["tr1"] = json!(0);
    args["tr2"] = json!(0);
    args["sharpness"] = json!(0.0);
    let runs = run_chunks(args, input, output, chunks);

    let whole = &runs[0].outputs[3].planes[0];
    let chunked = &runs[1].outputs[0].planes[0];
    assert_eq!(chunked, whole, "frame 3 keeps the same field in both runs");

    let settings = QtgmcReferenceSettings {
        tr0: 0,
        tr1: 0,
        tr2: 0,
        sharpness: 0.0,
        ..mode_settings(ReferenceInput::Repair)
    };
    let stream = QtgmcReferenceSettings {
        first_frame: 3,
        ..settings
    };
    let from_stream = expected(&closing, &stream, &cubic);
    let from_chunk = expected(&closing, &settings, &cubic);
    assert_ne!(
        from_stream.fields[0][0], from_chunk.fields[0][0],
        "the parity of frame 3 changes its output"
    );

    assert_mode_matches(&runs[1], &closing, &stream, "frame 3 onwards");
}

#[test]
fn odd_chunk_lengths_keep_the_right_parity() {
    let clip = Clip::badly("badly_objects_8");
    let parts = [clip.cut(5..6), clip.cut(5..8)];
    let borrowed: Vec<&Clip> = parts.iter().collect();

    for (mode, input) in &MODES[1..] {
        let runs = run_parts(mode_args(mode), &borrowed, 5);
        let settings = QtgmcReferenceSettings {
            first_frame: 5,
            ..mode_settings(*input)
        };

        for (run, part) in runs.iter().zip(&parts) {
            assert_eq!(run.outputs.len(), part.frames.len(), "{mode} {}", part.name);

            let label = format!("{mode} {}", part.name);
            assert_mode_matches(run, part, &settings, &label);
        }
    }
}

#[test]
fn a_stream_shorter_than_the_lag_flushes_every_frame() {
    let clip = Clip::badly("badly_objects_8");
    let parts: Vec<Clip> = (1..=5).map(|count| clip.cut(0..count)).collect();
    let borrowed: Vec<&Clip> = parts.iter().collect();

    for (mode, input) in MODES {
        let runs = run_parts(mode_args(mode), &borrowed, 0);
        let settings = mode_settings(input);

        for (run, part) in runs.iter().zip(&parts) {
            assert_eq!(run.outputs.len(), part.frames.len(), "{mode} {}", part.name);

            let label = format!("{mode} {}", part.name);
            assert_mode_matches(run, part, &settings, &label);
        }
    }
}

#[test]
fn progressive_modes_keep_every_frame_s_index_and_timestamp() {
    let clip = Clip::badly("badly_pan_8").cut(0..4);
    let input = progressive_format(&clip);
    let output = deinterlaced(input, false);

    for (mode, _) in MODES {
        let mut chunk = inputs(&clip, &input, 10);
        for frame in &mut chunk {
            frame.meta.chunk = 3;
        }
        let input_metas: Vec<FrameMeta> = chunk.iter().map(|frame| frame.meta.clone()).collect();

        let mut runs = run_chunks(mode_args(mode), input, output, vec![chunk]);
        let run = runs.pop().expect("one chunk was run");
        let found = metas(&run.outputs);
        assert_eq!(found, input_metas, "{mode}");

        let timestamps: Vec<Option<Duration>> = found.iter().map(|meta| meta.timestamp).collect();
        let expected_timestamps: Vec<Option<Duration>> = (10..14)
            .map(|frame| Some(Duration::from_millis(40 * frame)))
            .collect();
        assert_eq!(timestamps, expected_timestamps, "{mode}");
    }
}
