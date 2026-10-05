use crate::deinterlace::qtgmc::Parity;
use crate::motion::{BlockGrid, MotionSettings, PlaneScale};
use crate::testing::qtgmc::chain::{kept_parity, repair_mask, repaired_edi};
use crate::testing::qtgmc::reference::merge;
use crate::testing::qtgmc::{
    FieldClip,
    ProgressiveClip,
    QtgmcFieldSteps,
    QtgmcReferenceSettings,
    ReferenceInput,
    ReferenceMotion,
    badly_clips,
    field_clips,
    qtgmc_deltas,
    qtgmc_fixture,
    reference_bob,
    reference_qtgmc,
    reference_qtgmc_field,
    reference_shimmer,
    reference_temporal_limits,
};
use crate::testing::{
    ReferenceNeighbour,
    assert_plane_within,
    reference_compensate,
    reference_degrain,
    reference_scene_changed,
    reference_vertical_blur,
};

fn bob_interpolate(plane: &[f32], width: usize, height: usize, parity: Parity) -> Vec<f32> {
    reference_bob(plane, (width, height), parity)
}

fn clip(name: &str) -> FieldClip {
    field_clips()
        .into_iter()
        .find(|clip| clip.name == name)
        .expect("the field clip exists")
}

/// The first `count` frames of `clip` as f32 code values.
fn code_frames(clip: &FieldClip, count: usize) -> Vec<[Vec<f32>; 3]> {
    clip.frames
        .iter()
        .take(count)
        .map(|planes| {
            planes
                .each_ref()
                .map(|plane| plane.iter().map(|sample| *sample as f32).collect())
        })
        .collect()
}

fn plane_size(clip: &FieldClip, plane: usize) -> (usize, usize) {
    if plane == 0 {
        (clip.width, clip.height)
    } else {
        (clip.width / 2, clip.height / 2)
    }
}

#[test]
fn deltas_run_nearest_first_later_field_first() {
    assert_eq!(qtgmc_deltas(1), vec![1, -1]);
    assert_eq!(qtgmc_deltas(3), vec![1, -1, 2, -2, 3, -3]);
}

#[test]
fn the_reference_with_every_radius_zero_and_no_sharpening_is_the_interpolation() {
    let source = clip("field_objects_8");
    let frames = code_frames(&source, 3);
    let settings = QtgmcReferenceSettings {
        tr0: 0,
        tr1: 0,
        tr2: 0,
        sharpness: 0.0,
        ..QtgmcReferenceSettings::default()
    };

    let output = reference_qtgmc(
        &frames,
        (source.width, source.height),
        8,
        &settings,
        &bob_interpolate,
    );

    assert_eq!(output.len(), 2 * frames.len());
    for (field, planes) in output.iter().enumerate() {
        let parity = if field.is_multiple_of(2) {
            Parity::Top
        } else {
            Parity::Bottom
        };

        for plane in 0..3 {
            let (width, height) = plane_size(&source, plane);
            let edi = bob_interpolate(&frames[field / 2][plane], width, height, parity);
            let expected = reference_shimmer(&edi, &edi, width, height, 8);

            assert_eq!(planes[plane], expected, "field {field} plane {plane}");
        }
    }
}

#[test]
fn reference_qtgmc_is_deterministic() {
    let source = clip("field_noisy_8");
    let frames = code_frames(&source, 3);
    let settings = QtgmcReferenceSettings::default();
    let luma = (source.width, source.height);

    let first = reference_qtgmc(&frames, luma, source.bits, &settings, &bob_interpolate);
    let second = reference_qtgmc(&frames, luma, source.bits, &settings, &bob_interpolate);

    assert_eq!(first, second);
}

#[test]
fn a_field_s_steps_end_in_its_output() {
    let source = clip("field_pan_8");
    let frames = code_frames(&source, 3);
    let settings = QtgmcReferenceSettings::default();
    let luma = (source.width, source.height);

    let outputs = reference_qtgmc(&frames, luma, source.bits, &settings, &bob_interpolate);
    let steps = reference_qtgmc_field(&frames, luma, source.bits, &settings, &bob_interpolate, 0);

    assert_eq!(steps.output, outputs[0]);
    let deltas: Vec<i32> = steps.motion.iter().map(|found| found.delta).collect();
    assert_eq!(deltas, vec![1, 2], "the first field has no earlier neighbours");
}

#[test]
fn the_reference_changes_the_bob_where_it_smooths() {
    let source = clip("field_noisy_8");
    let frames = code_frames(&source, 3);
    let settings = QtgmcReferenceSettings::default();
    let luma = (source.width, source.height);

    let steps = reference_qtgmc_field(&frames, luma, source.bits, &settings, &bob_interpolate, 2);

    assert_ne!(steps.smoothed[0], steps.edi[0], "TR1 should smooth the noise");
    assert_ne!(steps.output[0], steps.edi[0]);
    assert!(
        steps.output[0]
            .iter()
            .all(|sample| (0.0..=255.0).contains(sample))
    );
}

#[test]
fn gray_frames_give_gray_fields() {
    let source = clip("field_pan_8");
    let frames: Vec<[Vec<f32>; 3]> = code_frames(&source, 2)
        .into_iter()
        .map(|[luma, _, _]| [luma, Vec::new(), Vec::new()])
        .collect();
    let colour = code_frames(&source, 2);
    let settings = QtgmcReferenceSettings::default();
    let luma = (source.width, source.height);

    let gray = reference_qtgmc(&frames, luma, source.bits, &settings, &bob_interpolate);
    let full = reference_qtgmc(&colour, luma, source.bits, &settings, &bob_interpolate);

    for (field, (gray_planes, full_planes)) in gray.iter().zip(&full).enumerate() {
        assert_eq!(gray_planes[0], full_planes[0], "field {field} luma");
        assert!(
            gray_planes[1].is_empty() && gray_planes[2].is_empty(),
            "field {field}"
        );
    }
}

/// The first field after a cut from `field_static_8` to `field_pan_8`.
const AFTER_CUT: usize = 4;

/// The neighbours `distance` fields away that are not flagged, read through their `edi`, as a degrain takes
/// them.
fn unflagged<'a>(
    steps: &'a QtgmcFieldSteps,
    distance: i32,
    plane: usize,
) -> (Vec<ReferenceNeighbour<'a>>, Vec<bool>) {
    let kept: Vec<&ReferenceMotion> = steps
        .motion
        .iter()
        .filter(|found| found.delta.abs() == distance && !found.flagged)
        .collect();
    let neighbours = kept
        .iter()
        .map(|found| ReferenceNeighbour {
            plane: &found.neighbour_edi[plane],
            vectors: &found.field.vectors,
            sads: &found.field.sads,
        })
        .collect();

    (neighbours, vec![false; kept.len()])
}

// TR0 is off, as its average reaches across the cut and would flag every neighbour of the field.
#[test]
fn a_neighbour_across_a_cut_is_flagged_and_left_out() {
    let before = clip("field_static_8");
    let after = clip("field_pan_8");
    let mut frames = code_frames(&before, 2);
    frames.extend(code_frames(&after, 2));
    let luma = (after.width, after.height);
    let settings = QtgmcReferenceSettings {
        tr0: 0,
        ..QtgmcReferenceSettings::default()
    };

    let steps = reference_qtgmc_field(&frames, luma, after.bits, &settings, &bob_interpolate, AFTER_CUT);

    let flags: Vec<(i32, bool)> = steps
        .motion
        .iter()
        .map(|found| (found.delta, found.flagged))
        .collect();
    assert_eq!(flags, vec![(1, false), (-1, true), (2, false), (-2, true)]);

    let grid = BlockGrid::new(&MotionSettings::default(), luma.0 as u32, luma.1 as u32);
    let pel = MotionSettings::default().pel;
    let degrain_settings = settings.tr1_degrain();
    for plane in 0..3 {
        let size = plane_size(&after, plane);
        let scale = if plane == 0 {
            PlaneScale::LUMA
        } else {
            PlaneScale {
                shift_x: 1,
                shift_y: 1,
            }
        };
        let edi = &steps.edi[plane];

        let next = &steps.motion[0];
        let compensated = reference_compensate(
            &next.neighbour_edi[plane],
            size,
            &next.field.vectors,
            &grid,
            pel,
            scale,
        );
        let (upper, lower) = reference_temporal_limits(edi, &[&compensated]);
        assert_eq!(steps.upper[plane], upper, "plane {plane} upper");
        assert_eq!(steps.lower[plane], lower, "plane {plane} lower");

        let degrain_at = |distance: i32| {
            let (neighbours, flagged) = unflagged(&steps, distance, plane);
            reference_degrain(
                edi,
                size,
                &neighbours,
                &flagged,
                &grid,
                &degrain_settings,
                pel,
                scale,
                plane > 0,
            )
        };
        let near = degrain_at(1);
        let far = degrain_at(2);
        let blended = merge(&near, &far, 0.2);
        let smoothed = merge(&blended, edi, 0.0625);
        assert_eq!(steps.smoothed[plane], smoothed, "plane {plane} tr1");
    }
}

fn badly_clip(name: &str) -> ProgressiveClip {
    badly_clips()
        .into_iter()
        .find(|clip| clip.name == name)
        .expect("the badly deinterlaced clip exists")
}

fn float_planes(planes: &[Vec<i32>; 3]) -> [Vec<f32>; 3] {
    planes
        .each_ref()
        .map(|plane| plane.iter().map(|sample| *sample as f32).collect())
}

/// The first `count` frames of `clip` as f32 code values.
fn progressive_frames(clip: &ProgressiveClip, count: usize) -> Vec<[Vec<f32>; 3]> {
    clip.frames.iter().take(count).map(float_planes).collect()
}

// Reversed repair keeps, top field first, the fields that repair keeps bottom field first.
#[test]
fn the_reference_repair_field_pattern_matches_havsfunc() {
    let inputs = qtgmc_fixture("inputs", "frames");

    for (name, input) in [
        ("tff", ReferenceInput::Repair),
        ("bff", ReferenceInput::RepairReversed),
    ] {
        let outputs = qtgmc_fixture("repair", name);

        for (source, expected) in inputs.iter().zip(&outputs) {
            // VapourSynth's Weave leaves the last frame of an odd count without a partner field.
            let paired_frames = source.frames.len() / 2 * 2;

            for index in 0..paired_frames {
                let planes = &source.frames[index];
                let frame: [Vec<f32>; 3] = std::array::from_fn(|plane| {
                    planes[plane]
                        .samples
                        .iter()
                        .map(|sample| *sample as f32)
                        .collect()
                });
                let luma = (planes[0].width, planes[0].height);
                let mask = vec![1.0; luma.0 * luma.1];
                let parity = kept_parity(input, index);

                let edi = repaired_edi(&frame, luma, parity, &mask, &bob_interpolate);

                let wanted: Vec<f32> = expected.frames[index][0]
                    .samples
                    .iter()
                    .map(|sample| *sample as f32)
                    .collect();
                let label = format!("repair/{name} {} frame {index}", source.clip);
                assert_plane_within(&edi[0], &wanted, 1.0, luma.0 as u32, &label);
                assert_eq!(edi[1], frame[1], "{label} keeps its chroma");
                assert_eq!(edi[2], frame[2], "{label} keeps its chroma");
            }
        }
    }
}

#[test]
fn progressive_input_with_every_radius_zero_and_no_sharpening_is_rep2_of_the_frame() {
    let source = badly_clip("badly_objects_8");
    let frames = progressive_frames(&source, 3);
    let settings = QtgmcReferenceSettings {
        tr0: 0,
        tr1: 0,
        tr2: 0,
        sharpness: 0.0,
        input: ReferenceInput::Progressive,
        ..QtgmcReferenceSettings::default()
    };
    let luma = (source.width, source.height);

    let output = reference_qtgmc(&frames, luma, source.bits, &settings, &bob_interpolate);

    assert_eq!(output.len(), frames.len());
    for (index, planes) in output.iter().enumerate() {
        for plane in 0..3 {
            let width = if plane == 0 {
                source.width
            } else {
                source.width / 2
            };
            let height = frames[index][plane].len() / width;
            let frame = &frames[index][plane];
            let expected = reference_shimmer(frame, frame, width, height, source.bits);

            assert_eq!(planes[plane], expected, "frame {index} plane {plane}");
        }
    }
}

#[test]
fn the_last_repaired_frame_keeps_its_original_luma() {
    let source = badly_clip("badly_pan_8");
    let frames = progressive_frames(&source, 3);
    let settings = QtgmcReferenceSettings {
        input: ReferenceInput::Repair,
        ..QtgmcReferenceSettings::default()
    };
    let luma = (source.width, source.height);

    let outputs = reference_qtgmc(&frames, luma, source.bits, &settings, &bob_interpolate);
    assert_eq!(outputs.len(), frames.len());

    let last = reference_qtgmc_field(&frames, luma, source.bits, &settings, &bob_interpolate, 2);
    let deltas: Vec<i32> = last.motion.iter().map(|found| found.delta).collect();
    assert_eq!(deltas, vec![-1, -2], "the last frame has no later neighbours");
    assert_eq!(last.edi, frames[2]);

    let first = reference_qtgmc_field(&frames, luma, source.bits, &settings, &bob_interpolate, 0);
    assert_ne!(
        first.edi[0], frames[0][0],
        "an earlier frame rebuilds where it moves"
    );
    assert_eq!(
        first.edi[1..],
        frames[0][1..],
        "an earlier frame keeps its chroma"
    );
}

#[test]
fn the_repair_mask_flags_by_mvtools_default_rule() {
    let luma = (128, 64);
    let grid = BlockGrid::new(&MotionSettings::default(), luma.0 as u32, luma.1 as u32);
    let chain_rule = QtgmcReferenceSettings::SCENE_CHANGE;
    let mask_rule = QtgmcReferenceSettings::MASK_SCENE_CHANGE;
    let chain_threshold = chain_rule.threshold(grid.blksize);
    let mask_threshold = mask_rule.threshold(grid.blksize);

    let between = vec![(chain_threshold + mask_threshold) / 2.0; grid.blocks()];
    assert!(reference_scene_changed(&between, &grid, chain_rule));
    assert!(!reference_scene_changed(&between, &grid, mask_rule));
    let mask = repair_mask(Some(&between), &grid, luma);
    assert!(
        mask.iter().all(|weight| *weight > 0.0),
        "only QTGMC's rule flags it"
    );

    let above = vec![mask_threshold * 2.0; grid.blocks()];
    assert!(reference_scene_changed(&above, &grid, mask_rule));
    let mask = repair_mask(Some(&above), &grid, luma);
    assert!(mask.iter().all(|weight| *weight == 0.0), "MVTools' rule flags it");

    let missing = repair_mask(None, &grid, luma);
    assert_eq!(missing, vec![0.0; luma.0 * luma.1]);
}

#[test]
fn each_progressive_mode_bobs_its_frame_as_qtgmc_does() {
    let source = badly_clip("badly_objects_8");
    let frames = progressive_frames(&source, 3);
    let luma = (source.width, source.height);

    let modes = [
        ReferenceInput::Progressive,
        ReferenceInput::Repair,
        ReferenceInput::RepairReversed,
    ];
    for input in modes {
        let settings = QtgmcReferenceSettings {
            input,
            ..QtgmcReferenceSettings::default()
        };

        for (entry, frame) in frames.iter().enumerate() {
            let steps = reference_qtgmc_field(&frames, luma, source.bits, &settings, &bob_interpolate, entry);

            let expected = match input {
                ReferenceInput::Progressive => frame[0].clone(),
                _ => reference_vertical_blur(&frame[0], luma.0, luma.1),
            };
            assert_eq!(steps.bob, expected, "{input:?} entry {entry}");
        }
    }
}
