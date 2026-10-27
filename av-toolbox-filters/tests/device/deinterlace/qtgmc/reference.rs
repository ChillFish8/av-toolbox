use av_toolbox_cubek::testing::QtgmcReferenceSettings;
use av_toolbox_gpu::FieldOrder;
use serde_json::json;

use super::{
    Clip,
    Eedi3Reference,
    assert_matches,
    assert_motion_fields,
    assert_paired_outputs,
    cubic,
    deinterlaced,
    expected,
    inputs,
    interlaced,
    reference_defaults,
    run_chunks,
    run_clip,
};

/// The frame at which the scene-cut clip changes scene.
const CUT_FRAME: usize = 4;

/// A clip of `first`'s frames before `at` and `second`'s frames from it.
fn joined(first: &Clip, second: &Clip, at: usize) -> Clip {
    let mut frames = first.frames[..at].to_vec();
    frames.extend_from_slice(&second.frames[at..]);
    let name = format!("{} cut to {} at frame {at}", first.name, second.name);

    Clip {
        name,
        frames,
        ..first.clone()
    }
}

/// One frame of `width`, its even rows from `top` and its odd rows from `bottom`, plane by plane.
fn woven(top: &[Vec<i32>; 3], bottom: &[Vec<i32>; 3], width: usize) -> [Vec<i32>; 3] {
    [0, 1, 2].map(|plane| {
        let plane_width = if plane == 0 { width } else { width / 2 };
        let mut planes = top[plane].clone();
        if plane_width == 0 || planes.is_empty() {
            return planes;
        }

        let rows = planes.len() / plane_width;
        for row in (1..rows).step_by(2) {
            let span = row * plane_width..(row + 1) * plane_width;
            planes[span.clone()].copy_from_slice(&bottom[plane][span]);
        }

        planes
    })
}

/// `clip`'s fields shown bottom first, rewoven so the top-first reference meets them in that order.
///
/// Frame `n` of the result holds the top field of frame `n - 1` and the bottom field of frame `n`. Its
/// first and last frames are `clip`'s own, so the reference's field `k + 1` is the stage's field `k`, with
/// one extra field at each end.
fn bottom_first_reordered(clip: &Clip) -> Clip {
    let count = clip.frames.len();
    let mut frames = Vec::with_capacity(count + 1);
    frames.push(clip.frames[0].clone());

    for index in 1..count {
        let rewoven = woven(&clip.frames[index - 1], &clip.frames[index], clip.width);
        frames.push(rewoven);
    }

    frames.push(clip.frames[count - 1].clone());
    let name = format!("{} reordered bottom first", clip.name);

    Clip {
        name,
        frames,
        ..clip.clone()
    }
}

#[test]
fn the_stage_matches_the_reference() {
    let names = ["field_pan_8", "field_objects_8", "field_noisy_10"];
    let settings = reference_defaults();

    for name in names {
        let clip = Clip::named(name);
        let run = run_clip(json!({ "spatial": { "kind": "cubic" } }), &clip);

        let wanted = expected(&clip, &settings, &cubic);
        assert_matches(&run, &wanted, &clip, name);
    }
}

#[test]
fn the_stage_matches_the_reference_at_every_radius_extreme() {
    let clip = Clip::named("field_objects_8");
    let defaults = reference_defaults();
    let cases = [
        ("tr0", 0, QtgmcReferenceSettings { tr0: 0, ..defaults }),
        ("tr0", 2, QtgmcReferenceSettings { tr0: 2, ..defaults }),
        ("tr1", 0, QtgmcReferenceSettings { tr1: 0, ..defaults }),
        ("tr1", 2, QtgmcReferenceSettings { tr1: 2, ..defaults }),
        ("tr2", 0, QtgmcReferenceSettings { tr2: 0, ..defaults }),
        ("tr2", 3, QtgmcReferenceSettings { tr2: 3, ..defaults }),
    ];

    for (name, radius, settings) in cases {
        let mut args = json!({ "spatial": { "kind": "cubic" } });
        args[name] = json!(radius);
        let run = run_clip(args, &clip);

        let wanted = expected(&clip, &settings, &cubic);
        let label = format!("{name} {radius}");
        assert_matches(&run, &wanted, &clip, &label);
    }
}

#[test]
fn the_default_eedi3_path_matches_the_reference() {
    let clip = Clip::named("field_objects_8").cut(0..3);
    let run = run_clip(json!({}), &clip);

    let eedi3 = Eedi3Reference::new(clip.bits);
    let interpolate = |plane: &[f32], width, height, parity| eedi3.interpolate(plane, width, height, parity);
    let settings = reference_defaults();
    let wanted = expected(&clip, &settings, &interpolate);
    assert_matches(&run, &wanted, &clip, "eedi3");
}

#[test]
fn a_scene_cut_inside_a_chunk_matches_the_reference() {
    let first = Clip::named("field_static_8");
    let second = Clip::named("field_pan_8");
    let clip = joined(&first, &second, CUT_FRAME);
    let run = run_clip(json!({ "spatial": { "kind": "cubic" } }), &clip);

    let settings = reference_defaults();
    let wanted = expected(&clip, &settings, &cubic);
    let after_cut = &wanted.motion[2 * CUT_FRAME];
    let flagged = after_cut.iter().all(|motion| motion.flagged);
    assert!(
        flagged,
        "every neighbour of the first field after the cut is a scene change"
    );

    assert_matches(&run, &wanted, &clip, &clip.name);
}

/// Compares only the interior fields, because the reweave adds a field at each end.
///
/// A field's motion reads the bobs up to `max_tr + tr0` fields away and its output those up to the lag, so
/// the fields nearer an end read the added fields. The top-first tests cover the edges.
#[test]
fn bottom_field_first_matches_the_reference_reordered() {
    let clip = Clip::named("field_objects_8");
    let input = interlaced(&clip, FieldOrder::BottomFirst);
    let output = deinterlaced(input, true);
    let chunk = inputs(&clip, &input, 0);
    let mut runs = run_chunks(
        json!({ "spatial": { "kind": "cubic" } }),
        input,
        output,
        vec![chunk],
    );
    let run = runs.pop().expect("one chunk was run");

    let reordered = bottom_first_reordered(&clip);
    let settings = reference_defaults();
    let wanted = expected(&reordered, &settings, &cubic);

    let motion_reach = (settings.max_tr() + settings.tr0) as usize;
    let lag = motion_reach + settings.tr2 as usize;
    let fields = clip.field_count();
    let motion_fields =
        (motion_reach..fields - motion_reach).map(|field| (field, wanted.motion[field + 1].as_slice()));
    assert_motion_fields(&run.motion, motion_fields, &reordered.name);

    let pairs = (lag..fields - lag).map(|field| (field, &run.outputs[field], &wanted.fields[field + 1]));
    assert_paired_outputs(pairs, &clip, wanted.input, &reordered.name);
}

#[test]
fn gray_input_matches_the_reference() {
    let clip = Clip::named("field_pan_8").gray();
    let run = run_clip(json!({ "spatial": { "kind": "cubic" } }), &clip);

    let settings = reference_defaults();
    let wanted = expected(&clip, &settings, &cubic);
    assert_matches(&run, &wanted, &clip, &clip.name);
}
