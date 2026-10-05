use crate::testing::motion::motion_clips;
use crate::testing::qtgmc::{FieldClip, field_clips};

const MOTION_INPUTS: [(&str, &[u8]); 6] = [
    ("pan_8", include_bytes!("../../fixtures/motion/inputs/pan_8.y")),
    ("pan_10", include_bytes!("../../fixtures/motion/inputs/pan_10.y")),
    (
        "objects_8",
        include_bytes!("../../fixtures/motion/inputs/objects_8.y"),
    ),
    (
        "noisy_8",
        include_bytes!("../../fixtures/motion/inputs/noisy_8.y"),
    ),
    (
        "noisy_10",
        include_bytes!("../../fixtures/motion/inputs/noisy_10.y"),
    ),
    ("cut_8", include_bytes!("../../fixtures/motion/inputs/cut_8.y")),
];

fn plane_width(clip: &FieldClip, plane: usize) -> usize {
    if plane == 0 { clip.width } else { clip.width / 2 }
}

#[test]
fn field_clips_are_deterministic() {
    let first = field_clips();
    let second = field_clips();

    assert_eq!(first.len(), second.len());
    for (built, rebuilt) in first.iter().zip(&second) {
        assert_eq!(built.frames, rebuilt.frames, "{} frames", built.name);
        assert_eq!(built.truth, rebuilt.truth, "{} truth", built.name);
    }
}

#[test]
fn field_clips_hold_two_fields_a_frame() {
    let names: Vec<&str> = field_clips().iter().map(|clip| clip.name).collect();
    assert_eq!(
        names,
        [
            "field_pan_8",
            "field_objects_8",
            "field_noisy_8",
            "field_noisy_10",
            "field_static_8"
        ]
    );

    for clip in field_clips() {
        assert_eq!((clip.width, clip.height), (128, 64), "{}", clip.name);
        assert_eq!(clip.frames.len(), 8, "{}", clip.name);
        assert_eq!(clip.truth.len(), 2 * clip.frames.len(), "{}", clip.name);

        let sizes = clip.frames[0].each_ref().map(Vec::len);
        assert_eq!(sizes, [128 * 64, 64 * 32, 64 * 32], "{}", clip.name);
    }
}

#[test]
fn motion_clips_are_unchanged() {
    for clip in motion_clips() {
        let (_, bytes) = MOTION_INPUTS
            .iter()
            .find(|(name, _)| *name == clip.name)
            .expect("every motion clip has an input fixture");
        let (pairs, _) = bytes.as_chunks::<2>();
        let committed: Vec<i32> = pairs
            .iter()
            .map(|pair| i32::from(u16::from_le_bytes(*pair)))
            .collect();
        let built: Vec<i32> = clip
            .frames
            .iter()
            .flat_map(|planes| planes[0].iter().copied())
            .collect();

        assert_eq!(built, committed, "{} differs from av-denoise's input", clip.name);
    }
}

#[test]
fn every_field_of_a_clip_matches_its_truth_rows() {
    for clip in field_clips() {
        for (field, truth) in clip.truth.iter().enumerate() {
            let frame = &clip.frames[field / 2];

            for plane in 0..3 {
                let width = plane_width(&clip, plane);
                let rows = frame[plane].len() / width;

                for row in (field % 2..rows).step_by(2) {
                    let span = row * width..(row + 1) * width;
                    assert_eq!(
                        frame[plane][span.clone()],
                        truth[plane][span],
                        "{} field {field} plane {plane} row {row}",
                        clip.name
                    );
                }
            }
        }
    }
}

#[test]
fn a_moving_clip_changes_between_its_fields() {
    let clips = field_clips();
    let pan = clips
        .iter()
        .find(|clip| clip.name == "field_pan_8")
        .expect("field_pan_8");

    assert_ne!(pan.truth[0][0], pan.truth[1][0]);
}

#[test]
fn the_static_clip_holds_still_with_lines_in_both_fields() {
    let clips = field_clips();
    let still = clips
        .iter()
        .find(|clip| clip.name == "field_static_8")
        .expect("field_static_8");

    for truth in &still.truth {
        assert_eq!(*truth, still.truth[0]);
    }

    let luma = &still.truth[0][0];
    let line = |row: usize| luma[row * still.width..(row + 1) * still.width].to_vec();
    let level = line(2)[0];
    assert!(line(2).iter().all(|sample| *sample == level));
    assert_eq!(line(7), line(2), "a line on an odd row");
    assert_ne!(line(3), line(2));
}

/// How far noise of 3 codes at 8 bits can move a sample, past the furthest Box-Muller draw of 8.6 sigma
/// with a code of rounding.
fn noise_bound(bits: u32) -> i32 {
    let sigma = 3.0 * (1u32 << (bits - 8)) as f64;

    (8.6 * sigma).ceil() as i32 + 1
}

#[test]
fn clean_matches_truth_on_clean_clips() {
    for clip in field_clips() {
        if clip.name.starts_with("field_noisy") {
            continue;
        }

        assert_eq!(clip.clean, clip.truth, "{}", clip.name);
    }
}

#[test]
fn clean_differs_from_truth_only_by_noise_on_noisy_clips() {
    let clips = field_clips();
    let noisy: Vec<&FieldClip> = clips
        .iter()
        .filter(|clip| clip.name.starts_with("field_noisy"))
        .collect();
    assert_eq!(noisy.len(), 2);

    for clip in noisy {
        let bound = noise_bound(clip.bits);

        let mut largest = 0;
        for (truth, clean) in clip.truth.iter().zip(&clip.clean) {
            for plane in 0..3 {
                for (noisy_sample, clean_sample) in truth[plane].iter().zip(&clean[plane]) {
                    largest = largest.max((noisy_sample - clean_sample).abs());
                }
            }
        }

        assert!(largest > 0, "{} has no noise", clip.name);
        assert!(
            largest <= bound,
            "{} moves a sample by {largest}, over {bound}",
            clip.name
        );
    }
}
