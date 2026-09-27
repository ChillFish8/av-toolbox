use std::num::NonZeroUsize;
use std::time::Duration;

use av_scenechange::Rational32;
use av_toolbox_gpu::{FrameFormat, FrameLayout, HostFrame, SampleFormat, Subsampling, testing_context};

use super::clip::{THREE_SCENES, clip, flat_clip, grey8, grey8_of, yuv420_ten_bit};
use crate::scene::args::{DetectArgs, Mode, Speed};
use crate::scene::window::{Decision, LumaWindow, Settings, frame_rate};

fn natural(speed: Speed, min_scene_frames: usize, max_scene_frames: Option<usize>) -> DetectArgs {
    let max_scene_frames = max_scene_frames.and_then(NonZeroUsize::new);

    DetectArgs {
        mode: Mode::Natural,
        speed,
        min_scene_frames,
        max_scene_frames,
    }
}

fn decide(format: &FrameFormat, args: &DetectArgs, scenes: &[(u64, u64)]) -> Vec<Decision> {
    let context = testing_context();
    let frames = clip(&context, format, scenes);

    decide_frames(format, args, frames)
}

fn decide_frames(format: &FrameFormat, args: &DetectArgs, frames: Vec<HostFrame>) -> Vec<Decision> {
    let settings = Settings::for_stream(format, args).expect("the clip is readable");
    let mut window = LumaWindow::new(settings);

    let mut decisions = Vec::new();
    for frame in &frames {
        let decided = window.push(frame).expect("the luma copies");
        decisions.extend(decided);
    }

    let rest = window.finish();
    decisions.extend(rest);

    decisions
}

fn scene_starts(decisions: &[Decision]) -> Vec<usize> {
    decisions
        .iter()
        .enumerate()
        .filter(|(_, decision)| decision.starts_scene)
        .map(|(position, _)| position)
        .collect()
}

#[test]
fn every_frame_gets_one_decision() {
    let args = natural(Speed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, THREE_SCENES);

    assert_eq!(decisions.len(), 90);
}

#[test]
fn hard_cuts_start_scenes_at_eight_bits() {
    let args = natural(Speed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, THREE_SCENES);

    assert_eq!(scene_starts(&decisions), vec![0, 30, 60]);
}

#[test]
fn hard_cuts_start_scenes_at_ten_bits_in_yuv420() {
    let args = natural(Speed::Standard, 0, None);

    let decisions = decide(&yuv420_ten_bit(), &args, THREE_SCENES);

    assert_eq!(scene_starts(&decisions), vec![0, 30, 60]);
}

#[test]
fn the_first_frame_and_the_last_four_go_unscored() {
    let args = natural(Speed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, THREE_SCENES);

    let unscored: Vec<usize> = decisions
        .iter()
        .enumerate()
        .filter(|(_, decision)| decision.score.is_none())
        .map(|(position, _)| position)
        .collect();
    assert_eq!(unscored, vec![0, 86, 87, 88, 89]);
}

#[test]
fn a_cut_scores_above_the_frame_before_it() {
    let args = natural(Speed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, THREE_SCENES);

    let before = decisions[29].score.expect("scored").get();
    let cut = decisions[30].score.expect("scored").get();
    assert!(cut > before, "cut {cut}, before {before}");
}

#[test]
fn a_minimum_scene_length_suppresses_a_close_cut() {
    let args = natural(Speed::Standard, 24, None);

    let decisions = decide(&grey8(), &args, &[(1, 30), (2, 10), (3, 30)]);

    assert_eq!(scene_starts(&decisions), vec![0, 30]);
}

#[test]
fn a_maximum_scene_length_forces_a_cut() {
    let args = natural(Speed::Standard, 0, Some(20));

    let decisions = decide(&grey8(), &args, &[(1, 50)]);

    assert_eq!(scene_starts(&decisions), vec![0, 20, 40]);
}

#[test]
fn a_single_frame_is_one_scene() {
    let args = natural(Speed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, &[(1, 1)]);

    assert_eq!(decisions, vec![Decision::FIRST]);
}

#[test]
fn fast_speed_scores_frames() {
    let args = natural(Speed::Fast, 0, None);

    let decisions = decide(&grey8(), &args, THREE_SCENES);

    let scored = decisions
        .iter()
        .filter(|decision| decision.score.is_some())
        .count();
    assert_eq!(scored, 85);
}

#[test]
fn the_frame_rate_comes_from_the_first_two_timestamps() {
    let first = Some(Duration::from_millis(0));
    let second = Some(Duration::from_millis(40));

    let rate = frame_rate(first, second);

    assert_eq!(rate, Rational32::new(25, 1));
}

#[test]
fn the_frame_rate_falls_back_to_thirty() {
    let first = Some(Duration::from_millis(40));
    let huge = Some(Duration::from_secs(100_000));
    let fallback = Rational32::new(30, 1);

    assert_eq!(frame_rate(None, first), fallback);
    assert_eq!(frame_rate(first, first), fallback);
    assert_eq!(frame_rate(Some(Duration::ZERO), huge), fallback);
}

#[test]
fn a_flat_clip_scores_only_finite_values() {
    let args = natural(Speed::Standard, 0, None);
    let context = testing_context();
    let format = grey8();
    let frames = flat_clip(&context, &format, 40);

    let decisions = decide_frames(&format, &args, frames);

    let non_finite = decisions
        .iter()
        .filter_map(|decision| decision.score)
        .filter(|score| !score.get().is_finite())
        .count();
    assert_eq!(non_finite, 0, "every score should be finite");
}

#[test]
fn a_height_not_a_multiple_of_sixty_four_does_not_panic() {
    let args = natural(Speed::Standard, 0, None);
    let format = grey8_of(96, 40);

    decide(&format, &args, THREE_SCENES);
}

#[test]
fn rgb_and_float_frames_are_refused() {
    let args = natural(Speed::Standard, 0, None);
    let rgb = FrameFormat {
        layout: FrameLayout::Rgb,
        ..grey8()
    };
    let float = FrameFormat {
        sample: SampleFormat::F32,
        ..grey8()
    };

    assert!(Settings::for_stream(&rgb, &args).is_err());
    assert!(Settings::for_stream(&float, &args).is_err());
}

#[test]
fn chroma_planes_are_refused() {
    let args = natural(Speed::Standard, 0, None);
    let chroma = FrameFormat {
        layout: FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        ..grey8()
    };

    assert!(Settings::for_stream(&chroma, &args).is_err());
}
