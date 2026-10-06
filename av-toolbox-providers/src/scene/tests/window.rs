use std::num::NonZeroUsize;
use std::time::Duration;

use av_scenechange::Rational32;
use av_toolbox_gpu::{
    Channel,
    FrameContext,
    FrameFormat,
    FrameLayout,
    HostFrame,
    SampleFormat,
    Subsampling,
    testing_context,
};

use crate::provider::SceneSpeed;
use crate::scene::window::{Decision, DetectOptions, LumaWindow, Settings, frame_rate};
use crate::tests::clip::{THREE_SCENES, clip, flat_clip, grey8, grey8_of, yuv420_ten_bit};

fn natural(speed: SceneSpeed, min_scene_frames: usize, max_scene_frames: Option<usize>) -> DetectOptions {
    let max_scene_frames = max_scene_frames.and_then(NonZeroUsize::new);

    DetectOptions {
        speed,
        min_scene_frames,
        max_scene_frames,
    }
}

fn decide(format: &FrameFormat, args: &DetectOptions, scenes: &[(u64, u64)]) -> Vec<Decision> {
    let context = testing_context();
    let frames = clip(&context, format, scenes);

    decide_frames(format, args, frames)
}

fn decide_frames(format: &FrameFormat, args: &DetectOptions, frames: Vec<HostFrame>) -> Vec<Decision> {
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
    let args = natural(SceneSpeed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, THREE_SCENES);

    assert_eq!(decisions.len(), 90);
}

#[test]
fn hard_cuts_start_scenes_at_eight_bits() {
    let args = natural(SceneSpeed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, THREE_SCENES);

    assert_eq!(scene_starts(&decisions), vec![0, 30, 60]);
}

#[test]
fn hard_cuts_start_scenes_at_ten_bits_in_yuv420() {
    let args = natural(SceneSpeed::Standard, 0, None);

    let decisions = decide(&yuv420_ten_bit(), &args, THREE_SCENES);

    assert_eq!(scene_starts(&decisions), vec![0, 30, 60]);
}

#[test]
fn the_first_frame_and_the_last_four_go_unscored() {
    let args = natural(SceneSpeed::Standard, 0, None);

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
    let args = natural(SceneSpeed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, THREE_SCENES);

    let before = decisions[29].score.expect("scored").get();
    let cut = decisions[30].score.expect("scored").get();
    assert!(cut > before, "cut {cut}, before {before}");
}

#[test]
fn a_minimum_scene_length_suppresses_a_close_cut() {
    let args = natural(SceneSpeed::Standard, 24, None);

    let decisions = decide(&grey8(), &args, &[(1, 30), (2, 10), (3, 30)]);

    assert_eq!(scene_starts(&decisions), vec![0, 30]);
}

#[test]
fn a_maximum_scene_length_forces_a_cut() {
    let args = natural(SceneSpeed::Standard, 0, Some(20));

    let decisions = decide(&grey8(), &args, &[(1, 50)]);

    assert_eq!(scene_starts(&decisions), vec![0, 20, 40]);
}

#[test]
fn a_single_frame_is_one_scene() {
    let args = natural(SceneSpeed::Standard, 0, None);

    let decisions = decide(&grey8(), &args, &[(1, 1)]);

    assert_eq!(decisions, vec![Decision::FIRST]);
}

#[test]
fn fast_speed_scores_frames() {
    let args = natural(SceneSpeed::Fast, 0, None);

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
    let args = natural(SceneSpeed::Standard, 0, None);
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
    let args = natural(SceneSpeed::Standard, 0, None);
    let format = grey8_of(96, 40);

    decide(&format, &args, THREE_SCENES);
}

fn to_float_luma(context: &FrameContext, format: &FrameFormat, frame: &HostFrame) -> HostFrame {
    let codes = frame.plane(Channel::Y).expect("the clip carries luma");

    let mut float_frame = context.alloc_host_frame(format, frame.meta().clone());
    let luma = float_frame
        .plane_mut(Channel::Y)
        .expect("the float frame carries luma");
    for (index, code) in codes.bytes().iter().enumerate() {
        let value = f32::from(*code) / 255.0;
        let at = index * 4;
        luma[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    float_frame.freeze()
}

#[test]
fn float_luma_finds_the_same_cuts_as_integer_luma() {
    let args = natural(SceneSpeed::Standard, 0, None);
    let context = testing_context();
    let integer_format = grey8();
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..grey8()
    };
    let integer_frames = clip(&context, &integer_format, THREE_SCENES);
    let float_frames: Vec<HostFrame> = integer_frames
        .iter()
        .map(|frame| to_float_luma(&context, &float_format, frame))
        .collect();

    let integer = decide_frames(&integer_format, &args, integer_frames);
    let float = decide_frames(&float_format, &args, float_frames);

    assert_eq!(scene_starts(&float), vec![0, 30, 60]);
    assert_eq!(scene_starts(&float), scene_starts(&integer));
}

#[test]
fn rgb_and_half_float_frames_are_refused() {
    let args = natural(SceneSpeed::Standard, 0, None);
    let rgb = FrameFormat {
        layout: FrameLayout::Rgb,
        ..grey8()
    };
    let half_float = FrameFormat {
        sample: SampleFormat::F16,
        ..grey8()
    };

    assert!(Settings::for_stream(&rgb, &args).is_err());
    assert!(Settings::for_stream(&half_float, &args).is_err());
}

#[test]
fn chroma_planes_are_refused() {
    let args = natural(SceneSpeed::Standard, 0, None);
    let chroma = FrameFormat {
        layout: FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        ..grey8()
    };

    assert!(Settings::for_stream(&chroma, &args).is_err());
}
