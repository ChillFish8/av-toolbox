use av_toolbox_core::{ProviderDefinition, StageDefinition};
use av_toolbox_gpu::{FrameContext, FrameFormat, HostFrame, ResolvedFormats, StreamFormat, testing_context};

use super::{
    Chain,
    SCORE_LAYER,
    breaks_of,
    chunks_in,
    demands,
    finish,
    limited,
    process,
    score_demands,
    serve,
};
use crate::scene::score::SceneScoreProvider;
use crate::scene::window::{DetectOptions, LumaWindow, Settings};
use crate::scene::{SceneCut, SceneRequest, SceneScore, SceneSpeed};
use crate::split::provider::SceneSplitProvider;
use crate::split::rule::split_lengths;
use crate::tests::clip::{THREE_SCENES, clip, grey8};

/// A long scene with a short one before it, so the cut at frame 10 comes soon after the first.
const CLOSE_CUT: &[(u64, u64)] = &[(1, 10), (2, 30)];

fn serve_clip(requests: &[SceneRequest], scenes: &[(u64, u64)]) -> Vec<HostFrame> {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), scenes);

    serve(&mut context, &grey8(), requests, frames)
}

/// The chunk ids `split_lengths` gives each of `scenes`, counting on across them.
fn split_ids(frames: &[HostFrame], scenes: &[std::ops::Range<usize>], limit: usize) -> Vec<u64> {
    let mut ids = Vec::new();
    let mut next = 0;
    for scene in scenes {
        let scores: Vec<Option<SceneScore>> = frames[scene.clone()].iter().map(score_of).collect();

        for length in split_lengths(&scores, limit) {
            ids.extend(std::iter::repeat_n(next, length));
            next += 1;
        }
    }

    ids
}

/// The score the split read for `frame`.
fn score_of(frame: &HostFrame) -> Option<SceneScore> {
    let provided = &frame.meta().provided;
    let cut = provided
        .get::<SceneCut>(SCORE_LAYER)
        .expect("every frame is scored");

    cut.score
}

/// The scene starts a detector with its own minimum scene length finds on `scenes`.
fn detector_starts(min_scene_frames: usize, scenes: &[(u64, u64)]) -> Vec<usize> {
    let context = testing_context();
    let frames = clip(&context, &grey8(), scenes);
    let options = DetectOptions {
        speed: SceneSpeed::Standard,
        min_scene_frames,
        max_scene_frames: None,
    };
    let settings = Settings::for_stream(&grey8(), &options).expect("the clip is readable");
    let mut window = LumaWindow::new(settings);

    let mut decisions = Vec::new();
    for frame in &frames {
        let decided = window.push(frame).expect("the luma copies");
        decisions.extend(decided);
    }

    let rest = window.finish();
    decisions.extend(rest);

    decisions
        .iter()
        .enumerate()
        .filter(|(_, decision)| decision.starts_scene)
        .map(|(position, _)| position)
        .collect()
}

/// The most frames received but not yet emitted after any call, over a run of `scenes`.
fn peak_held(
    context: &mut FrameContext,
    format: &FrameFormat,
    requests: &[SceneRequest],
    scenes: &[(u64, u64)],
) -> usize {
    let frames = clip(context, format, scenes);
    let mut chain = Chain::new(requests, format);

    let mut received = 0;
    let mut emitted = 0;
    let mut peak = 0;
    for frame in frames {
        received += 1;
        emitted += process(&mut chain, context, frame).len();
        peak = peak.max(received - emitted);
    }

    emitted += finish(&mut chain, context).len();
    assert_eq!(emitted, received, "every frame is emitted by the end");

    peak
}

/// The frames the score and split providers together say they may hold.
fn held_frames(requests: &[SceneRequest]) -> usize {
    let args = demands(requests);
    let score_args = score_demands(&args);
    let input = StreamFormat::Frames(grey8());
    let formats = ResolvedFormats::new(vec![Some(input)], Some(input));

    let scored = SceneScoreProvider::held_frames(&score_args, &formats);
    let split = SceneSplitProvider::held_frames(&args, &formats);

    scored + split
}

#[test]
fn natural_layer_breaks_at_every_cut() {
    let emitted = serve_clip(&[SceneRequest::natural()], THREE_SCENES);

    let expected: Vec<u64> = (0..90).map(|index| index / 30).collect();
    assert_eq!(chunks_in(&emitted, 0), expected);
}

#[test]
fn a_cut_soon_after_the_last_is_kept_in_natural() {
    let emitted = serve_clip(&[SceneRequest::natural()], CLOSE_CUT);

    let chunks = chunks_in(&emitted, 0);
    assert_eq!(breaks_of(&chunks), vec![0, 10]);
}

#[test]
fn a_minimum_drops_close_cuts() {
    let emitted = serve_clip(&[limited(24, None)], CLOSE_CUT);

    assert_eq!(chunks_in(&emitted, 0), vec![0; 40]);
}

#[test]
fn the_minimum_matches_the_detector_own_minimum() {
    for scenes in [THREE_SCENES, CLOSE_CUT] {
        let emitted = serve_clip(&[limited(24, None)], scenes);

        let chunks = chunks_in(&emitted, 0);
        assert_eq!(breaks_of(&chunks), detector_starts(24, scenes), "{scenes:?}");
    }
}

#[test]
fn limited_splits_match_split_lengths() {
    let emitted = serve_clip(&[limited(0, Some(12))], THREE_SCENES);

    let expected = split_ids(&emitted, &[0..30, 30..60, 60..90], 12);
    assert_eq!(chunks_in(&emitted, 0), expected);
    assert!(expected.len() == 90 && expected[89] > 5, "{expected:?}");
}

#[test]
fn a_long_scene_is_closed_at_four_times_the_limit() {
    let emitted = serve_clip(&[limited(0, Some(10))], &[(1, 60)]);

    let expected = split_ids(&emitted, &[0..40, 40..60], 10);
    assert_eq!(chunks_in(&emitted, 0), expected);
}

#[test]
fn held_frames_cover_the_most_held() {
    let mut context = testing_context();

    let natural = [SceneRequest::natural()];
    let split = [limited(0, Some(12))];

    let natural_peak = peak_held(&mut context, &grey8(), &natural, THREE_SCENES);
    let split_peak = peak_held(&mut context, &grey8(), &split, &[(1, 60)]);

    assert_eq!(natural_peak, held_frames(&natural));
    assert_eq!(split_peak, held_frames(&split));
    assert_eq!((natural_peak, split_peak), (5, 53));
}

#[test]
fn layers_together_hold_no_more_than_the_longest_alone() {
    let mut context = testing_context();
    let requests = [limited(0, Some(5)), limited(0, Some(12)), SceneRequest::natural()];

    let peak = peak_held(&mut context, &grey8(), &requests, &[(1, 60)]);

    assert_eq!(peak, held_frames(&requests));
    assert_eq!(peak, 53);
}

#[test]
fn a_seconds_limit_without_a_frame_rate_is_refused() {
    let args = demands(&[SceneRequest::limited_to_seconds(10.0)]);

    let checked = SceneSplitProvider::check(&args, &grey8());

    let error = checked.expect_err("refused").to_string();
    assert!(error.contains("no frame rate"), "{error}");
    assert!(error.contains("max_frames"), "{error}");
}
