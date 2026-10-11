use crate::tests::support::{SIZE, color_space, distorted_frame, gpu_id, planes, reference_frame};
use crate::{Cvvdp, CvvdpOptions, DisplayModel, Planes};

const FRAMES: usize = 5;

fn options(display: DisplayModel) -> CvvdpOptions {
    CvvdpOptions {
        fps: 25.0,
        resize_to_display: false,
        display,
    }
}

fn cvvdp_with(display: DisplayModel) -> Cvvdp {
    let reference_space = color_space();
    let distorted_space = color_space();
    let options = options(display);
    let gpu_id = gpu_id();

    Cvvdp::new(reference_space, distorted_space, &options, gpu_id).expect("makes the metric")
}

fn cvvdp() -> Cvvdp {
    cvvdp_with(DisplayModel::StandardFhd)
}

/// Scores the same frame pair `count` times, returning the last running score.
fn score_frames(metric: &mut Cvvdp, reference: &Planes, distorted: &Planes, count: usize) -> f64 {
    let mut score = 0.0;
    for _ in 0..count {
        score = metric.compute(reference, distorted).expect("scores");
    }

    score
}

#[test]
fn identical_frames_score_near_10_on_cvvdp() {
    let reference = reference_frame();
    let reference_planes = planes(&reference);
    let mut metric = cvvdp();

    let score = score_frames(&mut metric, &reference_planes, &reference_planes, FRAMES);

    assert!(score > 9.5, "{score}");
}

#[test]
fn distorted_frames_score_lower_on_cvvdp() {
    let reference = reference_frame();
    let distorted = distorted_frame();
    let reference_planes = planes(&reference);
    let distorted_planes = planes(&distorted);
    let mut identical_metric = cvvdp();
    let mut distorted_metric = cvvdp();

    let identical = score_frames(
        &mut identical_metric,
        &reference_planes,
        &reference_planes,
        FRAMES,
    );
    let degraded = score_frames(
        &mut distorted_metric,
        &reference_planes,
        &distorted_planes,
        FRAMES,
    );

    assert!(
        degraded < identical,
        "distorted {degraded}, identical {identical}"
    );
}

#[test]
fn reset_and_reset_score_make_a_used_metric_score_like_a_fresh_one() {
    let reference = reference_frame();
    let distorted = distorted_frame();
    let reference_planes = planes(&reference);
    let distorted_planes = planes(&distorted);
    let mut fresh = cvvdp();
    let mut used = cvvdp();

    let expected = score_frames(&mut fresh, &reference_planes, &distorted_planes, FRAMES);

    score_frames(&mut used, &reference_planes, &reference_planes, FRAMES);
    used.reset().expect("resets the history");
    used.reset_score().expect("resets the score");
    let actual = score_frames(&mut used, &reference_planes, &distorted_planes, FRAMES);

    assert!(
        (actual - expected).abs() < 1e-3,
        "used {actual}, fresh {expected}"
    );
}

#[test]
fn reset_score_alone_restarts_the_running_score() {
    let reference = reference_frame();
    let distorted = distorted_frame();
    let reference_planes = planes(&reference);
    let distorted_planes = planes(&distorted);
    let mut metric = cvvdp();

    let degraded = score_frames(&mut metric, &reference_planes, &distorted_planes, FRAMES);
    metric.reset_score().expect("resets the score");
    let restarted = score_frames(&mut metric, &reference_planes, &reference_planes, FRAMES);

    assert!(
        restarted > degraded,
        "restarted {restarted}, distorted {degraded}"
    );
}

#[test]
fn loaded_history_frames_are_not_scored() {
    let reference = reference_frame();
    let reference_planes = planes(&reference);
    let mut metric = cvvdp();

    for _ in 0..FRAMES {
        metric
            .load_temporal(&reference_planes, &reference_planes)
            .expect("loads a history frame");
    }

    let score = metric
        .compute(&reference_planes, &reference_planes)
        .expect("scores");

    assert!(score > 9.5, "{score}");
}

#[test]
fn a_cvvdp_map_matches_the_frame_size_and_is_near_zero_for_identical_frames() {
    let reference = reference_frame();
    let reference_planes = planes(&reference);
    let mut metric = cvvdp();
    let mut map = metric.distortion_map();

    metric
        .compute_with_map(&reference_planes, &reference_planes, &mut map)
        .expect("scores with a map");

    let peak = map.data().iter().copied().fold(0.0f32, f32::max);
    assert_eq!((map.width(), map.height()), (SIZE, SIZE));
    assert_eq!(map.data().len(), map.width() * map.height());
    assert!(peak < 0.01, "{peak}");
}

#[test]
fn a_custom_display_initialises_and_scores() {
    let reference = reference_frame();
    let reference_planes = planes(&reference);
    let display = DisplayModel::Custom {
        name: "standard_fhd".to_string(),
        json: r#"{"standard_fhd": {"name": "a dimmer standard_fhd", "max_luminance": 100.0}}"#.to_string(),
    };
    let mut metric = cvvdp_with(display);

    let score = metric
        .compute(&reference_planes, &reference_planes)
        .expect("scores");

    assert!(score.is_finite(), "{score}");
}
