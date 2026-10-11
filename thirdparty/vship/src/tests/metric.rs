use crate::tests::support::{
    SIZE,
    butteraugli,
    color_space,
    distorted_frame,
    gpu_id,
    planes,
    reference_frame,
};
use crate::{Butteraugli, Ssimulacra2, device_count, device_info};

#[test]
fn vship_lists_a_named_device() {
    let count = device_count().expect("counts devices");
    let gpu_id = gpu_id();
    let info = device_info(gpu_id).expect("describes the test device");

    assert!(count > 0);
    assert!(!info.name.is_empty());
}

#[test]
fn identical_frames_score_near_100_on_ssimulacra2() {
    let reference = reference_frame();
    let reference_planes = planes(&reference);
    let gpu_id = gpu_id();
    let mut metric = Ssimulacra2::new(color_space(), color_space(), gpu_id).expect("makes the metric");

    let score = metric
        .compute(&reference_planes, &reference_planes)
        .expect("scores");

    assert!(score > 99.5, "{score}");
}

#[test]
fn a_distorted_frame_scores_lower_on_ssimulacra2() {
    let reference = reference_frame();
    let distorted = distorted_frame();
    let reference_planes = planes(&reference);
    let distorted_planes = planes(&distorted);
    let gpu_id = gpu_id();
    let mut metric = Ssimulacra2::new(color_space(), color_space(), gpu_id).expect("makes the metric");

    let score = metric
        .compute(&reference_planes, &distorted_planes)
        .expect("scores");

    assert!(score < 99.0, "{score}");
}

#[test]
fn identical_frames_score_0_on_butteraugli() {
    let reference = reference_frame();
    let reference_planes = planes(&reference);
    let gpu_id = gpu_id();
    let mut metric =
        Butteraugli::new(color_space(), color_space(), 2, 203.0, gpu_id).expect("makes the metric");

    let score = metric
        .compute(&reference_planes, &reference_planes)
        .expect("scores");

    assert!(score.norm_inf.abs() < 1e-6, "{score:?}");
}

#[test]
fn a_distorted_frame_scores_higher_on_butteraugli() {
    let reference = reference_frame();
    let distorted = distorted_frame();
    let reference_planes = planes(&reference);
    let distorted_planes = planes(&distorted);
    let gpu_id = gpu_id();
    let mut metric =
        Butteraugli::new(color_space(), color_space(), 2, 203.0, gpu_id).expect("makes the metric");

    let score = metric
        .compute(&reference_planes, &distorted_planes)
        .expect("scores");

    assert!(score.norm_inf > 0.1, "{score:?}");
    assert!(score.norm_q > 0.0, "{score:?}");
}

#[test]
fn a_device_past_the_last_is_an_error() {
    let count = device_count().expect("counts devices");

    let error = Ssimulacra2::new(color_space(), color_space(), count).expect_err("a missing device worked");

    assert_ne!(error.code(), 0);
    assert!(!error.message().is_empty());
}

#[test]
fn planes_that_do_not_fit_are_refused_before_vship() {
    let reference = reference_frame();
    let short = [
        reference[0][..10].to_vec(),
        reference[1].clone(),
        reference[2].clone(),
    ];
    let reference_planes = planes(&reference);
    let short_planes = planes(&short);
    let gpu_id = gpu_id();
    let mut metric = Ssimulacra2::new(color_space(), color_space(), gpu_id).expect("makes the metric");

    let error = metric
        .compute(&reference_planes, &short_planes)
        .expect_err("short planes scored");

    assert_eq!(error.code(), 0);
    assert!(error.message().contains("the distorted frame plane 0"), "{error}");
}

#[test]
fn a_butteraugli_map_matches_the_frame_size() {
    let metric = butteraugli();

    let map = metric.distortion_map();

    assert_eq!((map.width(), map.height()), (SIZE, SIZE));
    assert_eq!(map.data().len(), SIZE * SIZE);
}

#[test]
fn identical_frames_give_an_all_zero_butteraugli_map() {
    let reference = reference_frame();
    let reference_planes = planes(&reference);
    let mut metric = butteraugli();
    let mut map = metric.distortion_map();

    metric
        .compute_with_map(&reference_planes, &reference_planes, &mut map)
        .expect("scores with a map");

    let peak = map.data().iter().copied().fold(0.0f32, f32::max);
    assert!(peak.abs() < 1e-6, "{peak}");
}

#[test]
fn a_distorted_butteraugli_map_peaks_at_norm_inf() {
    let reference = reference_frame();
    let distorted = distorted_frame();
    let reference_planes = planes(&reference);
    let distorted_planes = planes(&distorted);
    let mut metric = butteraugli();
    let mut map = metric.distortion_map();

    let score = metric
        .compute_with_map(&reference_planes, &distorted_planes, &mut map)
        .expect("scores with a map");

    let peak = map.data().iter().copied().fold(0.0f32, f32::max);
    let peak = f64::from(peak);
    assert!(peak > 0.0, "{peak}");
    assert!(
        (peak - score.norm_inf).abs() < 1e-3,
        "peak {peak}, norm_inf {}",
        score.norm_inf
    );
}

#[test]
fn computing_with_a_map_scores_like_computing_without() {
    let reference = reference_frame();
    let distorted = distorted_frame();
    let reference_planes = planes(&reference);
    let distorted_planes = planes(&distorted);
    let mut metric = butteraugli();
    let mut map = metric.distortion_map();

    let plain = metric
        .compute(&reference_planes, &distorted_planes)
        .expect("scores");
    let mapped = metric
        .compute_with_map(&reference_planes, &distorted_planes, &mut map)
        .expect("scores with a map");

    assert!(
        (plain.norm_q - mapped.norm_q).abs() < 1e-9,
        "{plain:?} {mapped:?}"
    );
    assert!(
        (plain.norm_inf - mapped.norm_inf).abs() < 1e-9,
        "{plain:?} {mapped:?}"
    );
}
