use crate::{
    Butteraugli,
    ChromaLocation,
    ColorSpace,
    Planes,
    Range,
    SampleType,
    Ssimulacra2,
    device_count,
    device_info,
};

const SIZE: usize = 256;

fn gpu_id() -> i32 {
    let selector = std::env::var("AVTB_DEVICE").unwrap_or_default();
    let (kind, ordinal) = selector.split_once(':').unwrap_or((selector.as_str(), "0"));
    let wants_integrated = match kind {
        "discrete" => false,
        "integrated" => true,
        _ => return 0,
    };

    let ordinal: usize = ordinal.parse().expect("AVTB_DEVICE has a numeric ordinal");

    let count = device_count().expect("counts devices");
    let matching: Vec<i32> = (0..count)
        .filter(|&candidate| {
            let info = device_info(candidate).expect("describes a device");
            info.integrated == wants_integrated
        })
        .collect();

    *matching
        .get(ordinal)
        .expect("AVTB_DEVICE names a device Vship lists")
}

fn color_space() -> ColorSpace {
    ColorSpace {
        sample: SampleType::Uint8,
        width: SIZE as i64,
        height: SIZE as i64,
        range: Range::Limited,
        sub_sampling_w: 1,
        sub_sampling_h: 1,
        chroma_location: ChromaLocation::Left,
        matrix: 1,
        transfer: 1,
        primaries: 1,
    }
}

/// A smooth gradient with fine detail, so every metric has structure to compare.
fn reference_frame() -> [Vec<u8>; 3] {
    let mut y_plane = vec![0u8; SIZE * SIZE];
    for row in 0..SIZE {
        for column in 0..SIZE {
            let gradient = 16 + (row + column) * 200 / (2 * SIZE);
            let detail = if (row / 4 + column / 4) % 2 == 0 { 10 } else { 0 };
            y_plane[row * SIZE + column] = (gradient + detail) as u8;
        }
    }

    let chroma_plane = vec![128u8; (SIZE / 2) * (SIZE / 2)];

    [y_plane, chroma_plane.clone(), chroma_plane]
}

/// The reference with its luma detail blurred away.
fn distorted_frame() -> [Vec<u8>; 3] {
    let [mut y_plane, u_plane, v_plane] = reference_frame();
    for row in 0..SIZE {
        for column in 0..SIZE {
            let gradient = 16 + (row + column) * 200 / (2 * SIZE);
            y_plane[row * SIZE + column] = (gradient + 5) as u8;
        }
    }

    [y_plane, u_plane, v_plane]
}

fn planes(frame: &[Vec<u8>; 3]) -> Planes<'_> {
    Planes {
        data: [&frame[0], &frame[1], &frame[2]],
        strides: [SIZE, SIZE / 2, SIZE / 2],
    }
}

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
