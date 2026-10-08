use av_toolbox_cubek::deblock::pp7::Pp7Mode;
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_cubek::testing::{Pp7Case, reference_pp7};
use av_toolbox_gpu::{ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use super::{centred_chroma, h264_planes, plane_samples, qed_layout, run_decoded, stored_as_f32};
use crate::support::frame;

const PP7_STAGE: &str = "deblock/pp7";

/// Every plane of a `format` frame deblocked by pp7 on the CPU, with `qp` and `mode`.
fn pp7_reference(
    format: &FrameFormat,
    planes: &[Vec<f64>],
    samples: MatrixSamples,
    qp: f64,
    mode: Pp7Mode,
) -> Vec<Vec<f64>> {
    let layout = qed_layout(format.layout);
    let case = Pp7Case {
        width: format.width,
        height: format.height,
        layout,
        samples,
        qp,
        mode,
    };

    reference_pp7(&case, planes)
}

#[test]
fn pp7_on_a_420_frame_matches_the_reference_in_every_mode() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(64, 48, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);
    let modes = [
        ("hard", Pp7Mode::Hard),
        ("soft", Pp7Mode::Soft),
        ("medium", Pp7Mode::Medium),
    ];

    for (name, mode) in modes {
        let args = json!({ "mode": name, "qp": 2.5 });

        let found = run_decoded(PP7_STAGE, input, samples, args, &planes);

        let expected = pp7_reference(&input, &planes, samples, 2.5, mode);
        assert_eq!(found, expected, "mode {name}");
        assert_ne!(found[0], planes[0], "mode {name}: luma came back unchanged");
    }
}

#[test]
fn pp7_leaves_planes_not_selected_unchanged() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(64, 48, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);
    let args = json!({ "planes": ["y"] });

    let found = run_decoded(PP7_STAGE, input, samples, args, &planes);

    let expected = pp7_reference(&input, &planes, samples, 2.0, Pp7Mode::Hard);
    assert_eq!(found[0], expected[0], "luma");
    assert_eq!(found[1], planes[1], "the unselected U plane changed");
    assert_eq!(found[2], planes[2], "the unselected V plane changed");
}

#[test]
fn pp7_returns_a_70x46_frame_at_its_own_size() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let input = frame(70, 46, layout, ten_bit, ChromaLocation::Left);
    let samples = MatrixSamples::U16 { bits: 10 };
    let planes = h264_planes(&input, samples);

    let found = run_decoded(PP7_STAGE, input, samples, json!({}), &planes);

    let expected = pp7_reference(&input, &planes, samples, 2.0, Pp7Mode::Hard);
    assert_eq!(found.len(), 3);
    for (index, plane) in found.iter().enumerate() {
        let (width, height) = plane_samples(&input, index);
        let label = format!("plane {index}");

        assert_eq!(
            plane.len(),
            (width * height) as usize,
            "{label} is {width}x{height}"
        );
        assert_eq!(*plane, expected[index], "{label}");
    }
}

#[test]
fn pp7_on_an_odd_420_frame_matches_the_reference() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(71, 45, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_decoded(PP7_STAGE, input, samples, json!({}), &planes);

    let expected = pp7_reference(&input, &planes, samples, 2.0, Pp7Mode::Hard);
    let chroma_size = plane_samples(&input, 1);
    assert_eq!(chroma_size, (36, 23));
    assert_eq!(found, expected);
}

#[test]
fn pp7_on_a_422_frame_aligns_chroma_by_8_across_and_16_down() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv422,
    };
    let input = frame(46, 22, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);
    let args = json!({ "mode": "medium", "qp": 4 });

    let found = run_decoded(PP7_STAGE, input, samples, args, &planes);

    let expected = pp7_reference(&input, &planes, samples, 4.0, Pp7Mode::Medium);
    let chroma_size = plane_samples(&input, 1);
    assert_eq!(chroma_size, (23, 22));
    assert_eq!(found, expected);
}

#[test]
fn pp7_on_a_16_bit_rgb_frame_deblocks_every_plane() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = frame(64, 48, FrameLayout::Rgb, sixteen_bit, ChromaLocation::Left);
    let samples = MatrixSamples::U16 { bits: 16 };
    let planes = h264_planes(&input, samples);
    let args = json!({ "mode": "soft" });

    let found = run_decoded(PP7_STAGE, input, samples, args, &planes);

    let expected = pp7_reference(&input, &planes, samples, 2.0, Pp7Mode::Soft);
    for index in 0..3 {
        let label = format!("rgb plane {index}");

        assert_eq!(found[index], expected[index], "{label}");
        assert_ne!(found[index], planes[index], "{label} came back unchanged");
    }
}

#[test]
fn pp7_on_a_float_frame_matches_the_reference_exactly() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = frame(70, 46, layout, SampleFormat::F32, ChromaLocation::Left);
    let samples = MatrixSamples::F32;
    let unshifted = h264_planes(&input, samples);
    let centred = centred_chroma(unshifted);
    let planes: Vec<Vec<f64>> = centred
        .iter()
        .map(|plane| stored_as_f32(plane, samples))
        .collect();
    let args = json!({ "qp": 13.37 });

    let found = run_decoded(PP7_STAGE, input, samples, args, &planes);

    let expected = pp7_reference(&input, &planes, samples, 13.37, Pp7Mode::Hard);
    assert_eq!(found, expected);
}

#[test]
fn pp7_on_a_frame_smaller_than_one_tile_comes_back_its_own_size() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(6, 4, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_decoded(PP7_STAGE, input, samples, json!({}), &planes);

    let expected = pp7_reference(&input, &planes, samples, 2.0, Pp7Mode::Hard);
    for (index, plane) in found.iter().enumerate() {
        let (width, height) = plane_samples(&input, index);

        assert_eq!(
            plane.len(),
            (width * height) as usize,
            "plane {index} is {width}x{height}"
        );
    }

    assert_eq!(found, expected);
}
