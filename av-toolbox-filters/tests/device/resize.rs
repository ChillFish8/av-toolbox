use av_toolbox_cubek::resize::{Kernel, Taps, taps};
use av_toolbox_cubek::testing::reference_resize_float;
use av_toolbox_gpu::{ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{
    ROUND_TRIP,
    assert_codes_within,
    float_bytes,
    float_samples,
    frame,
    planes_as_codes,
    run_once,
    widened,
};

fn planes_of(format: &FrameFormat, fill: impl Fn(usize, usize) -> u8) -> Vec<Vec<u8>> {
    format
        .channels()
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            (0..samples).map(|sample| fill(index, sample)).collect()
        })
        .collect()
}

fn resized_to(input: &FrameFormat, width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        width,
        height,
        ..*input
    }
}

/// The F32 frame an integer `input` resizes into.
fn resized_into_float(input: &FrameFormat, width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        sample: SampleFormat::F32,
        ..resized_to(input, width, height)
    }
}

/// Runs `stage` on one integer frame into F32, reading every output plane back as `bits` deep codes.
fn run_into_codes_at(
    stage: &str,
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
    bits: u32,
) -> Vec<Vec<f32>> {
    let found = run_once(stage, input, output, args, planes);

    planes_as_codes(&output, &found, bits)
}

/// Eight-bit samples as f32 code values.
fn codes_of(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample)).collect()
}

/// How far a resized code may stray from the reference, which adds its taps up in another order.
const SUMMED: f32 = 0.01;

/// Taps for one axis of a centre-aligned resize, for luma when `factor` is 1, or else for chroma
/// sited `siting` luma samples past `factor * i`.
fn axis_taps(start: f32, length: f32, output: u32, factor: u32, siting: f32, kernel: &Kernel) -> Taps {
    let scale = length / output as f32;
    let luma_position = start + (siting + 0.5) * scale - 0.5;
    let offset = (luma_position - siting) / factor as f32;
    let plane_output = output / factor;

    taps(plane_output, scale, offset, kernel)
}

/// Vertical stripes whose edges sit on the same luma columns in every plane of a 16x8 4:2:0 frame.
fn aligned_stripes(plane: usize, sample: usize) -> u8 {
    let bright = match plane {
        0 => sample % 8 >= 4,
        _ => sample % 4 >= 2,
    };

    if bright { 240 } else { 16 }
}

fn check_420_doubling(location: ChromaLocation, siting_x: f32) {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(16, 8, layout, SampleFormat::U8, location);
    let output = resized_into_float(&input, 32, 16);
    let planes = planes_of(&input, aligned_stripes);
    let args = json!({ "width": 32, "height": 16 });

    let found = run_into_codes_at("resize/bilinear", input, output, args, &planes, 8);

    let kernel = Kernel::Bilinear;
    let luma_codes = codes_of(&planes[0]);
    let chroma_codes = codes_of(&planes[1]);
    let luma_across = axis_taps(0.0, 16.0, 32, 1, 0.0, &kernel);
    let luma_down = axis_taps(0.0, 8.0, 16, 1, 0.0, &kernel);
    let luma = reference_resize_float(&luma_codes, 16, 8, &luma_across, &luma_down);
    let chroma_across = axis_taps(0.0, 16.0, 32, 2, siting_x, &kernel);
    let chroma_down = axis_taps(0.0, 8.0, 16, 2, 0.5, &kernel);
    let chroma = reference_resize_float(&chroma_codes, 8, 4, &chroma_across, &chroma_down);

    assert_codes_within(&found[0], &luma, SUMMED, "Y");
    assert_codes_within(&found[1], &chroma, SUMMED, "U");
    assert_codes_within(&found[2], &chroma, SUMMED, "V");
}

#[test]
fn a_left_sited_420_frame_keeps_its_chroma_siting_through_a_doubling() {
    check_420_doubling(ChromaLocation::Left, 0.0);
}

#[test]
fn a_centre_sited_420_frame_keeps_its_chroma_siting_through_a_doubling() {
    check_420_doubling(ChromaLocation::Center, 0.5);
}

#[test]
fn a_crop_window_picks_out_its_region() {
    let input = frame(8, 8, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let output = resized_into_float(&input, 4, 2);
    let planes = planes_of(&input, |_, sample| sample as u8);
    let args =
        json!({ "width": 4, "height": 2, "src_left": 2, "src_top": 4, "src_width": 4, "src_height": 2 });

    let found = run_into_codes_at("resize/point", input, output, args, &planes, 8);

    let expected = codes_of(&[34, 35, 36, 37, 42, 43, 44, 45]);
    assert_codes_within(&found[0], &expected, ROUND_TRIP, "Y");
}

#[test]
fn a_window_left_of_the_frame_reflects() {
    let input = frame(8, 4, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let output = resized_into_float(&input, 8, 4);
    let planes = planes_of(&input, |_, sample| (sample % 8 * 30) as u8);
    let args = json!({ "width": 8, "height": 4, "src_left": -2 });

    let found = run_into_codes_at("resize/point", input, output, args, &planes, 8);

    let row = codes_of(&[60, 30, 0, 30, 60, 90, 120, 150]);
    let expected: Vec<f32> = row.iter().copied().cycle().take(32).collect();
    assert_codes_within(&found[0], &expected, ROUND_TRIP, "Y");
}

#[test]
fn an_identity_resize_hands_back_the_input() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(8, 4, layout, SampleFormat::U8, ChromaLocation::Left);
    let planes = planes_of(&input, |plane, sample| (plane * 50 + sample) as u8);
    let args = json!({ "width": 8, "height": 4 });

    let found = run_once("resize/spline36", input, input, args, &planes);

    assert_eq!(found, planes);
}

#[test]
fn an_odd_sized_420_frame_resizes_every_plane() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(15, 9, layout, SampleFormat::U8, ChromaLocation::Left);
    let output = resized_into_float(&input, 16, 10);
    let planes = planes_of(&input, |plane, sample| (plane * 40 + sample * 3) as u8);
    let args = json!({ "width": 16, "height": 10 });

    let found = run_into_codes_at("resize/spline16", input, output, args, &planes, 8);

    let kernel = Kernel::Spline16;
    let luma_codes = codes_of(&planes[0]);
    let chroma_codes = codes_of(&planes[1]);
    let luma_across = axis_taps(0.0, 15.0, 16, 1, 0.0, &kernel);
    let luma_down = axis_taps(0.0, 9.0, 10, 1, 0.0, &kernel);
    let luma = reference_resize_float(&luma_codes, 15, 9, &luma_across, &luma_down);
    let chroma_across = axis_taps(0.0, 15.0, 16, 2, 0.0, &kernel);
    let chroma_down = axis_taps(0.0, 9.0, 10, 2, 0.5, &kernel);
    let chroma = reference_resize_float(&chroma_codes, 8, 5, &chroma_across, &chroma_down);

    assert_codes_within(&found[0], &luma, SUMMED, "Y");
    assert_codes_within(&found[1], &chroma, SUMMED, "U");
}

#[test]
fn a_resized_eight_bit_clip_lands_on_full_range() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(16, 8, layout, SampleFormat::U8, ChromaLocation::Left);
    let output = resized_into_float(&input, 32, 16);
    let planes = planes_of(&input, |plane, _| if plane == 0 { 235 } else { 128 });
    let args = json!({ "width": 32, "height": 16 });

    let found = run_once("resize/spline36", input, output, args, &planes);

    let luma = float_samples(&found[0]);
    let u_plane = float_samples(&found[1]);
    let v_plane = float_samples(&found[2]);
    assert!(luma.iter().all(|sample| (sample - 1.0).abs() < 1e-6), "{luma:?}");
    assert!(u_plane.iter().all(|sample| sample.abs() < 1e-6), "{u_plane:?}");
    assert!(v_plane.iter().all(|sample| sample.abs() < 1e-6), "{v_plane:?}");
}

#[test]
fn a_float_420_frame_resizes_with_its_chroma_centred() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(15, 9, layout, SampleFormat::F32, ChromaLocation::Left);
    let output = resized_to(&input, 32, 18);
    let luma: Vec<f32> = (0..15 * 9).map(|sample| (sample % 7) as f32 / 6.0).collect();
    let chroma: Vec<f32> = (0..8 * 5).map(|sample| (sample % 5) as f32 / 4.0 - 0.5).collect();
    let planes = [float_bytes(&luma), float_bytes(&chroma), float_bytes(&chroma)];
    let args = json!({ "width": 32, "height": 18 });

    let found = run_once("resize/lanczos", input, output, args, &planes);

    let kernel = Kernel::Lanczos { taps: 3 };
    let luma_across = axis_taps(0.0, 15.0, 32, 1, 0.0, &kernel);
    let luma_down = axis_taps(0.0, 9.0, 18, 1, 0.0, &kernel);
    let expected_luma = reference_resize_float(&luma, 15, 9, &luma_across, &luma_down);
    let chroma_across = axis_taps(0.0, 15.0, 32, 2, 0.0, &kernel);
    let chroma_down = axis_taps(0.0, 9.0, 18, 2, 0.5, &kernel);
    let expected_chroma = reference_resize_float(&chroma, 8, 5, &chroma_across, &chroma_down);

    let found_luma = float_samples(&found[0]);
    let found_u = float_samples(&found[1]);
    let found_v = float_samples(&found[2]);
    assert_codes_within(&found_luma, &expected_luma, 1e-5, "Y");
    assert_codes_within(&found_u, &expected_chroma, 1e-5, "U");
    assert_codes_within(&found_v, &expected_chroma, 1e-5, "V");
}

#[test]
fn a_display_sized_float_map_scales_back_to_the_clip() {
    let input = frame(
        384,
        216,
        FrameLayout::Grey,
        SampleFormat::F32,
        ChromaLocation::Left,
    );
    let output = resized_to(&input, 192, 108);
    let map: Vec<f32> = (0..384 * 216)
        .map(|sample| if sample % 384 < 192 { 0.0 } else { 0.004 })
        .collect();
    let planes = [float_bytes(&map)];
    let args = json!({ "width": 192, "height": 108 });

    let found = run_once("resize/bilinear", input, output, args, &planes);

    let kernel = Kernel::Bilinear;
    let across = axis_taps(0.0, 384.0, 192, 1, 0.0, &kernel);
    let down = axis_taps(0.0, 216.0, 108, 1, 0.0, &kernel);
    let expected = reference_resize_float(&map, 384, 216, &across, &down);

    let resized = float_samples(&found[0]);
    assert_codes_within(&resized, &expected, 1e-5, "Y");
}

#[test]
fn a_ten_bit_frame_resizes_at_its_depth() {
    let sample = SampleFormat::U16 { depth: Depth::TEN };
    let input = frame(8, 8, FrameLayout::Grey, sample, ChromaLocation::Left);
    let output = resized_into_float(&input, 20, 8);
    let luma: Vec<u16> = (0..64)
        .map(|index| if index % 8 >= 4 { 1023 } else { 0 })
        .collect();
    let bytes: Vec<u8> = luma.iter().flat_map(|value| value.to_le_bytes()).collect();
    let args = json!({ "width": 20, "height": 8, "taps": 4 });

    let found = run_into_codes_at("resize/lanczos", input, output, args, &[bytes], 10);

    let kernel = Kernel::Lanczos { taps: 4 };
    let codes = widened(&luma);
    let across = axis_taps(0.0, 8.0, 20, 1, 0.0, &kernel);
    let down = axis_taps(0.0, 8.0, 8, 1, 0.0, &kernel);
    let expected = reference_resize_float(&codes, 8, 8, &across, &down);

    let brightest = found[0].iter().copied().fold(0.0, f32::max);
    assert!(
        brightest > 1023.0,
        "the ringing past white is kept, found {brightest}"
    );
    assert_codes_within(&found[0], &expected, SUMMED, "Y");
}
