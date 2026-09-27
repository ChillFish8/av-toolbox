use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::testing::reference_warp_sharp;
use av_toolbox_cubek::warp::{BlurKind, WarpParams};
use av_toolbox_gpu::{ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{frame, from_bytes, luma_is_shared, run_once, to_bytes};

fn yuv420(sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(48, 32, layout, sample, ChromaLocation::Left)
}

/// Every plane of `format`, a busy pattern wrapped at `peak`, as u16 values.
fn patterned(format: &FrameFormat, peak: u32) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let column = index % width;
                    let row = index / width;
                    let value = column * 37 + row * 11 + (column * row) % 17 + plane as u32 * 29;
                    (value % (peak + 1)) as u16
                })
                .collect()
        })
        .collect()
}

fn as_bytes(values: &[Vec<u16>], element: Element) -> Vec<Vec<u8>> {
    values.iter().map(|plane| to_bytes(plane, element)).collect()
}

/// Runs `warp/awarpsharp` on `input` and checks each plane against the reference, or against the
/// input where the plane is not selected.
fn matches_reference(input: FrameFormat, args: serde_json::Value, expected: WarpParams, selected: &[bool]) {
    let peak = expected.peak() as u32;
    let values = patterned(&input, peak);
    let planes = as_bytes(&values, expected.element);

    let warped = run_once("warp/awarpsharp", input, input, args, &planes);

    for (plane, channel) in input.channels().iter().enumerate() {
        let (width, height) = input.channel_dimensions(*channel).expect("carried");
        let actual = from_bytes(&warped[plane], expected.element);
        let label = channel.to_string();

        if selected[plane] {
            let reference = reference_warp_sharp(&values[plane], width, height, &expected);
            assert_eq!(actual, reference, "{label}");
        } else {
            assert_eq!(actual, values[plane], "{label} should be untouched");
        }
    }
}

fn default_params(element: Element, bits: u32) -> WarpParams {
    WarpParams {
        element,
        bits,
        thresh: 128,
        blur: BlurKind::R6,
        passes: 2,
        depth: 16,
    }
}

#[test]
fn eight_and_ten_bit_clips_match_the_reference() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let eight_bit_input = yuv420(SampleFormat::U8);
    let ten_bit_input = yuv420(ten_bit);
    let eight_bit_params = default_params(Element::U8, 8);
    let ten_bit_params = default_params(Element::U16, 10);
    let every = [true, true, true];

    matches_reference(eight_bit_input, json!({}), eight_bit_params, &every);
    matches_reference(ten_bit_input, json!({}), ten_bit_params, &every);
}

#[test]
fn the_r2_blur_runs_three_passes_by_default() {
    let input = frame(48, 32, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let expected = WarpParams {
        blur: BlurKind::R2,
        passes: 3,
        ..default_params(Element::U8, 8)
    };

    matches_reference(input, json!({ "type": "r2" }), expected, &[true]);
}

#[test]
fn only_the_named_rgb_plane_changes() {
    let input = frame(48, 32, FrameLayout::Rgb, SampleFormat::U8, ChromaLocation::Left);
    let expected = default_params(Element::U8, 8);
    let args = json!({ "planes": ["g"] });

    matches_reference(input, args, expected, &[false, true, false]);
}

#[test]
fn an_unselected_luma_plane_is_shared() {
    let input = yuv420(SampleFormat::U8);
    let args = json!({ "planes": ["u", "v"] });

    assert!(luma_is_shared("warp/awarpsharp", input, input, args));
}
