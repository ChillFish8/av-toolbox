use av_toolbox_cubek::dehalo::alpha::AlphaParams;
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::testing::{reference_dehalo_alpha, reference_narrow};
use av_toolbox_gpu::{Channel, ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{frame, from_bytes, luma_is_shared, run_once, to_bytes};

fn alpha(bits: u32, ss: f32) -> AlphaParams {
    AlphaParams {
        rx: 2.0,
        ry: 2.0,
        darkstr: 1.0,
        brightstr: 1.0,
        lowsens: 50.0,
        highsens: 50.0,
        ss,
        bits,
    }
}

/// Every plane of `format`: a step with a halo on each row, shifted per plane, wrapped at `peak`.
fn haloed(format: &FrameFormat, peak: u32) -> Vec<Vec<u16>> {
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
                    let base = if column < width / 2 { 40 } else { 200 };
                    let ring = if column.abs_diff(width / 2) < 3 { 30 } else { 0 };
                    let value = (base + ring + row * 3 + plane as u32 * 7) * (peak + 1) / 256;
                    value.min(peak) as u16
                })
                .collect()
        })
        .collect()
}

fn assert_within_one(actual: &[u16], expected: &[u16], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");

    for (index, (found, wanted)) in actual.iter().zip(expected).enumerate() {
        assert!(
            found.abs_diff(*wanted) <= 1,
            "{label} sample {index}: {found} against {wanted}"
        );
    }
}

/// Runs `dehalo/alpha` and checks each plane against the reference or against its input.
fn matches_reference(input: FrameFormat, args: serde_json::Value, expected: AlphaParams, selected: &[bool]) {
    let peak = (1u32 << expected.bits) - 1;
    let element = if expected.bits == 8 {
        Element::U8
    } else {
        Element::U16
    };
    let values = haloed(&input, peak);
    let planes: Vec<Vec<u8>> = values.iter().map(|plane| to_bytes(plane, element)).collect();

    let dehaloed = run_once("dehalo/alpha", input, input, args, &planes);

    for (plane, channel) in input.channels().iter().enumerate() {
        let (width, height) = input.channel_dimensions(*channel).expect("carried");
        let actual = from_bytes(&dehaloed[plane], element);
        let label = channel.to_string();

        if !selected[plane] {
            assert_eq!(actual, values[plane], "{label} should be untouched");
            continue;
        }

        let widened: Vec<f32> = values[plane].iter().map(|value| f32::from(*value)).collect();
        let reference = reference_dehalo_alpha(&widened, width, height, &expected);
        let rounded = reference_narrow(&reference, peak as f32);
        assert_within_one(&actual, &rounded, &label);
    }
}

fn yuv420(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

#[test]
fn luma_is_dehaloed_by_default_at_8_and_10_bits() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let eight_bit_input = yuv420(64, 48, SampleFormat::U8);
    let ten_bit_input = yuv420(64, 48, ten_bit);
    let args = json!({});
    let selected = [true, false, false];

    matches_reference(eight_bit_input, args.clone(), alpha(8, 1.5), &selected);
    matches_reference(ten_bit_input, args, alpha(10, 1.5), &selected);
}

#[test]
fn sixteen_bit_matches_the_reference() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = frame(64, 48, FrameLayout::Grey, sixteen_bit, ChromaLocation::Left);
    let args = json!({});
    let selected = [true];

    matches_reference(input, args, alpha(16, 1.5), &selected);
}

#[test]
fn repair_mode_one_is_used_without_supersampling() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "ss": 1.0 });
    let selected = [true];

    matches_reference(input, args, alpha(8, 1.0), &selected);
}

#[test]
fn a_small_chroma_plane_is_dehaloed() {
    let input = yuv420(20, 12, SampleFormat::U8);
    let args = json!({ "planes": ["u"] });
    let selected = [false, true, false];

    matches_reference(input, args, alpha(8, 1.5), &selected);
}

#[test]
fn a_flat_frame_is_unchanged() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = frame(64, 48, FrameLayout::Grey, sixteen_bit, ChromaLocation::Left);
    let peak: u16 = 65535;
    let samples = input.channel_samples(Channel::Y).expect("a listed channel");
    let values: Vec<u16> = vec![peak; samples];
    let planes = vec![to_bytes(&values, Element::U16)];
    let args = json!({});

    let dehaloed = run_once("dehalo/alpha", input, input, args, &planes);
    let actual = from_bytes(&dehaloed[0], Element::U16);

    assert!(actual.iter().all(|sample| *sample == peak));
}

#[test]
fn unselected_chroma_is_shared() {
    let input = yuv420(64, 48, SampleFormat::U8);
    let args = json!({ "planes": ["u", "v"] });

    assert!(luma_is_shared("dehalo/alpha", input, input, args));
}
