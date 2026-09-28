use av_toolbox_cubek::dehalo::dering::{DeringParams, ring_mask_sample};
use av_toolbox_cubek::dehalo::edge_clean::threshold_sample;
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::neighbourhood::{Extreme, Shape, Spread};
use av_toolbox_cubek::rgvs::RepairMode;
use av_toolbox_cubek::testing::{
    reference_extreme,
    reference_hq_dering,
    reference_hysteresis,
    reference_median,
    reference_narrow,
    reference_prewitt,
    reference_spread,
};
use av_toolbox_gpu::{Channel, ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{frame, from_bytes, luma_is_shared, run_once, to_bytes};

fn yuv420(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

fn dering(bits: u32) -> DeringParams {
    DeringParams {
        mrad: 1,
        msmooth: 1,
        incedge: false,
        mthr: 60.0,
        minp: 1,
        nrmode: 1,
        sharp: 1,
        drrep: Some(RepairMode::TwentyFour),
        thr: 12.0,
        darkthr: 3.0,
        elast: 2.0,
        bits,
    }
}

/// A step edge with alternating ringing beside it, plus a gentle ramp down each plane, shifted
/// per plane and clamped to `peak`.
fn ringing(format: &FrameFormat, peak: u32) -> Vec<Vec<u16>> {
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
                    let middle = width / 2;
                    let base: i32 = if column < middle { 60 } else { 180 };
                    let distance = column.abs_diff(middle);

                    let mut ring = 0i32;
                    if distance > 0 && distance <= 6 {
                        let magnitude = 24 - (distance as i32 - 1) * 3;
                        ring = if distance % 2 == 0 { magnitude } else { -magnitude };
                    }

                    let ramp = (row % 16) as i32;
                    let shifted = base + ring + ramp + plane as i32 * 5;
                    let scaled = shifted * (peak as i32 + 1) / 256;

                    scaled.clamp(0, peak as i32) as u16
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

/// Runs `dering/hq` and checks each plane against the reference or against its input.
fn matches_dering_reference(
    input: FrameFormat,
    args: serde_json::Value,
    expected: DeringParams,
    selected: &[bool],
) {
    let peak = (1u32 << expected.bits) - 1;
    let element = if expected.bits == 8 {
        Element::U8
    } else {
        Element::U16
    };
    let values = ringing(&input, peak);
    let planes: Vec<Vec<u8>> = values.iter().map(|plane| to_bytes(plane, element)).collect();

    let derung = run_once("dering/hq", input, input, args, &planes);

    for (plane, channel) in input.channels().iter().enumerate() {
        let (width, height) = input.channel_dimensions(*channel).expect("carried");
        let actual = from_bytes(&derung[plane], element);
        let label = channel.to_string();

        if !selected[plane] {
            assert_eq!(actual, values[plane], "{label} should be untouched");
            continue;
        }

        let widened: Vec<f32> = values[plane].iter().map(|value| f32::from(*value)).collect();
        let reference = reference_hq_dering(&widened, width, height, &expected);
        let rounded = reference_narrow(&reference, peak as f32);
        assert_within_one(&actual, &rounded, &label);
    }
}

#[test]
fn defaults_match_the_reference_at_8_and_10_bits() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let eight_bit_input = yuv420(64, 48, SampleFormat::U8);
    let ten_bit_input = yuv420(64, 48, ten_bit);
    let args = json!({});
    let selected = [true, false, false];

    matches_dering_reference(eight_bit_input, args.clone(), dering(8), &selected);
    matches_dering_reference(ten_bit_input, args, dering(10), &selected);
}

#[test]
fn sixteen_bit_grey_matches_the_reference() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = frame(64, 48, FrameLayout::Grey, sixteen_bit, ChromaLocation::Left);
    let args = json!({});
    let selected = [true];

    matches_dering_reference(input, args, dering(16), &selected);
}

#[test]
fn each_sharp_matches_the_reference() {
    for sharp in [0u32, 2, 3] {
        let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
        let args = json!({ "sharp": sharp });
        let selected = [true];
        let expected = DeringParams { sharp, ..dering(8) };

        matches_dering_reference(input, args, expected, &selected);
    }
}

#[test]
fn each_minp_matches_the_reference() {
    for minp in [0u32, 2, 3, 4] {
        let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
        let args = json!({ "minp": minp });
        let selected = [true];
        let expected = DeringParams { minp, ..dering(8) };

        matches_dering_reference(input, args, expected, &selected);
    }
}

#[test]
fn incedge_on_matches_the_reference() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "incedge": true });
    let selected = [true];
    let expected = DeringParams {
        incedge: true,
        ..dering(8)
    };

    matches_dering_reference(input, args, expected, &selected);
}

#[test]
fn mrad_zero_with_msmooth_zero_matches_the_reference() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "mrad": 0, "msmooth": 0 });
    let selected = [true];
    let expected = DeringParams {
        mrad: 0,
        msmooth: 0,
        ..dering(8)
    };

    matches_dering_reference(input, args, expected, &selected);
}

#[test]
fn each_drrep_matches_the_reference() {
    let cases = [
        (0, None),
        (1, Some(RepairMode::One)),
        (2, Some(RepairMode::Two)),
        (13, Some(RepairMode::Thirteen)),
        (17, Some(RepairMode::Seventeen)),
    ];

    for (number, mode) in cases {
        let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
        let args = json!({ "drrep": number });
        let selected = [true];
        let expected = DeringParams {
            drrep: mode,
            ..dering(8)
        };

        matches_dering_reference(input, args, expected, &selected);
    }
}

#[test]
fn nrmode_three_matches_the_reference() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "nrmode": 3 });
    let selected = [true];
    let expected = DeringParams {
        nrmode: 3,
        ..dering(8)
    };

    matches_dering_reference(input, args, expected, &selected);
}

#[test]
fn a_one_wide_plane_matches_the_reference() {
    let input = frame(1, 6, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({});
    let selected = [true];

    matches_dering_reference(input, args, dering(8), &selected);
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

    let derung = run_once("dering/hq", input, input, args, &planes);
    let actual = from_bytes(&derung[0], Element::U16);

    assert!(actual.iter().all(|sample| *sample == peak));
}

#[test]
fn unselected_chroma_is_shared() {
    let input = yuv420(64, 48, SampleFormat::U8);
    let args = json!({ "planes": ["u", "v"] });

    assert!(luma_is_shared("dering/hq", input, input, args));
}

/// The ring mask havsfunc's `HQDeringmod` merges with, computed on the CPU the way
/// [reference_hq_dering] builds it internally.
fn ring_mask_values(input: &[f32], width: u32, height: u32, params: &DeringParams) -> Vec<f32> {
    let peak = params.peak();
    let edges = reference_prewitt(input, width, height, peak);
    let floor = params.scale(params.mthr);
    let pm: Vec<f32> = edges
        .iter()
        .map(|edge| threshold_sample(*edge, floor, f32::MAX, peak, false))
        .collect();
    let pm_median = reference_median(&pm, width, height, 1);
    let fmask = reference_hysteresis(&pm_median, &pm, width, height, peak);

    let mut outer = fmask.clone();
    for _ in 0..params.mrad {
        outer = reference_extreme(&outer, width, height, Extreme::Maximum, Shape::Square);
    }
    for _ in 0..params.msmooth {
        outer = reference_spread(&outer, width, height, Spread::Inflate);
    }

    let inner = match params.minp {
        0 => fmask.clone(),
        1 => {
            let inflated = reference_spread(&fmask, width, height, Spread::Inflate);

            reference_extreme(&inflated, width, height, Extreme::Minimum, Shape::Square)
        },
        2 => reference_extreme(&fmask, width, height, Extreme::Minimum, Shape::Square),
        3 => {
            let inflated = reference_spread(&fmask, width, height, Spread::Inflate);
            let once = reference_extreme(&inflated, width, height, Extreme::Minimum, Shape::Square);

            reference_extreme(&once, width, height, Extreme::Minimum, Shape::Square)
        },
        4 => {
            let once = reference_extreme(&fmask, width, height, Extreme::Minimum, Shape::Square);

            reference_extreme(&once, width, height, Extreme::Minimum, Shape::Square)
        },
        _ => unreachable!("minp is bounded to 0..=4 by dering_params"),
    };

    (0..input.len())
        .map(|index| ring_mask_sample(outer[index], inner[index], params.incedge, peak))
        .collect()
}

#[test]
fn the_default_ring_mask_neither_stays_zero_nor_fires_everywhere() {
    let width = 64;
    let height = 48;
    let input = frame(
        width,
        height,
        FrameLayout::Grey,
        SampleFormat::U8,
        ChromaLocation::Left,
    );
    let peak = 255u32;
    let values = ringing(&input, peak);
    let widened: Vec<f32> = values[0].iter().map(|value| f32::from(*value)).collect();
    let params = dering(8);

    let ring_mask = ring_mask_values(&widened, width, height, &params);

    assert!(
        ring_mask.iter().any(|value| *value > 0.0),
        "the ring mask never fires"
    );
    assert!(ring_mask.contains(&0.0), "the ring mask is set everywhere");
}
