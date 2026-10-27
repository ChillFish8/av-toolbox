use av_toolbox_cubek::dehalo::dering::DeringParams;
use av_toolbox_cubek::rgvs::RepairMode;
use av_toolbox_cubek::testing::{reference_hq_dering, reference_narrow, reference_ring_mask};
use av_toolbox_gpu::{Channel, ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{
    assert_a_flat_frame_is_unchanged,
    assert_planes_match,
    f32_parity,
    frame,
    luma_is_shared,
};

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

/// Runs `dering/hq` and checks each plane against the reference or against its input.
fn matches_dering_reference(
    input: FrameFormat,
    args: serde_json::Value,
    expected: DeringParams,
    selected: &[bool],
) {
    let peak = (1u32 << expected.bits) - 1;
    let values = ringing(&input, peak);
    let reference = |codes: &[f32], width, height| reference_hq_dering(codes, width, height, &expected);

    assert_planes_match(
        "dering/hq",
        input,
        args,
        expected.bits,
        selected,
        &values,
        reference,
    );
}

/// The fewest luma samples the reference must change on [ringing]'s 64x48 fixture with the default
/// `dering/hq` parameters, so a no-op reference could not pass. The fixture actually changes 88 of
/// its 3072 samples, so this leaves a wide margin.
const MINIMUM_CHANGED_SAMPLES: usize = 50;

/// Asserts that [reference_hq_dering] changes a real number of luma samples on `input`, rather than
/// leaving it untouched.
fn assert_a_real_number_of_samples_changed(input: FrameFormat, params: &DeringParams) {
    let peak = (1u32 << params.bits) - 1;
    let values = ringing(&input, peak);
    let (width, height) = input.channel_dimensions(Channel::Y).expect("a listed channel");
    let widened: Vec<f32> = values[0].iter().map(|value| f32::from(*value)).collect();

    let reference = reference_hq_dering(&widened, width, height, params);
    let rounded = reference_narrow(&reference, peak as f32);
    let changed = rounded
        .iter()
        .zip(&values[0])
        .filter(|(after, before)| **after != **before)
        .count();

    assert!(
        changed >= MINIMUM_CHANGED_SAMPLES,
        "only {changed} of {} samples changed",
        width * height
    );
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
    assert_a_real_number_of_samples_changed(eight_bit_input, &dering(8));
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
    assert_a_flat_frame_is_unchanged("dering/hq");
}

#[test]
fn f32_matches_sixteen_bit() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = yuv420(64, 48, sixteen_bit);
    let values = ringing(&input, 65535);

    let parity = f32_parity("dering/hq", input, json!({}), &values);

    assert!(parity.largest <= 1.0, "{parity:?}");
}

#[test]
fn unselected_chroma_is_shared() {
    let input = yuv420(64, 48, SampleFormat::F32);
    let args = json!({ "planes": ["u", "v"] });

    assert!(luma_is_shared("dering/hq", input, input, args));
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

    let ring_mask = reference_ring_mask(&widened, width, height, &params);

    assert!(
        ring_mask.iter().any(|value| *value > 0.0),
        "the ring mask never fires"
    );
    assert!(ring_mask.contains(&0.0), "the ring mask is set everywhere");
}
