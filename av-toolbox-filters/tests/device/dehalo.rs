use av_toolbox_cubek::dehalo::alpha::AlphaParams;
use av_toolbox_cubek::dehalo::edge_clean::EdgeCleanParams;
use av_toolbox_cubek::dehalo::fine::FineParams;
use av_toolbox_cubek::dehalo::yahr::YahrParams;
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::rgvs::RepairMode;
use av_toolbox_cubek::testing::{
    reference_dehalo_alpha,
    reference_edge_clean,
    reference_fine_dehalo,
    reference_narrow,
    reference_yahr,
};
use av_toolbox_gpu::{
    Channel,
    ChromaLocation,
    ColourRange,
    Depth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    Subsampling,
};
use serde_json::json;

use crate::support::{
    assert_a_flat_frame_is_unchanged,
    assert_codes_within,
    assert_planes_match,
    encoded,
    f32_parity,
    float_samples,
    frame,
    luma_is_shared,
    run_into_codes,
    run_once,
    widened,
};

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

/// A step with a halo on each row of `format`, shifted per plane and clamped to `peak`.
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
                    let value = (base + ring + (row % 8) * 3 + plane as u32 * 7) * (peak + 1) / 256;
                    value.min(peak) as u16
                })
                .collect()
        })
        .collect()
}

/// Runs `stage` on a haloed `input` of `bits` deep codes, and checks each selected plane against
/// `reference` and the rest against the input.
fn matches(
    stage: &str,
    input: FrameFormat,
    args: serde_json::Value,
    bits: u32,
    selected: &[bool],
    reference: impl Fn(&[f32], u32, u32) -> Vec<f32>,
) {
    let peak = (1u32 << bits) - 1;
    let values = haloed(&input, peak);

    assert_planes_match(stage, input, args, bits, selected, &values, reference);
}

/// Checks `stage`'s F32 and 16-bit runs on one haloed clip differ by at most `largest`, with no more
/// than a `beyond_one` share of samples more than one code apart.
fn assert_f32_matches_sixteen_bit(stage: &str, largest: f32, beyond_one: f32) {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = yuv420(64, 48, sixteen_bit);
    let values = haloed(&input, 65535);

    let parity = f32_parity(stage, input, json!({}), &values);

    assert!(parity.largest <= largest, "{stage}: {parity:?}");
    assert!(parity.beyond_one <= beyond_one, "{stage}: {parity:?}");
}

/// Runs `dehalo/alpha` and checks each plane against the reference or against its input.
fn matches_reference(input: FrameFormat, args: serde_json::Value, expected: AlphaParams, selected: &[bool]) {
    let reference = |codes: &[f32], width, height| reference_dehalo_alpha(codes, width, height, &expected);

    matches("dehalo/alpha", input, args, expected.bits, selected, reference);
}

fn fine(bits: u32) -> FineParams {
    FineParams {
        alpha: alpha(bits, 1.5),
        thmi: 80.0,
        thma: 128.0,
        thlimi: 50.0,
        thlima: 100.0,
        contra: 0.0,
        excl: true,
        edgeproc: 0.0,
    }
}

/// Runs `dehalo/fine` and checks each plane against the reference or against its input.
fn matches_fine_reference(
    input: FrameFormat,
    args: serde_json::Value,
    expected: FineParams,
    selected: &[bool],
) {
    let reference = |codes: &[f32], width, height| reference_fine_dehalo(codes, width, height, &expected);

    matches(
        "dehalo/fine",
        input,
        args,
        expected.alpha.bits,
        selected,
        reference,
    );
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
    assert_a_flat_frame_is_unchanged("dehalo/alpha");
}

#[test]
fn unselected_chroma_is_shared() {
    let input = yuv420(64, 48, SampleFormat::F32);
    let args = json!({ "planes": ["u", "v"] });

    assert!(luma_is_shared("dehalo/alpha", input, input, args));
}

#[test]
fn alpha_f32_matches_sixteen_bit() {
    assert_f32_matches_sixteen_bit("dehalo/alpha", 1.0, 0.0);
}

#[test]
fn an_unfiltered_chroma_plane_becomes_f32_centred_on_zero() {
    let mut input = yuv420(64, 48, SampleFormat::U8);
    input.colour.range = ColourRange::Limited;
    let haloed_planes = haloed(&input, 255);
    let flat_chroma = vec![128u16; haloed_planes[1].len()];
    let values = [haloed_planes[0].clone(), flat_chroma.clone(), flat_chroma];
    let planes = encoded(&input, &values);
    let output = FrameFormat {
        sample: SampleFormat::F32,
        ..input
    };

    let found = run_once("dehalo/alpha", input, output, json!({}), &planes);

    for bytes in &found[1..] {
        let samples = float_samples(bytes);
        assert!(samples.iter().all(|sample| sample.abs() <= 1e-6), "{samples:?}");
    }
}

#[test]
fn eight_bit_through_f32_rounds_back_to_todays_output() {
    let input = yuv420(64, 48, SampleFormat::U8);
    let (width, height) = input.channel_dimensions(Channel::Y).expect("a listed channel");
    let values = haloed(&input, 255);
    let planes = encoded(&input, &values);
    let params = alpha(8, 1.5);

    let found = run_into_codes("dehalo/alpha", input, json!({}), &planes, 8);

    let rounded: Vec<f32> = found[0].iter().map(|code| code.round()).collect();
    let codes = widened(&values[0]);
    let reference = reference_dehalo_alpha(&codes, width, height, &params);
    let narrowed = reference_narrow(&reference, 255.0);
    let todays = widened(&narrowed);

    assert_codes_within(&rounded, &todays, 1.0, "Y");
}

#[test]
fn fine_dehalos_luma_by_default_at_8_and_10_bits() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let eight_bit_input = yuv420(64, 48, SampleFormat::U8);
    let ten_bit_input = yuv420(64, 48, ten_bit);
    let args = json!({});
    let selected = [true, false, false];

    matches_fine_reference(eight_bit_input, args.clone(), fine(8), &selected);
    matches_fine_reference(ten_bit_input, args, fine(10), &selected);
}

#[test]
fn fine_contrasharpening_and_edge_processing_match() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "contra": 1.0, "edgeproc": 0.5, "excl": false });
    let selected = [true];
    let expected = FineParams {
        contra: 1.0,
        edgeproc: 0.5,
        excl: false,
        ..fine(8)
    };

    matches_fine_reference(input, args, expected, &selected);
}

#[test]
fn fine_rounds_a_small_radius_to_one_pass() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "rx": 1.4 });
    let selected = [true];
    let alpha = AlphaParams {
        rx: 1.4,
        ry: 1.4,
        ..alpha(8, 1.5)
    };
    let expected = FineParams { alpha, ..fine(8) };

    matches_fine_reference(input, args, expected, &selected);
}

#[test]
fn fine_unselected_chroma_is_shared() {
    let input = yuv420(64, 48, SampleFormat::F32);
    let args = json!({ "planes": ["u", "v"] });

    assert!(luma_is_shared("dehalo/fine", input, input, args));
}

#[test]
fn fine_asymmetric_radii_reach_the_horizontal_passes() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "rx": 3.0, "ry": 1.0 });
    let selected = [true];
    let alpha_params = AlphaParams {
        rx: 3.0,
        ry: 1.0,
        ..alpha(8, 1.5)
    };
    let expected = FineParams {
        alpha: alpha_params,
        ..fine(8)
    };

    matches_fine_reference(input, args, expected, &selected);
}

#[test]
fn fine_sixteen_bit_matches_the_reference() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = frame(64, 48, FrameLayout::Grey, sixteen_bit, ChromaLocation::Left);
    let args = json!({});
    let selected = [true];

    matches_fine_reference(input, args, fine(16), &selected);
}

#[test]
fn fine_leaves_a_flat_frame_unchanged() {
    assert_a_flat_frame_is_unchanged("dehalo/fine");
}

#[test]
fn fine_f32_matches_sixteen_bit() {
    assert_f32_matches_sixteen_bit("dehalo/fine", 1.0, 0.0);
}

fn yahr(bits: u32) -> YahrParams {
    let element = if bits == 8 { Element::U8 } else { Element::U16 };

    YahrParams {
        blur: 2,
        depth: 32,
        element,
        bits,
    }
}

/// Runs `dehalo/yahr` and checks each plane against the reference or against its input.
fn matches_yahr_reference(
    input: FrameFormat,
    args: serde_json::Value,
    expected: YahrParams,
    selected: &[bool],
) {
    let reference = |codes: &[f32], width, height| reference_yahr(codes, width, height, &expected);

    matches("dehalo/yahr", input, args, expected.bits, selected, reference);
}

#[test]
fn yahr_f32_matches_sixteen_bit() {
    assert_f32_matches_sixteen_bit("dehalo/yahr", 1.0, 0.0);
}

#[test]
fn yahr_luma_is_cleaned_by_default_at_8_and_10_bits() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let eight_bit_input = yuv420(64, 48, SampleFormat::U8);
    let ten_bit_input = yuv420(64, 48, ten_bit);
    let args = json!({});
    let selected = [true, false, false];

    matches_yahr_reference(eight_bit_input, args.clone(), yahr(8), &selected);
    matches_yahr_reference(ten_bit_input, args, yahr(10), &selected);
}

#[test]
fn yahr_sixteen_bit_matches_the_reference() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = frame(64, 48, FrameLayout::Grey, sixteen_bit, ChromaLocation::Left);
    let args = json!({});
    let selected = [true];

    matches_yahr_reference(input, args, yahr(16), &selected);
}

#[test]
fn yahr_custom_blur_and_negative_depth_match() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "blur": 0, "depth": -20 });
    let selected = [true];
    let expected = YahrParams {
        blur: 0,
        depth: -20,
        ..yahr(8)
    };

    matches_yahr_reference(input, args, expected, &selected);
}

#[test]
fn yahr_a_tiny_plane_matches_the_reference() {
    let input = frame(2, 5, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({});
    let selected = [true];

    matches_yahr_reference(input, args, yahr(8), &selected);
}

#[test]
fn yahr_a_one_wide_plane_matches_the_reference() {
    let input = frame(1, 6, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({});
    let selected = [true];

    matches_yahr_reference(input, args, yahr(8), &selected);
}

#[test]
fn yahr_a_flat_frame_is_unchanged() {
    assert_a_flat_frame_is_unchanged("dehalo/yahr");
}

#[test]
fn yahr_unselected_chroma_is_shared() {
    let input = yuv420(64, 48, SampleFormat::F32);
    let args = json!({ "planes": ["u", "v"] });

    assert!(luma_is_shared("dehalo/yahr", input, input, args));
}

fn edge_clean(bits: u32) -> EdgeCleanParams {
    let element = if bits == 8 { Element::U8 } else { Element::U16 };

    EdgeCleanParams {
        strength: 10,
        rep: true,
        rmode: RepairMode::Seventeen,
        smode: false,
        hot: false,
        element,
        bits,
    }
}

/// Runs `dehalo/edge_clean` and checks each plane against the reference or against its input.
fn matches_edge_clean_reference(
    input: FrameFormat,
    args: serde_json::Value,
    expected: EdgeCleanParams,
    selected: &[bool],
) {
    let reference = |codes: &[f32], width, height| reference_edge_clean(codes, width, height, &expected);

    matches(
        "dehalo/edge_clean",
        input,
        args,
        expected.bits,
        selected,
        reference,
    );
}

#[test]
fn edge_clean_luma_is_cleaned_by_default_at_8_and_10_bits() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let eight_bit_input = yuv420(64, 48, SampleFormat::U8);
    let ten_bit_input = yuv420(64, 48, ten_bit);
    let args = json!({});
    let selected = [true, false, false];

    matches_edge_clean_reference(eight_bit_input, args.clone(), edge_clean(8), &selected);
    matches_edge_clean_reference(ten_bit_input, args, edge_clean(10), &selected);
}

/// `smode` and `hot` really move the result, so a comparison run with them on cannot pass by
/// accident against one that never took either path.
fn assert_smode_and_hot_change_the_result(input: &FrameFormat, expected: &EdgeCleanParams) {
    let peak = (1u32 << expected.bits) - 1;
    let (width, height) = input.channel_dimensions(Channel::Y).expect("a listed channel");
    let values = haloed(input, peak);
    let widened: Vec<f32> = values[0].iter().map(|value| f32::from(*value)).collect();

    let with_flags = reference_edge_clean(&widened, width, height, expected);
    let bare = EdgeCleanParams {
        smode: false,
        hot: false,
        ..*expected
    };
    let without_flags = reference_edge_clean(&widened, width, height, &bare);

    assert_ne!(with_flags, without_flags, "smode and hot made no difference");
}

#[test]
fn edge_clean_smode_and_hot_match_at_8_and_16_bits() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let eight_bit_input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let sixteen_bit_input = frame(64, 48, FrameLayout::Grey, sixteen_bit, ChromaLocation::Left);
    let args = json!({ "smode": true, "hot": true });
    let selected = [true];
    let expected_eight = EdgeCleanParams {
        smode: true,
        hot: true,
        ..edge_clean(8)
    };
    let expected_sixteen = EdgeCleanParams {
        smode: true,
        hot: true,
        ..edge_clean(16)
    };

    assert_smode_and_hot_change_the_result(&eight_bit_input, &expected_eight);
    assert_smode_and_hot_change_the_result(&sixteen_bit_input, &expected_sixteen);

    matches_edge_clean_reference(eight_bit_input, args.clone(), expected_eight, &selected);
    matches_edge_clean_reference(sixteen_bit_input, args, expected_sixteen, &selected);
}

#[test]
fn edge_clean_hot_alone_writes_straight_into_the_output() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "hot": true });
    let selected = [true];
    let expected = EdgeCleanParams {
        hot: true,
        ..edge_clean(8)
    };

    matches_edge_clean_reference(input, args, expected, &selected);
}

#[test]
fn edge_clean_a_tiny_plane_with_smode_and_hot_matches_the_reference() {
    let input = frame(1, 6, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "smode": true, "hot": true });
    let selected = [true];
    let expected = EdgeCleanParams {
        smode: true,
        hot: true,
        ..edge_clean(8)
    };

    matches_edge_clean_reference(input, args, expected, &selected);
}

#[test]
fn edge_clean_without_rep_matches_the_reference() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "rep": false });
    let selected = [true];
    let expected = EdgeCleanParams {
        rep: false,
        ..edge_clean(8)
    };

    matches_edge_clean_reference(input, args, expected, &selected);
}

#[test]
fn edge_clean_every_supported_repair_mode_matches() {
    let cases = [
        (1, RepairMode::One),
        (2, RepairMode::Two),
        (13, RepairMode::Thirteen),
        (24, RepairMode::TwentyFour),
    ];

    for (number, mode) in cases {
        let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
        let args = json!({ "rmode": number });
        let selected = [true];
        let expected = EdgeCleanParams {
            rmode: mode,
            ..edge_clean(8)
        };

        matches_edge_clean_reference(input, args, expected, &selected);
    }
}

#[test]
fn edge_clean_high_strength_with_smode_matches() {
    let input = frame(64, 48, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "strength": 250, "smode": true });
    let selected = [true];
    let expected = EdgeCleanParams {
        strength: 250,
        smode: true,
        ..edge_clean(8)
    };

    matches_edge_clean_reference(input, args, expected, &selected);
}

#[test]
fn edge_clean_a_flat_frame_is_unchanged() {
    assert_a_flat_frame_is_unchanged("dehalo/edge_clean");
}

#[test]
fn edge_clean_unselected_chroma_is_shared() {
    let input = yuv420(64, 48, SampleFormat::F32);
    let args = json!({ "planes": ["u", "v"] });

    assert!(luma_is_shared("dehalo/edge_clean", input, input, args));
}

#[test]
fn edge_clean_f32_matches_sixteen_bit() {
    assert_f32_matches_sixteen_bit("dehalo/edge_clean", 1.0, 0.0);
}
