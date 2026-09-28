use av_toolbox_core::FilterDefinition;
use av_toolbox_cubek::rgvs::RepairMode;
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth as BitDepth,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Offered,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::dering::Hq;
use crate::dering::hq::{HqArgs, dering_params};

fn args(value: serde_json::Value) -> HqArgs {
    serde_json::from_value(value).expect("parses")
}

fn format(width: u32, height: u32, layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

fn yuv420(width: u32, height: u32) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    format(width, height, layout, SampleFormat::U8)
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let input = yuv420(16, 8);
    let error = dering_params(&parsed, &input).expect_err("should be refused");

    error.to_string()
}

fn refusal_of(parsed: &HqArgs, input: &FrameFormat) -> String {
    let error = dering_params(parsed, input).expect_err("should be refused");

    error.to_string()
}

/// Fully populated arguments, matching the defaults, for tests that build `HqArgs` directly
/// rather than through JSON, since JSON cannot carry NaN or infinity.
fn default_args() -> HqArgs {
    HqArgs {
        mrad: 1,
        msmooth: 1,
        incedge: false,
        mthr: 60.0,
        minp: 1,
        nrmode: None,
        sharp: 1,
        drrep: 24,
        thr: 12.0,
        elast: 2.0,
        darkthr: None,
        planes: None,
    }
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

#[test]
fn the_defaults_are_havsfuncs() {
    let parsed = args(json!({}));
    let input = yuv420(640, 360);
    let params = dering_params(&parsed, &input).expect("valid");

    assert_eq!(params.mrad, 1);
    assert_eq!(params.msmooth, 1);
    assert!(!params.incedge);
    assert_eq!(params.mthr, 60.0);
    assert_eq!(params.minp, 1);
    assert_eq!(params.nrmode, 1);
    assert_eq!(params.sharp, 1);
    assert_eq!(params.drrep, Some(RepairMode::TwentyFour));
    assert_eq!(params.thr, 12.0);
    assert_eq!(params.elast, 2.0);
    assert_eq!(params.darkthr, 3.0);
}

#[test]
fn darkthr_defaults_to_a_quarter_of_thr() {
    let parsed = args(json!({ "thr": 20.0 }));
    let input = yuv420(640, 360);
    let params = dering_params(&parsed, &input).expect("valid");

    assert_eq!(params.darkthr, 5.0);
}

#[test]
fn an_explicit_darkthr_wins() {
    let parsed = args(json!({ "thr": 20.0, "darkthr": 1.0 }));
    let input = yuv420(640, 360);
    let params = dering_params(&parsed, &input).expect("valid");

    assert_eq!(params.darkthr, 1.0);
}

#[test]
fn nrmode_defaults_from_the_frame_size() {
    let parsed = args(json!({}));
    let large = yuv420(1280, 720);
    let small = yuv420(1024, 576);

    let large_params = dering_params(&parsed, &large).expect("valid");
    let small_params = dering_params(&parsed, &small).expect("valid");

    assert_eq!(large_params.nrmode, 2);
    assert_eq!(small_params.nrmode, 1);
}

#[test]
fn an_explicit_nrmode_wins_over_the_frame_size_default() {
    let parsed = args(json!({ "nrmode": 3 }));
    let large = yuv420(1280, 720);
    let params = dering_params(&parsed, &large).expect("valid");

    assert_eq!(params.nrmode, 3);
}

#[test]
fn mrad_above_the_limit_is_refused() {
    let message = refusal(json!({ "mrad": 17 }));

    assert!(
        message.contains("dering/hq") && message.contains("mrad") && message.contains("17"),
        "{message}"
    );
}

#[test]
fn msmooth_above_the_limit_is_refused() {
    let message = refusal(json!({ "msmooth": 17 }));

    assert!(
        message.contains("dering/hq") && message.contains("msmooth") && message.contains("17"),
        "{message}"
    );
}

#[test]
fn mthr_above_the_limit_is_refused() {
    let message = refusal(json!({ "mthr": 256.0 }));

    assert!(
        message.contains("dering/hq") && message.contains("mthr") && message.contains("256"),
        "{message}"
    );
}

#[test]
fn minp_above_the_limit_is_refused() {
    let message = refusal(json!({ "minp": 5 }));

    assert!(
        message.contains("dering/hq") && message.contains("minp") && message.contains("5"),
        "{message}"
    );
}

#[test]
fn nrmode_zero_is_refused() {
    let message = refusal(json!({ "nrmode": 0 }));

    assert!(
        message.contains("dering/hq") && message.contains("nrmode") && message.contains("0"),
        "{message}"
    );
}

#[test]
fn nrmode_four_is_refused() {
    let message = refusal(json!({ "nrmode": 4 }));

    assert!(
        message.contains("dering/hq") && message.contains("nrmode") && message.contains("4"),
        "{message}"
    );
}

#[test]
fn sharp_above_the_limit_is_refused() {
    let message = refusal(json!({ "sharp": 4 }));

    assert!(
        message.contains("dering/hq") && message.contains("sharp") && message.contains("4"),
        "{message}"
    );
}

#[test]
fn an_unsupported_drrep_is_refused() {
    let message = refusal(json!({ "drrep": 3 }));

    assert!(
        message.contains("dering/hq") && message.contains("drrep") && message.contains("3"),
        "{message}"
    );
}

#[test]
fn a_negative_thr_is_refused() {
    let message = refusal(json!({ "thr": -1.0 }));

    assert!(
        message.contains("dering/hq") && message.contains("thr") && message.contains("-1"),
        "{message}"
    );
}

#[test]
fn an_elast_below_one_is_refused() {
    let message = refusal(json!({ "elast": 0.5 }));

    assert!(
        message.contains("dering/hq") && message.contains("elast") && message.contains("0.5"),
        "{message}"
    );
}

#[test]
fn a_negative_darkthr_is_refused() {
    let message = refusal(json!({ "darkthr": -1.0 }));

    assert!(
        message.contains("dering/hq") && message.contains("darkthr") && message.contains("-1"),
        "{message}"
    );
}

#[test]
fn a_nan_thr_is_refused() {
    let parsed = HqArgs {
        thr: f32::NAN,
        ..default_args()
    };
    let input = yuv420(16, 8);
    let message = refusal_of(&parsed, &input);

    assert!(
        message.contains("dering/hq") && message.contains("thr") && message.contains("NaN"),
        "{message}"
    );
}

#[test]
fn an_infinite_elast_is_refused() {
    let parsed = HqArgs {
        elast: f32::INFINITY,
        ..default_args()
    };
    let input = yuv420(16, 8);
    let message = refusal_of(&parsed, &input);

    assert!(
        message.contains("dering/hq") && message.contains("elast") && message.contains("inf"),
        "{message}"
    );
}

#[test]
fn drrep_zero_gives_no_repair_mode() {
    let parsed = args(json!({ "drrep": 0 }));
    let input = yuv420(640, 360);
    let params = dering_params(&parsed, &input).expect("valid");

    assert_eq!(params.drrep, None);
}

#[test]
fn every_supported_drrep_is_accepted() {
    let cases = [
        (1, RepairMode::One),
        (2, RepairMode::Two),
        (13, RepairMode::Thirteen),
        (17, RepairMode::Seventeen),
        (24, RepairMode::TwentyFour),
    ];

    for (number, mode) in cases {
        let parsed = args(json!({ "drrep": number }));
        let input = yuv420(640, 360);
        let params = dering_params(&parsed, &input).expect("valid");

        assert_eq!(params.drrep, Some(mode));
    }
}

#[test]
fn the_output_format_is_the_input_format() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let inputs = [
        yuv420(16, 8),
        format(16, 8, FrameLayout::Grey, ten_bit),
        format(16, 8, chroma, ten_bit),
    ];

    for input in inputs {
        let offered = StreamFormat::Frames(input);
        let chosen = [Some(offered)];
        let outputs = Hq::produces(&capabilities, &parsed, &chosen).expect("valid");
        let sample = outputs.samples.preferences()[0];
        let output = outputs.template.with_sample(sample);

        assert_eq!(output, offered);
    }
}

#[test]
fn floats_are_left_out_and_masks_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..yuv420(16, 8)
    };
    let float = StreamFormat::Frames(float_format);
    let mask = StreamFormat::Mask(MaskFormat {
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let accepted = Hq::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("frames are readable");
    let refused = Hq::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}

#[test]
fn a_plane_the_layout_lacks_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u"] }));
    let input = format(16, 8, FrameLayout::Grey, SampleFormat::U8);
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Hq::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");

    assert!(error.to_string().contains("dering/hq"));
}
