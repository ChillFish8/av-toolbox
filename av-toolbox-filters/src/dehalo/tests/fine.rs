use av_toolbox_core::FilterDefinition;
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

use crate::dehalo::Fine;
use crate::dehalo::fine::{FineArgs, fine_params};

fn args(value: serde_json::Value) -> FineArgs {
    serde_json::from_value(value).expect("parses")
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let error = fine_params(&parsed, SampleFormat::U8).expect_err("should be refused");

    error.to_string()
}

fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 16,
        height: 8,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn yuv420() -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    format(layout, SampleFormat::U8)
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
    let params = fine_params(&parsed, SampleFormat::U8).expect("valid");

    assert_eq!((params.alpha.rx, params.alpha.ry), (2.0, 2.0));
    assert_eq!(
        (params.thmi, params.thma, params.thlimi, params.thlima),
        (80.0, 128.0, 50.0, 100.0)
    );
    assert_eq!((params.alpha.darkstr, params.alpha.brightstr), (1.0, 1.0));
    assert_eq!((params.contra, params.excl, params.edgeproc), (0.0, true, 0.0));
    assert_eq!(
        (params.alpha.lowsens, params.alpha.highsens, params.alpha.ss),
        (50.0, 50.0, 1.5)
    );
}

#[test]
fn ry_follows_rx_unless_given() {
    let following = args(json!({ "rx": 3.0 }));
    let separate = args(json!({ "rx": 3.0, "ry": 1.5 }));

    let followed = fine_params(&following, SampleFormat::U8).expect("valid");
    let given = fine_params(&separate, SampleFormat::U8).expect("valid");

    assert_eq!(followed.alpha.ry, 3.0);
    assert_eq!(given.alpha.ry, 1.5);
}

#[test]
fn out_of_range_and_crossed_thresholds_are_refused() {
    let cases = [
        (json!({ "rx": 0.5 }), "rx", "0.5"),
        (json!({ "ry": 0.5 }), "ry", "0.5"),
        (json!({ "thmi": 130.0 }), "thmi", "130"),
        (json!({ "thlimi": 100.0 }), "thlimi", "100"),
        (json!({ "thma": 256.0 }), "thma", "256"),
        (json!({ "contra": -1.0 }), "contra", "-1"),
        (json!({ "edgeproc": -0.5 }), "edgeproc", "-0.5"),
        (json!({ "darkstr": -1.0 }), "darkstr", "-1"),
    ];

    for (value, name, expected) in cases {
        let message = refusal(value);

        assert!(
            message.contains("dehalo/fine") && message.contains(name) && message.contains(expected),
            "{message}"
        );
    }
}

#[test]
fn a_threshold_below_zero_is_refused() {
    let message = refusal(json!({ "thmi": -1.0 }));

    assert!(
        message.contains("dehalo/fine") && message.contains("thmi"),
        "{message}"
    );
}

#[test]
fn thlima_above_255_is_refused() {
    let message = refusal(json!({ "thlima": 256.0 }));

    assert!(
        message.contains("dehalo/fine") && message.contains("thlima"),
        "{message}"
    );
}

#[test]
fn the_depth_comes_from_the_sample() {
    let parsed = args(json!({}));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };

    let params = fine_params(&parsed, ten_bit).expect("valid");

    assert_eq!(params.alpha.bits, 10);
}

#[test]
fn non_finite_arguments_are_refused() {
    let mut nan_rx = args(json!({}));
    nan_rx.rx = f32::NAN;
    let nan_error = fine_params(&nan_rx, SampleFormat::U8).expect_err("should be refused");
    let nan_message = nan_error.to_string();

    let mut infinite_edgeproc = args(json!({}));
    infinite_edgeproc.edgeproc = f32::INFINITY;
    let infinite_error = fine_params(&infinite_edgeproc, SampleFormat::U8).expect_err("should be refused");
    let infinite_message = infinite_error.to_string();

    assert!(
        nan_message.contains("dehalo/fine") && nan_message.contains("rx"),
        "{nan_message}"
    );
    assert!(
        infinite_message.contains("dehalo/fine") && infinite_message.contains("edgeproc"),
        "{infinite_message}"
    );
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
        yuv420(),
        format(FrameLayout::Grey, ten_bit),
        format(FrameLayout::Rgb, SampleFormat::U8),
        format(chroma, ten_bit),
    ];

    for input in inputs {
        let offered = StreamFormat::Frames(input);
        let chosen = [Some(offered)];
        let outputs = Fine::produces(&capabilities, &parsed, &chosen).expect("valid");
        let sample = outputs.samples.preferences()[0];
        let output = outputs.template.with_sample(sample);

        assert_eq!(output, offered);
    }
}

#[test]
fn a_plane_the_layout_lacks_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u"] }));
    let input = format(FrameLayout::Grey, SampleFormat::U8);
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Fine::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");

    assert!(error.to_string().contains("dehalo/fine"));
}

#[test]
fn a_plane_size_that_overflows_is_refused_through_produces() {
    let capabilities = capabilities();
    // rx 1 leaves the halo-free size at the plane's own width, and a width this close to `u32::MAX`
    // overflows once it is rounded up to a multiple of 4.
    let parsed = args(json!({ "rx": 1.0 }));
    let input = FrameFormat {
        width: u32::MAX,
        height: 8,
        layout: FrameLayout::Grey,
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    };
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Fine::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");

    assert!(error.to_string().contains("dehalo/fine"), "{error}");
}

#[test]
fn a_plane_wider_than_the_supersampled_limit_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let input = FrameFormat {
        width: 11000,
        height: 16,
        layout: FrameLayout::Grey,
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    };
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Fine::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");
    let message = error.to_string();

    assert!(
        message.contains("dehalo/fine") && message.contains("11000") && !message.contains("ss"),
        "{message}"
    );
}

#[test]
fn floats_are_left_out_and_masks_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..yuv420()
    };
    let float = StreamFormat::Frames(float_format);
    let mask = StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let accepted =
        Fine::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("frames are readable");
    let refused = Fine::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}
