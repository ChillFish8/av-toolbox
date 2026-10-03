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

use crate::dehalo::Yahr;
use crate::dehalo::yahr::{YahrArgs, yahr_params};

fn args(value: serde_json::Value) -> YahrArgs {
    serde_json::from_value(value).expect("parses")
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let error = yahr_params(&parsed, SampleFormat::U8).expect_err("should be refused");

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
    let params = yahr_params(&parsed, SampleFormat::U8).expect("valid");

    assert_eq!(params.blur, 2);
    assert_eq!(params.depth, 32);
    assert_eq!(parsed.planes, None);
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (json!({ "blur": 101 }), "blur", "101"),
        (json!({ "depth": 128 }), "depth", "128"),
        (json!({ "depth": -129 }), "depth", "-129"),
    ];

    for (value, name, expected) in cases {
        let message = refusal(value);

        assert!(
            message.contains("dehalo/yahr") && message.contains(name) && message.contains(expected),
            "{message}"
        );
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
        yuv420(),
        format(FrameLayout::Grey, ten_bit),
        format(chroma, ten_bit),
    ];

    for input in inputs {
        let offered = StreamFormat::Frames(input);
        let chosen = [Some(offered)];
        let outputs = Yahr::produces(&capabilities, &parsed, &chosen).expect("valid");
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
        ..yuv420()
    };
    let float = StreamFormat::Frames(float_format);
    let mask = StreamFormat::Mask(MaskFormat {
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let accepted =
        Yahr::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("frames are readable");
    let refused = Yahr::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}

#[test]
fn a_plane_the_layout_lacks_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u"] }));
    let input = format(FrameLayout::Grey, SampleFormat::U8);
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Yahr::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");

    assert!(error.to_string().contains("dehalo/yahr"));
}
