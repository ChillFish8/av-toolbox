use av_toolbox_core::FilterDefinition;
use av_toolbox_cubek::warp::BlurKind;
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

use crate::warp::Awarpsharp;
use crate::warp::awarpsharp::{AwarpsharpArgs, BlurType, warp_settings};

fn args(value: serde_json::Value) -> AwarpsharpArgs {
    serde_json::from_value(value).expect("parses")
}

fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 16,
        height: 8,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

fn yuv420() -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    format(layout, SampleFormat::U8)
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let input = yuv420();
    let error = warp_settings(&parsed, &input).expect_err("should be refused");

    error.to_string()
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
fn the_defaults_are_the_plugins() {
    let parsed = args(json!({}));

    assert_eq!(parsed.thresh, 128);
    assert_eq!(parsed.blur, None);
    assert_eq!(parsed.blur_type, BlurType::R6);
    assert_eq!(parsed.depth, 16);
    assert_eq!(parsed.planes, None);
}

#[test]
fn the_blur_type_parses_under_type() {
    let parsed = args(json!({ "type": "r2" }));

    assert_eq!(parsed.blur_type, BlurType::R2);
}

#[test]
fn the_default_blur_count_follows_the_type() {
    let input = yuv420();
    let wide_args = args(json!({}));
    let narrow_args = args(json!({ "type": "r2" }));
    let chosen_args = args(json!({ "type": "r2", "blur": 5 }));

    let wide = warp_settings(&wide_args, &input).expect("valid");
    let narrow = warp_settings(&narrow_args, &input).expect("valid");
    let chosen = warp_settings(&chosen_args, &input).expect("valid");

    assert_eq!((wide.params.blur, wide.params.passes), (BlurKind::R6, 2));
    assert_eq!((narrow.params.blur, narrow.params.passes), (BlurKind::R2, 3));
    assert_eq!(chosen.params.passes, 5);
}

#[test]
fn every_plane_is_warped_by_default() {
    let input = yuv420();
    let parsed = args(json!({}));

    let settings = warp_settings(&parsed, &input).expect("valid");

    assert_eq!(settings.selected, vec![true, true, true]);
}

#[test]
fn named_planes_limit_the_warp() {
    let input = yuv420();
    let parsed = args(json!({ "planes": ["y"] }));

    let settings = warp_settings(&parsed, &input).expect("valid");

    assert_eq!(settings.selected, vec![true, false, false]);
}

#[test]
fn out_of_range_arguments_are_refused() {
    let thresh = refusal(json!({ "thresh": 256 }));
    let blur = refusal(json!({ "blur": 101 }));
    let high = refusal(json!({ "depth": 128 }));
    let low = refusal(json!({ "depth": -129 }));

    assert!(thresh.contains("thresh") && thresh.contains("256"), "{thresh}");
    assert!(blur.contains("blur") && blur.contains("101"), "{blur}");
    assert!(high.contains("depth") && high.contains("128"), "{high}");
    assert!(low.contains("depth") && low.contains("-129"), "{low}");
}

#[test]
fn the_ten_bit_params_carry_the_depth() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let input = format(FrameLayout::Grey, ten_bit);
    let parsed = args(json!({ "depth": -20 }));

    let settings = warp_settings(&parsed, &input).expect("valid");

    assert_eq!(settings.params.bits, 10);
    assert_eq!(settings.params.depth, -20);
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
        let outputs = Awarpsharp::produces(&capabilities, &parsed, &chosen).expect("valid");
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
        Awarpsharp::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("frames are readable");
    let refused = Awarpsharp::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}
