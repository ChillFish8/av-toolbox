use av_toolbox_core::{FilterDefinition, StageError};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    ColourRange,
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

use crate::resample::Depth;
use crate::resample::depth::{DepthArgs, Dither};

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn frames(layout: FrameLayout, sample: SampleFormat) -> StreamFormat {
    StreamFormat::Frames(FrameFormat {
        width: 16,
        height: 8,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    })
}

fn yuv420(sample: SampleFormat) -> StreamFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frames(layout, sample)
}

fn args(value: serde_json::Value) -> Result<DepthArgs, serde_json::Error> {
    serde_json::from_value(value)
}

fn produced(input: StreamFormat, value: serde_json::Value) -> Result<StreamFormat, StageError> {
    let capabilities = capabilities();
    let arguments = args(value).expect("parses");
    let chosen = [Some(input)];
    let outputs = Depth::produces(&capabilities, &arguments, &chosen)?;
    let sample = outputs.samples.preferences()[0];

    Ok(outputs.template.with_sample(sample))
}

#[test]
fn dither_defaults_to_blue_noise() {
    let parsed = args(json!({ "depth": 8 })).expect("parses");

    assert_eq!(parsed.dither, Dither::BlueNoise);
}

#[test]
fn every_dither_name_parses() {
    for (name, dither) in [
        ("none", Dither::None),
        ("ordered", Dither::Ordered),
        ("blue_noise", Dither::BlueNoise),
    ] {
        let parsed = args(json!({ "depth": 8, "dither": name })).expect("parses");

        assert_eq!(parsed.dither, dither);
    }
}

#[test]
fn an_unknown_argument_is_refused() {
    let refused = args(json!({ "depth": 8, "bits": 8 }));

    assert!(refused.is_err());
}

#[test]
fn ten_bits_gives_a_ten_bit_u16_and_keeps_everything_else() {
    let input = yuv420(SampleFormat::U8);
    let output = produced(input, json!({ "depth": 10 })).expect("converts");
    let converted = output.as_frames().expect("frames");
    let original = input.as_frames().expect("frames");

    assert_eq!(converted.sample, SampleFormat::U16 { depth: BitDepth::TEN });
    assert_eq!(converted.layout, original.layout);
    assert_eq!(converted.colour, original.colour);
}

#[test]
fn eight_bits_gives_u8() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let input = frames(FrameLayout::Grey, ten_bit);
    let output = produced(input, json!({ "depth": 8 })).expect("converts");

    assert_eq!(output.sample(), SampleFormat::U8);
}

#[test]
fn a_chroma_clip_is_converted() {
    let layout = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let input = frames(layout, SampleFormat::U8);

    let output = produced(input, json!({ "depth": 12 })).expect("converts");

    assert_eq!(output.as_frames().expect("frames").layout, layout);
}

#[test]
fn a_depth_outside_eight_to_sixteen_is_refused() {
    for depth in [0, 7, 17] {
        let input = yuv420(SampleFormat::U8);
        let error = produced(input, json!({ "depth": depth })).expect_err("refused");

        assert!(error.to_string().contains("between 8 and 16"), "{error}");
    }
}

#[test]
fn floats_are_accepted_and_masks_refused() {
    let capabilities = capabilities();
    let arguments = args(json!({ "depth": 10 })).expect("parses");
    let inputs = [Offered::Unknown];
    let float = yuv420(SampleFormat::F32);
    let mask = StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let accepted =
        Depth::accepts_input(&capabilities, &arguments, 0, float, &inputs).expect("frames are readable");
    let refused = Depth::accepts_input(&capabilities, &arguments, 0, mask, &inputs);

    assert!(accepted.accepts(SampleFormat::F32));
    assert!(refused.is_err());
}

#[test]
fn a_float_grey_clip_becomes_eight_bit() {
    let input = frames(FrameLayout::Grey, SampleFormat::F32);
    let output = produced(input, json!({ "depth": 8, "dither": "none" })).expect("converts");
    let converted = output.as_frames().expect("frames");

    assert_eq!(converted.sample, SampleFormat::U8);
    assert_eq!(converted.layout, FrameLayout::Grey);
}

fn with_range(format: StreamFormat, range: ColourRange) -> StreamFormat {
    let frames = format.as_frames().expect("frames");
    let colour = ColourInfo {
        range,
        ..frames.colour
    };

    StreamFormat::Frames(FrameFormat { colour, ..*frames })
}

fn offered(input: StreamFormat, value: serde_json::Value) -> Result<Vec<SampleFormat>, StageError> {
    let capabilities = capabilities();
    let arguments = args(value).expect("parses");
    let chosen = [Some(input)];
    let outputs = Depth::produces(&capabilities, &arguments, &chosen)?;

    Ok(outputs.samples.preferences().to_vec())
}

#[test]
fn depth_accepts_f32_as_a_target() {
    let input = yuv420(SampleFormat::U8);
    let samples = offered(input, json!({ "depth": "f32" })).expect("converts");

    assert_eq!(samples, [SampleFormat::F32]);
}

#[test]
fn at_least_offers_every_deeper_integer_narrowest_first() {
    let input = yuv420(SampleFormat::U8);
    let samples = offered(input, json!({ "depth": 10, "at_least": true })).expect("converts");

    let expected: Vec<SampleFormat> = (10..=16)
        .map(|bits| {
            let depth = BitDepth::new(bits).expect("valid");
            SampleFormat::U16 { depth }
        })
        .collect();

    assert_eq!(samples, expected);
}

#[test]
fn at_least_with_a_float_target_is_refused() {
    let input = yuv420(SampleFormat::U8);
    let refused = offered(input, json!({ "depth": "f32", "at_least": true }));

    assert!(refused.is_err());
}

#[test]
fn a_float_output_keeps_the_input_range_label() {
    let narrow = yuv420(SampleFormat::U8);
    let input = with_range(narrow, ColourRange::Limited);
    let output = produced(input, json!({ "depth": "f32" })).expect("converts");
    let converted = output.as_frames().expect("frames");

    assert_eq!(converted.sample, SampleFormat::F32);
    assert_eq!(converted.colour.range, ColourRange::Limited);
}

#[test]
fn a_float_input_returns_to_its_labelled_range() {
    let float = yuv420(SampleFormat::F32);
    let input = with_range(float, ColourRange::Limited);
    let output = produced(input, json!({ "depth": 10 })).expect("converts");
    let converted = output.as_frames().expect("frames");

    assert_eq!(converted.sample, SampleFormat::U16 { depth: BitDepth::TEN });
    assert_eq!(converted.colour.range, ColourRange::Limited);
}
