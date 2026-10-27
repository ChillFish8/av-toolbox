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

use crate::resample::MinDepth;
use crate::resample::min_depth::MinDepthArgs;

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn yuv420(sample: SampleFormat) -> StreamFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

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

fn args(value: serde_json::Value) -> Result<MinDepthArgs, serde_json::Error> {
    serde_json::from_value(value)
}

fn produced(
    input: StreamFormat,
    value: serde_json::Value,
) -> Result<StreamFormat, av_toolbox_core::StageError> {
    let capabilities = capabilities();
    let arguments = args(value).expect("parses");
    let chosen = [Some(input)];
    let outputs = MinDepth::produces(&capabilities, &arguments, &chosen)?;
    let sample = outputs.samples.preferences()[0];

    Ok(outputs.template.with_sample(sample))
}

#[test]
fn a_dither_is_refused() {
    let refused = args(json!({ "depth": 10, "dither": "ordered" }));

    assert!(refused.is_err());
}

#[test]
fn eight_bits_is_raised_to_the_minimum_and_keeps_everything_else() {
    let input = yuv420(SampleFormat::U8);
    let output = produced(input, json!({ "depth": 10 })).expect("raises");
    let raised = output.as_frames().expect("frames");
    let original = input.as_frames().expect("frames");

    assert_eq!(raised.sample, SampleFormat::U16 { depth: BitDepth::TEN });
    assert_eq!(raised.layout, original.layout);
    assert_eq!(raised.colour, original.colour);
}

#[test]
fn nine_bits_is_raised_to_the_minimum() {
    let nine_bit = BitDepth::new(9).expect("valid");
    let input = yuv420(SampleFormat::U16 { depth: nine_bit });
    let output = produced(input, json!({ "depth": 10 })).expect("raises");

    assert_eq!(output.sample(), SampleFormat::U16 { depth: BitDepth::TEN });
}

#[test]
fn a_depth_at_or_above_the_minimum_is_kept() {
    let twelve_bit = BitDepth::new(12).expect("valid");
    let kept = [
        SampleFormat::U16 { depth: BitDepth::TEN },
        SampleFormat::U16 { depth: twelve_bit },
    ];

    for sample in kept {
        let input = yuv420(sample);
        let output = produced(input, json!({ "depth": 10 })).expect("keeps");

        assert_eq!(output.sample(), sample);
    }
}

#[test]
fn a_minimum_of_eight_keeps_eight_bits() {
    let input = yuv420(SampleFormat::U8);
    let output = produced(input, json!({ "depth": 8 })).expect("keeps");

    assert_eq!(output.sample(), SampleFormat::U8);
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
fn min_depth_refuses_a_float_clip_with_an_explanation() {
    let input = yuv420(SampleFormat::F32);

    for depth in [8, 10, 16] {
        let error = produced(input, json!({ "depth": depth })).expect_err("refused");

        assert!(error.to_string().contains("resample/depth"), "{error}");
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
        MinDepth::accepts_input(&capabilities, &arguments, 0, float, &inputs).expect("frames are readable");
    let refused = MinDepth::accepts_input(&capabilities, &arguments, 0, mask, &inputs);

    assert!(accepted.accepts(SampleFormat::F32));
    assert!(refused.is_err());
}
