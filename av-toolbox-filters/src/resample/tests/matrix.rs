use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition, StageError};
use av_toolbox_cubek::matrix::{Coefficients, Direction, MatrixSamples, SampleRange};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    ColourRange,
    Depth as BitDepth,
    FormatSet,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Matrix as StreamMatrix,
    Offered,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::resample::Matrix;
use crate::resample::matrix::{MatrixArgs, MatrixName, RangeName, TargetName, converted};

const STAGE: &str = "resample/matrix";

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn args(value: serde_json::Value) -> MatrixArgs {
    serde_json::from_value(value).expect("parses")
}

fn format(
    layout: FrameLayout,
    sample: SampleFormat,
    matrix: StreamMatrix,
    range: ColourRange,
) -> FrameFormat {
    let colour = ColourInfo {
        matrix,
        range,
        ..ColourInfo::default()
    };

    FrameFormat {
        width: 16,
        height: 8,
        layout,
        sample,
        colour,
        frame_rate: None,
        field_order: None,
    }
}

fn yuv(subsampling: Subsampling) -> FrameLayout {
    FrameLayout::Yuv { subsampling }
}

fn bt709_yuv444() -> FrameFormat {
    let layout = yuv(Subsampling::Yuv444);

    format(
        layout,
        SampleFormat::U8,
        StreamMatrix::Bt709,
        ColourRange::Limited,
    )
}

fn rgb() -> FrameFormat {
    format(
        FrameLayout::Rgb,
        SampleFormat::U8,
        StreamMatrix::Identity,
        ColourRange::Full,
    )
}

fn produced(input: FrameFormat, value: serde_json::Value) -> Result<FrameFormat, StageError> {
    let capabilities = capabilities();
    let arguments = args(value);
    let chosen = [Some(StreamFormat::Frames(input))];
    let outputs = Matrix::produces(&capabilities, &arguments, &chosen)?;
    let output = outputs.template.with_sample(input.sample);
    let frames = output.as_frames().expect("frames");

    Ok(*frames)
}

fn refusal(input: FrameFormat, value: serde_json::Value) -> String {
    let error = produced(input, value).expect_err("should be refused");

    error.to_string()
}

fn accepted(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let capabilities = capabilities();
    let arguments = args(json!({ "to": "rgb" }));
    let inputs = [Offered::Unknown];

    Matrix::accepts_input(&capabilities, &arguments, 0, offered, &inputs)
}

#[test]
fn to_is_required_and_the_rest_default_to_the_stream() {
    let missing = serde_json::from_value::<MatrixArgs>(json!({}));
    let parsed = args(json!({ "to": "yuv" }));

    assert!(missing.is_err());
    assert_eq!(parsed.to, TargetName::Yuv);
    assert_eq!(parsed.matrix, None);
    assert_eq!(parsed.range, None);
}

#[test]
fn every_matrix_and_range_name_parses() {
    let parsed = args(json!({ "to": "yuv", "matrix": "bt2020_ncl", "range": "full" }));
    let bt709 = args(json!({ "to": "yuv", "matrix": "bt709" }));
    let bt601 = args(json!({ "to": "yuv", "matrix": "bt601", "range": "limited" }));

    assert_eq!(parsed.matrix, Some(MatrixName::Bt2020Ncl));
    assert_eq!(parsed.range, Some(RangeName::Full));
    assert_eq!(bt709.matrix, Some(MatrixName::Bt709));
    assert_eq!(bt601.matrix, Some(MatrixName::Bt601));
    assert_eq!(bt601.range, Some(RangeName::Limited));
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<MatrixArgs>(json!({ "to": "rgb", "kernel": "point" }));
    let error = parsed.expect_err("kernel is not an argument");
    let message = error.to_string();

    assert!(message.contains("kernel"), "{message}");
}

#[test]
fn going_to_rgb_gives_full_range_rgb_with_the_identity_matrix() {
    let output = produced(bt709_yuv444(), json!({ "to": "rgb" })).expect("accepted");

    assert_eq!(output.layout, FrameLayout::Rgb);
    assert_eq!(output.colour.matrix, StreamMatrix::Identity);
    assert_eq!(output.colour.range, ColourRange::Full);
    assert_eq!(output.sample, SampleFormat::U8);
    assert_eq!((output.width, output.height), (16, 8));
}

#[test]
fn going_to_rgb_reads_the_stream_matrix_and_range() {
    let input = bt709_yuv444();
    let arguments = args(json!({ "to": "rgb" }));

    let planned = converted(&input, &arguments).expect("accepted");

    assert_eq!(planned.conversion.coefficients, Coefficients::Bt709);
    assert_eq!(planned.conversion.direction, Direction::ToRgb);
    assert_eq!(planned.conversion.input_range, SampleRange::Limited);
    assert_eq!(planned.conversion.output_range, SampleRange::Full);
    assert_eq!(planned.conversion.samples, MatrixSamples::U8);
}

#[test]
fn the_arguments_override_the_stream_matrix_and_range() {
    let input = bt709_yuv444();
    let arguments = args(json!({ "to": "rgb", "matrix": "bt601", "range": "full" }));

    let planned = converted(&input, &arguments).expect("accepted");

    assert_eq!(planned.conversion.coefficients, Coefficients::Bt601);
    assert_eq!(planned.conversion.input_range, SampleRange::Full);
}

#[test]
fn going_to_yuv_takes_the_matrix_and_defaults_to_limited_range() {
    let output = produced(rgb(), json!({ "to": "yuv", "matrix": "bt2020_ncl" })).expect("accepted");
    let full = produced(rgb(), json!({ "to": "yuv", "matrix": "bt709", "range": "full" })).expect("accepted");

    assert_eq!(output.layout, yuv(Subsampling::Yuv444));
    assert_eq!(output.colour.matrix, StreamMatrix::Bt2020Ncl);
    assert_eq!(output.colour.range, ColourRange::Limited);
    assert_eq!(full.colour.range, ColourRange::Full);
}

#[test]
fn an_rgb_stream_of_unspecified_range_is_read_as_full_range() {
    let input = format(
        FrameLayout::Rgb,
        SampleFormat::U8,
        StreamMatrix::Identity,
        ColourRange::Unspecified,
    );
    let arguments = args(json!({ "to": "yuv", "matrix": "bt709" }));

    let planned = converted(&input, &arguments).expect("accepted");

    assert_eq!(planned.conversion.input_range, SampleRange::Full);
    assert_eq!(planned.conversion.output_range, SampleRange::Limited);
}

#[test]
fn a_limited_rgb_stream_is_read_as_limited() {
    let input = format(
        FrameLayout::Rgb,
        SampleFormat::U8,
        StreamMatrix::Identity,
        ColourRange::Limited,
    );
    let arguments = args(json!({ "to": "yuv", "matrix": "bt709" }));

    let planned = converted(&input, &arguments).expect("accepted");

    assert_eq!(planned.conversion.input_range, SampleRange::Limited);
}

#[test]
fn a_ten_bit_frame_converts_at_ten_bits() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let layout = yuv(Subsampling::Yuv444);
    let input = format(layout, ten_bit, StreamMatrix::Bt709, ColourRange::Limited);
    let arguments = args(json!({ "to": "rgb" }));

    let planned = converted(&input, &arguments).expect("accepted");

    assert_eq!(planned.conversion.samples, MatrixSamples::U16 { bits: 10 });
}

#[test]
fn subsampled_yuv_is_refused_naming_resample_chroma() {
    let layout = yuv(Subsampling::Yuv420);
    let input = format(
        layout,
        SampleFormat::U8,
        StreamMatrix::Bt709,
        ColourRange::Limited,
    );

    let message = refusal(input, json!({ "to": "rgb" }));

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("4:2:0"), "{message}");
    assert!(message.contains("resample/chroma"), "{message}");
}

#[test]
fn an_unspecified_or_identity_matrix_without_matrix_is_refused() {
    let layout = yuv(Subsampling::Yuv444);
    let unspecified = format(
        layout,
        SampleFormat::U8,
        StreamMatrix::Unspecified,
        ColourRange::Limited,
    );
    let identity = format(
        layout,
        SampleFormat::U8,
        StreamMatrix::Identity,
        ColourRange::Limited,
    );

    let unspecified_message = refusal(unspecified, json!({ "to": "rgb" }));
    let identity_message = refusal(identity, json!({ "to": "rgb" }));
    let with_matrix = produced(unspecified, json!({ "to": "rgb", "matrix": "bt709" }));

    for message in [unspecified_message, identity_message] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("matrix"), "{message}");
    }

    assert!(with_matrix.is_ok());
}

#[test]
fn an_unspecified_range_without_range_is_refused_going_to_rgb() {
    let layout = yuv(Subsampling::Yuv444);
    let input = format(
        layout,
        SampleFormat::U8,
        StreamMatrix::Bt709,
        ColourRange::Unspecified,
    );

    let message = refusal(input, json!({ "to": "rgb" }));
    let with_range = produced(input, json!({ "to": "rgb", "range": "limited" }));

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("range"), "{message}");
    assert!(with_range.is_ok());
}

#[test]
fn going_to_yuv_without_matrix_is_refused() {
    let message = refusal(rgb(), json!({ "to": "yuv" }));

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("matrix"), "{message}");
}

#[test]
fn grey_and_a_layout_already_at_its_target_are_refused() {
    let grey = format(
        FrameLayout::Grey,
        SampleFormat::U8,
        StreamMatrix::Bt709,
        ColourRange::Limited,
    );

    let grey_message = refusal(grey, json!({ "to": "rgb" }));
    let rgb_message = refusal(rgb(), json!({ "to": "rgb" }));
    let yuv_message = refusal(bt709_yuv444(), json!({ "to": "yuv", "matrix": "bt709" }));

    assert!(grey_message.contains("grey"), "{grey_message}");
    assert!(rgb_message.contains("already"), "{rgb_message}");
    assert!(yuv_message.contains("already"), "{yuv_message}");

    for message in [grey_message, rgb_message, yuv_message] {
        assert!(message.contains(STAGE), "{message}");
    }
}

#[test]
fn integers_and_floats_are_accepted_and_a_mask_is_refused() {
    let layout = yuv(Subsampling::Yuv444);
    let samples = [
        SampleFormat::U8,
        SampleFormat::U16 {
            depth: BitDepth::SIXTEEN,
        },
        SampleFormat::F32,
    ];
    let half = format(
        layout,
        SampleFormat::F16,
        StreamMatrix::Bt709,
        ColourRange::Limited,
    );
    let mask = StreamFormat::Mask(MaskFormat {
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    for sample in samples {
        let input = format(layout, sample, StreamMatrix::Bt709, ColourRange::Limited);
        let set = accepted(StreamFormat::Frames(input)).expect("frames are readable");

        assert_eq!(set, FormatSet::of([sample]));
    }

    let half_set = accepted(StreamFormat::Frames(half)).expect("frames are readable");
    let mask_error = accepted(mask).expect_err("a mask is refused");
    let mask_message = mask_error.to_string();

    assert!(half_set.is_empty());
    assert!(mask_message.contains(STAGE), "{mask_message}");
    assert!(mask_message.contains("mask"), "{mask_message}");
}

#[test]
fn a_float_frame_converts_as_float() {
    let layout = yuv(Subsampling::Yuv444);
    let input = format(
        layout,
        SampleFormat::F32,
        StreamMatrix::Bt709,
        ColourRange::Limited,
    );
    let arguments = args(json!({ "to": "rgb" }));

    let planned = converted(&input, &arguments).expect("accepted");

    assert_eq!(planned.conversion.samples, MatrixSamples::F32);
}

#[test]
fn the_stage_converts_frame_by_frame() {
    let arguments = args(json!({ "to": "rgb" }));

    assert_eq!(Matrix::parallelism(&arguments), Parallelism::Frame);
}

#[test]
fn an_unspecified_range_float_stream_converts_to_rgb() {
    let layout = yuv(Subsampling::Yuv444);
    let input = format(
        layout,
        SampleFormat::F32,
        StreamMatrix::Bt709,
        ColourRange::Unspecified,
    );
    let arguments = args(json!({ "to": "rgb" }));

    let planned = converted(&input, &arguments).expect("accepted");

    assert_eq!(planned.conversion.input_range, SampleRange::Full);
}
