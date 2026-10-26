use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition};
use av_toolbox_cubek::deblock::h264::thresholds;
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_gpu::{
    ColourInfo,
    Depth as BitDepth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    StreamFormat,
    Subsampling,
};
use serde_json::json;

use crate::deblock::H264;
use crate::deblock::h264::{H264Args, accepted_samples, checked_deblocking, padded_sizes};
use crate::sample_checks::{assert_integers_and_floats_accepted_and_mask_refused, capabilities_computing};

const STAGE: &str = "deblock/h264";

fn args(value: serde_json::Value) -> H264Args {
    serde_json::from_value(value).expect("parses")
}

fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 48,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn yuv(subsampling: Subsampling) -> FrameLayout {
    FrameLayout::Yuv { subsampling }
}

fn yuv420() -> FrameFormat {
    let layout = yuv(Subsampling::Yuv420);

    format(layout, SampleFormat::U8)
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let arguments = args(value);
    let error = checked_deblocking(&arguments, &input).expect_err("should be refused");

    error.to_string()
}

#[test]
fn the_defaults_are_quant_25_no_offsets_and_every_plane() {
    let parsed = args(json!({}));
    let input = yuv420();

    let deblocking = checked_deblocking(&parsed, &input).expect("accepted");

    assert_eq!((parsed.quant, parsed.aoffset, parsed.boffset), (25, 0, 0));
    assert_eq!(parsed.planes, None);
    assert_eq!(deblocking.selected, vec![true, true, true]);
    assert_eq!(deblocking.thresholds, thresholds(25, 0, 0, MatrixSamples::U8));
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<H264Args>(json!({ "strength": 3 }));
    let error = parsed.expect_err("strength is not an argument");
    let message = error.to_string();

    assert!(message.contains("strength"), "{message}");
}

#[test]
fn quant_outside_0_to_60_is_refused_naming_the_range() {
    for quant in [-1, 61] {
        let value = json!({ "quant": quant });
        let input = yuv420();
        let message = refusal(value, input);

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("quant"), "{message}");
        assert!(message.contains("between 0 and 60"), "{message}");
    }

    for quant in [0, 60] {
        let arguments = args(json!({ "quant": quant }));
        let input = yuv420();
        let checked = checked_deblocking(&arguments, &input);

        assert!(checked.is_ok(), "quant {quant} is in range");
    }
}

#[test]
fn any_offset_is_accepted_and_clamped_into_the_tables() {
    let arguments = args(json!({ "quant": 40, "aoffset": 1000, "boffset": -1000 }));
    let input = yuv420();

    let deblocking = checked_deblocking(&arguments, &input).expect("accepted");

    assert_eq!(deblocking.thresholds, thresholds(60, 0, -60, MatrixSamples::U8));
}

#[test]
fn bad_planes_are_refused() {
    let no_planes = json!({ "planes": [] });
    let repeated = json!({ "planes": ["u", "u"] });
    let absent = json!({ "planes": ["r"] });
    let input = yuv420();
    let empty = refusal(no_planes, input);
    let twice = refusal(repeated, input);
    let missing = refusal(absent, input);

    for message in [empty, twice, missing] {
        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("planes"), "{message}");
    }
}

#[test]
fn planes_choose_which_planes_are_deblocked() {
    let arguments = args(json!({ "planes": ["v", "y"] }));
    let input = yuv420();

    let deblocking = checked_deblocking(&arguments, &input).expect("accepted");

    assert_eq!(deblocking.selected, vec![true, false, true]);
}

#[test]
fn integers_and_floats_are_accepted_and_a_mask_is_refused() {
    assert_integers_and_floats_accepted_and_mask_refused(STAGE, FrameLayout::Grey, format, accepted_samples);
}

#[test]
fn a_ten_bit_frame_is_deblocked_with_ten_bit_thresholds() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let input = format(FrameLayout::Grey, ten_bit);
    let arguments = args(json!({}));

    let deblocking = checked_deblocking(&arguments, &input).expect("accepted");

    let samples = MatrixSamples::U16 { bits: 10 };
    assert_eq!(deblocking.samples, samples);
    assert_eq!(deblocking.thresholds, thresholds(25, 0, 0, samples));
}

#[test]
fn a_float_frame_is_deblocked_as_float() {
    let input = format(FrameLayout::Rgb, SampleFormat::F32);
    let arguments = args(json!({ "quant": 51 }));

    let deblocking = checked_deblocking(&arguments, &input).expect("accepted");

    assert_eq!(deblocking.samples, MatrixSamples::F32);
    assert_eq!(deblocking.thresholds, thresholds(51, 0, 0, MatrixSamples::F32));
}

#[test]
fn each_plane_is_padded_to_its_share_of_the_padded_frame() {
    let odd_420 = FrameFormat {
        width: 70,
        height: 46,
        ..yuv420()
    };
    let odd_422 = FrameFormat {
        width: 71,
        height: 45,
        layout: yuv(Subsampling::Yuv422),
        ..yuv420()
    };

    let every_420 = padded_sizes(&odd_420, &[true, true, true]);
    let chroma_420 = padded_sizes(&odd_420, &[false, true, false]);
    let every_422 = padded_sizes(&odd_422, &[true, true, true]);

    assert_eq!(every_420, vec![(72, 48), (36, 24), (36, 24)]);
    assert_eq!(chroma_420, vec![(36, 24)]);
    assert_eq!(every_422, vec![(72, 48), (36, 48), (36, 48)]);
}

#[test]
fn the_output_is_the_input_format_and_the_stage_runs_frame_by_frame() {
    let capabilities = capabilities_computing(&[]);
    let arguments = args(json!({}));
    let input = yuv420();
    let chosen = [Some(StreamFormat::Frames(input))];

    let outputs = H264::produces(&capabilities, &arguments, &chosen).expect("accepted");

    let output = outputs.template.with_sample(input.sample);
    assert_eq!(output.as_frames(), Some(&input));
    assert_eq!(H264::parallelism(&arguments), Parallelism::Frame);
}
