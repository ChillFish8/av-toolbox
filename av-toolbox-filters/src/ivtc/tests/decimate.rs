use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition, StageError};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameRate,
    OutputFormats,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::ivtc::Decimate;
use crate::ivtc::decimate::DecimateArgs;

const STAGE: &str = "ivtc/decimate";

fn args(value: serde_json::Value) -> DecimateArgs {
    serde_json::from_value(value).expect("parses")
}

fn yuv420(width: u32, height: u32) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    FrameFormat {
        width,
        height,
        layout,
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: FrameRate::new(30000, 1001),
        field_order: Some(FieldOrder::Progressive),
    }
}

fn matched() -> FrameFormat {
    yuv420(64, 32)
}

fn grey() -> FrameFormat {
    FrameFormat {
        layout: FrameLayout::Grey,
        ..matched()
    }
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(64, 64)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn produces(value: serde_json::Value, input: FrameFormat) -> Result<OutputFormats, StageError> {
    let capabilities = capabilities();
    let parsed = args(value);
    let chosen = [Some(StreamFormat::Frames(input))];

    Decimate::produces(&capabilities, &parsed, &chosen)
}

fn produced(value: serde_json::Value, input: FrameFormat) -> FrameFormat {
    let outputs = produces(value, input).expect("should be accepted");
    let format = outputs.template.with_sample(input.sample);

    *format.as_frames().expect("frames")
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let error = produces(value, input).expect_err("should be refused");

    error.to_string()
}

fn assert_refused_naming(value: serde_json::Value, name: &str) {
    let message = refusal(value, matched());

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains(name), "{message}");
}

#[test]
fn the_defaults_are_vdecimates() {
    let parsed = args(json!({}));

    assert_eq!(parsed.cycle, 5);
    assert_eq!(parsed.chroma, None);
    assert_eq!(parsed.dupthresh, 1.1);
    assert_eq!(parsed.scthresh, 15.0);
    assert_eq!(parsed.blockx, 32);
    assert_eq!(parsed.blocky, 32);
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<DecimateArgs>(json!({ "dryrun": true }));
    let error = parsed.expect_err("dryrun is not an argument");
    let message = error.to_string();

    assert!(message.contains("dryrun"), "{message}");
}

#[test]
fn a_cycle_outside_two_to_25_is_refused() {
    assert_refused_naming(json!({ "cycle": 1 }), "cycle");
    assert_refused_naming(json!({ "cycle": 26 }), "cycle");
}

#[test]
fn the_cycle_bounds_are_accepted() {
    assert!(produces(json!({ "cycle": 2 }), matched()).is_ok());
    assert!(produces(json!({ "cycle": 25 }), matched()).is_ok());
}

#[test]
fn a_dupthresh_outside_a_percentage_is_refused() {
    assert_refused_naming(json!({ "dupthresh": -0.5 }), "dupthresh");
    assert_refused_naming(json!({ "dupthresh": 100.5 }), "dupthresh");
}

#[test]
fn a_scthresh_outside_a_percentage_is_refused() {
    assert_refused_naming(json!({ "scthresh": -1.0 }), "scthresh");
    assert_refused_naming(json!({ "scthresh": 101.0 }), "scthresh");
}

#[test]
fn a_block_that_is_not_a_power_of_two_is_refused() {
    assert_refused_naming(json!({ "blockx": 12 }), "blockx");
    assert_refused_naming(json!({ "blocky": 24 }), "blocky");
}

#[test]
fn a_block_outside_four_to_512_is_refused() {
    assert_refused_naming(json!({ "blockx": 2 }), "blockx");
    assert_refused_naming(json!({ "blocky": 1024 }), "blocky");
}

#[test]
fn chroma_on_a_grey_stream_is_refused() {
    let message = refusal(json!({ "chroma": true }), grey());

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("chroma"), "{message}");
}

#[test]
fn a_grey_stream_is_accepted_without_chroma() {
    assert!(produces(json!({}), grey()).is_ok());
    assert!(produces(json!({ "chroma": false }), grey()).is_ok());
}

#[test]
fn an_rgb_layout_is_refused() {
    let input = FrameFormat {
        layout: FrameLayout::Rgb,
        ..matched()
    };
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
}

#[test]
fn a_luma_half_a_block_across_is_refused() {
    let message = refusal(json!({}), yuv420(16, 32));

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("blockx"), "{message}");
}

#[test]
fn a_luma_half_a_block_down_is_refused() {
    let message = refusal(json!({}), yuv420(64, 16));

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("blocky"), "{message}");
}

#[test]
fn a_luma_just_over_half_a_block_is_accepted() {
    assert!(produces(json!({}), yuv420(18, 18)).is_ok());
}

#[test]
fn thirty_frames_a_second_become_twenty_four() {
    let output = produced(json!({}), matched());

    assert_eq!(output.frame_rate, FrameRate::new(24000, 1001));
    assert_eq!(output.width, 64);
    assert_eq!(output.height, 32);
    assert_eq!(output.layout, matched().layout);
}

#[test]
fn the_output_rate_is_reduced() {
    let input = FrameFormat {
        frame_rate: FrameRate::new(30, 1),
        ..matched()
    };

    let output = produced(json!({ "cycle": 3 }), input);

    assert_eq!(output.frame_rate, FrameRate::new(20, 1));
}

#[test]
fn an_unknown_rate_stays_unknown() {
    let input = FrameFormat {
        frame_rate: None,
        ..matched()
    };

    let output = produced(json!({}), input);

    assert_eq!(output.frame_rate, None);
}

#[test]
fn it_holds_a_cycle_and_the_frame_before() {
    let parsed = args(json!({ "cycle": 7 }));
    let input = StreamFormat::Frames(matched());
    let formats = ResolvedFormats::new(vec![Some(input)], Some(input));

    assert_eq!(Decimate::held_frames(&parsed, &formats), 8);
}

#[test]
fn it_runs_over_the_whole_stream() {
    let parsed = args(json!({}));

    assert_eq!(Decimate::parallelism(&parsed), Parallelism::Stream);
}
