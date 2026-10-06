use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition, StageError};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth as BitDepth,
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

use crate::deinterlace::spatial::Spatial;
use crate::ivtc::Match;
use crate::ivtc::matching::{Fallback, MatchArgs, MatchingField};

const STAGE: &str = "ivtc/match";

fn args(value: serde_json::Value) -> MatchArgs {
    serde_json::from_value(value).expect("parses")
}

fn yuv420(width: u32, height: u32, field_order: Option<FieldOrder>) -> FrameFormat {
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
        field_order,
    }
}

fn telecined() -> FrameFormat {
    yuv420(64, 32, Some(FieldOrder::TopFirst))
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

    Match::produces(&capabilities, &parsed, &chosen)
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
    let message = refusal(value, telecined());

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains(name), "{message}");
}

#[test]
fn the_defaults_are_vfms() {
    let parsed = args(json!({}));

    assert_eq!(parsed.order, None);
    assert_eq!(parsed.field, MatchingField::Same);
    assert_eq!(parsed.mode, 1);
    assert!(parsed.mchroma);
    assert_eq!(parsed.cthresh, 9);
    assert_eq!(parsed.mi, 80);
    assert!(parsed.chroma);
    assert_eq!(parsed.blockx, 16);
    assert_eq!(parsed.blocky, 16);
    assert_eq!(parsed.y0, 16);
    assert_eq!(parsed.y1, 16);
    assert_eq!(parsed.scthresh, 12.0);
    assert_eq!(parsed.micmatch, 1);
    assert!(matches!(
        parsed.fallback,
        Fallback::Bwdif {
            spatial: Spatial::Cubic {}
        }
    ));
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<MatchArgs>(json!({ "rate": "single" }));
    let error = parsed.expect_err("rate is not an argument");
    let message = error.to_string();

    assert!(message.contains("rate"), "{message}");
}

#[test]
fn field_takes_same_or_opposite() {
    let same = args(json!({ "field": "same" }));
    let opposite = args(json!({ "field": "opposite" }));

    assert_eq!(same.field, MatchingField::Same);
    assert_eq!(opposite.field, MatchingField::Opposite);
}

#[test]
fn a_fallback_of_none_is_accepted() {
    let parsed = args(json!({ "fallback": { "kind": "none" } }));

    assert!(matches!(parsed.fallback, Fallback::None {}));
    assert!(produces(json!({ "fallback": { "kind": "none" } }), telecined()).is_ok());
}

#[test]
fn a_bwdif_fallback_takes_a_spatial_prediction() {
    let parsed = args(json!({ "fallback": { "kind": "bwdif", "spatial": { "kind": "sangnom" } } }));

    assert!(matches!(
        parsed.fallback,
        Fallback::Bwdif {
            spatial: Spatial::Sangnom(_)
        }
    ));
}

#[test]
fn a_mode_past_five_is_refused() {
    assert_refused_naming(json!({ "mode": 6 }), "mode");
}

#[test]
fn a_micmatch_past_two_is_refused() {
    assert_refused_naming(json!({ "micmatch": 3 }), "micmatch");
}

#[test]
fn a_cthresh_outside_its_range_is_refused() {
    assert_refused_naming(json!({ "cthresh": -2 }), "cthresh");
    assert_refused_naming(json!({ "cthresh": 256 }), "cthresh");
}

#[test]
fn the_cthresh_bounds_are_accepted() {
    assert!(produces(json!({ "cthresh": -1 }), telecined()).is_ok());
    assert!(produces(json!({ "cthresh": 255 }), telecined()).is_ok());
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
fn an_mi_past_the_block_size_is_refused() {
    assert_refused_naming(json!({ "blockx": 8, "blocky": 4, "mi": 33 }), "mi");
    assert!(produces(json!({ "blockx": 8, "blocky": 4, "mi": 32 }), telecined()).is_ok());
}

#[test]
fn a_y0_past_y1_is_refused() {
    assert_refused_naming(json!({ "y0": 20, "y1": 10 }), "y0");
}

#[test]
fn a_scthresh_outside_a_percentage_is_refused() {
    assert_refused_naming(json!({ "scthresh": -1.0 }), "scthresh");
    assert_refused_naming(json!({ "scthresh": 100.5 }), "scthresh");
}

#[test]
fn an_unknown_field_order_without_order_is_refused() {
    let input = yuv420(64, 32, None);
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("order"), "{message}");
}

#[test]
fn order_overrides_a_progressive_flag() {
    let input = yuv420(64, 32, Some(FieldOrder::Progressive));

    assert!(produces(json!({ "order": "bottom" }), input).is_ok());
}

#[test]
fn an_rgb_layout_is_refused() {
    let input = FrameFormat {
        layout: FrameLayout::Rgb,
        ..telecined()
    };
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
}

#[test]
fn a_grey_layout_is_accepted() {
    let input = FrameFormat {
        layout: FrameLayout::Grey,
        ..telecined()
    };

    assert!(produces(json!({}), input).is_ok());
}

#[test]
fn an_odd_height_is_refused() {
    let input = FrameFormat {
        layout: FrameLayout::Grey,
        ..yuv420(64, 33, Some(FieldOrder::TopFirst))
    };
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("33"), "{message}");
}

#[test]
fn an_odd_chroma_height_is_refused_naming_its_plane() {
    let input = yuv420(64, 34, Some(FieldOrder::TopFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("U"), "{message}");
    assert!(message.contains("17"), "{message}");
}

#[test]
fn a_frame_shorter_than_a_block_is_refused() {
    let input = yuv420(64, 8, Some(FieldOrder::TopFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("blocky"), "{message}");
}

#[test]
fn a_frame_narrower_than_the_field_compare_is_refused() {
    let input = yuv420(8, 32, Some(FieldOrder::TopFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("16 columns"), "{message}");
}

#[test]
fn an_odd_width_under_subsampled_chroma_is_refused() {
    let input = yuv420(65, 32, Some(FieldOrder::TopFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("65"), "{message}");
}

#[test]
fn a_bad_fallback_setting_is_refused_naming_the_stage() {
    let value = json!({ "fallback": { "kind": "bwdif", "spatial": { "kind": "eedi2", "maxd": 30 } } });
    let message = refusal(value, telecined());

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("maxd"), "{message}");
}

#[test]
fn a_bad_fallback_threshold_count_names_the_stage() {
    let value = json!({ "fallback": { "kind": "bwdif", "spatial": { "kind": "eedi3", "vthresh": [1, 2] } } });
    let parsed = serde_json::from_value::<MatchArgs>(value);
    let message = parsed.expect_err("two thresholds").to_string();

    assert_eq!(message, format!("{STAGE} needs vthresh of 3 numbers, got 2"));
}

#[test]
fn the_output_is_progressive_at_the_input_rate() {
    let output = produced(json!({}), telecined());

    assert_eq!(output.field_order, Some(FieldOrder::Progressive));
    assert_eq!(output.frame_rate, FrameRate::new(30000, 1001));
    assert_eq!(output.width, 64);
    assert_eq!(output.height, 32);
}

#[test]
fn ten_bit_frames_are_accepted() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let input = FrameFormat {
        sample: ten_bit,
        ..telecined()
    };

    assert!(produces(json!({}), input).is_ok());
}

#[test]
fn it_holds_two_frames_and_runs_by_chunk() {
    let parsed = args(json!({}));
    let input = StreamFormat::Frames(telecined());
    let formats = ResolvedFormats::new(vec![Some(input)], Some(input));

    assert_eq!(Match::held_frames(&parsed, &formats), 2);
    assert_eq!(Match::parallelism(&parsed), Parallelism::Chunk(None));
}
