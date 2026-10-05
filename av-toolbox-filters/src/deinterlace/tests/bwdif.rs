use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition, StageError};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth as BitDepth,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameRate,
    MaskFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::deinterlace::Bwdif;
use crate::deinterlace::bwdif::{BwdifArgs, Order, Rate};
use crate::deinterlace::spatial::Spatial;

const STAGE: &str = "deinterlace/bwdif";

fn args(value: serde_json::Value) -> BwdifArgs {
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
        frame_rate: FrameRate::new(25, 1),
        field_order,
    }
}

fn interlaced() -> FrameFormat {
    yuv420(16, 8, Some(FieldOrder::TopFirst))
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn produces(value: serde_json::Value, input: FrameFormat) -> Result<OutputFormats, StageError> {
    let capabilities = capabilities();
    let parsed = args(value);
    let chosen = [Some(StreamFormat::Frames(input))];

    Bwdif::produces(&capabilities, &parsed, &chosen)
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

#[test]
fn the_defaults_are_the_stream_order_double_rate_on_every_plane() {
    let parsed = args(json!({}));

    assert_eq!(parsed.order, None);
    assert_eq!(parsed.rate, Rate::Double);
    assert_eq!(parsed.planes, None);
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<BwdifArgs>(json!({ "field": 1 }));
    let error = parsed.expect_err("field is not an argument");
    let message = error.to_string();

    assert!(message.contains("field"), "{message}");
}

#[test]
fn an_unknown_field_order_without_order_is_refused() {
    let input = yuv420(16, 8, None);
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("order"), "{message}");
    assert!(message.contains("unknown"), "{message}");
}

#[test]
fn a_progressive_stream_without_order_is_refused() {
    let input = yuv420(16, 8, Some(FieldOrder::Progressive));
    let message = refusal(json!({}), input);

    assert!(message.contains("order"), "{message}");
    assert!(message.contains("progressive"), "{message}");
}

#[test]
fn order_overrides_a_progressive_flag() {
    let input = yuv420(16, 8, Some(FieldOrder::Progressive));
    let output = produced(json!({ "order": "bottom" }), input);

    assert_eq!(output.field_order, Some(FieldOrder::Progressive));
}

#[test]
fn the_output_is_progressive_at_twice_the_rate() {
    let output = produced(json!({}), interlaced());

    assert_eq!(output.field_order, Some(FieldOrder::Progressive));
    assert_eq!(output.frame_rate, FrameRate::new(50, 1));
    assert_eq!(output.width, 16);
    assert_eq!(output.height, 8);
}

#[test]
fn a_ntsc_rate_doubles_its_numerator() {
    let input = FrameFormat {
        frame_rate: FrameRate::new(30000, 1001),
        ..interlaced()
    };
    let output = produced(json!({}), input);

    assert_eq!(output.frame_rate, FrameRate::new(60000, 1001));
}

#[test]
fn single_rate_keeps_the_rate() {
    let output = produced(json!({ "rate": "single" }), interlaced());

    assert_eq!(output.frame_rate, FrameRate::new(25, 1));
}

#[test]
fn an_unknown_rate_stays_unknown() {
    let input = FrameFormat {
        frame_rate: None,
        ..interlaced()
    };
    let output = produced(json!({}), input);

    assert_eq!(output.frame_rate, None);
}

#[test]
fn an_even_denominator_is_halved_rather_than_the_numerator_doubled() {
    // A rate the reduction leaves with an even denominator, so its numerator is odd.
    let input = FrameFormat {
        frame_rate: FrameRate::new(u32::MAX, 2),
        ..interlaced()
    };
    let output = produced(json!({}), input);

    assert_eq!(output.frame_rate, FrameRate::new(u32::MAX, 1));
}

#[test]
fn a_doubled_rate_that_does_not_fit_is_refused() {
    let input = FrameFormat {
        frame_rate: FrameRate::new(u32::MAX, 1),
        ..interlaced()
    };
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("frame rate"), "{message}");
}

#[test]
fn an_odd_chroma_height_is_refused_naming_its_plane() {
    let input = yuv420(720, 486, Some(FieldOrder::BottomFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("U"), "{message}");
    assert!(message.contains("243"), "{message}");
}

#[test]
fn luma_alone_deinterlaces_an_odd_chroma_height() {
    let input = yuv420(720, 486, Some(FieldOrder::BottomFirst));
    let accepted = produces(json!({ "planes": ["y"] }), input);

    assert!(accepted.is_ok());
}

#[test]
fn a_plane_shorter_than_four_rows_is_refused() {
    let input = FrameFormat {
        layout: FrameLayout::Grey,
        ..yuv420(16, 2, Some(FieldOrder::TopFirst))
    };
    let message = refusal(json!({}), input);

    assert!(message.contains("at least 4 rows"), "{message}");
}

#[test]
fn floats_are_left_out_and_masks_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..interlaced()
    };
    let float = StreamFormat::Frames(float_format);
    let mask = StreamFormat::Mask(MaskFormat {
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let accepted = Bwdif::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("readable");
    let refused = Bwdif::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}

#[test]
fn ten_bit_frames_are_accepted() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let input = FrameFormat {
        sample: ten_bit,
        ..interlaced()
    };

    assert!(produces(json!({}), input).is_ok());
}

#[test]
fn it_holds_two_frames_and_runs_by_chunk() {
    let parsed = args(json!({}));
    let input = StreamFormat::Frames(interlaced());
    let formats = ResolvedFormats::new(vec![Some(input)], Some(input));

    assert_eq!(Bwdif::held_frames(&parsed, &formats), 2);
    assert_eq!(Bwdif::parallelism(&parsed), Parallelism::Chunk);
}

#[test]
fn order_takes_top_or_bottom() {
    assert_eq!(args(json!({ "order": "top" })).order, Some(Order::Top));
    assert_eq!(args(json!({ "order": "bottom" })).order, Some(Order::Bottom));
}

#[test]
fn an_explicit_eedi3_without_mdis_uses_aa_eedi3_s_mdis() {
    let parsed = args(json!({ "spatial": { "kind": "eedi3", "alpha": 0.3 } }));
    let Spatial::Eedi3(spatial) = parsed.spatial else {
        panic!("spatial should be EEDI3");
    };

    assert_eq!(spatial.mdis, 20);
}
