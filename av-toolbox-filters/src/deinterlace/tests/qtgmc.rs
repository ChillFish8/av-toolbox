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

use crate::deinterlace::Qtgmc;
use crate::deinterlace::bwdif::Order;
use crate::deinterlace::qtgmc::{Input, QtgmcArgs};
use crate::deinterlace::spatial::Spatial;

fn args(value: serde_json::Value) -> QtgmcArgs {
    serde_json::from_value(value).expect("parses")
}

fn frame(width: u32, height: u32, layout: FrameLayout, field_order: Option<FieldOrder>) -> FrameFormat {
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

fn yuv420(width: u32, height: u32, field_order: Option<FieldOrder>) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, field_order)
}

fn interlaced() -> FrameFormat {
    yuv420(64, 64, Some(FieldOrder::TopFirst))
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

    Qtgmc::produces(&capabilities, &parsed, &chosen)
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

fn refusal_names(value: serde_json::Value, name: &str) {
    let message = refusal(value, interlaced());

    assert!(message.contains("deinterlace/qtgmc"), "{message}");
    assert!(message.contains(name), "{message}");
}

#[test]
fn the_defaults_are_slower() {
    let parsed = args(json!({}));

    assert_eq!(parsed.order, None);
    assert_eq!(parsed.rate, None);
    assert!(matches!(parsed.spatial, Spatial::Eedi3(_)));
    assert_eq!(parsed.tr0, 2);
    assert_eq!(parsed.tr1, 2);
    assert_eq!(parsed.tr2, 1);
    assert_eq!(parsed.sharpness, 1.0);
    assert_eq!(parsed.thsad1, 640.0);
    assert_eq!(parsed.thsad2, 256.0);
}

#[test]
fn the_default_spatial_searches_eight_pixels() {
    let parsed = args(json!({}));
    let Spatial::Eedi3(spatial) = parsed.spatial else {
        panic!("the default spatial should be EEDI3");
    };

    assert_eq!(spatial.mdis, 8);
}

#[test]
fn an_explicit_eedi3_without_mdis_uses_qtgmc_s_mdis() {
    let parsed = args(json!({ "spatial": { "kind": "eedi3", "alpha": 0.3 } }));
    let Spatial::Eedi3(spatial) = parsed.spatial else {
        panic!("spatial should be EEDI3");
    };

    assert_eq!(spatial.mdis, 8);
}

#[test]
fn cubic_spatial_is_accepted() {
    let value = json!({ "spatial": { "kind": "cubic" } });

    assert!(produces(value, interlaced()).is_ok());
}

#[test]
fn unknown_arguments_are_refused() {
    let parsed = serde_json::from_value::<QtgmcArgs>(json!({ "preset": "slow" }));
    let error = parsed.expect_err("preset is not an argument");
    let message = error.to_string();

    assert!(message.contains("preset"), "{message}");
}

#[test]
fn tr0_above_two_is_refused() {
    refusal_names(json!({ "tr0": 3 }), "tr0");
}

#[test]
fn tr1_above_two_is_refused() {
    refusal_names(json!({ "tr1": 3 }), "tr1");
}

#[test]
fn tr2_above_three_is_refused() {
    refusal_names(json!({ "tr2": 4 }), "tr2");
}

#[test]
fn the_largest_radii_are_accepted() {
    let value = json!({ "tr0": 2, "tr1": 2, "tr2": 3 });

    assert!(produces(value, interlaced()).is_ok());
}

#[test]
fn a_negative_sharpness_is_refused() {
    refusal_names(json!({ "sharpness": -0.5 }), "sharpness");
}

#[test]
fn a_non_finite_sharpness_is_refused() {
    let parsed = QtgmcArgs {
        sharpness: f32::NAN,
        ..args(json!({}))
    };
    let chosen = [Some(StreamFormat::Frames(interlaced()))];
    let error = Qtgmc::produces(&capabilities(), &parsed, &chosen).expect_err("NaN is not finite");
    let message = error.to_string();

    assert!(message.contains("sharpness"), "{message}");
}

#[test]
fn a_zero_sharpness_is_accepted() {
    assert!(produces(json!({ "sharpness": 0.0 }), interlaced()).is_ok());
}

#[test]
fn a_thsad1_of_zero_is_refused() {
    refusal_names(json!({ "thsad1": 0.0 }), "thsad1");
}

#[test]
fn a_thsad2_below_zero_is_refused() {
    refusal_names(json!({ "thsad2": -1.0 }), "thsad2");
}

#[test]
fn a_non_finite_threshold_is_refused() {
    let parsed = QtgmcArgs {
        thsad2: f32::INFINITY,
        ..args(json!({}))
    };
    let chosen = [Some(StreamFormat::Frames(interlaced()))];
    let error = Qtgmc::produces(&capabilities(), &parsed, &chosen).expect_err("infinity is not finite");
    let message = error.to_string();

    assert!(message.contains("thsad2"), "{message}");
}

#[test]
fn a_float_format_is_refused() {
    let input = FrameFormat {
        sample: SampleFormat::F32,
        ..interlaced()
    };
    let message = refusal(json!({}), input);

    assert!(message.contains("integer YUV or Gray"), "{message}");
}

#[test]
fn an_rgb_format_is_refused() {
    let input = frame(64, 64, FrameLayout::Rgb, Some(FieldOrder::TopFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains("integer YUV or Gray"), "{message}");
}

#[test]
fn gray_and_sixteen_bit_frames_are_accepted() {
    let gray = frame(64, 64, FrameLayout::Grey, Some(FieldOrder::TopFirst));
    let sixteen = FrameFormat {
        sample: SampleFormat::U16 {
            depth: av_toolbox_gpu::Depth::SIXTEEN,
        },
        ..interlaced()
    };

    assert!(produces(json!({}), gray).is_ok());
    assert!(produces(json!({}), sixteen).is_ok());
}

#[test]
fn an_odd_plane_height_is_refused() {
    let input = yuv420(64, 70, Some(FieldOrder::TopFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains("deinterlace/qtgmc"), "{message}");
    assert!(message.contains("U"), "{message}");
    assert!(message.contains("35"), "{message}");
}

#[test]
fn fields_smaller_than_a_block_are_refused() {
    let input = yuv420(64, 24, Some(FieldOrder::TopFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains("block"), "{message}");
}

#[test]
fn a_frame_narrower_than_a_block_is_refused() {
    let input = yuv420(8, 64, Some(FieldOrder::TopFirst));
    let message = refusal(json!({}), input);

    assert!(message.contains("block"), "{message}");
}

#[test]
fn a_field_of_exactly_one_block_is_accepted() {
    let input = yuv420(16, 32, Some(FieldOrder::TopFirst));

    assert!(produces(json!({}), input).is_ok());
}

#[test]
fn a_plane_too_wide_for_the_spatial_is_refused() {
    let input = yuv420(65536, 32, Some(FieldOrder::TopFirst));
    let value = json!({ "spatial": { "kind": "eedi2" } });
    let message = refusal(value, input);

    assert!(message.contains("deinterlace/qtgmc"), "{message}");
    assert!(message.contains("65535"), "{message}");
}

#[test]
fn a_bad_spatial_setting_is_refused_under_the_stage() {
    let value = json!({ "spatial": { "kind": "eedi3", "nrad": 9 } });
    let message = refusal(value, interlaced());

    assert!(message.contains("deinterlace/qtgmc"), "{message}");
    assert!(message.contains("nrad"), "{message}");
}

#[test]
fn a_progressive_stream_without_order_is_refused() {
    let input = yuv420(64, 64, Some(FieldOrder::Progressive));
    let message = refusal(json!({}), input);

    assert!(message.contains("order"), "{message}");
    assert!(message.contains("progressive"), "{message}");
}

#[test]
fn an_explicit_order_overrides_a_progressive_stream() {
    let input = yuv420(64, 64, Some(FieldOrder::Progressive));
    let output = produced(json!({ "order": "bottom" }), input);

    assert_eq!(output.field_order, Some(FieldOrder::Progressive));
    assert_eq!(args(json!({ "order": "bottom" })).order, Some(Order::Bottom));
}

#[test]
fn double_rate_doubles_and_single_keeps_the_rate() {
    let doubled = produced(json!({}), interlaced());
    let single = produced(json!({ "rate": "single" }), interlaced());

    assert_eq!(doubled.frame_rate, FrameRate::new(50, 1));
    assert_eq!(doubled.field_order, Some(FieldOrder::Progressive));
    assert_eq!(single.frame_rate, FrameRate::new(25, 1));
    assert_eq!(single.field_order, Some(FieldOrder::Progressive));
}

#[test]
fn it_holds_frames_by_the_lag_and_runs_by_chunk() {
    let input = StreamFormat::Frames(interlaced());
    let formats = ResolvedFormats::new(vec![Some(input)], Some(input));

    let defaults = args(json!({}));
    assert_eq!(defaults.lag(), 5);
    assert_eq!(Qtgmc::held_frames(&defaults, &formats), 4);
    assert_eq!(Qtgmc::parallelism(&defaults), Parallelism::Chunk);

    let widest = args(json!({ "tr0": 2, "tr1": 2, "tr2": 3 }));
    assert_eq!(widest.lag(), 8);
    assert_eq!(Qtgmc::held_frames(&widest, &formats), 6);
    assert_eq!(Qtgmc::parallelism(&widest), Parallelism::Chunk);
}

#[test]
fn the_default_input_is_interlaced() {
    assert_eq!(args(json!({})).input, Input::Interlaced);
}

#[test]
fn every_input_parses() {
    let names = ["interlaced", "progressive", "repair", "repair_reversed"];
    let expected = [
        Input::Interlaced,
        Input::Progressive,
        Input::Repair,
        Input::RepairReversed,
    ];

    for (name, input) in names.into_iter().zip(expected) {
        assert_eq!(args(json!({ "input": name })).input, input);
    }
}

#[test]
fn rate_is_refused_with_progressive_input() {
    for name in ["progressive", "repair", "repair_reversed"] {
        let value = json!({ "input": name, "rate": "single", "order": "top" });
        let message = refusal(value, interlaced());

        assert!(message.contains("deinterlace/qtgmc"), "{message}");
        assert!(message.contains("rate"), "{message}");
    }
}

#[test]
fn progressive_input_keeps_the_rate() {
    for name in ["progressive", "repair", "repair_reversed"] {
        let output = produced(json!({ "input": name, "order": "top" }), interlaced());

        assert_eq!(output.frame_rate, FrameRate::new(25, 1));
        assert_eq!(output.field_order, Some(FieldOrder::Progressive));
    }
}

#[test]
fn repair_needs_order_on_a_progressive_stream() {
    let input = yuv420(64, 64, Some(FieldOrder::Progressive));

    for name in ["repair", "repair_reversed"] {
        let message = refusal(json!({ "input": name }), input);

        assert!(message.contains("order"), "{message}");

        let output = produced(json!({ "input": name, "order": "bottom" }), input);
        assert_eq!(output.field_order, Some(FieldOrder::Progressive));
    }
}

#[test]
fn progressive_needs_no_order() {
    let input = yuv420(64, 64, Some(FieldOrder::Progressive));
    let output = produced(json!({ "input": "progressive" }), input);

    assert_eq!(output.frame_rate, FrameRate::new(25, 1));
    assert_eq!(output.field_order, Some(FieldOrder::Progressive));
}

#[test]
fn held_frames_follow_each_mode_s_lag() {
    let input = StreamFormat::Frames(interlaced());
    let formats = ResolvedFormats::new(vec![Some(input)], Some(input));
    let cases = [
        ("interlaced", json!({}), 5, 4),
        ("progressive", json!({}), 5, 7),
        ("repair", json!({}), 6, 8),
        ("repair_reversed", json!({}), 6, 8),
        ("interlaced", json!({ "tr0": 2, "tr1": 2, "tr2": 3 }), 8, 6),
        ("progressive", json!({ "tr0": 2, "tr1": 2, "tr2": 3 }), 8, 10),
        ("repair", json!({ "tr0": 2, "tr1": 2, "tr2": 3 }), 8, 10),
    ];

    for (name, mut value, lag, held) in cases {
        value["input"] = json!(name);
        let parsed = args(value);

        assert_eq!(parsed.lag(), lag, "{name} lag");
        assert_eq!(Qtgmc::held_frames(&parsed, &formats), held, "{name} held frames");
    }
}

#[test]
fn progressive_input_accepts_an_odd_height() {
    let input = yuv420(64, 70, Some(FieldOrder::Progressive));

    assert!(produces(json!({ "input": "progressive" }), input).is_ok());
}

#[test]
fn progressive_input_checks_blocks_against_the_frame() {
    let input = yuv420(64, 24, Some(FieldOrder::TopFirst));

    assert!(produces(json!({ "input": "progressive" }), input).is_ok());

    let message = refusal(json!({ "input": "repair" }), input);
    assert!(message.contains("block"), "{message}");
}

#[test]
fn repair_refuses_an_odd_plane_height() {
    let input = yuv420(64, 70, Some(FieldOrder::TopFirst));
    let message = refusal(json!({ "input": "repair" }), input);

    assert!(message.contains("deinterlace/qtgmc"), "{message}");
    assert!(message.contains("35"), "{message}");
}
