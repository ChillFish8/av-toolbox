use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition, StageError};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth as BitDepth,
    ElementType,
    FormatSet,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Offered,
    OutputFormats,
    SampleFormat,
    StreamFormat,
    Subsampling,
};
use av_toolbox_models::Store;
use serde_json::json;

use crate::aa::Based;
use crate::aa::based::{BasedArgs, Eedi3Tuning, checked_frame_mask, planned};
use crate::aa::direction::Direction;
use crate::artcnn::ModelName;
use crate::cnn::PrecisionName;
use crate::sample_checks::capabilities_computing;

const STAGE: &str = "aa/based";

const DIRECTIONS: [&str; 3] = ["vertical", "horizontal", "both"];

fn args(value: serde_json::Value) -> BasedArgs {
    serde_json::from_value(value).expect("parses")
}

fn parse_error(value: serde_json::Value) -> String {
    let parsed = serde_json::from_value::<BasedArgs>(value);
    let error = parsed.expect_err("should be refused");

    error.to_string()
}

fn format(width: u32, height: u32, layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn grey(width: u32, height: u32) -> FrameFormat {
    format(width, height, FrameLayout::Grey, SampleFormat::U8)
}

fn yuv420(width: u32, height: u32) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    format(width, height, layout, SampleFormat::U8)
}

fn capabilities() -> Capabilities {
    capabilities_computing(&[ElementType::F32, ElementType::F16])
}

fn mask_format(width: u32, height: u32) -> StreamFormat {
    StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width,
        height,
        sample: SampleFormat::U8,
        chroma: None,
    })
}

fn produces_with(
    value: serde_json::Value,
    input: FrameFormat,
    mask: Option<StreamFormat>,
) -> Result<OutputFormats, StageError> {
    let capabilities = capabilities();
    let parsed = args(value);
    let frames = Some(StreamFormat::Frames(input));

    Based::produces(&capabilities, &parsed, &[frames, mask])
}

fn produces(value: serde_json::Value, input: FrameFormat) -> Result<OutputFormats, StageError> {
    produces_with(value, input, None)
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let error = produces(value, input).expect_err("should be refused");

    error.to_string()
}

fn accepted(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];

    Based::accepts_input(&capabilities, &parsed, 0, offered, &inputs)
}

#[test]
fn the_defaults_match_based_aa() {
    let parsed = args(json!({}));

    assert_eq!(parsed.mask_thr, 60);
    assert!(parsed.edge_mask);
    assert_eq!(parsed.pscale, 0.0);
    assert_eq!(parsed.model, ModelName::R8f64);
    assert_eq!(parsed.precision, PrecisionName::F16);
    assert!(parsed.postfilter);
    assert_eq!(parsed.direction, Direction::Both);
    assert_eq!(parsed.eedi3.alpha, 0.125);
    assert_eq!(parsed.eedi3.beta, 0.25);
    assert_eq!(parsed.eedi3.gamma, 40.0);
    assert_eq!(parsed.eedi3.nrad, 2);
    assert_eq!(parsed.eedi3.mdis, 20);
    assert_eq!(parsed.eedi3.vcheck, 2);
    assert_eq!(parsed.eedi3.vthresh, [12.0, 24.0, 4.0]);

    let input = yuv420(16, 16);
    let accepted = produces(json!({}), input);
    assert!(accepted.is_ok());
}

#[test]
fn an_eedi3_object_keeps_the_defaults_it_does_not_name() {
    let parsed = args(json!({ "eedi3": { "alpha": 0.3, "mdis": 10 } }));

    let expected = Eedi3Tuning {
        alpha: 0.3,
        mdis: 10,
        ..Eedi3Tuning::default()
    };
    assert_eq!(parsed.eedi3, expected);
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (
            json!({ "mask_thr": 256 }),
            "needs mask_thr between 0 and 255",
            "got 256",
        ),
        (
            json!({ "pscale": -0.1 }),
            "needs pscale between 0 and 1",
            "got -0.1",
        ),
        (
            json!({ "pscale": 1.1 }),
            "needs pscale between 0 and 1",
            "got 1.1",
        ),
        (
            json!({ "eedi3": { "alpha": -0.1 } }),
            "needs alpha of 0 or between 1e-30 and 1",
            "got -0.1",
        ),
        (
            json!({ "eedi3": { "alpha": 1.1, "beta": 0 } }),
            "needs alpha of 0 or between 1e-30 and 1",
            "got 1.1",
        ),
        (
            json!({ "eedi3": { "alpha": 0, "beta": 1.1 } }),
            "needs beta of 0 or between 1e-30 and 1",
            "got 1.1",
        ),
        (
            json!({ "eedi3": { "alpha": 0.6, "beta": 0.5 } }),
            "needs alpha and beta to add up to at most 1",
            "got 1.1",
        ),
        (
            json!({ "eedi3": { "gamma": -1 } }),
            "needs gamma between 0 and 1e30",
            "got -1.0",
        ),
        (
            json!({ "eedi3": { "gamma": 2e30 } }),
            "needs gamma between 0 and 1e30",
            "got 2e30",
        ),
        (
            json!({ "eedi3": { "nrad": 4 } }),
            "needs nrad between 0 and 3",
            "got 4",
        ),
        (
            json!({ "eedi3": { "mdis": 0 } }),
            "needs mdis between 1 and 40",
            "got 0",
        ),
        (
            json!({ "eedi3": { "mdis": 41 } }),
            "needs mdis between 1 and 40",
            "got 41",
        ),
        (
            json!({ "eedi3": { "vcheck": 4 } }),
            "needs vcheck between 0 and 3",
            "got 4",
        ),
        (
            json!({ "eedi3": { "vthresh": [0, 24, 4] } }),
            "needs a finite vthresh[0] of at least 1e-30",
            "got 0.0",
        ),
        (
            json!({ "eedi3": { "vthresh": [12, 0, 4] } }),
            "needs a finite vthresh[1] of at least 1e-30",
            "got 0.0",
        ),
        (
            json!({ "eedi3": { "vthresh": [12, 24, 1e-31] } }),
            "needs a finite vthresh[2] of at least 1e-30",
            "got 1e-31",
        ),
    ];

    for (value, fragment, expected) in cases {
        let input = grey(16, 16);
        let message = refusal(value, input);

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains(fragment), "{message}");
        assert!(message.ends_with(expected), "{message}");
    }
}

#[test]
fn the_range_bounds_are_accepted() {
    let cases = [
        json!({ "mask_thr": 0, "pscale": 0 }),
        json!({ "mask_thr": 255, "pscale": 1 }),
        json!({ "eedi3": { "alpha": 0.5, "beta": 0.5, "gamma": 0, "nrad": 0, "mdis": 1, "vcheck": 0 } }),
        json!({ "eedi3": { "gamma": 1e30, "nrad": 3, "mdis": 40, "vcheck": 3 } }),
    ];

    for value in cases {
        let input = grey(16, 16);
        let accepted = produces(value.clone(), input);

        assert!(accepted.is_ok(), "{value}");
    }
}

#[test]
fn an_unknown_model_or_precision_is_refused() {
    let model = parse_error(json!({ "model": "r4f32" }));
    let precision = parse_error(json!({ "precision": "f8" }));

    assert!(model.contains("unknown variant `r4f32`"), "{model}");
    assert!(precision.contains("unknown variant `f8`"), "{precision}");
}

#[test]
fn hp_and_planes_inside_eedi3_are_refused() {
    let hp = parse_error(json!({ "eedi3": { "hp": false } }));
    let planes = parse_error(json!({ "eedi3": { "planes": ["y"] } }));

    assert!(hp.contains("unknown field `hp`"), "{hp}");
    assert!(planes.contains("unknown field `planes`"), "{planes}");
}

#[test]
fn an_unknown_argument_and_planes_are_refused() {
    let unknown = parse_error(json!({ "mask_threshold": 60 }));
    let planes = parse_error(json!({ "planes": ["y"] }));

    assert!(unknown.contains("unknown field `mask_threshold`"), "{unknown}");
    assert!(planes.contains("unknown field `planes`"), "{planes}");
}

#[test]
fn an_eedi3_vthresh_of_the_wrong_length_is_refused() {
    let message = parse_error(json!({ "eedi3": { "vthresh": [12, 24] } }));

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("needs vthresh of 3 numbers"), "{message}");
    assert!(message.contains("got 2"), "{message}");
}

#[test]
fn edge_mask_false_with_a_bound_mask_is_refused() {
    let input = yuv420(16, 16);
    let mask = Some(mask_format(16, 16));

    let error = produces_with(json!({ "edge_mask": false }), input, mask).expect_err("nothing to merge");
    let without_mask = produces(json!({ "edge_mask": false }), input);
    let with_edge_mask = produces_with(json!({}), input, mask);

    let message = error.to_string();
    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("needs edge_mask = true"), "{message}");
    assert!(message.ends_with("got false"), "{message}");
    assert!(without_mask.is_ok());
    assert!(with_edge_mask.is_ok());
}

#[test]
fn a_mask_of_another_size_is_refused() {
    let input = yuv420(16, 16);
    let small = Some(mask_format(8, 4));

    let error = produces_with(json!({}), input, small).expect_err("too small");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("got 8x4"), "{message}");
}

#[test]
fn the_stage_runs_as_a_single_instance() {
    let parsed = args(json!({}));

    assert_eq!(Based::parallelism(&parsed), Parallelism::Stream);
}

#[test]
fn grey_and_yuv_are_accepted_and_the_output_is_the_input() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let sixteen_bit = SampleFormat::U16 {
        depth: BitDepth::SIXTEEN,
    };
    let layouts = [
        FrameLayout::Grey,
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv422,
        },
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        },
    ];

    for layout in layouts {
        for sample in [SampleFormat::U8, ten_bit, sixteen_bit] {
            let input = format(16, 16, layout, sample);
            let offered = StreamFormat::Frames(input);

            let accepted_set = accepted(offered).expect("accepted");
            let outputs = produces(json!({}), input).expect("valid");
            let chosen = outputs.samples.preferences()[0];
            let output = outputs.template.with_sample(chosen);

            assert!(accepted_set.accepts(sample), "{layout} {sample}");
            assert_eq!(output, offered, "{layout} {sample}");
        }
    }
}

#[test]
fn rgb_and_chroma_layouts_are_refused() {
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let layouts = [
        (FrameLayout::Rgb, "not rgb frames"),
        (chroma, "not chroma (4:2:0) frames"),
    ];

    for (layout, name) in layouts {
        let input = format(16, 16, layout, SampleFormat::U8);
        let offered = StreamFormat::Frames(input);

        let accept_error = accepted(offered).expect_err("the layout is refused");
        let produce_message = refusal(json!({}), input);

        for message in [accept_error.to_string(), produce_message] {
            assert!(message.contains(STAGE), "{message}");
            assert!(message.ends_with(name), "{message}");
        }
    }
}

#[test]
fn floats_are_left_out_and_a_mask_on_the_input_refused() {
    let float_input = format(16, 16, FrameLayout::Grey, SampleFormat::F32);
    let floats = StreamFormat::Frames(float_input);
    let mask = mask_format(16, 16);

    let float_set = accepted(floats).expect("frames are readable");
    let mask_error = accepted(mask).expect_err("a mask is refused");
    let float_message = refusal(json!({}), float_input);

    let float_name = format!("not {} frames", SampleFormat::F32);
    let mask_message = mask_error.to_string();
    assert!(float_set.is_empty());
    assert!(mask_message.contains(STAGE), "{mask_message}");
    assert!(mask_message.contains("not a mask"), "{mask_message}");
    assert!(float_message.contains(STAGE), "{float_message}");
    assert!(float_message.ends_with(&float_name), "{float_message}");
}

#[test]
fn odd_luma_sizes_are_accepted_in_every_direction() {
    let sizes = [grey(63, 45), grey(2, 2), grey(3, 5)];

    for direction in DIRECTIONS {
        for input in sizes {
            let accepted = produces(json!({ "direction": direction }), input);

            assert!(accepted.is_ok(), "{direction} {}x{}", input.width, input.height);
        }
    }
}

#[test]
fn a_luma_plane_under_2_in_either_dimension_is_refused() {
    let cases = [(grey(1, 16), "1x16"), (grey(16, 1), "16x1"), (grey(1, 1), "1x1")];

    for direction in DIRECTIONS {
        for (input, found) in cases {
            let message = refusal(json!({ "direction": direction }), input);
            let size = format!("got {found}");

            assert!(message.contains(STAGE), "{message}");
            assert!(
                message.contains("needs a luma plane of at least 2x2"),
                "{message}"
            );
            assert!(message.ends_with(&size), "{message}");
        }
    }
}

/// Doubled, a 32768 by 16384 luma plane needs more backpointer words than a `u32` indexes at `mdis` 40.
#[test]
fn a_luma_plane_too_large_to_index_once_doubled_is_refused() {
    let large = grey(32768, 16384);

    for direction in DIRECTIONS {
        let value = json!({ "direction": direction, "eedi3": { "mdis": 40 } });
        let message = refusal(value, large);
        let named_direction = format!("with mdis 40 and direction {direction}");

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains(&named_direction), "{message}");
        assert!(message.ends_with("got 32768x16384"), "{message}");
    }

    let nearer = produces(json!({ "direction": "vertical", "eedi3": { "mdis": 1 } }), large);
    assert!(nearer.is_ok());
}

#[test]
fn f16_without_f16_arithmetic_is_refused_suggesting_f32() {
    let capabilities = capabilities_computing(&[ElementType::F32]);
    let frames = Some(StreamFormat::Frames(grey(16, 16)));
    let f16 = args(json!({ "precision": "f16" }));
    let f32 = args(json!({ "precision": "f32" }));

    let error = Based::produces(&capabilities, &f16, &[frames, None]).expect_err("no f16 arithmetic");
    let with_f32 = Based::produces(&capabilities, &f32, &[frames, None]);

    let message = error.to_string();
    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("precision f16"), "{message}");
    assert!(message.ends_with("Use precision = \"f32\""), "{message}");
    assert!(with_f32.is_ok());
}

#[test]
fn a_missing_model_names_the_pull_command() {
    let root = tempfile::tempdir().expect("a temp dir");
    let path = root.path().to_path_buf();
    let store = Store::new(path, "v0.1.0");
    let capabilities = capabilities();
    let parsed = args(json!({ "model": "r16f96" }));
    let frames = Some(StreamFormat::Frames(grey(16, 16)));

    let error =
        planned(&capabilities, &parsed, &[frames, None], &store).expect_err("R16F96 is not installed");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("model = \"r16f96\""), "{message}");
    assert!(message.contains("ArtCNN_R16F96.bin"), "{message}");
    assert!(message.contains("av-toolbox pull artcnn"), "{message}");
}

#[test]
fn the_builtin_model_is_planned_without_a_store() {
    let root = tempfile::tempdir().expect("a temp dir");
    let path = root.path().to_path_buf();
    let store = Store::new(path, "v0.1.0");
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let input = grey(16, 16);
    let frames = Some(StreamFormat::Frames(input));

    let plan = planned(&capabilities, &parsed, &[frames, None], &store).expect("planned");

    assert_eq!(plan.input, input);
    assert!(!plan.masked);
}

#[test]
fn a_frame_without_its_bound_mask_is_refused() {
    let error = checked_frame_mask(true, None).expect_err("the mask is bound");
    let unbound = checked_frame_mask(false, None).expect("no mask is bound");

    let message = error.to_string();
    assert!(message.contains(STAGE), "{message}");
    assert!(
        message.ends_with("was handed a frame without its mask"),
        "{message}"
    );
    assert!(unbound.is_none());
}

#[test]
fn a_bound_mask_is_planned() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let frames = Some(StreamFormat::Frames(yuv420(16, 16)));
    let mask = Some(mask_format(16, 16));
    let store = Store::from_environment();

    let plan = planned(&capabilities, &parsed, &[frames, mask], &store).expect("planned");

    assert!(plan.masked);
}
