use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition, StageError};
use av_toolbox_cubek::artcnn::{BodyPath, Model, Precision};
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
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use av_toolbox_models::{Located, Store};
use serde_json::json;

use crate::upscale::Artcnn;
use crate::upscale::artcnn::{ArtcnnArgs, ModelName, PrecisionName, allocation_cap, body_path, locate};

const STAGE: &str = "upscale/artcnn";

fn args(value: serde_json::Value) -> ArtcnnArgs {
    serde_json::from_value(value).expect("parses")
}

fn format(width: u32, height: u32, layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

fn capabilities_computing(elements: &[ElementType]) -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .computable(elements.iter().copied())
        .build()
}

fn capabilities() -> Capabilities {
    capabilities_computing(&[ElementType::F32, ElementType::F16])
}

fn produced(
    capabilities: &Capabilities,
    input: FrameFormat,
    value: serde_json::Value,
) -> Result<FrameFormat, StageError> {
    let arguments = args(value);
    let chosen = [Some(StreamFormat::Frames(input))];
    let outputs = Artcnn::produces(capabilities, &arguments, &chosen)?;
    let output = outputs.template.with_sample(input.sample);
    let frames = output.as_frames().expect("frames");

    Ok(*frames)
}

fn accepted(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let capabilities = capabilities();
    let arguments = args(json!({}));
    let inputs = [Offered::Unknown];

    Artcnn::accepts_input(&capabilities, &arguments, 0, offered, &inputs)
}

#[test]
fn the_defaults_are_r8f64_at_f16() {
    let parsed = args(json!({}));

    assert_eq!(parsed.model, ModelName::R8f64);
    assert_eq!(parsed.precision, PrecisionName::F16);
}

#[test]
fn the_model_and_precision_parse_from_their_names() {
    let parsed = args(json!({ "model": "r16f96", "precision": "f32" }));

    assert_eq!(parsed.model, ModelName::R16f96);
    assert_eq!(parsed.precision, PrecisionName::F32);
}

#[test]
fn an_unknown_model_is_refused() {
    let value = json!({ "model": "r4f32" });

    let error = serde_json::from_value::<ArtcnnArgs>(value).expect_err("r4f32 is no model");
    let message = error.to_string();

    assert!(message.contains("r4f32"), "{message}");
}

#[test]
fn grey_and_every_yuv_subsampling_double() {
    let capabilities = capabilities();
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
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
    let samples = [SampleFormat::U8, ten_bit];
    let sizes = [(16, 8), (15, 9)];

    for layout in layouts {
        for sample in samples {
            for (width, height) in sizes {
                let input = format(width, height, layout, sample);

                let output = produced(&capabilities, input, json!({})).expect("upscalable");

                let expected = FrameFormat {
                    width: width * 2,
                    height: height * 2,
                    ..input
                };
                assert_eq!(output, expected, "{layout} {sample} {width}x{height}");
            }
        }
    }
}

#[test]
fn rgb_and_chroma_layouts_are_refused() {
    let capabilities = capabilities();
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let layouts = [(FrameLayout::Rgb, "rgb"), (chroma, "chroma")];

    for (layout, name) in layouts {
        let input = format(16, 8, layout, SampleFormat::U8);
        let offered = StreamFormat::Frames(input);

        let accept_error = accepted(offered).expect_err("the layout is refused");
        let produce_error = produced(&capabilities, input, json!({})).expect_err("the layout is refused");

        for message in [accept_error.to_string(), produce_error.to_string()] {
            assert!(message.contains(STAGE), "{message}");
            assert!(message.contains(name), "{message}");
        }
    }
}

#[test]
fn floats_are_left_out_and_a_mask_is_refused() {
    let capabilities = capabilities();
    let float_input = format(16, 8, FrameLayout::Grey, SampleFormat::F32);
    let floats = StreamFormat::Frames(float_input);
    let mask = StreamFormat::Mask(MaskFormat {
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let float_set = accepted(floats).expect("frames are readable");
    let mask_error = accepted(mask).expect_err("a mask is refused");
    let float_error = produced(&capabilities, float_input, json!({})).expect_err("floats are refused");

    let mask_message = mask_error.to_string();
    let float_message = float_error.to_string();
    let float_name = SampleFormat::F32.to_string();
    assert!(float_set.is_empty());
    assert!(mask_message.contains(STAGE), "{mask_message}");
    assert!(mask_message.contains("mask"), "{mask_message}");
    assert!(float_message.contains(STAGE), "{float_message}");
    assert!(float_message.contains(&float_name), "{float_message}");
}

#[test]
fn f16_without_f16_arithmetic_is_refused_suggesting_f32() {
    let capabilities = capabilities_computing(&[ElementType::F32]);
    let input = format(16, 8, FrameLayout::Grey, SampleFormat::U8);

    let error = produced(&capabilities, input, json!({ "precision": "f16" })).expect_err("no f16 arithmetic");
    let accepted = produced(&capabilities, input, json!({ "precision": "f32" }));

    let message = error.to_string();
    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("precision"), "{message}");
    assert!(message.contains("f16"), "{message}");
    assert!(message.contains("\"f32\""), "{message}");
    assert!(accepted.is_ok());
}

#[test]
fn a_missing_model_names_the_pull_command() {
    let root = tempfile::tempdir().expect("a temp dir");
    let path = root.path().to_path_buf();
    let store = Store::new(path, "v0.1.0");

    let error = locate(&store, Model::R16F96).expect_err("R16F96 is not installed");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("ArtCNN_R16F96.bin"), "{message}");
    assert!(message.contains("av-toolbox pull artcnn"), "{message}");
}

#[test]
fn the_builtin_model_is_found_without_a_store() {
    let root = tempfile::tempdir().expect("a temp dir");
    let path = root.path().to_path_buf();
    let store = Store::new(path, "v0.1.0");

    let located = locate(&store, Model::R8F64).expect("R8F64 is built in");

    assert!(matches!(located, Located::Builtin(_)));
}

#[test]
fn the_stage_runs_frames_in_parallel() {
    let arguments = args(json!({}));

    assert_eq!(Artcnn::parallelism(&arguments), Parallelism::Frame);
}

fn capabilities_multiplying(combinations: &[(ElementType, ElementType)]) -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .cooperative_matrix(combinations.iter().copied())
        .build()
}

#[test]
fn the_convolution_path_is_chosen_when_the_device_offers_it() {
    let f16_only = capabilities_multiplying(&[(ElementType::F16, ElementType::F32)]);
    let none = capabilities_multiplying(&[]);
    let both = capabilities_multiplying(&[
        (ElementType::F16, ElementType::F32),
        (ElementType::F32, ElementType::F32),
    ]);

    assert_eq!(body_path(&f16_only, Precision::F16), BodyPath::Convolution);
    assert_eq!(body_path(&f16_only, Precision::F32), BodyPath::Im2col);
    assert_eq!(body_path(&none, Precision::F16), BodyPath::Im2col);
    assert_eq!(body_path(&none, Precision::F32), BodyPath::Im2col);
    assert_eq!(body_path(&both, Precision::F16), BodyPath::Convolution);
    assert_eq!(body_path(&both, Precision::F32), BodyPath::Convolution);
}

#[test]
fn an_unset_allocation_limit_is_no_cap() {
    let unset = capabilities();
    let accelerator = default_accelerators()[0];
    let limited = Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .max_allocation(1 << 20)
        .build();

    assert_eq!(allocation_cap(&unset), usize::MAX);
    assert_eq!(allocation_cap(&limited), 1 << 20);
}
