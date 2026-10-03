use av_toolbox_core::{FilterDefinition, StageError};
use av_toolbox_cubek::aa::Eedi3Settings;
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth as BitDepth,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Offered,
    OutputFormats,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::aa::Eedi3;
use crate::aa::direction::Direction;
use crate::aa::eedi3::{Eedi3Args, checked_arguments};

const STAGE: &str = "aa/eedi3";

const DIRECTIONS: [&str; 3] = ["vertical", "horizontal", "both"];

fn args(value: serde_json::Value) -> Eedi3Args {
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
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn mask_format(width: u32, height: u32, sample: SampleFormat) -> StreamFormat {
    StreamFormat::Mask(MaskFormat {
        width,
        height,
        sample,
        chroma: None,
    })
}

fn produces(value: serde_json::Value, input: FrameFormat) -> Result<OutputFormats, StageError> {
    let capabilities = capabilities();
    let parsed = args(value);
    let chosen = [Some(StreamFormat::Frames(input)), None];

    Eedi3::produces(&capabilities, &parsed, &chosen)
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let error = produces(value, input).expect_err("should be refused");

    error.to_string()
}

#[test]
fn the_defaults_match_eedi3_on_the_first_plane_in_both_directions() {
    let parsed = args(json!({}));

    assert_eq!(parsed.alpha, 0.2);
    assert_eq!(parsed.beta, 0.25);
    assert_eq!(parsed.gamma, 20.0);
    assert_eq!(parsed.nrad, 2);
    assert_eq!(parsed.mdis, 20);
    assert_eq!(parsed.vcheck, 2);
    assert_eq!(parsed.vthresh, [32.0, 64.0, 4.0]);
    assert_eq!(parsed.planes, None);
    assert_eq!(parsed.direction, Direction::Both);

    let input = yuv420(16, 16);
    let accepted = produces(json!({}), input);
    assert!(accepted.is_ok());
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (
            json!({ "alpha": -0.1 }),
            "needs alpha of 0 or between 1e-30 and 1",
            "got -0.1",
        ),
        (
            json!({ "alpha": 1.1, "beta": 0 }),
            "needs alpha of 0 or between 1e-30 and 1",
            "got 1.1",
        ),
        (
            json!({ "alpha": 1e-31 }),
            "needs alpha of 0 or between 1e-30 and 1",
            "got 1e-31",
        ),
        (
            json!({ "beta": -0.1 }),
            "needs beta of 0 or between 1e-30 and 1",
            "got -0.1",
        ),
        (
            json!({ "alpha": 0, "beta": 1.1 }),
            "needs beta of 0 or between 1e-30 and 1",
            "got 1.1",
        ),
        (
            json!({ "beta": 1e-31 }),
            "needs beta of 0 or between 1e-30 and 1",
            "got 1e-31",
        ),
        (
            json!({ "alpha": 0.6, "beta": 0.5 }),
            "needs alpha and beta to add up to at most 1",
            "got 1.1",
        ),
        (
            json!({ "gamma": -1 }),
            "needs gamma between 0 and 1e30",
            "got -1.0",
        ),
        (
            json!({ "gamma": 2e30 }),
            "needs gamma between 0 and 1e30",
            "got 2e30",
        ),
        (json!({ "nrad": 4 }), "needs nrad between 0 and 3", "got 4"),
        (json!({ "mdis": 0 }), "needs mdis between 1 and 40", "got 0"),
        (json!({ "mdis": 41 }), "needs mdis between 1 and 40", "got 41"),
        (json!({ "vcheck": 4 }), "needs vcheck between 0 and 3", "got 4"),
        (
            json!({ "vthresh": [0, 64, 4] }),
            "needs a finite vthresh[0] of at least 1e-30",
            "got 0.0",
        ),
        (
            json!({ "vthresh": [32, 0, 4] }),
            "needs a finite vthresh[1] of at least 1e-30",
            "got 0.0",
        ),
        (
            json!({ "vthresh": [32, 64, 0] }),
            "needs a finite vthresh[2] of at least 1e-30",
            "got 0.0",
        ),
        (
            json!({ "vthresh": [1e-31, 64, 4] }),
            "needs a finite vthresh[0] of at least 1e-30",
            "got 1e-31",
        ),
        (
            json!({ "vthresh": [32, 1e-31, 4] }),
            "needs a finite vthresh[1] of at least 1e-30",
            "got 1e-31",
        ),
        (
            json!({ "vthresh": [32, 64, 1e-31] }),
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
fn a_vthresh_of_the_wrong_length_is_refused() {
    let parsed = serde_json::from_value::<Eedi3Args>(json!({ "vthresh": [32, 64] }));

    let error = parsed.expect_err("vthresh takes 3 numbers");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("needs vthresh of 3 numbers"), "{message}");
    assert!(message.ends_with("got 2"), "{message}");
}

fn range_bounds() -> [serde_json::Value; 12] {
    [
        json!({ "alpha": 0 }),
        json!({ "alpha": 1, "beta": 0 }),
        json!({ "alpha": 0.5, "beta": 0.5 }),
        json!({ "alpha": 0.25, "beta": 0.750000001 }),
        json!({ "alpha": 1e-30, "beta": 1e-30 }),
        json!({ "mdis": 1 }),
        json!({ "mdis": 40 }),
        json!({ "gamma": 0 }),
        json!({ "gamma": 1e30 }),
        json!({ "nrad": 0, "vcheck": 0 }),
        json!({ "nrad": 3, "vcheck": 3 }),
        json!({ "vthresh": [1e-30, 1e-30, 1e-30] }),
    ]
}

/// `0.25 + 0.750000001` passes 1 in f64 but is exactly 1 in f32.
#[test]
fn the_range_bounds_are_accepted() {
    let cases = range_bounds();

    for value in cases {
        let input = grey(16, 16);
        let accepted = produces(value.clone(), input);

        assert!(accepted.is_ok(), "{value}");
    }
}

#[test]
fn the_range_bounds_build_settings_at_every_depth() {
    for value in range_bounds() {
        let parsed = args(value);
        let arguments = checked_arguments(&parsed).expect("the bounds are accepted");

        for bits in [8, 16] {
            Eedi3Settings::new(&arguments, bits);
        }
    }
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<Eedi3Args>(json!({ "map": 1 }));

    let error = parsed.expect_err("map is not an argument");
    let message = error.to_string();

    assert!(message.contains("map"), "{message}");
}

#[test]
fn hp_is_refused() {
    let parsed = serde_json::from_value::<Eedi3Args>(json!({ "hp": false }));

    let error = parsed.expect_err("hp is not an argument");
    let message = error.to_string();

    assert!(message.contains("hp"), "{message}");
}

#[test]
fn floats_are_left_out_and_a_mask_on_the_input_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..yuv420(16, 16)
    };
    let float = StreamFormat::Frames(float_format);
    let mask = mask_format(16, 16, SampleFormat::U8);

    let accepted = Eedi3::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("readable");
    let refused = Eedi3::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}

#[test]
fn every_layout_is_accepted_and_the_output_is_the_input() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let sixteen_bit = SampleFormat::U16 {
        depth: BitDepth::SIXTEEN,
    };
    let yuv = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let layouts = [yuv, FrameLayout::Rgb, FrameLayout::Grey, chroma];

    for layout in layouts {
        for sample in [SampleFormat::U8, ten_bit, sixteen_bit] {
            let input = format(16, 16, layout, sample);
            let offered = StreamFormat::Frames(input);

            let outputs = produces(json!({}), input).expect("valid");
            let chosen = outputs.samples.preferences()[0];
            let output = outputs.template.with_sample(chosen);

            assert_eq!(output, offered, "{layout} {sample}");
        }
    }
}

#[test]
fn a_4_by_4_plane_is_accepted_in_every_direction() {
    let input = grey(4, 4);

    for direction in DIRECTIONS {
        let accepted = produces(json!({ "direction": direction }), input);

        assert!(accepted.is_ok(), "{direction}");
    }
}

#[test]
fn a_plane_under_4_in_either_dimension_is_refused_in_every_direction() {
    let cases = [(grey(2, 16), "2x16"), (grey(16, 2), "16x2"), (grey(2, 2), "2x2")];

    for direction in DIRECTIONS {
        for (input, found) in cases {
            let message = refusal(json!({ "direction": direction }), input);
            let size = format!("got {found}");

            assert!(message.contains(STAGE), "{message}");
            assert!(message.contains("plane Y"), "{message}");
            assert!(message.contains("at least 4x4"), "{message}");
            assert!(message.ends_with(&size), "{message}");
        }
    }
}

#[test]
fn a_small_chroma_plane_is_refused_when_selected() {
    let input = yuv420(6, 16);

    let chroma = refusal(json!({ "planes": ["u"], "direction": "vertical" }), input);
    let luma = produces(json!({ "planes": ["y"], "direction": "vertical" }), input);

    assert!(chroma.contains(STAGE), "{chroma}");
    assert!(chroma.contains("plane U"), "{chroma}");
    assert!(chroma.ends_with("got 3x8"), "{chroma}");
    assert!(luma.is_ok());
}

/// Past `u32::MAX` backpointer words, or samples, the device cannot index the plane.
#[test]
fn a_plane_too_large_to_index_is_refused() {
    let large = grey(65536, 32768);
    let huge = grey(65536, 65536);

    for direction in DIRECTIONS {
        let message = refusal(json!({ "direction": direction, "mdis": 40 }), large);
        let named_direction = format!("direction {direction}");

        assert!(message.contains(STAGE), "{message}");
        assert!(
            message.contains("plane Y small enough to index with mdis 40"),
            "{message}"
        );
        assert!(message.contains(&named_direction), "{message}");
        assert!(message.ends_with("got 65536x32768"), "{message}");
    }

    let nearer = produces(json!({ "direction": "vertical", "mdis": 1 }), large);
    let too_many_samples = refusal(json!({ "direction": "vertical", "mdis": 1 }), huge);

    assert!(nearer.is_ok());
    assert!(
        too_many_samples.ends_with("got 65536x65536"),
        "{too_many_samples}"
    );
}

#[test]
fn an_odd_height_is_refused_vertically_but_not_horizontally() {
    let input = grey(16, 15);

    let vertical = refusal(json!({ "direction": "vertical" }), input);
    let horizontal = produces(json!({ "direction": "horizontal" }), input);
    let both = refusal(json!({ "direction": "both" }), input);

    assert!(vertical.contains(STAGE), "{vertical}");
    assert!(vertical.contains("even height"), "{vertical}");
    assert!(vertical.contains("direction vertical"), "{vertical}");
    assert!(vertical.ends_with("got 15"), "{vertical}");
    assert!(horizontal.is_ok());
    assert!(both.contains("even height"), "{both}");
    assert!(both.contains("direction both"), "{both}");
}

#[test]
fn an_odd_width_is_refused_horizontally_but_not_vertically() {
    let input = grey(15, 16);

    let horizontal = refusal(json!({ "direction": "horizontal" }), input);
    let vertical = produces(json!({ "direction": "vertical" }), input);
    let both = refusal(json!({ "direction": "both" }), input);

    assert!(horizontal.contains(STAGE), "{horizontal}");
    assert!(horizontal.contains("even width"), "{horizontal}");
    assert!(horizontal.contains("direction horizontal"), "{horizontal}");
    assert!(horizontal.ends_with("got 15"), "{horizontal}");
    assert!(vertical.is_ok());
    assert!(both.contains("even width"), "{both}");
    assert!(both.contains("direction both"), "{both}");
}

#[test]
fn a_mask_of_another_size_is_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let frames = Some(StreamFormat::Frames(yuv420(16, 16)));
    let small = Some(mask_format(8, 4, SampleFormat::U8));

    let error = Eedi3::produces(&capabilities, &parsed, &[frames, small]).expect_err("too small");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("8x4"), "{message}");
}
