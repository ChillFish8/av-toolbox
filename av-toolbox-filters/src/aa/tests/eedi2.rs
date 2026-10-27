use av_toolbox_core::{FilterDefinition, StageError};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth as BitDepth,
    FormatSet,
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

use crate::aa::Eedi2;
use crate::aa::direction::Direction;
use crate::aa::eedi2::Eedi2Args;
use crate::sample_checks::assert_integers_and_floats_accepted_and_mask_refused;

const STAGE: &str = "aa/eedi2";

fn args(value: serde_json::Value) -> Eedi2Args {
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
        frame_rate: None,
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

    Eedi2::produces(&capabilities, &parsed, &chosen)
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let error = produces(value, input).expect_err("should be refused");

    error.to_string()
}

#[test]
fn the_defaults_match_eedi2_on_the_first_plane_in_both_directions() {
    let parsed = args(json!({}));

    assert_eq!(parsed.mthresh, 10);
    assert_eq!(parsed.lthresh, 20);
    assert_eq!(parsed.vthresh, 20);
    assert_eq!(parsed.estr, 2);
    assert_eq!(parsed.dstr, 4);
    assert_eq!(parsed.maxd, 24);
    assert_eq!(parsed.nt, 50);
    assert_eq!(parsed.pp, 1);
    assert_eq!(parsed.planes, None);
    assert_eq!(parsed.direction, Direction::Both);

    let input = yuv420(16, 16);
    let accepted = produces(json!({}), input);
    assert!(accepted.is_ok());
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (json!({ "maxd": 0 }), "needs maxd between 1 and 29", "got 0"),
        (json!({ "maxd": 30 }), "needs maxd between 1 and 29", "got 30"),
        (json!({ "pp": 2 }), "needs pp between 0 and 1", "got 2"),
        (json!({ "mthresh": -1 }), "needs mthresh of at least 0", "got -1"),
        (json!({ "lthresh": -1 }), "needs lthresh of at least 0", "got -1"),
        (json!({ "vthresh": -1 }), "needs vthresh of at least 0", "got -1"),
        (json!({ "estr": -1 }), "needs estr of at least 0", "got -1"),
        (json!({ "dstr": -1 }), "needs dstr of at least 0", "got -1"),
        (json!({ "nt": -1 }), "needs nt of at least 0", "got -1"),
    ];

    for (value, fragment, expected) in cases {
        let input = grey(16, 16);
        let message = refusal(value, input);

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains(fragment), "{message}");
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn the_range_bounds_are_accepted() {
    let cases = [
        json!({ "maxd": 1, "pp": 0 }),
        json!({ "maxd": 29, "pp": 1 }),
        json!({ "mthresh": 0, "lthresh": 0, "vthresh": 0, "estr": 0, "dstr": 0, "nt": 0 }),
    ];

    for value in cases {
        let input = grey(16, 16);
        let accepted = produces(value.clone(), input);

        assert!(accepted.is_ok(), "{value}");
    }
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<Eedi2Args>(json!({ "map": 1 }));

    let error = parsed.expect_err("map is not an argument");
    let message = error.to_string();

    assert!(message.contains("map"), "{message}");
}

#[test]
fn floats_are_accepted_and_a_mask_on_the_input_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let frame_format = |layout, sample| format(16, 16, layout, sample);
    let accepted = |offered| Eedi2::accepts_input(&capabilities, &parsed, 0, offered, &inputs);

    assert_integers_and_floats_accepted_and_mask_refused(STAGE, layout, frame_format, accepted);
}

#[test]
fn a_bound_mask_makes_the_output_f32() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let frames = Some(StreamFormat::Frames(yuv420(16, 16)));
    let mask = Some(mask_format(16, 16, SampleFormat::U8));

    let outputs = Eedi2::produces(&capabilities, &parsed, &[frames, mask]).expect("valid");

    assert_eq!(outputs.samples, FormatSet::of([SampleFormat::F32]));
}

#[test]
fn f32_input_gives_f32_output() {
    let input = format(16, 16, FrameLayout::Grey, SampleFormat::F32);

    let outputs = produces(json!({}), input).expect("valid");

    assert_eq!(outputs.samples, FormatSet::of([SampleFormat::F32]));
}

#[test]
fn every_layout_is_accepted_and_the_output_is_the_input() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let yuv = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let layouts = [yuv, FrameLayout::Rgb, FrameLayout::Grey, chroma];

    for layout in layouts {
        for sample in [SampleFormat::U8, ten_bit] {
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
fn the_smallest_planes_are_accepted_in_each_direction() {
    let cases = [
        ("vertical", grey(8, 14)),
        ("horizontal", grey(14, 8)),
        ("both", grey(14, 14)),
    ];

    for (direction, input) in cases {
        let accepted = produces(json!({ "direction": direction }), input);

        assert!(accepted.is_ok(), "{direction}");
    }
}

#[test]
fn a_plane_smaller_than_8_by_14_in_the_direction_it_runs_is_refused() {
    let cases = [
        ("vertical", grey(7, 14), "8x14", "7x14"),
        ("vertical", grey(8, 12), "8x14", "8x12"),
        ("horizontal", grey(14, 7), "14x8", "14x7"),
        ("horizontal", grey(12, 8), "14x8", "12x8"),
        ("both", grey(14, 12), "14x14", "14x12"),
        ("both", grey(12, 14), "14x14", "12x14"),
    ];

    for (direction, input, smallest, found) in cases {
        let message = refusal(json!({ "direction": direction }), input);
        let needed = format!("at least {smallest}");
        let named_direction = format!("direction {direction}");

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("plane Y"), "{message}");
        assert!(message.contains(&needed), "{message}");
        assert!(message.contains(&named_direction), "{message}");
        assert!(message.contains(found), "{message}");
    }
}

#[test]
fn a_small_chroma_plane_is_refused_when_selected() {
    let input = yuv420(32, 24);

    let chroma = refusal(json!({ "planes": ["u"], "direction": "vertical" }), input);
    let luma = produces(json!({ "planes": ["y"], "direction": "vertical" }), input);

    assert!(chroma.contains(STAGE), "{chroma}");
    assert!(chroma.contains("plane U"), "{chroma}");
    assert!(chroma.contains("16x12"), "{chroma}");
    assert!(luma.is_ok());
}

#[test]
fn a_row_wider_than_65535_is_refused_in_the_direction_it_runs() {
    let wide = grey(65536, 16);
    let tall = grey(16, 65536);

    let wide_vertical = refusal(json!({ "direction": "vertical" }), wide);
    let tall_horizontal = refusal(json!({ "direction": "horizontal" }), tall);
    let wide_horizontal = produces(json!({ "direction": "horizontal" }), wide);
    let tall_vertical = produces(json!({ "direction": "vertical" }), tall);

    assert!(wide_vertical.contains(STAGE), "{wide_vertical}");
    assert!(
        wide_vertical.contains("width of at most 65535"),
        "{wide_vertical}"
    );
    assert!(wide_vertical.contains("65536"), "{wide_vertical}");
    assert!(tall_horizontal.contains(STAGE), "{tall_horizontal}");
    assert!(
        tall_horizontal.contains("height of at most 65535"),
        "{tall_horizontal}"
    );
    assert!(tall_horizontal.contains("65536"), "{tall_horizontal}");
    assert!(wide_horizontal.is_ok());
    assert!(tall_vertical.is_ok());
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
    assert!(vertical.contains("15"), "{vertical}");
    assert!(horizontal.is_ok());
    assert!(both.contains("even height"), "{both}");
    assert!(both.contains("direction both"), "{both}");
}

#[test]
fn an_odd_width_is_refused_horizontally_but_not_vertically() {
    let input = grey(15, 16);

    let horizontal = refusal(json!({ "direction": "horizontal" }), input);
    let vertical = produces(json!({ "direction": "vertical" }), input);

    assert!(horizontal.contains(STAGE), "{horizontal}");
    assert!(horizontal.contains("even width"), "{horizontal}");
    assert!(horizontal.contains("direction horizontal"), "{horizontal}");
    assert!(horizontal.contains("15"), "{horizontal}");
    assert!(vertical.is_ok());
}

#[test]
fn a_mask_of_another_size_is_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let frames = Some(StreamFormat::Frames(yuv420(16, 16)));
    let small = Some(mask_format(8, 4, SampleFormat::U8));

    let error = Eedi2::produces(&capabilities, &parsed, &[frames, small]).expect_err("too small");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("8x4"), "{message}");
}
