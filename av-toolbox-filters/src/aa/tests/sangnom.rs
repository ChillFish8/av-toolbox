use av_toolbox_core::{FilterDefinition, StageError};
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

use crate::aa::Sangnom;
use crate::aa::direction::Direction;
use crate::aa::sangnom::SangnomArgs;

const STAGE: &str = "aa/sangnom";

fn args(value: serde_json::Value) -> SangnomArgs {
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

    Sangnom::produces(&capabilities, &parsed, &chosen)
}

fn refusal(value: serde_json::Value, input: FrameFormat) -> String {
    let error = produces(value, input).expect_err("should be refused");

    error.to_string()
}

#[test]
fn the_defaults_are_aa_48_on_the_first_plane_in_both_directions() {
    let parsed = args(json!({}));

    assert_eq!(parsed.aa, 48);
    assert_eq!(parsed.planes, None);
    assert_eq!(parsed.direction, Direction::Both);

    let input = yuv420(16, 8);
    let accepted = produces(json!({}), input);
    assert!(accepted.is_ok());
}

#[test]
fn aa_above_128_is_refused() {
    let input = yuv420(16, 8);
    let message = refusal(json!({ "aa": 129 }), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("needs aa between 0 and 128"), "{message}");
    assert!(message.contains("129"), "{message}");

    let largest = produces(json!({ "aa": 128 }), input);
    assert!(largest.is_ok());
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<SangnomArgs>(json!({ "strength": 3 }));

    let error = parsed.expect_err("strength is not an argument");
    let message = error.to_string();

    assert!(message.contains("strength"), "{message}");
}

#[test]
fn floats_are_left_out_and_a_mask_on_the_input_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..yuv420(16, 8)
    };
    let float = StreamFormat::Frames(float_format);
    let mask = mask_format(16, 8, SampleFormat::U8);

    let accepted = Sangnom::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("readable");
    let refused = Sangnom::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
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
            let input = format(16, 8, layout, sample);
            let offered = StreamFormat::Frames(input);

            let outputs = produces(json!({}), input).expect("valid");
            let chosen = outputs.samples.preferences()[0];
            let output = outputs.template.with_sample(chosen);

            assert_eq!(output, offered, "{layout} {sample}");
        }
    }
}

#[test]
fn an_odd_height_is_refused_vertically_but_not_horizontally() {
    let input = format(16, 9, FrameLayout::Grey, SampleFormat::U8);

    let vertical = refusal(json!({ "direction": "vertical" }), input);
    let horizontal = produces(json!({ "direction": "horizontal" }), input);
    let both = refusal(json!({ "direction": "both" }), input);

    assert!(vertical.contains(STAGE), "{vertical}");
    assert!(vertical.contains("direction"), "{vertical}");
    assert!(vertical.contains("vertical"), "{vertical}");
    assert!(vertical.contains("height"), "{vertical}");
    assert!(vertical.contains('9'), "{vertical}");
    assert!(horizontal.is_ok());
    assert!(both.contains(STAGE), "{both}");
    assert!(both.contains("direction both"), "{both}");
    assert!(both.contains("height"), "{both}");
    assert!(both.contains('9'), "{both}");
}

#[test]
fn an_odd_width_is_refused_horizontally_but_not_vertically() {
    let input = format(15, 8, FrameLayout::Grey, SampleFormat::U8);

    let horizontal = refusal(json!({ "direction": "horizontal" }), input);
    let vertical = produces(json!({ "direction": "vertical" }), input);

    assert!(horizontal.contains(STAGE), "{horizontal}");
    assert!(horizontal.contains("horizontal"), "{horizontal}");
    assert!(horizontal.contains("width"), "{horizontal}");
    assert!(horizontal.contains("15"), "{horizontal}");
    assert!(vertical.is_ok());
}

#[test]
fn a_subsampled_plane_uses_its_own_size() {
    let chroma_selected = json!({ "planes": ["u"], "direction": "vertical" });
    let luma_selected = json!({ "planes": ["y"], "direction": "vertical" });

    let even_clip = yuv420(16, 12);
    let odd_clip = yuv420(16, 10);

    let even_chroma = produces(chroma_selected.clone(), even_clip);
    let odd_chroma = refusal(chroma_selected, odd_clip);
    let luma_only = produces(luma_selected, odd_clip);

    assert!(even_chroma.is_ok());
    assert!(odd_chroma.contains(STAGE), "{odd_chroma}");
    assert!(odd_chroma.contains('U'), "{odd_chroma}");
    assert!(odd_chroma.contains("height"), "{odd_chroma}");
    assert!(odd_chroma.contains('5'), "{odd_chroma}");
    assert!(luma_only.is_ok());
}

#[test]
fn a_mask_of_another_size_is_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let frames = Some(StreamFormat::Frames(yuv420(16, 8)));
    let small = Some(mask_format(8, 4, SampleFormat::U8));

    let error = Sangnom::produces(&capabilities, &parsed, &[frames, small]).expect_err("too small");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("8x4"), "{message}");
}

#[test]
fn a_plane_listed_twice_is_refused() {
    let input = yuv420(16, 8);
    let message = refusal(json!({ "planes": ["u", "u"] }), input);

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains('U') && message.contains("twice"), "{message}");
}
