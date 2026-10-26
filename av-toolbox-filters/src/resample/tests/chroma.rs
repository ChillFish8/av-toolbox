use av_toolbox_core::FilterDefinition;
use av_toolbox_gpu::{
    Capabilities,
    ChromaLocation,
    ColourInfo,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Offered,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::resample::Chroma;
use crate::resample::chroma::{ChromaArgs, chroma_axes};

fn yuv(width: u32, height: u32, subsampling: Subsampling, location: ChromaLocation) -> FrameFormat {
    let colour = ColourInfo {
        chroma_location: location,
        ..ColourInfo::default()
    };

    FrameFormat {
        width,
        height,
        layout: FrameLayout::Yuv { subsampling },
        sample: SampleFormat::U8,
        colour,
        frame_rate: None,
        field_order: None,
    }
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn produced(
    input: FrameFormat,
    value: serde_json::Value,
) -> Result<FrameFormat, av_toolbox_core::StageError> {
    let capabilities = capabilities();
    let arguments: ChromaArgs = serde_json::from_value(value).expect("parses");
    let chosen = [Some(StreamFormat::Frames(input))];
    let outputs = Chroma::produces(&capabilities, &arguments, &chosen)?;
    let format = outputs.template.with_sample(input.sample);

    Ok(*format.as_frames().expect("frames"))
}

#[test]
fn left_sited_420_to_444_reads_between_chroma_samples() {
    let input = yuv(16, 8, Subsampling::Yuv420, ChromaLocation::Left);
    let axes = chroma_axes(
        &input,
        Subsampling::Yuv420,
        Subsampling::Yuv444,
        ChromaLocation::Left,
    );

    assert_eq!((axes.width, axes.height), (16, 8));
    assert_eq!(axes.across, (0.5, 0.0));
    assert_eq!(axes.down, (0.5, -0.25));
}

#[test]
fn left_to_center_moves_chroma_a_quarter_sample() {
    let input = yuv(16, 8, Subsampling::Yuv420, ChromaLocation::Left);
    let axes = chroma_axes(
        &input,
        Subsampling::Yuv420,
        Subsampling::Yuv420,
        ChromaLocation::Center,
    );

    assert_eq!((axes.width, axes.height), (8, 4));
    assert_eq!(axes.across, (1.0, 0.25));
    assert_eq!(axes.down, (1.0, 0.0));
}

#[test]
fn center_sited_444_to_420_averages_pairs() {
    let input = yuv(16, 8, Subsampling::Yuv444, ChromaLocation::Center);
    let axes = chroma_axes(
        &input,
        Subsampling::Yuv444,
        Subsampling::Yuv420,
        ChromaLocation::Center,
    );

    assert_eq!(axes.across, (2.0, 0.5));
    assert_eq!(axes.down, (2.0, 0.5));
}

#[test]
fn the_output_takes_the_new_subsampling_and_siting() {
    let input = yuv(16, 8, Subsampling::Yuv420, ChromaLocation::Left);
    let value = json!({ "subsampling": "4:4:4", "location": "center" });

    let output = produced(input, value).expect("converts");

    assert_eq!(
        output.layout,
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444
        }
    );
    assert_eq!(output.colour.chroma_location, ChromaLocation::Center);
    assert_eq!((output.width, output.height), (16, 8));
}

#[test]
fn the_siting_defaults_to_the_inputs() {
    let input = yuv(16, 8, Subsampling::Yuv444, ChromaLocation::TopLeft);
    let value = json!({ "subsampling": "4:2:0" });

    let output = produced(input, value).expect("converts");

    assert_eq!(output.colour.chroma_location, ChromaLocation::TopLeft);
}

#[test]
fn an_odd_width_is_refused_for_420() {
    let input = yuv(15, 8, Subsampling::Yuv444, ChromaLocation::Left);
    let value = json!({ "subsampling": "4:2:0" });

    let error = produced(input, value).expect_err("15 does not halve");

    assert!(error.to_string().contains("width"), "{error}");
    assert!(error.to_string().contains('2'), "{error}");
}

#[test]
fn only_yuv_clips_are_accepted() {
    let grey = FrameFormat {
        layout: FrameLayout::Grey,
        ..yuv(16, 8, Subsampling::Yuv444, ChromaLocation::Left)
    };
    let value = json!({ "subsampling": "4:4:4" });

    let refused = produced(grey, value);

    assert!(refused.is_err());
}

#[test]
fn rgb_is_refused() {
    let rgb = FrameFormat {
        layout: FrameLayout::Rgb,
        ..yuv(16, 8, Subsampling::Yuv444, ChromaLocation::Left)
    };
    let value = json!({ "subsampling": "4:4:4" });

    let refused = produced(rgb, value);

    assert!(refused.is_err());
}

#[test]
fn a_chroma_clip_is_refused() {
    let chroma = FrameFormat {
        layout: FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        ..yuv(16, 8, Subsampling::Yuv444, ChromaLocation::Left)
    };
    let value = json!({ "subsampling": "4:4:4" });

    let refused = produced(chroma, value);

    assert!(refused.is_err());
}

#[test]
fn a_mask_is_refused() {
    let capabilities = capabilities();
    let arguments: ChromaArgs = serde_json::from_value(json!({ "subsampling": "4:4:4" })).expect("parses");
    let inputs = [Offered::Unknown];
    let mask = StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let refused = Chroma::accepts_input(&capabilities, &arguments, 0, mask, &inputs);

    assert!(refused.is_err());
}

#[test]
fn floats_give_an_empty_set() {
    let capabilities = capabilities();
    let arguments: ChromaArgs = serde_json::from_value(json!({ "subsampling": "4:4:4" })).expect("parses");
    let inputs = [Offered::Unknown];
    let float = StreamFormat::Frames(FrameFormat {
        sample: SampleFormat::F32,
        ..yuv(16, 8, Subsampling::Yuv444, ChromaLocation::Left)
    });

    let accepted =
        Chroma::accepts_input(&capabilities, &arguments, 0, float, &inputs).expect("frames are readable");

    assert!(accepted.is_empty());
}

#[test]
fn a_kernel_setting_for_another_kernel_is_refused() {
    let input = yuv(16, 8, Subsampling::Yuv420, ChromaLocation::Left);
    let value = json!({ "subsampling": "4:4:4", "kernel": "spline36", "b": 0.0 });

    let refused = produced(input, value);

    assert!(refused.is_err());
}
