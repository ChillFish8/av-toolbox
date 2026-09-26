use av_toolbox_cubek::depth::{Conversion, Thresholds, plane_shift};
use av_toolbox_cubek::resize::{Kernel, taps};
use av_toolbox_cubek::testing::{reference_depth, reference_resize};
use av_toolbox_gpu::{
    Channel,
    ChromaLocation,
    ColourInfo,
    ColourRange,
    Depth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    Subsampling,
};
use serde_json::json;

use crate::support::{
    assert_within_one,
    every_plane_is_shared,
    frame,
    luma_is_shared,
    passes_through,
    patterned_planes,
    run_once,
};

/// Every plane of `format`, each sample `(index * 13) % 1024`.
fn ten_bit_values(format: &FrameFormat) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            (0..samples).map(|index| (index * 13 % 1024) as u16).collect()
        })
        .collect()
}

fn widen(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn widen_u8(bytes: &[u8]) -> Vec<u16> {
    bytes.iter().map(|sample| u16::from(*sample)).collect()
}

fn unwiden(bytes: &[u8]) -> Vec<u16> {
    let (pairs, _) = bytes.as_chunks::<2>();

    pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
}

/// Every plane of `format`, each sample `(index * 13) % 256`.
fn eight_bit_values(format: &FrameFormat) -> Vec<Vec<u8>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            (0..samples).map(|index| (index * 13 % 256) as u8).collect()
        })
        .collect()
}

#[test]
fn lowering_a_ten_bit_frame_matches_the_reference() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(32, 16, layout, ten_bit, ChromaLocation::Left);
    let output = FrameFormat {
        sample: SampleFormat::U8,
        ..input
    };
    let args = json!({ "depth": 8, "dither": "ordered" });

    let values = ten_bit_values(&input);
    let planes: Vec<Vec<u8>> = values.iter().map(|plane| widen(plane)).collect();
    let converted = run_once("resample/depth", input, output, args, &planes);

    let conversion = Conversion::new(10, 8, false);
    let thresholds = Thresholds::ordered();
    for (index, channel) in input.channels().iter().enumerate() {
        let (width, _) = input.channel_dimensions(*channel).expect("carried");
        let shift = plane_shift(index as u32);
        let expected = reference_depth(&values[index], width, &conversion, &thresholds, shift);
        let actual = widen_u8(&converted[index]);

        assert_eq!(actual, expected, "{channel}");
    }
}

#[test]
fn raising_the_depth_ignores_the_dither() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let narrow = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let colour = ColourInfo {
        range: ColourRange::Full,
        ..narrow.colour
    };
    let input = FrameFormat { colour, ..narrow };
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let output = FrameFormat {
        sample: ten_bit,
        ..input
    };
    let args = json!({ "depth": 10, "dither": "ordered" });

    let planes = eight_bit_values(&input);
    let converted = run_once("resample/depth", input, output, args, &planes);

    let conversion = Conversion::new(8, 10, true);
    let thresholds = Thresholds::rounding();
    for (index, channel) in input.channels().iter().enumerate() {
        let (width, _) = input.channel_dimensions(*channel).expect("carried");
        let shift = plane_shift(index as u32);
        let values = widen_u8(&planes[index]);
        let expected = reference_depth(&values, width, &conversion, &thresholds, shift);
        let actual = unwiden(&converted[index]);

        assert_eq!(actual, expected, "{channel}");
    }
}

#[test]
fn an_unchanged_depth_hands_back_the_input() {
    let layout = FrameLayout::Grey;
    let input = frame(16, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({ "depth": 8 });

    let same = passes_through("resample/depth", input, input, args);

    assert!(same);
}

#[test]
fn every_kernel_matches_the_reference_between_subsamplings() {
    let kernels = [
        ("point", Kernel::Point),
        ("bilinear", Kernel::Bilinear),
        (
            "bicubic",
            Kernel::Bicubic {
                b: 1.0 / 3.0,
                c: 1.0 / 3.0,
            },
        ),
        ("lanczos", Kernel::Lanczos { taps: 3 }),
        ("spline16", Kernel::Spline16),
        ("spline36", Kernel::Spline36),
        ("spline64", Kernel::Spline64),
    ];
    let changes = [
        (Subsampling::Yuv420, Subsampling::Yuv444),
        (Subsampling::Yuv444, Subsampling::Yuv420),
        (Subsampling::Yuv422, Subsampling::Yuv420),
    ];

    for (name, kernel) in kernels {
        for (from, to) in changes {
            let input = frame(
                32,
                16,
                FrameLayout::Yuv { subsampling: from },
                SampleFormat::U8,
                ChromaLocation::Left,
            );
            let output = FrameFormat {
                layout: FrameLayout::Yuv { subsampling: to },
                ..input
            };
            let target = json!(to.to_string());
            let args = json!({ "subsampling": target, "kernel": name });

            let planes = patterned_planes(&input);
            let resampled = run_once("resample/chroma", input, output, args, &planes);

            let (in_width, in_height) = input.channel_dimensions(Channel::U).expect("carried");
            let (out_width, out_height) = output.channel_dimensions(Channel::U).expect("carried");
            let (factor_in_x, factor_in_y) = from.factors();
            let (factor_out_x, factor_out_y) = to.factors();
            let (siting_in_x, siting_in_y) = ChromaLocation::Left.offsets(factor_in_x, factor_in_y);
            let (siting_out_x, siting_out_y) = ChromaLocation::Left.offsets(factor_out_x, factor_out_y);
            let scale_x = factor_out_x as f32 / factor_in_x as f32;
            let scale_y = factor_out_y as f32 / factor_in_y as f32;
            let offset_x = (siting_out_x - siting_in_x) / factor_in_x as f32;
            let offset_y = (siting_out_y - siting_in_y) / factor_in_y as f32;
            let across = taps(out_width, scale_x, offset_x, &kernel);
            let down = taps(out_height, scale_y, offset_y, &kernel);

            assert_eq!(resampled[0], planes[0], "{name}: luma must be untouched");
            for index in [1, 2] {
                let expected = reference_resize(&planes[index], in_width, in_height, &across, &down);
                assert_within_one(&resampled[index], &expected, name);
            }
        }
    }
}

#[test]
fn the_luma_plane_is_shared_not_copied() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let output = FrameFormat {
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        },
        ..input
    };
    let args = json!({ "subsampling": "4:4:4" });

    let shared = luma_is_shared("resample/chroma", input, output, args);

    assert!(shared);
}

#[test]
fn a_new_siting_at_444_relabels_every_plane_instead_of_resampling() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = frame(32, 16, layout, SampleFormat::U8, ChromaLocation::Left);
    let colour = ColourInfo {
        chroma_location: ChromaLocation::Center,
        ..input.colour
    };
    let output = FrameFormat { colour, ..input };
    let args = json!({ "subsampling": "4:4:4", "location": "center" });

    let shared = every_plane_is_shared("resample/chroma", input, output, args);

    assert!(shared);
}
