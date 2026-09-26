use av_toolbox_cubek::depth::{Conversion, Thresholds, plane_shift};
use av_toolbox_cubek::testing::reference_depth;
use av_toolbox_gpu::{
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

use crate::support::{frame, passes_through, run_once};

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
