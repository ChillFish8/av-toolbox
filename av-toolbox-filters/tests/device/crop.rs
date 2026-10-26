use av_toolbox_gpu::{Channel, ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{frame, passes_through, run_once};

/// Every plane of `format`, each byte `(index * 7 + plane * 13) % 251`, so every sample is distinct across a row.
fn byte_planes(format: &FrameFormat) -> Vec<Vec<u8>> {
    format
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let bytes = format.channel_bytes(*channel).expect("a listed channel");
            (0..bytes)
                .map(|index| ((index * 7 + plane * 13) % 251) as u8)
                .collect()
        })
        .collect()
}

/// The window of each plane a crop of `left` and `top` to `output` keeps, read on the host.
fn expected_window(
    input: &FrameFormat,
    output: &FrameFormat,
    planes: &[Vec<u8>],
    left: u32,
    top: u32,
) -> Vec<Vec<u8>> {
    let sample_bytes = input.sample.bytes();

    input
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let (in_width, _) = input.channel_dimensions(*channel).expect("listed");
            let (out_width, out_height) = output.channel_dimensions(*channel).expect("listed");
            let (plane_left, plane_top) = plane_offset(input, *channel, left, top);
            let mut window = Vec::new();

            for row in 0..out_height {
                let start = ((row + plane_top) * in_width + plane_left) as usize * sample_bytes;
                let end = start + out_width as usize * sample_bytes;
                window.extend_from_slice(&planes[plane][start..end]);
            }

            window
        })
        .collect()
}

fn plane_offset(format: &FrameFormat, channel: Channel, left: u32, top: u32) -> (u32, u32) {
    let FrameLayout::Yuv { subsampling } = format.layout else {
        return (left, top);
    };

    if channel == Channel::Y {
        return (left, top);
    }

    let (factor_x, factor_y) = subsampling.factors();
    (left / factor_x, top / factor_y)
}

fn check_crop(layout: FrameLayout, sample: SampleFormat) {
    let input = frame(16, 8, layout, sample, ChromaLocation::Left);
    let output = FrameFormat {
        width: 10,
        height: 4,
        ..input
    };
    let planes = byte_planes(&input);
    let args = json!({ "left": 2, "right": 4, "top": 2, "bottom": 2 });

    let found = run_once("crop/manual", input, output, args, &planes);

    let expected = expected_window(&input, &output, &planes, 2, 2);
    assert_eq!(found, expected, "{layout:?} {sample}");
}

#[test]
fn every_layout_and_sample_keeps_exactly_its_window() {
    let layouts = [
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv422,
        },
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        },
        FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        FrameLayout::Grey,
        FrameLayout::Rgb,
    ];
    let samples = [
        SampleFormat::U8,
        SampleFormat::U16 { depth: Depth::TEN },
        SampleFormat::F32,
    ];

    for layout in layouts {
        for sample in samples {
            check_crop(layout, sample);
        }
    }
}

#[test]
fn a_crop_of_nothing_passes_the_frame_through() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(16, 8, layout, SampleFormat::U8, ChromaLocation::Left);
    let args = json!({});

    let shared = passes_through("crop/manual", input, input, args);

    assert!(shared);
}
