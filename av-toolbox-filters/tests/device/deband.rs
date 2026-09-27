use av_toolbox_cubek::deband::{DebandParams, salt};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::testing::reference_deband;
use av_toolbox_gpu::{ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{frame, luma_is_shared, passes_through, run_once_at};

fn yuv420(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

/// Every plane of `format`, a diagonal ramp wrapped at `peak`, as u16 values.
fn ramps(format: &FrameFormat, peak: u32) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let column = index % width;
                    let row = index / width;
                    (((column * 3 + row * 5) / 8) % (peak + 1)) as u16
                })
                .collect()
        })
        .collect()
}

fn to_bytes(values: &[u16], element: Element) -> Vec<u8> {
    match element {
        Element::U8 => values.iter().map(|value| *value as u8).collect(),
        Element::U16 => values.iter().flat_map(|value| value.to_le_bytes()).collect(),
    }
}

fn from_bytes(bytes: &[u8], element: Element) -> Vec<u16> {
    match element {
        Element::U8 => bytes.iter().map(|value| u16::from(*value)).collect(),
        Element::U16 => {
            let (pairs, _) = bytes.as_chunks::<2>();
            pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
        },
    }
}

fn assert_mostly_within_one(actual: &[u16], expected: &[u16], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");

    let close = actual
        .iter()
        .zip(expected)
        .filter(|(got, want)| got.abs_diff(**want) <= 1)
        .count();
    let total = expected.len();
    let needed = (total as f64 * 0.999).ceil() as usize;

    assert!(close >= needed, "{label}: {close} of {total} samples within 1");
}

/// Runs `deband/placebo` on a ramp clip at `index` and checks every plane against the reference.
fn matches_reference(input: FrameFormat, element: Element, peak: u32, index: u64) {
    let args = json!({
        "iterations": 2,
        "threshold": [6.0, 4.0, 4.0],
        "grain": [4.0, 2.0, 2.0],
        "seed": 11,
    });
    let values = ramps(&input, peak);
    let planes: Vec<Vec<u8>> = values.iter().map(|plane| to_bytes(plane, element)).collect();

    let debanded = run_once_at("deband/placebo", input, input, args, &planes, index);

    let thresholds = [6.0, 4.0, 4.0];
    let grains = [4.0, 2.0, 2.0];
    let neutrals = [
        16.0 * (peak + 1) as f32 / 256.0,
        (peak + 1) as f32 / 2.0,
        (peak + 1) as f32 / 2.0,
    ];
    for (plane, channel) in input.channels().iter().enumerate() {
        let (width, height) = input.channel_dimensions(*channel).expect("carried");
        let params = DebandParams {
            element,
            iterations: 2,
            radius: 16.0,
            threshold: thresholds[plane],
            grain: grains[plane],
            neutral: neutrals[plane],
            peak: peak as f32,
        };
        let plane_salt = salt(index, 11, plane as u32);
        let expected = reference_deband(&values[plane], width, height, &params, plane_salt);
        let actual = from_bytes(&debanded[plane], element);
        let label = channel.to_string();

        assert_mostly_within_one(&actual, &expected, &label);
    }
}

#[test]
fn eight_and_ten_bit_clips_match_the_reference() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };

    matches_reference(yuv420(64, 32, SampleFormat::U8), Element::U8, 255, 0);
    matches_reference(yuv420(64, 32, ten_bit), Element::U16, 1023, 7);
}

#[test]
fn an_odd_sized_clip_matches_the_reference() {
    matches_reference(yuv420(33, 17, SampleFormat::U8), Element::U8, 255, 3);
}

#[test]
fn the_frame_index_moves_the_grain() {
    let input = yuv420(64, 32, SampleFormat::U8);
    let planes: Vec<Vec<u8>> = input
        .channels()
        .iter()
        .map(|channel| {
            let samples = input.channel_samples(*channel).expect("a listed channel");
            vec![128u8; samples]
        })
        .collect();
    let args = json!({ "grain": 20.0 });

    let first = run_once_at("deband/placebo", input, input, args.clone(), &planes, 4);
    let again = run_once_at("deband/placebo", input, input, args.clone(), &planes, 4);
    let next = run_once_at("deband/placebo", input, input, args, &planes, 5);

    assert_eq!(first, again);
    assert_ne!(first[0], next[0]);
}

#[test]
fn a_skipped_plane_is_shared() {
    let input = yuv420(64, 32, SampleFormat::U8);
    let args = json!({ "threshold": [0.0, 3.0, 3.0], "grain": [0.0, 4.0, 4.0] });

    assert!(luma_is_shared("deband/placebo", input, input, args));
}

#[test]
fn a_frame_with_every_plane_skipped_passes_through() {
    let input = yuv420(64, 32, SampleFormat::U8);
    let args = json!({ "threshold": 0.0, "grain": 0.0 });

    assert!(passes_through("deband/placebo", input, input, args));
}
