use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_gpu::{ColourRange, FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::support::{
    SelectFrames,
    backend,
    built_mask,
    f32_grey,
    float_bytes,
    mask_format,
    sixteen_bit_grey,
    wide_bytes,
};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const OUTLIER_SHARE: f32 = 0.0;

/// 16-bit codes with a ramp, a bright box with a darker core and some noise, so every family
/// has edges, detail and lines to find.
fn scene() -> Vec<u16> {
    let mut state = 12345u32;

    (0..HEIGHT)
        .flat_map(|row| (0..WIDTH).map(move |column| (row, column)))
        .map(|(row, column)| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((state >> 16) % 600) as u16;
            let ramp = (column * 300 + 4096) as u16;
            let boxed = (20..60).contains(&column) && (12..44).contains(&row);
            let core = (30..50).contains(&column) && (20..36).contains(&row);

            match (boxed, core) {
                (_, true) => 20000 + noise,
                (true, false) => 50000 + noise,
                _ => ramp + noise,
            }
        })
        .collect()
}

fn masks(stage: &str, args: &Value, format: FrameFormat, plane: Vec<u8>) -> Vec<u8> {
    let backend = backend();
    let output = mask_format(WIDTH, HEIGHT);
    let inputs = [StreamFormat::Frames(format)];
    let mask = built_mask(&backend, stage, args.clone(), &inputs, output);
    let frames = vec![vec![plane]];
    let mut selected = backend.build(SelectFrames { mask, format, frames });

    selected.remove(0)
}

/// The share of samples differing by more than one level.
fn differing_share(actual: &[u8], expected: &[u8]) -> f32 {
    assert_eq!(actual.len(), expected.len());

    let differing = actual
        .iter()
        .zip(expected)
        .filter(|(got, want)| got.abs_diff(**want) > 1)
        .count();

    differing as f32 / expected.len() as f32
}

fn assert_twin(stage: &str, args: Value) {
    let codes = scene();
    let twin_plane = wide_bytes(&codes);
    let twin_format = sixteen_bit_grey(WIDTH, HEIGHT);
    let expected = masks(stage, &args, twin_format, twin_plane);
    assert!(expected.iter().any(|level| *level > 0), "{stage} found nothing");

    let ranges = [
        (ColourRange::Limited, SampleRange::Limited),
        (ColourRange::Full, SampleRange::Full),
    ];
    for (label, range) in ranges {
        let float_plane = float_bytes(&codes, range);
        let float_format = f32_grey(WIDTH, HEIGHT, label);
        let actual = masks(stage, &args, float_format, float_plane);

        let share = differing_share(&actual, &expected);
        assert!(
            share <= OUTLIER_SHARE,
            "{stage} at {label:?} differs from its sixteen bit twin on {share}"
        );
    }
}

#[test]
fn an_f32_clip_masks_like_its_sixteen_bit_twin() {
    assert_twin("detail/mask", json!({}));
    assert_twin("edges/sobel", json!({}));
    assert_twin("edges/retinex", json!({}));
    assert_twin("lines/canny", json!({}));
}
