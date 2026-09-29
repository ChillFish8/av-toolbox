use av_toolbox_cubek::neighbourhood::{Expansion, Extreme, Shape, expansion_passes};
use av_toolbox_cubek::testing::reference_extreme;
use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::reference::square;
use crate::support::{SelectFrames, backend, built_mask, grey, mask_format, ten_bit_grey, wide_bytes};

fn run(args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    let backend = backend();
    let output = mask_format(format.width, format.height);
    let mask = built_mask(
        &backend,
        "detail/range",
        args,
        &[StreamFormat::Frames(format)],
        output,
    );

    backend.build(SelectFrames { mask, format, frames })
}

fn code_values(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample)).collect()
}

/// Mirrors `store_mask`, writing `round(clamp(v, 0, peak) * 255 / peak)` as u8.
fn store(values: &[f32], peak: f32) -> Vec<u8> {
    values
        .iter()
        .map(|value| {
            let clamped = value.clamp(0.0, peak);

            (clamped * 255.0 / peak).round() as u8
        })
        .collect()
}

/// Runs `passes` of `extreme` over the CPU, feeding each pass into the next.
fn expanded(input: &[f32], width: u32, height: u32, passes: &[Shape], extreme: Extreme) -> Vec<f32> {
    let mut plane = input.to_vec();
    for shape in passes {
        plane = reference_extreme(&plane, width, height, extreme, *shape);
    }

    plane
}

/// The local maximum minus the local minimum over `radius`, times `multi`, stored at `peak`.
fn reference_range_mask(
    luma: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    multi: f32,
    peak: f32,
) -> Vec<u8> {
    let passes = expansion_passes(radius, radius, Expansion::Ellipse);

    let maximum = expanded(luma, width, height, &passes, Extreme::Maximum);
    let minimum = expanded(luma, width, height, &passes, Extreme::Minimum);

    let scaled: Vec<f32> = maximum
        .iter()
        .zip(&minimum)
        .map(|(largest, smallest)| (largest - smallest) * multi)
        .collect();

    store(&scaled, peak)
}

/// A busy pattern that never repeats within a row or a column, so its local maximum and minimum
/// differ almost everywhere.
fn busy_frame(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|row| {
            (0..width).map(move |column| {
                let value = (row * 37 + column * 17 + row * column) % 251;

                value as u8
            })
        })
        .collect()
}

#[test]
fn detail_range_matches_the_reference() {
    let width = 24;
    let height = 24;
    let peak = 255.0;
    let square_luma = square(width, height);
    let busy_luma = busy_frame(width, height);

    for radius in [1u32, 2, 5] {
        for luma in [&square_luma, &busy_luma] {
            let luma_values = code_values(luma);
            let expected = reference_range_mask(&luma_values, width, height, radius, 1.0, peak);

            let format = grey(width, height);
            let frames = vec![vec![luma.to_vec()]];
            let request = json!({ "radius": radius });
            let masks = run(request, format, frames);

            assert_eq!(
                masks[0], expected,
                "radius {radius} did not match the reference exactly"
            );
        }
    }
}

#[test]
fn a_flat_frame_gives_an_empty_range_mask() {
    let width = 16;
    let height = 16;
    let luma = vec![128u8; (width * height) as usize];

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run(json!({}), format, frames);

    assert!(
        masks[0].iter().all(|sample| *sample == 0),
        "a flat frame found a range"
    );
}

/// A 10-bit frame split into a `500` half and a `600` half, a single 100-code step.
fn ten_bit_step_frame(width: u32, height: u32) -> Vec<u16> {
    let half = width / 2;

    (0..height)
        .flat_map(|_row| (0..width).map(move |column| if column < half { 500u16 } else { 600u16 }))
        .collect()
}

/// At radius 1, the samples beside the step hold `100`, which stores as
/// `round(100 * 255 / 1023) = round(24.93) = 25`.
#[test]
fn ten_bit_range_stores_scaled() {
    let width = 10;
    let height = 4;
    let luma = ten_bit_step_frame(width, height);

    let format = ten_bit_grey(width, height);
    let wide_plane = wide_bytes(&luma);
    let frames = vec![vec![wide_plane]];
    let request = json!({ "radius": 1 });
    let masks = run(request, format, frames);

    for row in 0..height {
        let left_of_step = (row * width + 4) as usize;
        let right_of_step = (row * width + 5) as usize;

        assert_eq!(
            masks[0][left_of_step], 25,
            "row {row} left of the step did not scale to 25"
        );
        assert_eq!(
            masks[0][right_of_step], 25,
            "row {row} right of the step did not scale to 25"
        );
    }
}

#[test]
fn multi_scales_and_saturates() {
    let width = 24;
    let height = 24;
    let luma = square(width, height);

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run(json!({ "multi": 50.0 }), format, frames);

    let edge_samples: Vec<u8> = masks[0].iter().copied().filter(|sample| *sample != 0).collect();
    assert!(!edge_samples.is_empty(), "no edge was found");
    assert!(
        edge_samples.iter().all(|sample| *sample == 255),
        "an edge sample did not saturate to 255"
    );
}

/// At `multi = 0.5` the square's edge range of 160 becomes 80, neither 0 nor saturated.
#[test]
fn a_fractional_multi_scales_without_saturating() {
    let width = 24;
    let height = 24;
    let peak = 255.0;
    let luma = square(width, height);
    let luma_values = code_values(&luma);
    let expected = reference_range_mask(&luma_values, width, height, 2, 0.5, peak);
    assert!(expected.contains(&80), "the reference holds no scaled edge of 80");

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run(json!({ "multi": 0.5 }), format, frames);

    assert_eq!(
        masks[0], expected,
        "multi 0.5 did not match the reference exactly"
    );
}
