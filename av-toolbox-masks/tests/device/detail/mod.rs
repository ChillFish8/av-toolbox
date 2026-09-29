mod mask;
mod neo;
mod range;

use av_toolbox_cubek::neighbourhood::{Expansion, Extreme, Shape, expansion_passes};
use av_toolbox_cubek::testing::reference_extreme;
use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::Value;

use crate::support::{SelectFrames, backend, built_mask, mask_format};

fn run_stage(stage: &str, args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    let backend = backend();
    let output = mask_format(format.width, format.height);
    let mask = built_mask(&backend, stage, args, &[StreamFormat::Frames(format)], output);

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

/// The local maximum minus the local minimum over `radius`.
fn reference_range_values(luma: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let passes = expansion_passes(radius, radius, Expansion::Ellipse);

    let maximum = expanded(luma, width, height, &passes, Extreme::Maximum);
    let minimum = expanded(luma, width, height, &passes, Extreme::Minimum);

    maximum
        .iter()
        .zip(&minimum)
        .map(|(largest, smallest)| largest - smallest)
        .collect()
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

/// The blur radius vs-jetpack gives `sigma`, `ceil(6 * sigma + 1) / 2`.
fn jetpack_blur_radius(sigma: f32) -> u32 {
    let reach = (6.0 * sigma + 1.0).ceil();

    reach as u32 / 2
}

/// A smooth gradient on the left half and a busy texture on the right half.
fn detail_scene(width: u32, height: u32) -> Vec<u8> {
    let half = width / 2;

    (0..height)
        .flat_map(|row| {
            (0..width).map(move |column| {
                if column < half {
                    return (100 + column / 64) as u8;
                }

                ((row * 37 + column * 17 + row * column) % 251) as u8
            })
        })
        .collect()
}

/// The number of `masks` samples that differ from `expected`.
fn differing_count(masks: &[u8], expected: &[u8]) -> usize {
    masks
        .iter()
        .zip(expected)
        .filter(|(got, want)| got != want)
        .count()
}

/// The share of the samples in `columns` of every row that satisfy `matches`.
fn column_share(
    masks: &[u8],
    width: u32,
    columns: std::ops::Range<u32>,
    matches: impl Fn(u8) -> bool,
) -> f32 {
    let selected: Vec<u8> = masks
        .chunks(width as usize)
        .flat_map(|row| row[columns.start as usize..columns.end as usize].iter().copied())
        .collect();
    let matching = selected.iter().filter(|sample| matches(**sample)).count();

    matching as f32 / selected.len() as f32
}

fn wide_code_values(samples: &[u16]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample)).collect()
}

/// How far the nearest `cleaned` value sits from a store half, and how many values sit within
/// `NEAR_HALF` of one, once stored at `peak` and computed in f64. A value that close can round the
/// other way on a device.
fn store_half_margin(cleaned: &[f32], peak: f32) -> (f64, usize) {
    let distances: Vec<f64> = cleaned
        .iter()
        .map(|value| store_half_distance(*value, peak))
        .collect();

    let closest = distances.iter().copied().fold(f64::INFINITY, f64::min);
    let near = distances.iter().filter(|distance| **distance < NEAR_HALF).count();

    (closest, near)
}

/// How far `value` sits from a store half once stored at `peak`, computed in f64.
fn store_half_distance(value: f32, peak: f32) -> f64 {
    let stored = f64::from(value).clamp(0.0, f64::from(peak)) * 255.0 / f64::from(peak);
    let fraction = stored - stored.floor();

    (fraction - 0.5).abs()
}

/// How close to a store half counts as near it, well above the device's float error at 255.
const NEAR_HALF: f64 = 0.001;
