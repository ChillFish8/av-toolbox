mod based;
mod eedi2;
mod eedi3;
mod sangnom;

use av_toolbox_gpu::{ChromaLocation, FrameFormat, FrameLayout, SampleFormat, Subsampling};

use crate::support::{element_of, frame, from_bytes};

/// The directions an anti-aliasing stage runs along, as its arguments name them.
pub(super) const DIRECTIONS: [&str; 3] = ["vertical", "horizontal", "both"];

pub(super) fn grey(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    frame(width, height, FrameLayout::Grey, sample, ChromaLocation::Left)
}

pub(super) fn yuv420(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

pub(super) fn yuv422(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv422,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

pub(super) fn bits_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        _ => 8,
    }
}

pub(super) fn decoded(format: &FrameFormat, planes: &[Vec<u8>]) -> Vec<Vec<u16>> {
    let element = element_of(format.sample);

    planes.iter().map(|plane| from_bytes(plane, element)).collect()
}

/// A plane with a one-sample-wide line of 220 on a flat 40, falling one row every 6 columns.
///
/// The line is too shallow for its rows to join up, so it is drawn as a staircase of flat runs.
pub(super) fn one_sample_diagonal(width: u32, height: u32) -> Vec<u16> {
    let first_row = height / 4;

    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            let line_row = first_row + column / 6;

            if row == line_row { 220 } else { 40 }
        })
        .collect()
}

/// The largest absolute difference between two vertically adjacent samples of a `width` wide plane.
///
/// Over a flat background, only the samples around a drawn line differ, so this is the tallest step
/// the line leaves.
pub(super) fn largest_vertical_step(plane: &[u16], width: u32) -> u16 {
    let width = width as usize;
    let upper_rows = plane.chunks_exact(width);
    let lower_rows = plane.chunks_exact(width).skip(1);

    upper_rows
        .zip(lower_rows)
        .flat_map(|(upper, lower)| upper.iter().zip(lower))
        .map(|(above, below)| above.abs_diff(*below))
        .max()
        .unwrap_or(0)
}

/// A deterministic hash of a sample's position, for noise.
pub(super) fn scramble(column: u32, row: u32) -> u32 {
    let mut state = column.wrapping_mul(0x9e37_79b1) ^ row.wrapping_mul(0x85eb_ca77);
    state ^= state >> 15;
    state = state.wrapping_mul(0x2c1b_3c6d);
    state ^= state >> 12;

    state
}

/// Diagonal bands of `low` and `high` with some noise, `slope` columns per row.
pub(super) fn noisy_diagonals(
    width: u32,
    height: u32,
    slope: u32,
    low: u32,
    high: u32,
    noise: u32,
) -> Vec<u16> {
    let mut plane = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let band = (column + slope * row) / 9 % 2;
            let base = if band == 0 { low } else { high };
            let jitter = scramble(column, row) % (noise + 1);
            let value = base + jitter;
            plane.push(value as u16);
        }
    }

    plane
}
