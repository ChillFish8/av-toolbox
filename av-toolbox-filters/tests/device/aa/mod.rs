mod sangnom;

use av_toolbox_gpu::{ChromaLocation, FrameFormat, FrameLayout, SampleFormat, Subsampling};

use crate::support::{element_of, frame, from_bytes};

pub(super) fn grey(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    frame(width, height, FrameLayout::Grey, sample, ChromaLocation::Left)
}

pub(super) fn yuv420(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

pub(super) fn bits_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U8 => 8,
        _ => 10,
    }
}

pub(super) fn decoded(format: &FrameFormat, planes: &[Vec<u8>]) -> Vec<Vec<u16>> {
    let element = element_of(format.sample);

    planes.iter().map(|plane| from_bytes(plane, element)).collect()
}

/// Swaps the axes of a `width` by `height` plane, giving a `height` by `width` one.
pub(super) fn transposed(plane: &[f32], width: u32, height: u32) -> Vec<f32> {
    let width = width as usize;
    let height = height as usize;
    let mut output = vec![0.0; plane.len()];

    for row in 0..height {
        for column in 0..width {
            output[column * height + row] = plane[row * width + column];
        }
    }

    output
}

/// Panics at the first sample of a `width` wide plane where `found` and `expected` differ.
pub(super) fn assert_exact(found: &[u16], expected: &[u16], width: u32, label: &str) {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong sample count");

    let differing = found
        .iter()
        .zip(expected)
        .position(|(actual, wanted)| actual != wanted);
    if let Some(index) = differing {
        let column = index as u32 % width;
        let row = index as u32 / width;
        panic!(
            "{label} differs first at {column},{row}: found {}, expected {}",
            found[index], expected[index],
        );
    }
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
