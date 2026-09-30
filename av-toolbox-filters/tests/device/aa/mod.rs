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
