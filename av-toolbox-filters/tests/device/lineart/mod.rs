mod darken;
mod thin;
mod toon;
mod vinverse;

use av_toolbox_cubek::depth::Element;
use av_toolbox_gpu::{Channel, ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};

use crate::support::{frame, from_bytes, to_bytes};

pub(super) const WIDTH: u32 = 64;

pub(super) const HEIGHT: u32 = 48;

pub(super) fn ten_bit() -> SampleFormat {
    SampleFormat::U16 { depth: Depth::TEN }
}

pub(super) fn yuv420(sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(WIDTH, HEIGHT, layout, sample, ChromaLocation::Left)
}

pub(super) fn element_of(sample: SampleFormat) -> Element {
    match sample {
        SampleFormat::U8 => Element::U8,
        _ => Element::U16,
    }
}

pub(super) fn peak_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U8 => 255,
        _ => 1023,
    }
}

pub(super) fn encoded(format: &FrameFormat, values: &[Vec<u16>]) -> Vec<Vec<u8>> {
    let element = element_of(format.sample);

    values.iter().map(|plane| to_bytes(plane, element)).collect()
}

/// How far a plane is from the reference.
pub(super) struct Agreement {
    pub(super) largest: u16,
    pub(super) off_by_one: usize,
}

pub(super) fn agreement(actual: &[u16], expected: &[u16]) -> Agreement {
    assert_eq!(actual.len(), expected.len());

    let differences: Vec<u16> = actual
        .iter()
        .zip(expected)
        .map(|(found, wanted)| found.abs_diff(*wanted))
        .collect();
    let largest = differences.iter().copied().max().unwrap_or(0);
    let off_by_one = differences.iter().filter(|difference| **difference == 1).count();

    Agreement { largest, off_by_one }
}

/// The samples of `plane` that differ from `other`.
pub(super) fn changed(plane: &[u16], other: &[u16]) -> usize {
    plane
        .iter()
        .zip(other)
        .filter(|(left, right)| left != right)
        .count()
}

/// A scene of three bands of rows, scaled to the sample's depth.
///
/// - The top band is a flat 200 with 1-wide lines. Three of them sit 1 code below, on and 1 code
///   above `cap - thr`, so their closing straddles `x + thr`. The others are deeper.
/// - The middle band is hashed noise, which puts lines of every depth in every direction.
/// - The bottom band is a flat 230, above the cap, with dark 1-wide and 2-wide lines.
pub(super) fn lined_scene(format: &FrameFormat) -> Vec<Vec<u16>> {
    let sample = format.sample;
    let peak = peak_of(sample);
    let scale = (peak + 1) / 256;
    let cap = 191 * scale;
    let thr = 4 * scale;
    let depths = [
        cap - thr - 1,
        cap - thr,
        cap - thr + 1,
        150 * scale,
        100 * scale,
        20 * scale,
    ];

    format
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let column = index % width;
                    let row = index / width;
                    let band = row * 3 / height;
                    let value = match band {
                        0 if column % 4 == 2 => depths[(column / 4) as usize % depths.len()],
                        0 => 200 * scale,
                        1 => {
                            let hash = (index + plane as u32 * 7).wrapping_mul(2_654_435_761) >> 20;
                            (60 + hash % 140) * scale
                        },
                        _ if column % 8 == 3 => 150 * scale,
                        _ if column % 8 == 6 || column % 8 == 7 => 100 * scale,
                        _ => 230 * scale,
                    };

                    value.min(peak) as u16
                })
                .collect()
        })
        .collect()
}

/// The distance from a rounding boundary within which an f32 result may round the other way.
///
/// It is about 16 f32 steps at 1023, generous for the dozen operations a sample goes through.
pub(super) const RISK: f32 = 1e-3;

/// How close the reference's samples sit to a rounding boundary.
pub(super) struct Rounding {
    /// The indices of the samples within `RISK` of one, boundary included.
    pub(super) at_risk: Vec<usize>,
    /// The distance of the closest sample that is not exactly on one.
    pub(super) nearest: f32,
}

/// How close the samples of `filtered` sit to a rounding boundary.
pub(super) fn rounding_of(filtered: &[f32]) -> Rounding {
    let margins: Vec<f32> = filtered
        .iter()
        .map(|value| ((value - value.floor()) - 0.5).abs())
        .collect();
    let at_risk: Vec<usize> = margins
        .iter()
        .enumerate()
        .filter(|(_, margin)| **margin < RISK)
        .map(|(index, _)| index)
        .collect();
    let nearest = margins
        .iter()
        .copied()
        .filter(|margin| *margin > 0.0)
        .fold(f32::MAX, f32::min);

    Rounding { at_risk, nearest }
}

/// A `yuv420` frame whose planes are `background`, with a 1-wide vertical line of `line` down the
/// middle column of the luma plane.
pub(super) fn faded_line_planes(format: &FrameFormat, background: u16, line: u16) -> Vec<Vec<u8>> {
    let element = element_of(format.sample);

    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            let values: Vec<u16> = (0..width * height)
                .map(|index| {
                    let on_line = *channel == Channel::Y && index % width == width / 2;
                    if on_line { line } else { background }
                })
                .collect();

            to_bytes(&values, element)
        })
        .collect()
}

/// The luma samples of a 64x48 frame, as a plane of codes.
pub(super) fn luma_codes(planes: &[Vec<u8>], sample: SampleFormat) -> Vec<u16> {
    let element = element_of(sample);

    from_bytes(&planes[0], element)
}
