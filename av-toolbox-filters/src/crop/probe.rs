use std::collections::HashMap;
use std::fmt;

use av_toolbox_core::{ProbeDefinition, Seeker, StageError};
use av_toolbox_cubek::values::FloatCodes;
use av_toolbox_gpu::{Channel, ColourRange, FrameFormat, FrameLayout, HostFrame, SampleFormat};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::Edges;
use crate::codes::{FLOAT_CODE_BITS, sample_range};

/// Finds the black bars around a source's picture.
pub(crate) struct Crop;

/// How many frames to measure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct CropRequest {
    pub(crate) samples: u32,
}

/// The bars found on each edge, and the size of the frames they were found on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CropBox {
    pub(crate) edges: Edges,
    pub(crate) measured: (u32, u32),
}

impl fmt::Display for CropBox {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Edges {
            left,
            right,
            top,
            bottom,
        } = self.edges;
        let (width, height) = self.measured;

        write!(
            formatter,
            "top {top}, bottom {bottom}, left {left}, right {right} of {width}x{height}"
        )
    }
}

/// What makes a row or column a bar, in the source's own sample values.
struct BarTest {
    black: u64,
    tolerance: u64,
}

/// One frame's luma, widened to `u16`.
struct Luma {
    samples: Vec<u16>,
    width: usize,
    height: usize,
}

/// The frames measured for `samples` asked of a clip of `total`, spread to skip its start and end.
pub(crate) fn sample_indices(total: usize, samples: u32) -> Vec<usize> {
    let wanted = samples as usize;
    if total <= wanted {
        return (0..total).collect();
    }

    let step = 1024 / (u64::from(samples) + 1);
    let length = total as u64;

    (1..=u64::from(samples))
        .map(|position| (length * position * step / 1024) as usize)
        .collect()
}

fn not_luma(layout: FrameLayout) -> StageError {
    let message = format!("crop/auto needs a YUV or grey source, not {layout:?}");

    StageError::msg(message)
}

fn different_sizes(first: (u32, u32), second: (u32, u32)) -> StageError {
    let message = format!(
        "crop measured {}x{} at one source and {}x{} at another",
        first.0, first.1, second.0, second.1
    );

    StageError::msg(message)
}

/// The bar test for `format`, or why its frames cannot be measured.
fn bar_test(format: &FrameFormat) -> Result<BarTest, StageError> {
    if !matches!(format.layout, FrameLayout::Yuv { .. } | FrameLayout::Grey) {
        return Err(not_luma(format.layout));
    }

    let bits = match format.sample.max_value() {
        Some(largest) => u16::BITS - largest.leading_zeros(),
        None => FLOAT_CODE_BITS,
    };
    let shift = bits - 8;
    let tolerance = 16u64 << shift;
    let black = match format.colour.range {
        ColourRange::Full => 0,
        ColourRange::Limited | ColourRange::Unspecified => tolerance,
    };

    Ok(BarTest { black, tolerance })
}

/// F32 luma as whole 16-bit codes in the frame's range, held to `0..=65535`.
fn float_codes(bytes: &[u8], range: ColourRange) -> Vec<u16> {
    let mapping = FloatCodes::new(FLOAT_CODE_BITS, sample_range(range), false);
    let (words, _) = bytes.as_chunks::<4>();

    words
        .iter()
        .map(|word| {
            let sample = f32::from_le_bytes(*word);
            let code = sample * mapping.scale + mapping.offset;
            code.round().clamp(0.0, mapping.peak) as u16
        })
        .collect()
}

fn luma_of(frame: &HostFrame) -> Luma {
    let format = frame.format();
    let plane = frame.plane(Channel::Y).expect("a YUV or grey frame carries luma");
    let bytes = plane.bytes();
    let samples = match format.sample {
        SampleFormat::U8 => bytes.iter().map(|value| u16::from(*value)).collect(),
        SampleFormat::F32 => float_codes(bytes, format.colour.range),
        _ => {
            let (pairs, _) = bytes.as_chunks::<2>();
            pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
        },
    };

    Luma {
        samples,
        width: plane.width() as usize,
        height: plane.height() as usize,
    }
}

/// Whether a line of samples is dark and flat enough to be a bar.
fn is_bar(test: &BarTest, line: impl Iterator<Item = u16> + Clone) -> bool {
    let raised = line.map(|value| u64::from(value).max(test.black));
    let count = raised.clone().count() as u64;
    let total: u64 = raised.clone().sum();
    let mean = total / count.max(1);

    let dark = mean < test.black + test.tolerance;
    let flat = raised.clone().all(|value| value <= mean + test.tolerance);

    dark && flat
}

fn row_is_bar(luma: &Luma, test: &BarTest, row: usize) -> bool {
    let start = row * luma.width;
    let line = luma.samples[start..start + luma.width].iter().copied();

    is_bar(test, line)
}

fn column_is_bar(luma: &Luma, test: &BarTest, column: usize) -> bool {
    let line = luma.samples.iter().skip(column).step_by(luma.width).copied();

    is_bar(test, line)
}

/// The bars of one frame, or `None` when every line is a bar.
fn measure(luma: &Luma, test: &BarTest) -> Option<Edges> {
    let top = (0..luma.height)
        .take_while(|row| row_is_bar(luma, test, *row))
        .count();
    if top == luma.height {
        return None;
    }

    let bottom = (0..luma.height)
        .rev()
        .take_while(|row| row_is_bar(luma, test, *row))
        .count();
    let left = (0..luma.width)
        .take_while(|column| column_is_bar(luma, test, *column))
        .count();
    if left == luma.width {
        return None;
    }

    let right = (0..luma.width)
        .rev()
        .take_while(|column| column_is_bar(luma, test, *column))
        .count();

    Some(Edges {
        left: left as u32,
        right: right as u32,
        top: top as u32,
        bottom: bottom as u32,
    })
}

fn smallest(first: Edges, second: Edges) -> Edges {
    Edges {
        left: first.left.min(second.left),
        right: first.right.min(second.right),
        top: first.top.min(second.top),
        bottom: first.bottom.min(second.bottom),
    }
}

/// Measures the frames one request asks for, reusing frames an earlier request measured.
fn crop_for(
    seeker: &mut dyn Seeker,
    test: &BarTest,
    samples: u32,
    measured: &mut HashMap<usize, Option<Edges>>,
) -> Result<Edges, StageError> {
    let indices = sample_indices(seeker.frame_count(), samples);
    let mut found: Option<Edges> = None;

    for index in indices {
        if found.is_some_and(|edges| edges.is_zero()) {
            break;
        }

        let edges = match measured.get(&index) {
            Some(edges) => *edges,
            None => {
                let frame = seeker.read(index)?;
                let luma = luma_of(&frame);
                let edges = measure(&luma, test);
                measured.insert(index, edges);
                edges
            },
        };

        let Some(edges) = edges else {
            continue;
        };

        let narrowed = match found {
            Some(earlier) => smallest(earlier, edges),
            None => edges,
        };
        found = Some(narrowed);
    }

    Ok(found.unwrap_or_default())
}

impl ProbeDefinition for Crop {
    type Request = CropRequest;
    type Output = CropBox;

    fn kind() -> &'static str {
        "crop"
    }

    fn describe(request: &Self::Request) -> String {
        format!("{} samples", request.samples)
    }

    fn probe(seeker: &mut dyn Seeker, requests: &[Self::Request]) -> Result<Vec<Self::Output>, StageError> {
        let format = *seeker.format();
        let test = bar_test(&format)?;
        let mut measured = HashMap::new();
        let mut boxes = Vec::with_capacity(requests.len());

        for request in requests {
            let edges = crop_for(seeker, &test, request.samples, &mut measured)?;
            let found = CropBox {
                edges,
                measured: (format.width, format.height),
            };
            boxes.push(found);
        }

        Ok(boxes)
    }

    fn combine(outputs: Vec<Self::Output>) -> Result<Self::Output, StageError> {
        let mut outputs = outputs.into_iter();
        let Some(mut combined) = outputs.next() else {
            return Ok(CropBox {
                edges: Edges::default(),
                measured: (0, 0),
            });
        };

        for other in outputs {
            if other.measured != combined.measured {
                return Err(different_sizes(combined.measured, other.measured));
            }

            combined.edges = smallest(combined.edges, other.edges);
        }

        Ok(combined)
    }
}
