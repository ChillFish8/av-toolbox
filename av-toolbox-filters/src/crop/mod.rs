mod auto;
mod instance;
mod manual;
mod probe;
#[cfg(test)]
mod tests;

use av_toolbox_core::StageError;
use av_toolbox_gpu::{FormatSet, FrameFormat, FrameLayout, StreamFormat};
use schemars::JsonSchema;
use serde::Deserialize;

pub(crate) use self::auto::Auto;
pub(crate) use self::instance::Cropper;
pub(crate) use self::manual::Manual;
pub(crate) use self::probe::{Crop, CropBox, CropRequest};

/// How many samples a crop removes from each edge.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Edges {
    /// Samples removed from the left edge.
    #[serde(default)]
    pub(crate) left: u32,
    /// Samples removed from the right edge.
    #[serde(default)]
    pub(crate) right: u32,
    /// Samples removed from the top edge.
    #[serde(default)]
    pub(crate) top: u32,
    /// Samples removed from the bottom edge.
    #[serde(default)]
    pub(crate) bottom: u32,
}

impl Edges {
    pub(crate) fn is_zero(&self) -> bool {
        *self == Self::default()
    }
}

fn not_frames(stage: &str) -> StageError {
    let message = format!("{stage} crops frames, not masks");

    StageError::msg(message)
}

fn missing_input(stage: &str) -> StageError {
    let message = format!("{stage} needs its input bound");

    StageError::msg(message)
}

fn nothing_left(stage: &str, axis: &str, removed: u64, size: u32) -> StageError {
    let message = format!("{stage} removes {removed} of {size} {axis}, leaving none");

    StageError::msg(message)
}

fn not_multiple(stage: &str, edge: &str, factor: u32, value: u32) -> StageError {
    let message =
        format!("{stage} needs `{edge}` to be a multiple of {factor} for this subsampling, got {value}");

    StageError::msg(message)
}

/// Accepts frames of any sample, since a crop only copies them.
pub(crate) fn accepts_frames_of_any_sample(
    offered: StreamFormat,
    stage: &str,
) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames(stage));
    };

    Ok(FormatSet::of([format.sample]))
}

/// The input format, once it is known to be frames.
pub(crate) fn checked_input(chosen: &[Option<StreamFormat>], stage: &str) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input(stage));
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames(stage));
    };

    Ok(*format)
}

/// The size one axis keeps, refused when nothing is left.
fn kept(stage: &str, axis: &str, size: u32, first: u32, second: u32) -> Result<u32, StageError> {
    let removed = u64::from(first) + u64::from(second);
    if removed >= u64::from(size) {
        return Err(nothing_left(stage, axis, removed, size));
    }

    Ok(size - first - second)
}

/// Refuses edges a YUV input's subsampling cannot split evenly.
fn check_subsampling(input: &FrameFormat, edges: &Edges, stage: &str) -> Result<(), StageError> {
    let FrameLayout::Yuv { subsampling } = input.layout else {
        return Ok(());
    };

    let (factor_x, factor_y) = subsampling.factors();
    let checks = [
        ("left", edges.left, factor_x),
        ("right", edges.right, factor_x),
        ("top", edges.top, factor_y),
        ("bottom", edges.bottom, factor_y),
    ];

    for (edge, value, factor) in checks {
        if !value.is_multiple_of(factor) {
            return Err(not_multiple(stage, edge, factor, value));
        }
    }

    Ok(())
}

/// The format a crop of `edges` leaves of `input`.
pub(crate) fn cropped(input: &FrameFormat, edges: &Edges, stage: &str) -> Result<FrameFormat, StageError> {
    let width = kept(stage, "columns", input.width, edges.left, edges.right)?;
    let height = kept(stage, "rows", input.height, edges.top, edges.bottom)?;
    check_subsampling(input, edges, stage)?;

    Ok(FrameFormat {
        width,
        height,
        ..*input
    })
}
