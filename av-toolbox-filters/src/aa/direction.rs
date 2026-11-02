use std::fmt;

use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::aa::{merge_fields, transpose};
use av_toolbox_gpu::{Channel, Client, FrameContext, FrameFormat};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::scratch::Scratch;
use crate::sizing::Sizing;

/// The axes an anti-aliasing stage works along.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Direction {
    /// Vertically, then horizontally on the vertical result.
    #[default]
    Both,
    /// Rebuilding rows from the rows around them.
    Vertical,
    /// Rebuilding columns from the columns around them.
    Horizontal,
}

impl Direction {
    pub(crate) fn runs_vertically(self) -> bool {
        matches!(self, Self::Both | Self::Vertical)
    }

    pub(crate) fn runs_horizontally(self) -> bool {
        matches!(self, Self::Both | Self::Horizontal)
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Both => "both",
            Self::Vertical => "vertical",
            Self::Horizontal => "horizontal",
        };

        formatter.write_str(name)
    }
}

fn odd_size(stage: &str, channel: Channel, direction: Direction, dimension: &str, size: u32) -> StageError {
    let message = format!(
        "{stage} needs an even {dimension} on plane {channel} with direction {direction}, got {size}"
    );

    StageError::msg(message)
}

pub(crate) fn missing_plane(stage: &str, channel: Channel) -> StageError {
    let message = format!("{stage} was handed a format without a size for plane {channel}");

    StageError::msg(message)
}

/// Refuses a selected plane whose height is odd when running vertically, or whose width is odd when
/// running horizontally.
pub(crate) fn checked_dimensions(
    stage: &str,
    format: &FrameFormat,
    selected: &[bool],
    direction: Direction,
) -> Result<(), StageError> {
    let channels = format.channels();
    let chosen = channels
        .iter()
        .zip(selected)
        .filter(|(_, is_selected)| **is_selected);
    for (channel, _) in chosen {
        let Some((width, height)) = format.channel_dimensions(*channel) else {
            return Err(missing_plane(stage, *channel));
        };

        if direction.runs_vertically() && !height.is_multiple_of(2) {
            return Err(odd_size(stage, *channel, direction, "height", height));
        }

        if direction.runs_horizontally() && !width.is_multiple_of(2) {
            return Err(odd_size(stage, *channel, direction, "width", width));
        }
    }

    Ok(())
}

/// One field pass of an anti-aliaser.
pub(crate) trait FieldPass {
    /// Keeps the rows of `input` whose parity is `offset` and rebuilds the others into `output`.
    ///
    /// Both are f32 planes of one size, with an even height.
    fn run(&self, context: &FrameContext, input: DevicePlane<'_>, output: DevicePlane<'_>, offset: u32);

    /// Runs the pass keeping the even rows into `even_output` and the one keeping the odd rows into
    /// `odd_output`.
    fn run_both(
        &self,
        context: &FrameContext,
        input: DevicePlane<'_>,
        even_output: DevicePlane<'_>,
        odd_output: DevicePlane<'_>,
    ) {
        self.run(context, input, even_output, 0);
        self.run(context, input, odd_output, 1);
    }
}

/// Anti-aliases `input` into `output` along `direction`.
///
/// A vertical run averages the `vertical` pass keeping the even rows with the one keeping the odd rows.
/// A horizontal run does the same with the `horizontal` pass on the transposed plane.
pub(crate) fn run_direction(
    client: &Client,
    context: &FrameContext,
    vertical: &impl FieldPass,
    horizontal: &impl FieldPass,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    direction: Direction,
) {
    match direction {
        Direction::Vertical => run_vertical(client, context, vertical, input, output),
        Direction::Horizontal => run_horizontal(client, context, horizontal, input, output),
        Direction::Both => {
            let vertical_result = Scratch::new::<f32>(context, input.width(), input.height());
            let vertical_plane = vertical_result.plane();

            run_vertical(client, context, vertical, input, vertical_plane);
            run_horizontal(client, context, horizontal, vertical_plane, output);
        },
    }
}

/// Averages the field pass keeping the even rows with the one keeping the odd rows.
fn run_vertical(
    client: &Client,
    context: &FrameContext,
    pass: &impl FieldPass,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let width = input.width();
    let height = input.height();
    let even_kept = Scratch::new::<f32>(context, width, height);
    let odd_kept = Scratch::new::<f32>(context, width, height);
    let even_plane = even_kept.plane();
    let odd_plane = odd_kept.plane();

    pass.run_both(context, input, even_plane, odd_plane);

    let sizing = Sizing(context);
    merge_fields(client, &sizing, even_plane, odd_plane, output);
}

/// Anti-aliases the transposed plane vertically and transposes the result back.
fn run_horizontal(
    client: &Client,
    context: &FrameContext,
    pass: &impl FieldPass,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let width = input.width();
    let height = input.height();
    let sizing = Sizing(context);
    let transposed = Scratch::new::<f32>(context, height, width);
    let filtered = Scratch::new::<f32>(context, height, width);
    let transposed_plane = transposed.plane();
    let filtered_plane = filtered.plane();

    transpose(client, &sizing, input, transposed_plane);
    run_vertical(client, context, pass, transposed_plane, filtered_plane);
    transpose(client, &sizing, filtered_plane, output);
}
