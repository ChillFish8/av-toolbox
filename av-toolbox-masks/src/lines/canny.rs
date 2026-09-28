use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_cubek::blur::{BlurPlanes, Gaussian};
use av_toolbox_cubek::labels::{CountPlanes, KeepPlanes, LabelPlanes, Settled, count, keep, label};
use av_toolbox_cubek::sobel::{SobelPlanes, sobel};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    Mask as GpuMask,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::ServerError;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::luma::{Luma, accepts_luma, load_luma, produces_luma_mask};
use crate::sizing::{Scratch, Sizing, mask_plane};

const NAME: &str = "lines/canny";

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CannyArgs {
    /// Edges weaker than this, in U8 gradient units, are never lines.
    #[serde(default = "default_low")]
    low: f32,
    /// Edges stronger than this, in U8 gradient units, are always lines.
    #[serde(default = "default_high")]
    high: f32,
}

fn default_low() -> f32 {
    50.0
}

fn default_high() -> f32 {
    150.0
}

impl CannyArgs {
    fn validate(&self) -> Result<(), StageError> {
        if self.low < 0.0 {
            let message = format!("{NAME} needs low at or above 0, got {}", self.low);
            return Err(StageError::msg(message));
        }

        if self.high < self.low {
            let message = format!(
                "{NAME} needs high at or above low, got {} and {}",
                self.high, self.low
            );
            return Err(StageError::msg(message));
        }

        Ok(())
    }
}

/// The gradient magnitude at a pixel, or 0 outside the plane.
#[cube]
fn magnitude_at(
    horizontal: &Array<f32>,
    vertical: &Array<f32>,
    width: u32,
    height: u32,
    column: i32,
    row: i32,
) -> f32 {
    let mut magnitude = 0.0f32;

    if column >= 0 && column < width as i32 && row >= 0 && row < height as i32 {
        let index = row as u32 * width + column as u32;
        magnitude = f32::abs(horizontal[index as usize]) + f32::abs(vertical[index as usize]);
    }

    magnitude
}

/// Keeps pixels stronger than the neighbour behind them and at least the one ahead along their
/// gradient, marking candidates and strong edges.
#[cube(launch_unchecked)]
fn suppress(
    horizontal: &Array<f32>,
    vertical: &Array<f32>,
    candidates: &mut Array<u8>,
    strong: &mut Array<u8>,
    width: u32,
    height: u32,
    low: f32,
    high: f32,
) {
    if ABSOLUTE_POS >= candidates.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = (position % width) as i32;
    let row = (position / width) as i32;
    let across = horizontal[ABSOLUTE_POS];
    let down = vertical[ABSOLUTE_POS];
    let magnitude = f32::abs(across) + f32::abs(down);

    let mut step_x = 1i32;
    let mut step_y = 0i32;
    if f32::abs(down) > f32::abs(across) * 2.4142135 {
        step_x = 0i32;
        step_y = 1i32;
    } else if f32::abs(down) > f32::abs(across) * 0.41421357 {
        if across * down > 0.0 {
            step_y = 1i32;
        } else {
            step_y = -1i32;
        }
    }

    let ahead = magnitude_at(horizontal, vertical, width, height, column + step_x, row + step_y);
    let behind = magnitude_at(horizontal, vertical, width, height, column - step_x, row - step_y);
    let kept = magnitude > behind && magnitude >= ahead;

    if kept && magnitude > low {
        candidates[ABSOLUTE_POS] = u8::cast_from(255u32);
    } else {
        candidates[ABSOLUTE_POS] = u8::cast_from(0u32);
    }

    if kept && magnitude > high {
        strong[ABSOLUTE_POS] = u8::cast_from(255u32);
    } else {
        strong[ABSOLUTE_POS] = u8::cast_from(0u32);
    }
}

struct CannyMask<R: Runtime> {
    client: ComputeClient<R>,
    luma: Luma,
    smooth: Gaussian,
    low: f32,
    high: f32,
}

impl<R: Runtime> StageInstance for CannyMask<R> {}

fn no_frame() -> StageError {
    StageError::msg("lines/canny expected a frame in slot 0")
}

fn unreadable_labels(error: ServerError) -> StageError {
    StageError::with_source("lines/canny could not read back its edge labels", error)
}

impl<R: Runtime> Mask for CannyMask<R> {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(no_frame)?;

        let context = &*context;
        let sizing = Sizing(context);
        let width = self.luma.width;
        let height = self.luma.height;
        let samples = self.luma.samples();

        let luma = Scratch::new::<f32>(context, width, height);
        let luma_plane = luma.plane();
        load_luma(&self.client, &sizing, NAME, &frame, &self.luma, luma_plane)?;

        let temporary = Scratch::new::<f32>(context, width, height);
        let smoothed = Scratch::new::<f32>(context, width, height);
        let smooth_planes = BlurPlanes {
            input: luma_plane,
            scratch: temporary.plane(),
            output: smoothed.plane(),
        };
        self.smooth.apply(&self.client, &sizing, smooth_planes);

        let horizontal = Scratch::new::<f32>(context, width, height);
        let vertical = Scratch::new::<f32>(context, width, height);
        let sobel_planes = SobelPlanes {
            input: smoothed.plane(),
            horizontal: horizontal.plane(),
            vertical: vertical.plane(),
        };
        sobel(&self.client, &sizing, sobel_planes);

        let candidates = Scratch::new::<u8>(context, width, height);
        let strong = Scratch::new::<u8>(context, width, height);
        let horizontal_read = horizontal.plane().array::<R>();
        let vertical_read = vertical.plane().array::<R>();
        let candidates_write = candidates.plane().array::<R>();
        let strong_write = strong.plane().array::<R>();
        let cubes = sizing.0.cube_count(samples);
        let shape = sizing.0.cube_dim();

        // Safety: every plane holds `samples` elements of its declared type and the kernel writes
        // nothing past them.
        unsafe {
            suppress::launch_unchecked::<R>(
                &self.client,
                cubes,
                shape,
                horizontal_read,
                vertical_read,
                candidates_write,
                strong_write,
                width,
                height,
                self.low,
                self.high,
            );
        }

        let labels = Scratch::new::<u32>(context, width, height);
        let changed = Scratch::line::<u32>(context, 1);
        let label_planes = LabelPlanes {
            marks: candidates.plane(),
            labels: labels.plane(),
            changed: changed.plane(),
        };
        let settled = label(&self.client, &sizing, label_planes).map_err(unreadable_labels)?;
        if let Settled::Capped { passes } = settled {
            tracing::warn!("lines/canny stopped joining edges after {passes} passes, so some may be split");
        }

        let counts = Scratch::new::<u32>(context, width, height);
        let count_planes = CountPlanes {
            labels: labels.plane(),
            marks: strong.plane(),
            counts: counts.plane(),
        };
        count(&self.client, &sizing, count_planes);

        let meta = frame.meta().clone();
        let format = self.luma.mask_format();
        let output = context.alloc_mask(&format, meta);
        let keep_planes = KeepPlanes {
            labels: labels.plane(),
            counts: counts.plane(),
            output: mask_plane(&output),
        };
        keep(&self.client, &sizing, keep_planes, 1);

        Ok(output)
    }
}

/// Finds edges with Canny's method.
pub(crate) struct Canny;

impl StageDefinition for Canny {
    type Args = CannyArgs;

    fn package() -> &'static str {
        "lines"
    }

    fn identifier() -> &'static str {
        "canny"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MaskDefinition for Canny {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_luma(NAME, offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        args.validate()?;

        produces_luma_mask(NAME, chosen)
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        args.validate()?;

        let luma = Luma::from_formats(NAME, formats)?;
        let client = context.client().clone();
        let smooth = Gaussian::with_radius(&client, 0.8, 1);

        Ok(Box::new(CannyMask {
            client,
            luma,
            smooth,
            low: args.low / 255.0,
            high: args.high / 255.0,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn args(value: serde_json::Value) -> CannyArgs {
        serde_json::from_value(value).expect("the args parse")
    }

    #[test]
    fn the_defaults_are_low_50_and_high_150() {
        let defaults = args(json!({}));

        assert_eq!(defaults.low, 50.0);
        assert_eq!(defaults.high, 150.0);
        assert!(defaults.validate().is_ok());
    }

    #[test]
    fn a_high_below_low_is_refused() {
        let error = args(json!({ "low": 200.0 }))
            .validate()
            .expect_err("high must be at least low");

        assert!(
            error
                .to_string()
                .contains("lines/canny needs high at or above low")
        );
    }

    #[test]
    fn a_negative_low_is_refused() {
        assert!(args(json!({ "low": -1.0 })).validate().is_err());
    }
}
