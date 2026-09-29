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
use av_toolbox_cubek::DevicePlane;
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
use schemars::JsonSchema;
use serde::Deserialize;

use super::range;
use crate::luma::{Luma, accepts_luma, load_luma_code_values, produces_luma_mask, store_mask};
use crate::sizing::{Scratch, Sizing, mask_plane};

const NAME: &str = "detail/range";

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct RangeArgs {
    /// How far, in samples, the local maximum and minimum reach.
    #[serde(default = "default_radius")]
    radius: u32,
    /// Multiplies the range before it is stored.
    #[serde(default = "default_multi")]
    multi: f32,
}

fn default_radius() -> u32 {
    2
}

fn default_multi() -> f32 {
    1.0
}

impl RangeArgs {
    fn validate(&self) -> Result<(), StageError> {
        if !(1..=16).contains(&self.radius) {
            let message = format!("{NAME} needs radius between 1 and 16, got {}", self.radius);
            return Err(StageError::msg(message));
        }

        if !self.multi.is_finite() || self.multi <= 0.0 {
            let message = format!("{NAME} needs multi finite and above 0, got {}", self.multi);
            return Err(StageError::msg(message));
        }

        Ok(())
    }
}

#[cube(launch_unchecked)]
fn scale_kernel(input: &Array<f32>, output: &mut Array<f32>, multi: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * multi;
}

/// Writes `input * multi` into `output`.
fn scale<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    multi: f32,
) {
    let samples = output.samples();
    let read = input.array::<R>();
    let written = output.array::<R>();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: `input` and `output` each hold `samples` f32 values.
    unsafe {
        scale_kernel::launch_unchecked::<R>(client, cubes, shape, read, written, multi);
    }
}

fn no_frame() -> StageError {
    StageError::msg(format!("{NAME} expected a frame in slot 0"))
}

struct RangeMask<R: Runtime> {
    client: ComputeClient<R>,
    luma: Luma,
    radius: u32,
    multi: f32,
}

impl<R: Runtime> StageInstance for RangeMask<R> {}

impl<R: Runtime> Mask for RangeMask<R> {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(no_frame)?;

        let context = &*context;
        let sizing = Sizing(context);
        let width = self.luma.width;
        let height = self.luma.height;
        let peak = self.luma.peak();

        let luma_scratch = Scratch::new::<f32>(context, width, height);
        let luma_plane = luma_scratch.plane();
        load_luma_code_values(&self.client, &sizing, NAME, &frame, &self.luma, luma_plane)?;

        let range_scratch = Scratch::new::<f32>(context, width, height);
        let maximum_scratch = Scratch::new::<f32>(context, width, height);
        let output_scratch = Scratch::new::<f32>(context, width, height);
        let range_plane = range_scratch.plane();
        let maximum_plane = maximum_scratch.plane();
        let difference_plane = output_scratch.plane();
        range(
            &self.client,
            &sizing,
            luma_plane,
            range_plane,
            maximum_plane,
            difference_plane,
            self.radius,
        );

        // `range_plane` is only the range's scratch, so it is free to hold the scaled range.
        let scaled_plane = range_plane;
        scale(&self.client, &sizing, difference_plane, scaled_plane, self.multi);

        let meta = frame.meta().clone();
        let format = self.luma.mask_format();
        let output = context.alloc_mask(&format, meta);
        let output_plane = mask_plane(&output);
        store_mask(&self.client, &sizing, scaled_plane, output_plane, peak);

        Ok(output)
    }
}

/// Marks where the local maximum and minimum of the luma differ.
pub(crate) struct Range;

impl StageDefinition for Range {
    type Args = RangeArgs;

    fn package() -> &'static str {
        "detail"
    }

    fn identifier() -> &'static str {
        "range"
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

impl MaskDefinition for Range {
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

        Ok(Box::new(RangeMask {
            client,
            luma,
            radius: args.radius,
            multi: args.multi,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn args(value: serde_json::Value) -> RangeArgs {
        serde_json::from_value(value).expect("the args parse")
    }

    fn refusal(args: RangeArgs) -> String {
        let error = args.validate().expect_err("the args should be refused");

        error.to_string()
    }

    #[test]
    fn the_defaults_are_radius_2_multi_1() {
        let defaults = args(json!({}));

        assert_eq!(defaults.radius, 2);
        assert_eq!(defaults.multi, 1.0);
        assert!(defaults.validate().is_ok());
    }

    #[test]
    fn a_radius_of_0_is_refused() {
        let too_small = args(json!({ "radius": 0 }));
        let message = refusal(too_small);

        assert!(
            message.contains("detail/range needs radius between 1 and 16, got 0"),
            "{message}"
        );
    }

    #[test]
    fn a_radius_of_17_is_refused() {
        let too_large = args(json!({ "radius": 17 }));
        let message = refusal(too_large);

        assert!(
            message.contains("detail/range needs radius between 1 and 16, got 17"),
            "{message}"
        );
    }

    #[test]
    fn a_zero_multi_is_refused() {
        let zero = args(json!({ "multi": 0.0 }));
        let message = refusal(zero);

        assert!(
            message.contains("detail/range needs multi finite and above 0, got 0"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_multi_is_refused() {
        let negative = args(json!({ "multi": -1.0 }));
        let message = refusal(negative);

        assert!(
            message.contains("detail/range needs multi finite and above 0, got -1"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_multi_is_refused() {
        let defaults = args(json!({}));
        let nan_args = RangeArgs {
            multi: f32::NAN,
            ..defaults
        };
        let message = refusal(nan_args);

        assert!(
            message.contains("detail/range needs multi finite and above 0, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn an_infinite_multi_is_refused() {
        let defaults = args(json!({}));
        let infinite_args = RangeArgs {
            multi: f32::INFINITY,
            ..defaults
        };
        let message = refusal(infinite_args);

        assert!(
            message.contains("detail/range needs multi finite and above 0, got inf"),
            "{message}"
        );
    }
}
