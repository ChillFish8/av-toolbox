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
use av_toolbox_cubek::edges::{EdgeLimits, Operator, binarize, edges};
use av_toolbox_cubek::neighbourhood::NeighbourPlanes;
use av_toolbox_cubek::rgvs::{GrainBorder, RemoveGrainMode, RemoveGrainPlanes, remove_grain};
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
use cubecl::prelude::{ComputeClient, Runtime};
use schemars::JsonSchema;
use serde::Deserialize;

use super::{join_maximum, range};
use crate::gauss::gaussian_for;
use crate::luma::{Luma, accepts_luma, load_luma_code_values, produces_luma_mask, store_mask};
use crate::sizing::{Scratch, Sizing, mask_plane};

const NAME: &str = "detail/mask";

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DetailMaskArgs {
    /// The fraction of peak the local range binarizes at.
    #[serde(default = "default_brz_mm")]
    brz_mm: f32,
    /// The fraction of peak the Kirsch edges binarize at.
    #[serde(default = "default_brz_ed")]
    brz_ed: f32,
    /// How far, in samples, the local range reaches.
    #[serde(default = "default_radius")]
    radius: u32,
    /// The deviation the luma is blurred at first. Left out, the luma is not blurred.
    ///
    /// Blurring first keeps shallow banding steps out of the mask.
    #[serde(default)]
    sigma: Option<f32>,
}

fn default_brz_mm() -> f32 {
    0.025
}

fn default_brz_ed() -> f32 {
    0.045
}

fn default_radius() -> u32 {
    3
}

impl DetailMaskArgs {
    fn validate(&self) -> Result<(), StageError> {
        if !(0.0..=1.0).contains(&self.brz_mm) {
            let message = format!("{NAME} needs brz_mm between 0 and 1, got {}", self.brz_mm);
            return Err(StageError::msg(message));
        }

        if !(0.0..=1.0).contains(&self.brz_ed) {
            let message = format!("{NAME} needs brz_ed between 0 and 1, got {}", self.brz_ed);
            return Err(StageError::msg(message));
        }

        if !(1..=16).contains(&self.radius) {
            let message = format!("{NAME} needs radius between 1 and 16, got {}", self.radius);
            return Err(StageError::msg(message));
        }

        if let Some(sigma) = self.sigma
            && !(sigma.is_finite() && sigma > 0.0 && sigma <= 64.0)
        {
            let message = format!("{NAME} needs sigma finite, above 0 and at most 64, got {sigma}");
            return Err(StageError::msg(message));
        }

        Ok(())
    }
}

fn no_frame() -> StageError {
    StageError::msg(format!("{NAME} expected a frame in slot 0"))
}

struct DetailMasker<R: Runtime> {
    client: ComputeClient<R>,
    luma: Luma,
    blur: Option<Gaussian>,
    radius: u32,
    range_threshold: f32,
    edge_threshold: f32,
}

impl<R: Runtime> StageInstance for DetailMasker<R> {}

impl<R: Runtime> Mask for DetailMasker<R> {
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

        let first_scratch = Scratch::new::<f32>(context, width, height);
        let second_scratch = Scratch::new::<f32>(context, width, height);
        let third_scratch = Scratch::new::<f32>(context, width, height);
        let first_plane = first_scratch.plane();
        let second_plane = second_scratch.plane();
        let third_plane = third_scratch.plane();

        // `first_plane` is only the blur's scratch here.
        let mut blurred_scratch = None;
        if let Some(gaussian) = self.blur.as_ref() {
            let output_scratch = Scratch::new::<f32>(context, width, height);
            let blur_planes = BlurPlanes {
                input: luma_plane,
                scratch: first_plane,
                output: output_scratch.plane(),
            };
            gaussian.apply(&self.client, &sizing, blur_planes);

            blurred_scratch = Some(output_scratch);
        }

        let base_plane = match &blurred_scratch {
            Some(scratch) => scratch.plane(),
            None => luma_plane,
        };

        range(
            &self.client,
            &sizing,
            base_plane,
            first_plane,
            second_plane,
            third_plane,
            self.radius,
        );
        let range_planes = NeighbourPlanes {
            input: third_plane,
            output: first_plane,
        };
        binarize(&self.client, &sizing, range_planes, self.range_threshold, peak);

        let open_limits = EdgeLimits::open(peak);
        let kirsch_planes = NeighbourPlanes {
            input: base_plane,
            output: second_plane,
        };
        edges(
            &self.client,
            &sizing,
            kirsch_planes,
            Operator::Kirsch,
            open_limits,
        );
        let edge_planes = NeighbourPlanes {
            input: second_plane,
            output: third_plane,
        };
        binarize(&self.client, &sizing, edge_planes, self.edge_threshold, peak);

        join_maximum(&self.client, &sizing, first_plane, third_plane, second_plane);

        let smooth_planes = RemoveGrainPlanes {
            input: second_plane,
            output: first_plane,
        };
        remove_grain(
            &self.client,
            &sizing,
            smooth_planes,
            RemoveGrainMode::TwentyTwo,
            GrainBorder::Reflect,
        );
        let clean_planes = RemoveGrainPlanes {
            input: first_plane,
            output: third_plane,
        };
        remove_grain(
            &self.client,
            &sizing,
            clean_planes,
            RemoveGrainMode::Eleven,
            GrainBorder::Reflect,
        );

        let meta = frame.meta().clone();
        let format = self.luma.mask_format();
        let output = context.alloc_mask(&format, meta);
        let output_plane = mask_plane(&output);
        store_mask(&self.client, &sizing, third_plane, output_plane, peak);

        Ok(output)
    }
}

/// Marks texture and edges in the luma, as vsmasktools' `detail_mask` does.
pub(crate) struct DetailMask;

impl StageDefinition for DetailMask {
    type Args = DetailMaskArgs;

    fn package() -> &'static str {
        "detail"
    }

    fn identifier() -> &'static str {
        "mask"
    }

    fn description() -> &'static str {
        "Marks texture and edges in the luma."
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

impl MaskDefinition for DetailMask {
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
        let peak = luma.peak();

        let blur = args.sigma.map(|sigma| gaussian_for(&client, sigma));
        let range_threshold = (args.brz_mm * peak).round();
        let edge_threshold = (args.brz_ed * peak).round();

        Ok(Box::new(DetailMasker {
            client,
            luma,
            blur,
            radius: args.radius,
            range_threshold,
            edge_threshold,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn args(value: serde_json::Value) -> DetailMaskArgs {
        serde_json::from_value(value).expect("the args parse")
    }

    fn refusal(args: DetailMaskArgs) -> String {
        let error = args.validate().expect_err("the args should be refused");

        error.to_string()
    }

    #[test]
    fn the_defaults_match_vsmasktools() {
        let defaults = args(json!({}));

        assert_eq!(defaults.brz_mm, 0.025);
        assert_eq!(defaults.brz_ed, 0.045);
        assert_eq!(defaults.radius, 3);
        assert_eq!(defaults.sigma, None);
        assert!(defaults.validate().is_ok());
    }

    #[test]
    fn a_brz_mm_above_1_is_refused() {
        let too_high = args(json!({ "brz_mm": 1.5 }));
        let message = refusal(too_high);

        assert!(
            message.contains("detail/mask needs brz_mm between 0 and 1, got 1.5"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_brz_mm_is_refused() {
        let negative = args(json!({ "brz_mm": -0.1 }));
        let message = refusal(negative);

        assert!(
            message.contains("detail/mask needs brz_mm between 0 and 1, got -0.1"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_brz_mm_is_refused() {
        let defaults = args(json!({}));
        let nan = DetailMaskArgs {
            brz_mm: f32::NAN,
            ..defaults
        };
        let message = refusal(nan);

        assert!(
            message.contains("detail/mask needs brz_mm between 0 and 1, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn a_brz_ed_above_1_is_refused() {
        let too_high = args(json!({ "brz_ed": 1.5 }));
        let message = refusal(too_high);

        assert!(
            message.contains("detail/mask needs brz_ed between 0 and 1, got 1.5"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_brz_ed_is_refused() {
        let negative = args(json!({ "brz_ed": -0.1 }));
        let message = refusal(negative);

        assert!(
            message.contains("detail/mask needs brz_ed between 0 and 1, got -0.1"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_brz_ed_is_refused() {
        let defaults = args(json!({}));
        let nan = DetailMaskArgs {
            brz_ed: f32::NAN,
            ..defaults
        };
        let message = refusal(nan);

        assert!(
            message.contains("detail/mask needs brz_ed between 0 and 1, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn a_radius_of_0_is_refused() {
        let too_small = args(json!({ "radius": 0 }));
        let message = refusal(too_small);

        assert!(
            message.contains("detail/mask needs radius between 1 and 16, got 0"),
            "{message}"
        );
    }

    #[test]
    fn a_radius_of_17_is_refused() {
        let too_large = args(json!({ "radius": 17 }));
        let message = refusal(too_large);

        assert!(
            message.contains("detail/mask needs radius between 1 and 16, got 17"),
            "{message}"
        );
    }

    #[test]
    fn a_zero_sigma_is_refused() {
        let zero = args(json!({ "sigma": 0.0 }));
        let message = refusal(zero);

        assert!(
            message.contains("detail/mask needs sigma finite, above 0 and at most 64, got 0"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_sigma_is_refused() {
        let negative = args(json!({ "sigma": -1.0 }));
        let message = refusal(negative);

        assert!(
            message.contains("detail/mask needs sigma finite, above 0 and at most 64, got -1"),
            "{message}"
        );
    }

    #[test]
    fn a_sigma_above_64_is_refused() {
        let too_wide = args(json!({ "sigma": 65.0 }));
        let message = refusal(too_wide);

        assert!(
            message.contains("detail/mask needs sigma finite, above 0 and at most 64, got 65"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_sigma_is_refused() {
        let defaults = args(json!({}));
        let nan = DetailMaskArgs {
            sigma: Some(f32::NAN),
            ..defaults
        };
        let message = refusal(nan);

        assert!(
            message.contains("detail/mask needs sigma finite, above 0 and at most 64, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn a_sigma_of_64_is_accepted() {
        let widest = args(json!({ "sigma": 64.0 }));

        assert!(widest.validate().is_ok());
    }
}
