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
use av_toolbox_cubek::edges::{EdgeLimits, Operator, add_capped, binarize, edges};
use av_toolbox_cubek::histogram::BINS;
use av_toolbox_cubek::neighbourhood::{Extreme, NeighbourPlanes, Shape, extreme};
use av_toolbox_cubek::reduce::scratch_len;
use av_toolbox_cubek::retinex::{RetinexBlurs, RetinexPlanes, RetinexSettings, retinex};
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
use cubecl::prelude::Client;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::gauss::jetpack_radius;
use crate::luma::{
    Luma,
    accepts_luma,
    binarize_threshold,
    load_luma_code_values,
    produces_luma_mask,
    store_mask,
};
use crate::sizing::{Scratch, Sizing, mask_plane};

const NAME: &str = "edges/retinex";

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct RetinexArgs {
    /// The deviation the retinex response is blurred at before its edges are found. 0 skips it.
    #[serde(default = "default_sigma")]
    sigma: f32,
    /// The deviations the retinex blurs the luma at to find its response.
    #[serde(default = "default_sigmas")]
    sigmas: Vec<f32>,
    /// The fraction of samples clipped from the dark end of the response.
    #[serde(default = "default_lower_thr")]
    lower_thr: f32,
    /// The fraction of samples clipped from the bright end of the response.
    #[serde(default = "default_upper_thr")]
    upper_thr: f32,
    /// The fraction of peak the edges binarize at. 0 skips it.
    #[serde(default = "default_brz")]
    brz: f32,
}

fn default_sigma() -> f32 {
    1.0
}

fn default_sigmas() -> Vec<f32> {
    vec![50.0, 200.0, 350.0]
}

fn default_lower_thr() -> f32 {
    0.001
}

fn default_upper_thr() -> f32 {
    0.005
}

fn default_brz() -> f32 {
    0.122
}

impl RetinexArgs {
    fn validate(&self) -> Result<(), StageError> {
        if !(0.0..=64.0).contains(&self.sigma) {
            let message = format!("{NAME} needs sigma between 0 and 64, got {}", self.sigma);
            return Err(StageError::msg(message));
        }

        if !(1..=8).contains(&self.sigmas.len()) {
            let message = format!("{NAME} needs 1 to 8 sigmas, got {}", self.sigmas.len());
            return Err(StageError::msg(message));
        }

        for sigma in &self.sigmas {
            if !(0.5..=1024.0).contains(sigma) {
                let message = format!("{NAME} needs each of sigmas between 0.5 and 1024, got {sigma}");
                return Err(StageError::msg(message));
            }
        }

        if !self.lower_thr.is_finite() || self.lower_thr < 0.0 {
            let message = format!(
                "{NAME} needs lower_thr finite and at least 0, got {}",
                self.lower_thr
            );
            return Err(StageError::msg(message));
        }

        if !self.upper_thr.is_finite() || self.upper_thr < 0.0 {
            let message = format!(
                "{NAME} needs upper_thr finite and at least 0, got {}",
                self.upper_thr
            );
            return Err(StageError::msg(message));
        }

        let total_thr = self.lower_thr + self.upper_thr;
        if total_thr >= 1.0 {
            let message = format!("{NAME} needs lower_thr + upper_thr below 1, got {total_thr}");
            return Err(StageError::msg(message));
        }

        if !(0.0..=1.0).contains(&self.brz) {
            let message = format!("{NAME} needs brz between 0 and 1, got {}", self.brz);
            return Err(StageError::msg(message));
        }

        Ok(())
    }
}

/// The radius the blur runs at, or `None` when `sigma` is 0 and the blur is skipped.
fn blur_radius(sigma: f32) -> Option<u32> {
    if sigma <= 0.0 {
        return None;
    }

    Some(jetpack_radius(sigma))
}

fn no_frame() -> StageError {
    StageError::msg(format!("{NAME} expected a frame in slot 0"))
}

struct RetinexMask {
    client: Client,
    luma: Luma,
    settings: RetinexSettings,
    retinex_blurs: RetinexBlurs,
    blur: Option<Gaussian>,
    binarize_threshold: Option<f32>,
}

impl StageInstance for RetinexMask {}

impl Mask for RetinexMask {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(no_frame)?;

        let context = &*context;
        let sizing = Sizing(context);
        let width = self.luma.width;
        let height = self.luma.height;
        let samples = self.luma.samples();
        let peak = self.luma.peak();

        let luma_scratch = Scratch::new::<f32>(context, width, height);
        let luma_plane = luma_scratch.plane();
        load_luma_code_values(&self.client, &sizing, NAME, &frame, &self.luma, luma_plane)?;

        let normalised_scratch = Scratch::new::<f32>(context, width, height);
        let blurred_scratch = Scratch::new::<f32>(context, width, height);
        let msr_scratch = Scratch::new::<f32>(context, width, height);
        let ret_scratch = Scratch::new::<f32>(context, width, height);
        let scratch_length = scratch_len(samples);
        let first_scratch = Scratch::line::<f32>(context, scratch_length);
        let second_scratch = Scratch::line::<f32>(context, scratch_length);
        let range_scratch = Scratch::line::<f32>(context, 2);
        let bins_scratch = Scratch::line::<u32>(context, BINS as usize);
        let low_scratch = Scratch::line::<f32>(context, 1);
        let high_scratch = Scratch::line::<f32>(context, 1);

        let retinex_planes = RetinexPlanes {
            input: luma_plane,
            normalised: normalised_scratch.plane(),
            blurred: blurred_scratch.plane(),
            msr: msr_scratch.plane(),
            output: ret_scratch.plane(),
            first: first_scratch.plane(),
            second: second_scratch.plane(),
            range: range_scratch.plane(),
            bins: bins_scratch.plane(),
            low: low_scratch.plane(),
            high: high_scratch.plane(),
        };
        retinex(
            &self.client,
            &sizing,
            retinex_planes,
            &self.settings,
            &self.retinex_blurs,
        );

        let ret_plane = ret_scratch.plane();
        let mut blurred_ret_scratch = None;
        if let Some(gaussian) = self.blur.as_ref() {
            let temporary_scratch = Scratch::new::<f32>(context, width, height);
            let output_scratch = Scratch::new::<f32>(context, width, height);
            let blur_planes = BlurPlanes {
                input: ret_plane,
                scratch: temporary_scratch.plane(),
                output: output_scratch.plane(),
            };
            gaussian.apply(&self.client, &sizing, blur_planes);

            blurred_ret_scratch = Some(output_scratch);
        }
        let lines_source_plane = match &blurred_ret_scratch {
            Some(scratch) => scratch.plane(),
            None => ret_plane,
        };

        let open_limits = EdgeLimits::open(peak);
        let prewitt_scratch = Scratch::new::<f32>(context, width, height);
        let prewitt_planes = NeighbourPlanes {
            input: lines_source_plane,
            output: prewitt_scratch.plane(),
        };
        edges(
            &self.client,
            &sizing,
            prewitt_planes,
            Operator::Prewitt,
            open_limits,
        );

        let lines_scratch = Scratch::new::<f32>(context, width, height);
        let lines_planes = NeighbourPlanes {
            input: prewitt_scratch.plane(),
            output: lines_scratch.plane(),
        };
        extreme(
            &self.client,
            &sizing,
            lines_planes,
            Extreme::Minimum,
            Shape::Corners,
        );

        let kirsch_scratch = Scratch::new::<f32>(context, width, height);
        let kirsch_planes = NeighbourPlanes {
            input: luma_plane,
            output: kirsch_scratch.plane(),
        };
        edges(
            &self.client,
            &sizing,
            kirsch_planes,
            Operator::Kirsch,
            open_limits,
        );

        let joined_scratch = Scratch::new::<f32>(context, width, height);
        add_capped(
            &self.client,
            &sizing,
            lines_scratch.plane(),
            kirsch_scratch.plane(),
            joined_scratch.plane(),
            peak,
        );

        let mut binarized_scratch = None;
        if let Some(threshold) = self.binarize_threshold {
            let output_scratch = Scratch::new::<f32>(context, width, height);
            let binarize_planes = NeighbourPlanes {
                input: joined_scratch.plane(),
                output: output_scratch.plane(),
            };
            binarize(&self.client, &sizing, binarize_planes, threshold, peak);

            binarized_scratch = Some(output_scratch);
        }
        let final_plane = match &binarized_scratch {
            Some(scratch) => scratch.plane(),
            None => joined_scratch.plane(),
        };

        let meta = frame.meta().clone();
        let format = self.luma.mask_format();
        let output = context.alloc_mask(&format, meta);
        let output_plane = mask_plane(&output);
        store_mask(&self.client, &sizing, final_plane, output_plane, peak);

        Ok(output)
    }
}

/// Finds edges in a retinex-lit copy of the luma, which brings out lines in dark areas.
pub(crate) struct Retinex;

impl StageDefinition for Retinex {
    type Args = RetinexArgs;

    fn package() -> &'static str {
        "edges"
    }

    fn identifier() -> &'static str {
        "retinex"
    }

    fn description() -> &'static str {
        "Finds edges after evening out the lighting, bringing out lines in dark areas."
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

impl MaskDefinition for Retinex {
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

    fn build(
        context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        args.validate()?;

        let luma = Luma::from_formats(NAME, formats)?;
        let client = context.client().clone();
        let peak = luma.peak();

        let gaussian_radius = blur_radius(args.sigma);
        let blur = gaussian_radius.map(|radius| Gaussian::with_radius(&client, args.sigma, radius));
        let binarize_threshold = binarize_threshold(args.brz, peak);
        let settings = RetinexSettings::new(args.sigmas, args.lower_thr, args.upper_thr, peak);
        let retinex_blurs = RetinexBlurs::new(&client, &settings);

        Ok(Box::new(RetinexMask {
            client,
            luma,
            settings,
            retinex_blurs,
            blur,
            binarize_threshold,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn args(value: serde_json::Value) -> RetinexArgs {
        serde_json::from_value(value).expect("the args parse")
    }

    fn refusal(args: RetinexArgs) -> String {
        let error = args.validate().expect_err("the args should be refused");

        error.to_string()
    }

    #[test]
    fn the_defaults_match_vsmasktools() {
        let defaults = args(json!({}));

        assert_eq!(defaults.sigma, 1.0);
        assert_eq!(defaults.sigmas, vec![50.0, 200.0, 350.0]);
        assert_eq!(defaults.lower_thr, 0.001);
        assert_eq!(defaults.upper_thr, 0.005);
        assert_eq!(defaults.brz, 0.122);
        assert!(defaults.validate().is_ok());
    }

    #[test]
    fn a_negative_sigma_is_refused() {
        let negative = args(json!({ "sigma": -1.0 }));
        let message = refusal(negative);

        assert!(
            message.contains("edges/retinex needs sigma between 0 and 64, got -1"),
            "{message}"
        );
    }

    #[test]
    fn a_sigma_above_64_is_refused() {
        let too_wide = args(json!({ "sigma": 65.0 }));
        let message = refusal(too_wide);

        assert!(
            message.contains("edges/retinex needs sigma between 0 and 64, got 65"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_sigma_is_refused() {
        let defaults = args(json!({}));
        let nan_args = RetinexArgs {
            sigma: f32::NAN,
            ..defaults
        };
        let message = refusal(nan_args);

        assert!(
            message.contains("edges/retinex needs sigma between 0 and 64, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn an_empty_sigmas_is_refused() {
        let empty = args(json!({ "sigmas": [] }));
        let message = refusal(empty);

        assert!(
            message.contains("edges/retinex needs 1 to 8 sigmas, got 0"),
            "{message}"
        );
    }

    #[test]
    fn nine_sigmas_are_refused() {
        let nine = args(json!({ "sigmas": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] }));
        let message = refusal(nine);

        assert!(
            message.contains("edges/retinex needs 1 to 8 sigmas, got 9"),
            "{message}"
        );
    }

    #[test]
    fn a_sigmas_value_of_0_4_is_refused() {
        let low = args(json!({ "sigmas": [0.4] }));
        let message = refusal(low);

        assert!(
            message.contains("edges/retinex needs each of sigmas between 0.5 and 1024, got 0.4"),
            "{message}"
        );
    }

    #[test]
    fn a_sigmas_value_above_1024_is_refused() {
        let high = args(json!({ "sigmas": [1025.0] }));
        let message = refusal(high);

        assert!(
            message.contains("edges/retinex needs each of sigmas between 0.5 and 1024, got 1025"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_lower_thr_is_refused() {
        let negative = args(json!({ "lower_thr": -0.1 }));
        let message = refusal(negative);

        assert!(
            message.contains("edges/retinex needs lower_thr finite and at least 0, got -0.1"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_upper_thr_is_refused() {
        let negative = args(json!({ "upper_thr": -0.1 }));
        let message = refusal(negative);

        assert!(
            message.contains("edges/retinex needs upper_thr finite and at least 0, got -0.1"),
            "{message}"
        );
    }

    #[test]
    fn thresholds_summing_to_1_are_refused() {
        let summed = args(json!({ "lower_thr": 0.5, "upper_thr": 0.5 }));
        let message = refusal(summed);

        assert!(
            message.contains("edges/retinex needs lower_thr + upper_thr below 1, got 1"),
            "{message}"
        );
    }

    #[test]
    fn a_brz_above_1_is_refused() {
        let too_high = args(json!({ "brz": 1.5 }));
        let message = refusal(too_high);

        assert!(
            message.contains("edges/retinex needs brz between 0 and 1, got 1.5"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_brz_is_refused() {
        let negative = args(json!({ "brz": -0.1 }));
        let message = refusal(negative);

        assert!(
            message.contains("edges/retinex needs brz between 0 and 1, got -0.1"),
            "{message}"
        );
    }

    #[test]
    fn a_zero_sigma_skips_the_blur() {
        assert_eq!(blur_radius(0.0), None);
    }
}
