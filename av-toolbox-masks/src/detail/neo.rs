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
use av_toolbox_cubek::bilateral::{BilateralPlanes, BilateralSettings, bilateral};
use av_toolbox_cubek::blur::{BlurPlanes, Gaussian};
use av_toolbox_cubek::edges::{EdgeLimits, Operator, add_capped, binarize, edges};
use av_toolbox_cubek::neighbourhood::{ChainPlanes, NeighbourPlanes, Spread, Step, chain};
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

use super::excess_over;
use crate::gauss::gaussian_for;
use crate::luma::{
    Luma,
    accepts_luma,
    binarize_threshold,
    load_luma_code_values,
    produces_luma_mask,
    store_mask,
};
use crate::sizing::{Scratch, Sizing, mask_plane};

const NAME: &str = "detail/neo";

const BILATERAL_SIGMA_R: f32 = 0.02;

const DETAIL_INFLATIONS: usize = 4;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct NeoArgs {
    /// The deviation of the blur and of the bilateral's spatial reach.
    #[serde(default = "default_sigma")]
    sigma: f32,
    /// The fraction of peak the detail binarizes at. 0 leaves the detail as it is.
    #[serde(default = "default_detail_brz")]
    detail_brz: f32,
    /// The fraction of peak the Prewitt lines binarize at. 0 leaves the lines as they are.
    #[serde(default = "default_lines_brz")]
    lines_brz: f32,
    /// The RemoveGrain mode that cleans the joined mask. One of 7, 11, 17 or 22.
    #[serde(default = "default_rg_mode")]
    rg_mode: u32,
}

fn default_sigma() -> f32 {
    5.0
}

fn default_detail_brz() -> f32 {
    0.005
}

fn default_lines_brz() -> f32 {
    0.08
}

fn default_rg_mode() -> u32 {
    17
}

impl NeoArgs {
    fn validate(&self) -> Result<(), StageError> {
        if !(0.5..=16.0).contains(&self.sigma) {
            let message = format!(
                "{NAME} needs sigma finite and between 0.5 and 16, got {}",
                self.sigma
            );
            return Err(StageError::msg(message));
        }

        if !(0.0..=1.0).contains(&self.detail_brz) {
            let message = format!("{NAME} needs detail_brz between 0 and 1, got {}", self.detail_brz);
            return Err(StageError::msg(message));
        }

        if !(0.0..=1.0).contains(&self.lines_brz) {
            let message = format!("{NAME} needs lines_brz between 0 and 1, got {}", self.lines_brz);
            return Err(StageError::msg(message));
        }

        if RemoveGrainMode::from_number(self.rg_mode).is_none() {
            let message = format!(
                "{NAME} needs rg_mode one of 7, 11, 17 or 22, got {}",
                self.rg_mode
            );
            return Err(StageError::msg(message));
        }

        Ok(())
    }
}

fn no_frame() -> StageError {
    StageError::msg(format!("{NAME} expected a frame in slot 0"))
}

fn no_mode(mode: u32) -> StageError {
    StageError::msg(format!("{NAME} has no RemoveGrain mode {mode}"))
}

struct NeoMasker<R: Runtime> {
    client: ComputeClient<R>,
    luma: Luma,
    gaussian: Gaussian,
    sigma: f32,
    detail_threshold: Option<f32>,
    lines_threshold: Option<f32>,
    mode: RemoveGrainMode,
}

impl<R: Runtime> StageInstance for NeoMasker<R> {}

impl<R: Runtime> Mask for NeoMasker<R> {
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

        let guide_scratch = Scratch::new::<f32>(context, width, height);
        let guided_scratch = Scratch::new::<f32>(context, width, height);
        let first_scratch = Scratch::new::<f32>(context, width, height);
        let second_scratch = Scratch::new::<f32>(context, width, height);
        let third_scratch = Scratch::new::<f32>(context, width, height);
        let first_plane = first_scratch.plane();
        let second_plane = second_scratch.plane();
        let third_plane = third_scratch.plane();

        // The luma blurred, which guides the bilateral. `first_plane` is only the blur's scratch here.
        let guide_plane = guide_scratch.plane();
        let blur_planes = BlurPlanes {
            input: luma_plane,
            scratch: first_plane,
            output: guide_plane,
        };
        self.gaussian.apply(&self.client, &sizing, blur_planes);

        let guided_plane = guided_scratch.plane();
        let bilateral_planes = BilateralPlanes {
            input: luma_plane,
            reference: guide_plane,
            output: guided_plane,
        };
        let settings = BilateralSettings {
            sigma_s: self.sigma,
            sigma_r: BILATERAL_SIGMA_R,
            peak,
        };
        bilateral(&self.client, &sizing, bilateral_planes, settings);

        excess_over(&self.client, &sizing, guided_plane, luma_plane, first_plane);
        let mut detail_steps = vec![Step::Spread(Spread::Deflate)];
        detail_steps.extend([Step::Spread(Spread::Inflate); DETAIL_INFLATIONS]);
        let detail_chain_planes = ChainPlanes {
            input: first_plane,
            scratch: second_plane,
            output: third_plane,
        };
        chain(&self.client, &sizing, detail_chain_planes, &detail_steps);

        let detail_plane = match self.detail_threshold {
            Some(threshold) => {
                let binary_planes = NeighbourPlanes {
                    input: third_plane,
                    output: first_plane,
                };
                binarize(&self.client, &sizing, binary_planes, threshold, peak);

                first_plane
            },
            None => third_plane,
        };

        let open_limits = EdgeLimits::open(peak);
        let prewitt_planes = NeighbourPlanes {
            input: luma_plane,
            output: second_plane,
        };
        edges(
            &self.client,
            &sizing,
            prewitt_planes,
            Operator::Prewitt,
            open_limits,
        );
        let lines_steps = [Step::Spread(Spread::Deflate), Step::Spread(Spread::Inflate)];
        let lines_chain_planes = ChainPlanes {
            input: second_plane,
            scratch: guide_plane,
            output: guided_plane,
        };
        chain(&self.client, &sizing, lines_chain_planes, &lines_steps);

        let lines_plane = match self.lines_threshold {
            Some(threshold) => {
                let binary_planes = NeighbourPlanes {
                    input: guided_plane,
                    output: second_plane,
                };
                binarize(&self.client, &sizing, binary_planes, threshold, peak);

                second_plane
            },
            None => guided_plane,
        };

        let joined_plane = guide_plane;
        add_capped(
            &self.client,
            &sizing,
            detail_plane,
            lines_plane,
            joined_plane,
            peak,
        );

        let clean_planes = RemoveGrainPlanes {
            input: joined_plane,
            output: third_plane,
        };
        remove_grain(
            &self.client,
            &sizing,
            clean_planes,
            self.mode,
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

/// Marks detail from a bilateral's difference and lines from Prewitt, as vsmasktools' `detail_mask_neo`
/// does.
pub(crate) struct Neo;

impl StageDefinition for Neo {
    type Args = NeoArgs;

    fn package() -> &'static str {
        "detail"
    }

    fn identifier() -> &'static str {
        "neo"
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

impl MaskDefinition for Neo {
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

        let gaussian = gaussian_for(&client, args.sigma);
        let detail_threshold = binarize_threshold(args.detail_brz, peak);
        let lines_threshold = binarize_threshold(args.lines_brz, peak);
        let Some(mode) = RemoveGrainMode::from_number(args.rg_mode) else {
            return Err(no_mode(args.rg_mode));
        };

        Ok(Box::new(NeoMasker {
            client,
            luma,
            gaussian,
            sigma: args.sigma,
            detail_threshold,
            lines_threshold,
            mode,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn args(value: serde_json::Value) -> NeoArgs {
        serde_json::from_value(value).expect("the args parse")
    }

    fn refusal(args: NeoArgs) -> String {
        let error = args.validate().expect_err("the args should be refused");

        error.to_string()
    }

    #[test]
    fn the_defaults_match_vsmasktools() {
        let defaults = args(json!({}));

        assert_eq!(defaults.sigma, 5.0);
        assert_eq!(defaults.detail_brz, 0.005);
        assert_eq!(defaults.lines_brz, 0.08);
        assert_eq!(defaults.rg_mode, 17);
        assert!(defaults.validate().is_ok());
    }

    #[test]
    fn a_sigma_below_half_is_refused() {
        let too_small = args(json!({ "sigma": 0.4 }));
        let message = refusal(too_small);

        assert!(
            message.contains("detail/neo needs sigma finite and between 0.5 and 16, got 0.4"),
            "{message}"
        );
    }

    #[test]
    fn a_sigma_above_16_is_refused() {
        let too_wide = args(json!({ "sigma": 16.5 }));
        let message = refusal(too_wide);

        assert!(
            message.contains("detail/neo needs sigma finite and between 0.5 and 16, got 16.5"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_sigma_is_refused() {
        let defaults = args(json!({}));
        let nan = NeoArgs {
            sigma: f32::NAN,
            ..defaults
        };
        let message = refusal(nan);

        assert!(
            message.contains("detail/neo needs sigma finite and between 0.5 and 16, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn an_infinite_sigma_is_refused() {
        let defaults = args(json!({}));
        let infinite = NeoArgs {
            sigma: f32::INFINITY,
            ..defaults
        };
        let message = refusal(infinite);

        assert!(
            message.contains("detail/neo needs sigma finite and between 0.5 and 16, got inf"),
            "{message}"
        );
    }

    #[test]
    fn the_sigma_limits_are_accepted() {
        let narrowest = args(json!({ "sigma": 0.5 }));
        let widest = args(json!({ "sigma": 16.0 }));

        assert!(narrowest.validate().is_ok());
        assert!(widest.validate().is_ok());
    }

    #[test]
    fn a_detail_brz_above_1_is_refused() {
        let too_high = args(json!({ "detail_brz": 1.5 }));
        let message = refusal(too_high);

        assert!(
            message.contains("detail/neo needs detail_brz between 0 and 1, got 1.5"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_detail_brz_is_refused() {
        let negative = args(json!({ "detail_brz": -0.1 }));
        let message = refusal(negative);

        assert!(
            message.contains("detail/neo needs detail_brz between 0 and 1, got -0.1"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_detail_brz_is_refused() {
        let defaults = args(json!({}));
        let nan = NeoArgs {
            detail_brz: f32::NAN,
            ..defaults
        };
        let message = refusal(nan);

        assert!(
            message.contains("detail/neo needs detail_brz between 0 and 1, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn a_lines_brz_above_1_is_refused() {
        let too_high = args(json!({ "lines_brz": 1.5 }));
        let message = refusal(too_high);

        assert!(
            message.contains("detail/neo needs lines_brz between 0 and 1, got 1.5"),
            "{message}"
        );
    }

    #[test]
    fn a_negative_lines_brz_is_refused() {
        let negative = args(json!({ "lines_brz": -0.1 }));
        let message = refusal(negative);

        assert!(
            message.contains("detail/neo needs lines_brz between 0 and 1, got -0.1"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_lines_brz_is_refused() {
        let defaults = args(json!({}));
        let nan = NeoArgs {
            lines_brz: f32::NAN,
            ..defaults
        };
        let message = refusal(nan);

        assert!(
            message.contains("detail/neo needs lines_brz between 0 and 1, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn the_brz_limits_are_accepted() {
        let off = args(json!({ "detail_brz": 0.0, "lines_brz": 0.0 }));
        let full = args(json!({ "detail_brz": 1.0, "lines_brz": 1.0 }));

        assert!(off.validate().is_ok());
        assert!(full.validate().is_ok());
    }

    #[test]
    fn an_rg_mode_of_1_is_refused() {
        let unsupported = args(json!({ "rg_mode": 1 }));
        let message = refusal(unsupported);

        assert!(
            message.contains("detail/neo needs rg_mode one of 7, 11, 17 or 22, got 1"),
            "{message}"
        );
    }

    #[test]
    fn an_rg_mode_of_12_is_refused() {
        let unsupported = args(json!({ "rg_mode": 12 }));
        let message = refusal(unsupported);

        assert!(
            message.contains("detail/neo needs rg_mode one of 7, 11, 17 or 22, got 12"),
            "{message}"
        );
    }

    #[test]
    fn an_rg_mode_of_24_is_refused() {
        let unsupported = args(json!({ "rg_mode": 24 }));
        let message = refusal(unsupported);

        assert!(
            message.contains("detail/neo needs rg_mode one of 7, 11, 17 or 22, got 24"),
            "{message}"
        );
    }

    #[test]
    fn every_supported_rg_mode_is_accepted() {
        for mode in [7, 11, 17, 22] {
            let supported = args(json!({ "rg_mode": mode }));

            assert!(supported.validate().is_ok(), "rg_mode {mode} was refused");
        }
    }

    #[test]
    fn zero_brz_turns_binarize_off() {
        let off = binarize_threshold(0.0, 255.0);

        assert_eq!(off, None);
    }

    #[test]
    fn a_brz_becomes_a_rounded_code_value() {
        let eight_bit_detail = binarize_threshold(0.005, 255.0);
        let eight_bit_lines = binarize_threshold(0.08, 255.0);
        let ten_bit_detail = binarize_threshold(0.005, 1023.0);
        let ten_bit_lines = binarize_threshold(0.08, 1023.0);

        assert_eq!(eight_bit_detail, Some(1.0));
        assert_eq!(eight_bit_lines, Some(20.0));
        assert_eq!(ten_bit_detail, Some(5.0));
        assert_eq!(ten_bit_lines, Some(82.0));
    }
}
