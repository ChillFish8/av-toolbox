use std::marker::PhantomData;

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
use av_toolbox_cubek::edges::{EdgeLimits, Operator, edges};
use av_toolbox_cubek::neighbourhood::NeighbourPlanes;
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

use crate::luma::{Luma, accepts_luma, load_luma_code_values, produces_luma_mask, store_mask};
use crate::sizing::{Scratch, Sizing, mask_plane};

/// What sets one `edges/*` stage apart from another.
pub(crate) trait EdgeOperator: 'static {
    const OPERATOR: Operator;

    fn identifier() -> &'static str;
}

pub(crate) struct Kirsch;

impl EdgeOperator for Kirsch {
    const OPERATOR: Operator = Operator::Kirsch;

    fn identifier() -> &'static str {
        "kirsch"
    }
}

pub(crate) struct Prewitt;

impl EdgeOperator for Prewitt {
    const OPERATOR: Operator = Operator::Prewitt;

    fn identifier() -> &'static str {
        "prewitt"
    }
}

pub(crate) struct Sobel;

impl EdgeOperator for Sobel {
    const OPERATOR: Operator = Operator::Sobel;

    fn identifier() -> &'static str {
        "sobel"
    }
}

fn stage_name<O: EdgeOperator>() -> String {
    format!("edges/{}", O::identifier())
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct EdgeArgs {
    /// Multiplies the edge strength before the thresholds.
    #[serde(default = "default_multi")]
    multi: f32,
    /// Strengths below this, on the 0 to 1 scale, become 0.
    #[serde(default)]
    lthr: f32,
    /// Strengths above this, on the 0 to 1 scale, become fully selected.
    #[serde(default = "default_hthr")]
    hthr: f32,
}

fn default_multi() -> f32 {
    1.0
}

fn default_hthr() -> f32 {
    1.0
}

impl EdgeArgs {
    fn validate(&self, stage: &str) -> Result<(), StageError> {
        if !self.multi.is_finite() || self.multi <= 0.0 {
            let message = format!("{stage} needs multi finite and above 0, got {}", self.multi);
            return Err(StageError::msg(message));
        }

        if !(0.0..=1.0).contains(&self.lthr) {
            let message = format!("{stage} needs lthr between 0 and 1, got {}", self.lthr);
            return Err(StageError::msg(message));
        }

        if !(0.0..=1.0).contains(&self.hthr) {
            let message = format!("{stage} needs hthr between 0 and 1, got {}", self.hthr);
            return Err(StageError::msg(message));
        }

        if self.hthr < self.lthr {
            let message = format!(
                "{stage} needs hthr at or above lthr, got {} and {}",
                self.hthr, self.lthr
            );
            return Err(StageError::msg(message));
        }

        Ok(())
    }

    /// The limits `edges/*` runs with, thresholds rounded to the nearest code value at `peak`.
    fn limits(&self, peak: f32) -> EdgeLimits {
        let low = (self.lthr * peak).round();
        let high = (self.hthr * peak).round();

        EdgeLimits {
            multi: self.multi,
            low,
            high,
            peak,
        }
    }
}

/// Finds edges through the operator `O` names.
pub(crate) struct Edges<O>(PhantomData<O>);

impl<O: EdgeOperator> StageDefinition for Edges<O> {
    type Args = EdgeArgs;

    fn package() -> &'static str {
        "edges"
    }

    fn identifier() -> &'static str {
        O::identifier()
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

impl<O: EdgeOperator> MaskDefinition for Edges<O> {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let stage = stage_name::<O>();

        accepts_luma(&stage, offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let stage = stage_name::<O>();
        args.validate(&stage)?;

        produces_luma_mask(&stage, chosen)
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        let stage = stage_name::<O>();
        args.validate(&stage)?;

        let luma = Luma::from_formats(&stage, formats)?;
        let client = context.client().clone();
        let limits = args.limits(luma.peak());

        Ok(Box::new(EdgesMask {
            client,
            stage,
            luma,
            operator: O::OPERATOR,
            limits,
        }))
    }
}

struct EdgesMask<R: Runtime> {
    client: ComputeClient<R>,
    stage: String,
    luma: Luma,
    operator: Operator,
    limits: EdgeLimits,
}

impl<R: Runtime> StageInstance for EdgesMask<R> {}

fn no_frame(stage: &str) -> StageError {
    let message = format!("{stage} expected a frame in slot 0");

    StageError::msg(message)
}

impl<R: Runtime> Mask for EdgesMask<R> {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let Some(frame) = inputs.take_frame(0) else {
            return Err(no_frame(&self.stage));
        };

        let context = &*context;
        let sizing = Sizing(context);
        let width = self.luma.width;
        let height = self.luma.height;

        let luma_scratch = Scratch::new::<f32>(context, width, height);
        let luma_plane = luma_scratch.plane();
        load_luma_code_values(&self.client, &sizing, &self.stage, &frame, &self.luma, luma_plane)?;

        let edge_scratch = Scratch::new::<f32>(context, width, height);
        let edge_plane = edge_scratch.plane();
        let edge_planes = NeighbourPlanes {
            input: luma_plane,
            output: edge_plane,
        };
        edges(&self.client, &sizing, edge_planes, self.operator, self.limits);

        let meta = frame.meta().clone();
        let format = self.luma.mask_format();
        let output = context.alloc_mask(&format, meta);
        let output_plane = mask_plane(&output);
        let peak = self.luma.peak();
        store_mask(&self.client, &sizing, edge_plane, output_plane, peak);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn args(value: serde_json::Value) -> EdgeArgs {
        serde_json::from_value(value).expect("the args parse")
    }

    #[test]
    fn the_defaults_are_multi_1_lthr_0_hthr_1() {
        let defaults = args(json!({}));

        assert_eq!(defaults.multi, 1.0);
        assert_eq!(defaults.lthr, 0.0);
        assert_eq!(defaults.hthr, 1.0);
        assert!(defaults.validate("edges/prewitt").is_ok());
    }

    #[test]
    fn a_zero_multi_is_refused() {
        let error = args(json!({ "multi": 0.0 }))
            .validate("edges/prewitt")
            .expect_err("zero multi is refused");

        assert!(
            error
                .to_string()
                .contains("edges/prewitt needs multi finite and above 0, got 0")
        );
    }

    #[test]
    fn a_negative_multi_is_refused() {
        let error = args(json!({ "multi": -1.0 }))
            .validate("edges/prewitt")
            .expect_err("negative multi is refused");

        assert!(
            error
                .to_string()
                .contains("edges/prewitt needs multi finite and above 0, got -1")
        );
    }

    #[test]
    fn a_nan_multi_is_refused() {
        let defaults = args(json!({}));
        let nan_args = EdgeArgs {
            multi: f32::NAN,
            ..defaults
        };

        let error = nan_args
            .validate("edges/prewitt")
            .expect_err("nan multi is refused");

        assert!(
            error
                .to_string()
                .contains("edges/prewitt needs multi finite and above 0, got NaN")
        );
    }

    #[test]
    fn an_infinite_multi_is_refused() {
        let defaults = args(json!({}));
        let infinite_args = EdgeArgs {
            multi: f32::INFINITY,
            ..defaults
        };

        let error = infinite_args
            .validate("edges/prewitt")
            .expect_err("infinite multi is refused");

        assert!(
            error
                .to_string()
                .contains("edges/prewitt needs multi finite and above 0, got inf")
        );
    }

    #[test]
    fn an_lthr_above_1_is_refused() {
        let error = args(json!({ "lthr": 1.5 }))
            .validate("edges/prewitt")
            .expect_err("lthr above 1 is refused");

        assert!(
            error
                .to_string()
                .contains("edges/prewitt needs lthr between 0 and 1, got 1.5")
        );
    }

    #[test]
    fn a_negative_lthr_is_refused() {
        let error = args(json!({ "lthr": -0.1 }))
            .validate("edges/prewitt")
            .expect_err("negative lthr is refused");

        assert!(
            error
                .to_string()
                .contains("edges/prewitt needs lthr between 0 and 1, got -0.1")
        );
    }

    #[test]
    fn an_hthr_above_1_is_refused() {
        let error = args(json!({ "hthr": 1.2 }))
            .validate("edges/prewitt")
            .expect_err("hthr above 1 is refused");

        assert!(
            error
                .to_string()
                .contains("edges/prewitt needs hthr between 0 and 1, got 1.2")
        );
    }

    #[test]
    fn an_hthr_below_lthr_is_refused() {
        let error = args(json!({ "lthr": 0.5, "hthr": 0.3 }))
            .validate("edges/prewitt")
            .expect_err("hthr below lthr is refused");

        assert!(
            error
                .to_string()
                .contains("edges/prewitt needs hthr at or above lthr, got 0.3 and 0.5")
        );
    }

    #[test]
    fn equal_thresholds_are_accepted() {
        let equal = args(json!({ "lthr": 0.4, "hthr": 0.4 }));

        assert!(equal.validate("edges/prewitt").is_ok());
    }

    #[test]
    fn a_threshold_rounds_to_a_code_value() {
        let tenth = args(json!({ "lthr": 0.1 }));

        let at_255 = tenth.limits(255.0);
        let at_1023 = tenth.limits(1023.0);

        assert_eq!(at_255.low, 26.0);
        assert_eq!(at_1023.low, 102.0);
    }
}
