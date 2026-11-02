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
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    Mask as GpuMask,
    MaskFormat,
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

use super::{accepts_mask, produces_mask};

const NAME: &str = "mask/threshold";

fn missing_mask() -> StageError {
    StageError::msg("mask/threshold expected a mask in slot 0")
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ThresholdArgs {
    /// Samples at or above this become 255, and the rest 0.
    #[serde(default = "default_cutoff")]
    pub(crate) cutoff: u8,
}

fn default_cutoff() -> u8 {
    128
}

#[cube(launch_unchecked)]
fn harden(input: &[u8], output: &mut [u8], cutoff: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let sample = u32::cast_from(input[ABSOLUTE_POS]);
    if sample >= cutoff {
        output[ABSOLUTE_POS] = 255u8;
    } else {
        output[ABSOLUTE_POS] = 0u8;
    }
}

struct ThresholdMask {
    client: Client,
    cutoff: u8,
}

impl StageInstance for ThresholdMask {}

impl Mask for ThresholdMask {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let input = inputs.take_mask(0).ok_or_else(missing_mask)?;

        let meta = input.meta().clone();
        let format = MaskFormat {
            chroma: None,
            ..*input.format()
        };

        let output = context.alloc_mask(&format, meta);

        let samples = output.plane().samples();
        let cubes = context.cube_count(samples);
        let shape = context.cube_dim();
        let read = input.plane().array::<u8>();
        let write = output.plane().array::<u8>();
        let cutoff = u32::from(self.cutoff);

        // Safety: both bindings name exactly as many elements as their plane holds, and the kernel
        // writes nothing past the length it is given.
        unsafe {
            harden::launch_unchecked(&self.client, cubes, shape, read, write, cutoff);
        }

        Ok(output)
    }
}

/// Hardens a mask to 0 or 255 at a cutoff.
pub(crate) struct Threshold;

impl StageDefinition for Threshold {
    type Args = ThresholdArgs;

    fn package() -> &'static str {
        "mask"
    }

    fn identifier() -> &'static str {
        "threshold"
    }

    fn description() -> &'static str {
        "Hardens a mask so every pixel is either fully selected or background."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Mask)]);
        SIGNATURE
    }
}

impl MaskDefinition for Threshold {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_mask(NAME, offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        produces_mask(NAME, chosen)
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        let client = context.client().clone();

        Ok(Box::new(ThresholdMask {
            client,
            cutoff: args.cutoff,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cutoff_defaults_to_128() {
        let args: ThresholdArgs = serde_json::from_value(serde_json::json!({})).expect("the defaults apply");

        assert_eq!(args.cutoff, 128);
    }

    #[test]
    fn an_unknown_argument_is_refused() {
        let parsed = serde_json::from_value::<ThresholdArgs>(serde_json::json!({ "cut": 3 }));

        assert!(parsed.is_err());
    }
}
