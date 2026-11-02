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

const NAME: &str = "mask/union";

fn missing_first_mask() -> StageError {
    StageError::msg("mask/union expected a mask in slot 0")
}

fn missing_mask() -> StageError {
    StageError::msg("mask/union expected a mask in every slot")
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnionArgs {}

#[cube(launch_unchecked)]
fn copy_mask(input: &[u8], output: &mut [u8]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
}

#[cube(launch_unchecked)]
fn keep_largest(input: &[u8], output: &mut [u8]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    if input[ABSOLUTE_POS] > output[ABSOLUTE_POS] {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

struct UnionMask {
    client: Client,
}

impl StageInstance for UnionMask {}

impl Mask for UnionMask {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let first = inputs.take_mask(0).ok_or_else(missing_first_mask)?;

        let meta = first.meta().clone();
        let format = MaskFormat {
            chroma: None,
            ..*first.format()
        };

        let output = context.alloc_mask(&format, meta);
        let samples = output.plane().samples();

        let read = first.plane().array::<u8>();
        let write = output.plane().array::<u8>();
        let cubes = context.cube_count(samples);
        let shape = context.cube_dim();

        // Safety: both bindings name exactly as many elements as their plane holds, and the kernel
        // writes nothing past the length it is given.
        unsafe {
            copy_mask::launch_unchecked(&self.client, cubes, shape, read, write);
        }

        for index in 1..inputs.len() {
            let input = inputs.take_mask(index).ok_or_else(missing_mask)?;

            let read = input.plane().array::<u8>();
            let write = output.plane().array::<u8>();
            let cubes = context.cube_count(samples);
            let shape = context.cube_dim();

            // Safety: both bindings name exactly as many elements as their plane holds, and the
            // kernel writes nothing past the length it is given.
            unsafe {
                keep_largest::launch_unchecked(&self.client, cubes, shape, read, write);
            }
        }

        Ok(output)
    }
}

/// Selects every pixel any of its masks selects, keeping the largest sample.
pub(crate) struct Union;

impl StageDefinition for Union {
    type Args = UnionArgs;

    fn package() -> &'static str {
        "mask"
    }

    fn identifier() -> &'static str {
        "union"
    }

    fn description() -> &'static str {
        "Joins masks, selecting every pixel any of them selects."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::with_variadic_of(
            &[InputParam::required("input").accepting(StreamType::Mask)],
            StreamType::Mask,
        );
        SIGNATURE
    }
}

impl MaskDefinition for Union {
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
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        let client = context.client().clone();

        Ok(Box::new(UnionMask { client }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_argument_is_refused() {
        let parsed = serde_json::from_value::<UnionArgs>(serde_json::json!({ "cut": 3 }));

        assert!(parsed.is_err());
    }
}
