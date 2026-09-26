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
    PlaneBuffer,
    ResolvedFormats,
    StreamFormat,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

use super::{accepts_mask, produces_mask};

const NAME: &str = "mask/vote";

fn missing_first_mask() -> StageError {
    StageError::msg("mask/vote expected a mask in slot 0")
}

fn missing_mask() -> StageError {
    StageError::msg("mask/vote expected a mask in every slot")
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct VoteArgs {
    /// How many inputs have to agree before a pixel is selected.
    #[serde(default = "default_votes")]
    pub(crate) votes: u32,
    /// A pixel votes when its sample is at or above this.
    #[serde(default = "default_threshold")]
    pub(crate) threshold: u8,
}

fn default_votes() -> u32 {
    2
}

fn default_threshold() -> u8 {
    56
}

impl VoteArgs {
    fn validate(&self, inputs: usize) -> Result<(), StageError> {
        let votes = self.votes as usize;
        if votes == 0 || votes > inputs {
            let message = format!(
                "{NAME} needs votes between 1 and its {inputs} inputs, got {}",
                self.votes,
            );
            return Err(StageError::msg(message));
        }

        Ok(())
    }
}

#[cube(launch_unchecked)]
fn clear_votes(counts: &mut Array<u32>) {
    if ABSOLUTE_POS >= counts.len() {
        terminate!();
    }

    counts[ABSOLUTE_POS] = 0u32;
}

#[cube(launch_unchecked)]
fn cast_vote(input: &Array<u8>, counts: &mut Array<u32>, threshold: u32) {
    if ABSOLUTE_POS >= counts.len() {
        terminate!();
    }

    let sample = u32::cast_from(input[ABSOLUTE_POS]);
    if sample >= threshold {
        counts[ABSOLUTE_POS] += 1u32;
    }
}

#[cube(launch_unchecked)]
fn tally_votes(counts: &Array<u32>, output: &mut Array<u8>, votes: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    if counts[ABSOLUTE_POS] >= votes {
        output[ABSOLUTE_POS] = u8::cast_from(255u32);
    } else {
        output[ABSOLUTE_POS] = u8::cast_from(0u32);
    }
}

struct VoteMask<R: Runtime> {
    client: ComputeClient<R>,
    votes: u32,
    threshold: u32,
}

impl<R: Runtime> StageInstance for VoteMask<R> {}

impl<R: Runtime> Mask for VoteMask<R> {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let first = inputs.take_mask(0).ok_or_else(missing_first_mask)?;

        let meta = first.meta().clone();
        let format = MaskFormat {
            chroma: None,
            ..*first.format()
        };

        let output = context.alloc_mask(&format, meta);
        let samples = output.plane().samples();

        let tally = context.scratch(samples * size_of::<u32>());
        let tally_handle = tally.handle().clone();

        // Safety: the tally was allocated to hold exactly `samples` u32 values.
        let counts = unsafe { ArrayArg::from_raw_parts(tally_handle, samples) };
        let cubes = context.cube_count(samples);
        let shape = context.cube_dim();

        // Safety: every binding names no more elements than its buffer holds, and the kernel writes
        // nothing past the length it is given.
        unsafe {
            clear_votes::launch_unchecked::<R>(&self.client, cubes, shape, counts);
        }

        self.cast(context, &first, &tally, samples);

        for index in 1..inputs.len() {
            let input = inputs.take_mask(index).ok_or_else(missing_mask)?;

            self.cast(context, &input, &tally, samples);
        }

        let tally_handle = tally.handle().clone();

        // Safety: the tally was allocated to hold exactly `samples` u32 values.
        let counts = unsafe { ArrayArg::from_raw_parts(tally_handle, samples) };
        let write = output.plane().array::<u8, R>();
        let cubes = context.cube_count(samples);
        let shape = context.cube_dim();

        // Safety: every binding names no more elements than its buffer holds, and the kernel writes
        // nothing past the length it is given.
        unsafe {
            tally_votes::launch_unchecked::<R>(&self.client, cubes, shape, counts, write, self.votes);
        }

        Ok(output)
    }
}

impl<R: Runtime> VoteMask<R> {
    fn cast(&self, context: &FrameContext, input: &GpuMask, tally: &PlaneBuffer, samples: usize) {
        let tally_handle = tally.handle().clone();

        // Safety: the tally was allocated to hold exactly `samples` u32 values.
        let counts = unsafe { ArrayArg::from_raw_parts(tally_handle, samples) };
        let read = input.plane().array::<u8, R>();
        let cubes = context.cube_count(samples);
        let shape = context.cube_dim();

        // Safety: every binding names no more elements than its buffer holds, and the kernel writes
        // nothing past the length it is given.
        unsafe {
            cast_vote::launch_unchecked::<R>(&self.client, cubes, shape, read, counts, self.threshold);
        }
    }
}

/// Selects the pixels at least `votes` of its masks select.
pub(crate) struct Vote;

impl StageDefinition for Vote {
    type Args = VoteArgs;

    fn package() -> &'static str {
        "mask"
    }

    fn identifier() -> &'static str {
        "vote"
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

impl MaskDefinition for Vote {
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
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        args.validate(chosen.len())?;

        produces_mask(NAME, chosen)
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        args.validate(formats.inputs().len())?;

        let client = context.client().clone();

        Ok(Box::new(VoteMask {
            client,
            votes: args.votes,
            threshold: u32::from(args.threshold),
        }))
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::SampleFormat;

    use super::*;

    fn capabilities() -> Capabilities {
        av_toolbox_gpu::testing_context().capabilities().clone()
    }

    fn masks(count: usize) -> Vec<Option<StreamFormat>> {
        let format = StreamFormat::Mask(MaskFormat {
            width: 4,
            height: 4,
            sample: SampleFormat::U8,
            chroma: None,
        });

        vec![Some(format); count]
    }

    #[test]
    fn the_defaults_are_two_votes_at_56() {
        let args: VoteArgs = serde_json::from_value(serde_json::json!({})).expect("the defaults apply");

        assert_eq!(args.votes, 2);
        assert_eq!(args.threshold, 56);
    }

    #[test]
    fn more_votes_than_inputs_is_refused() {
        let args = VoteArgs {
            votes: 4,
            threshold: 56,
        };

        let capabilities = capabilities();
        let inputs = masks(3);
        let error = Vote::produces(&capabilities, &args, &inputs).expect_err("4 of 3 cannot agree");

        assert!(error.to_string().contains("between 1 and its 3 inputs, got 4"));
    }

    #[test]
    fn zero_votes_is_refused() {
        let args = VoteArgs {
            votes: 0,
            threshold: 56,
        };

        let capabilities = capabilities();
        let inputs = masks(3);
        let refused = Vote::produces(&capabilities, &args, &inputs);

        assert!(refused.is_err());
    }
}
