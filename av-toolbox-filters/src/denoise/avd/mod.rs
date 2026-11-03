mod args;
mod grain;
mod planes;
mod runner;
#[cfg(test)]
mod tests;
mod tuning;

use std::marker::PhantomData;

use av_denoise_core::EngineSpec;
use av_toolbox_core::{
    ArtefactData,
    ArtefactDecl,
    ArtefactPiece,
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Parallelism,
    StageDefinition,
    StageError,
    StreamType,
    Tag,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use av_toolbox_providers::{SceneRequest, Scenes};

use self::args::engine_spec;
pub(crate) use self::args::{AvdArgs, Nl4dArgs, NlmeansArgs, NlmeansHqArgs};
use self::planes::{checked_layout, plane_groups, stage_geometry};
use self::runner::AvdFilter;
#[cfg(feature = "test-support")]
pub use self::tuning::{forget_tuning_passes, tuning_passes};
use crate::masked::checked_input;

fn not_frames(stage: &str) -> StageError {
    let message = format!("{stage} denoises frames, not masks");

    StageError::msg(message)
}

fn invalid_spec(stage: &str, error: av_denoise_core::Error) -> StageError {
    let message = format!("{stage} cannot denoise these frames: {error}");

    StageError::msg(message)
}

/// Refuses `input` when `args` cannot run on it, short of building the engine.
fn checked_stage<A: AvdArgs>(args: &A, input: &FrameFormat) -> Result<(), StageError> {
    let spec = stage_spec(args, input)?;

    spec.validate().map_err(|error| invalid_spec(A::STAGE, error))
}

/// The engine `args` denoises `input` with.
fn stage_spec<A: AvdArgs>(args: &A, input: &FrameFormat) -> Result<EngineSpec, StageError> {
    let groups = plane_groups(A::STAGE, args.planes(), input)?;
    let geometry = stage_geometry(input, groups);

    engine_spec(args, groups, geometry)
}

/// Denoises frames with one of av-denoise's engines, chosen by the arguments type.
pub(crate) struct Denoise<A>(PhantomData<A>);

impl<A: AvdArgs> StageDefinition for Denoise<A> {
    type Args = A;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        A::IDENTIFIER
    }

    fn description() -> &'static str {
        A::DESCRIPTION
    }

    fn tags() -> &'static [Tag] {
        &[Tag::Denoise]
    }

    fn artefacts() -> &'static [ArtefactDecl] {
        A::artefacts()
    }

    fn merge_artefact(name: &'static str, pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
        A::merge_artefact(name, pieces)
    }

    // Denoising across a cut blends two scenes, so it restarts at every cut.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::chunk::<Scenes>(SceneRequest::natural())
    }

    // A second scene in flight hides the time one engine spends waiting on the host.
    fn workers(_args: &Self::Args) -> Option<usize> {
        Some(2)
    }

    // A format the stage refuses never builds, so it holds nothing.
    fn held_frames(args: &Self::Args, formats: &ResolvedFormats) -> usize {
        let input = checked_input(A::STAGE, formats.inputs());
        let spec = input.and_then(|input| stage_spec(args, &input));

        spec.map_or(0, |spec| spec.max_held_frames())
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl<A: AvdArgs> FilterDefinition for Denoise<A> {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(format) = offered.as_frames() else {
            return Err(not_frames(A::STAGE));
        };

        checked_layout(A::STAGE, format.layout)?;

        Ok(FormatSet::of([format.sample]))
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(A::STAGE, chosen)?;
        checked_stage(args, &input)?;

        let format = StreamFormat::Frames(input);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([SampleFormat::F32]),
        })
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let input = checked_input(A::STAGE, formats.inputs())?;
        let filter = AvdFilter::new(context, &args, input)?;

        Ok(Box::new(filter))
    }
}
