use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
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
    Frame,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::engine::tests::support::StageLife;

// A filter that reads two inputs and passes the first through, standing in for a mask-and-frame
// filter, an overlay, or a composite: any two input stage whose producers can end up in
// different regions.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct JoinArgs {}

pub(crate) struct JoinInstance {
    _life: StageLife,
}

impl JoinInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for JoinInstance {}

impl Filter for JoinInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let base = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        inputs
            .take_frame(1)
            .ok_or_else(|| StageError::msg("expected a frame in slot 1"))?;

        Ok(vec![base])
    }
}

pub(crate) struct Join;

impl StageDefinition for Join {
    type Args = JoinArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "join"
    }

    // Per chunk, so a region headed by one routes by chunk. `RunWarning::DetectorOnWrongBranch`
    // and `RunWarning::NothingChunked` are only ever noticed on that path, and this is the
    // fixture every test of them is written around.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Chunk(None)
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base").accepting(StreamType::Frames),
            InputParam::required("overlay").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for Join {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(JoinInstance::new()))
    }
}

// A filter that passes its first input through and marks its second, optional one as the chunk
// authority. Leaving that optional input unbound is how a pipeline declares an authority nothing
// ever fills, which a frame routed region reads nothing from at all.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct ReferringArgs {
    /// Declares the stage per chunk rather than per frame, so one fixture covers both routings.
    #[serde(default)]
    per_chunk: bool,
}

pub(crate) struct ReferringInstance {
    _life: StageLife,
}

impl ReferringInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for ReferringInstance {}

impl Filter for ReferringInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

pub(crate) struct Referring;

impl StageDefinition for Referring {
    type Args = ReferringArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "referring"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.per_chunk {
            true => Parallelism::Chunk(None),
            false => Parallelism::Frame,
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("input").accepting(StreamType::Frames),
            InputParam::optional("reference")
                .accepting(StreamType::Frames)
                .chunk_authority(),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for Referring {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(ReferringInstance::new()))
    }
}
