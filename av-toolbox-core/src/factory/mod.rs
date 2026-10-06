mod builders;
mod definition;
mod meta;
mod prepared;
mod provider;

use av_toolbox_gpu::{
    Capabilities,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use serde_json::Value;

pub(crate) use self::definition::Definition;
pub use self::prepared::{
    PreparedFilter,
    PreparedMask,
    PreparedMetaSink,
    PreparedMetaTransform,
    PreparedProvider,
    PreparedSink,
    PreparedSource,
    PreparedTransform,
};
use crate::{
    BuildStageError,
    Crossing,
    InputArgument,
    InputDemand,
    InputSignature,
    OpenSlot,
    Parallelism,
    Residency,
    StageError,
    StageKind,
    Tag,
};

/// A description of a registered stage.
#[derive(Debug, Copy, Clone)]
pub struct StageInfo<'a> {
    kind: StageKind,
    package: &'static str,
    identifier: &'static str,
    full_identifier: &'a str,
    tags: &'static [Tag],
    schema: &'a Value,
    inputs: &'static InputSignature,
}

impl<'a> StageInfo<'a> {
    pub fn kind(&self) -> StageKind {
        self.kind
    }

    pub fn package(&self) -> &'static str {
        self.package
    }

    pub fn identifier(&self) -> &'static str {
        self.identifier
    }

    pub fn full_identifier(&self) -> &'a str {
        self.full_identifier
    }

    pub fn tags(&self) -> &'static [Tag] {
        self.tags
    }

    pub fn schema(&self) -> &'a Value {
        self.schema
    }

    pub fn inputs(&self) -> &'static InputSignature {
        self.inputs
    }
}

/// Prepares stages of one kind without naming their concrete definition.
pub trait StageFactory<P: ?Sized>: Send + Sync {
    fn info(&self) -> StageInfo<'_>;

    /// Validates and deserializes the arguments and binds the inputs, without needing a device.
    fn prepare(
        &self,
        args: &Value,
        inputs: &[InputArgument],
        slot: &OpenSlot,
    ) -> Result<Box<P>, BuildStageError>;

    /// Whether opening this stage's input, given `args`, would wait on a live stream.
    fn reads_stream(&self, _args: &Value) -> Result<bool, BuildStageError> {
        Ok(false)
    }
}

/// A prepared stage of any kind.
pub enum PreparedStage {
    Source(Box<dyn PreparedSource>),
    Filter(Box<dyn PreparedFilter>),
    Mask(Box<dyn PreparedMask>),
    Transform(Box<dyn PreparedTransform>),
    MetaTransform(Box<dyn PreparedMetaTransform>),
    Provider(Box<dyn PreparedProvider>),
    Sink(Box<dyn PreparedSink>),
    MetaSink(Box<dyn PreparedMetaSink>),
    /// A crossing that moves a stream between host and device.
    Crossing(Crossing),
}

impl PreparedStage {
    pub fn kind(&self) -> StageKind {
        match self {
            Self::Source(_) => StageKind::Source,
            Self::Filter(_) => StageKind::Filter,
            Self::Mask(_) => StageKind::Mask,
            Self::Transform(_) | Self::MetaTransform(_) | Self::Crossing(_) => StageKind::Transform,
            Self::Provider(_) => StageKind::Provider,
            Self::Sink(_) | Self::MetaSink(_) => StageKind::Sink,
        }
    }

    pub fn parallelism(&self) -> Parallelism {
        match self {
            Self::Source(prepared) => prepared.parallelism(),
            Self::Filter(prepared) => prepared.parallelism(),
            Self::Mask(prepared) => prepared.parallelism(),
            Self::Transform(prepared) => prepared.parallelism(),
            Self::MetaTransform(prepared) => prepared.parallelism(),
            // A provider reads the whole stream in order to supply it.
            Self::Provider(_) => Parallelism::Stream,
            Self::Sink(prepared) => prepared.parallelism(),
            Self::MetaSink(prepared) => prepared.parallelism(),
            // A crossing moves one item at a time and holds nothing between them.
            Self::Crossing(_) => Parallelism::Frame,
        }
    }

    pub fn chunk_authority(&self) -> Option<usize> {
        match self {
            Self::Source(prepared) => prepared.chunk_authority(),
            Self::Filter(prepared) => prepared.chunk_authority(),
            Self::Mask(prepared) => prepared.chunk_authority(),
            Self::Transform(prepared) => prepared.chunk_authority(),
            Self::MetaTransform(prepared) => prepared.chunk_authority(),
            Self::Provider(prepared) => prepared.chunk_authority(),
            Self::Sink(prepared) => prepared.chunk_authority(),
            Self::MetaSink(prepared) => prepared.chunk_authority(),
            // A crossing's one input is the only thing it could line up by.
            Self::Crossing(_) => Some(0),
        }
    }

    pub fn demands(&self) -> Vec<InputDemand> {
        match self {
            Self::Source(prepared) => prepared.demands(),
            Self::Filter(prepared) => prepared.demands(),
            Self::Mask(prepared) => prepared.demands(),
            Self::Transform(prepared) => prepared.demands(),
            Self::MetaTransform(prepared) => prepared.demands(),
            Self::Provider(_) => Vec::new(),
            Self::Sink(prepared) => prepared.demands(),
            Self::MetaSink(prepared) => prepared.demands(),
            Self::Crossing(_) => Vec::new(),
        }
    }

    /// How many workers this stage would like when the pipeline does not say.
    pub fn suggested_workers(&self) -> Option<usize> {
        match self {
            Self::Source(prepared) => prepared.suggested_workers(),
            Self::Filter(prepared) => prepared.suggested_workers(),
            Self::Mask(prepared) => prepared.suggested_workers(),
            Self::Transform(prepared) => prepared.suggested_workers(),
            Self::MetaTransform(prepared) => prepared.suggested_workers(),
            Self::Provider(_) => None,
            Self::Sink(prepared) => prepared.suggested_workers(),
            Self::MetaSink(prepared) => prepared.suggested_workers(),
            Self::Crossing(_) => None,
        }
    }

    pub fn held_frames(&self, formats: &ResolvedFormats) -> usize {
        match self {
            Self::Source(prepared) => prepared.held_frames(formats),
            Self::Filter(prepared) => prepared.held_frames(formats),
            Self::Mask(prepared) => prepared.held_frames(formats),
            Self::Transform(prepared) => prepared.held_frames(formats),
            Self::MetaTransform(prepared) => prepared.held_frames(formats),
            Self::Provider(prepared) => prepared.held_frames(formats),
            Self::Sink(prepared) => prepared.held_frames(formats),
            Self::MetaSink(prepared) => prepared.held_frames(formats),
            Self::Crossing(_) => 0,
        }
    }

    /// A source is never asked, because it has no inputs to be offered anything on.
    pub fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        match self {
            Self::Source(_) => Err(StageError::msg(
                "accepts_input was asked of a source, which has no inputs, so reaching this is a solver bug",
            )),
            Self::Filter(prepared) => prepared.accepts_input(capabilities, index, offered, inputs),
            Self::Mask(prepared) => prepared.accepts_input(capabilities, index, offered, inputs),
            Self::Transform(prepared) => prepared.accepts_input(capabilities, index, offered, inputs),
            Self::MetaTransform(prepared) => prepared.accepts_input(capabilities, index, offered, inputs),
            Self::Provider(prepared) => prepared.accepts_input(capabilities, offered),
            Self::Sink(prepared) => prepared.accepts_input(capabilities, index, offered, inputs),
            Self::MetaSink(prepared) => prepared.accepts_input(capabilities, index, offered, inputs),
            // A crossing never changes a format, so it accepts back whatever it was offered.
            Self::Crossing(_) => Ok(FormatSet::of([offered.sample()])),
        }
    }

    /// A sink is never asked, because it produces no stream.
    pub fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        match self {
            Self::Source(prepared) => prepared.produces(capabilities),
            Self::Filter(prepared) => prepared.produces(capabilities, chosen),
            Self::Mask(prepared) => prepared.produces(capabilities, chosen),
            Self::Transform(prepared) => prepared.produces(capabilities, chosen),
            Self::MetaTransform(prepared) => prepared.produces(capabilities, chosen),
            Self::Provider(prepared) => prepared.produces(chosen),
            Self::Sink(_) | Self::MetaSink(_) => Err(StageError::msg(
                "produces was asked of a sink, which produces no stream, so reaching this is a solver bug",
            )),
            Self::Crossing(_) => {
                let format = chosen.first().copied().flatten().ok_or_else(|| {
                    StageError::msg("a crossing was asked what it produces before its one input was chosen")
                })?;

                Ok(OutputFormats {
                    template: StreamTemplate::from_format(&format),
                    samples: FormatSet::of([format.sample()]),
                })
            },
        }
    }

    /// Which side this stage runs on.
    ///
    /// A metadata-only transform or sink reports [Residency::Either](crate::Residency::Either), and a
    /// crossing reports the side it leads to.
    pub fn residency(&self) -> Residency {
        match self {
            Self::Source(prepared) => prepared.residency(),
            Self::Filter(prepared) => prepared.residency(),
            Self::Mask(prepared) => prepared.residency(),
            Self::Transform(prepared) => prepared.residency(),
            Self::Provider(prepared) => prepared.residency(),
            Self::MetaTransform(_) | Self::MetaSink(_) => Residency::Either,
            Self::Sink(prepared) => prepared.residency(),
            Self::Crossing(crossing) => crossing.to_side(),
        }
    }
}
