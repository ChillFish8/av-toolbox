mod artefact;
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
    PreparedArtefactSink,
    PreparedFilter,
    PreparedMask,
    PreparedMetaSink,
    PreparedProvider,
    PreparedSink,
    PreparedSource,
};
use crate::{
    BuildStageError,
    Crossing,
    InputArgument,
    InputDemand,
    InputProbe,
    InputSignature,
    OpenSlot,
    Parallelism,
    Probed,
    Residency,
    Seeker,
    StageArtefacts,
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
    description: &'static str,
    schema: &'a Value,
    inputs: &'static InputSignature,
    artefacts: StageArtefacts,
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

    /// What this stage does, its summary line then any explanation.
    pub fn description(&self) -> &'static str {
        self.description
    }

    /// The first line of the description.
    pub fn summary(&self) -> &'static str {
        summary_of(self.description)
    }

    /// The explanation after the description's first blank line, when it has one.
    pub fn details(&self) -> Option<&'static str> {
        details_of(self.description)
    }

    pub fn schema(&self) -> &'a Value {
        self.schema
    }

    pub fn inputs(&self) -> &'static InputSignature {
        self.inputs
    }

    /// The artefacts this stage declares and how it merges them.
    pub fn artefacts(&self) -> StageArtefacts {
        self.artefacts
    }
}

fn summary_of(description: &str) -> &str {
    let first_line = description.lines().next().unwrap_or("");
    first_line.trim()
}

fn details_of(description: &str) -> Option<&str> {
    let (_, rest) = description.split_once("\n\n")?;
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return None;
    }

    Some(trimmed)
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
            Self::Crossing(_) => StageKind::Crossing,
            Self::Provider(_) => StageKind::Provider,
            Self::Sink(_) | Self::MetaSink(_) => StageKind::Sink,
        }
    }

    pub fn parallelism(&self) -> Parallelism {
        match self {
            Self::Source(prepared) => prepared.parallelism(),
            Self::Filter(prepared) => prepared.parallelism(),
            Self::Mask(prepared) => prepared.parallelism(),
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
            Self::Provider(prepared) => prepared.demands(),
            Self::Sink(prepared) => prepared.demands(),
            Self::MetaSink(prepared) => prepared.demands(),
            Self::Crossing(_) => Vec::new(),
        }
    }

    /// What this stage needs to learn from its sources before formats are solved.
    pub fn probes(&self) -> Vec<InputProbe> {
        match self {
            Self::Filter(prepared) => prepared.probes(),
            Self::Mask(prepared) => prepared.probes(),
            Self::Sink(prepared) => prepared.probes(),
            Self::MetaSink(prepared) => prepared.probes(),
            Self::Source(_) | Self::Provider(_) | Self::Crossing(_) => Vec::new(),
        }
    }

    /// Hands the stage what its probes found.
    pub fn settle(&mut self, probed: &Probed) -> Result<(), StageError> {
        match self {
            Self::Filter(prepared) => prepared.settle(probed),
            Self::Mask(prepared) => prepared.settle(probed),
            Self::Sink(prepared) => prepared.settle(probed),
            Self::MetaSink(prepared) => prepared.settle(probed),
            Self::Source(_) | Self::Provider(_) | Self::Crossing(_) => Ok(()),
        }
    }

    /// A seeker over this stage's input, when it is a source that can seek.
    pub fn seeker(&self) -> Result<Option<Box<dyn Seeker>>, StageError> {
        match self {
            Self::Source(prepared) => prepared.seeker(),
            _ => Ok(None),
        }
    }

    /// How many workers this stage would like when the pipeline does not say.
    pub fn suggested_workers(&self) -> Option<usize> {
        match self {
            Self::Source(prepared) => prepared.suggested_workers(),
            Self::Filter(prepared) => prepared.suggested_workers(),
            Self::Mask(prepared) => prepared.suggested_workers(),
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
    /// A metadata-only sink reports [Residency::Either](crate::Residency::Either), and a
    /// crossing reports the side it leads to.
    pub fn residency(&self) -> Residency {
        match self {
            Self::Source(prepared) => prepared.residency(),
            Self::Filter(prepared) => prepared.residency(),
            Self::Mask(prepared) => prepared.residency(),
            Self::Provider(prepared) => prepared.residency(),
            Self::MetaSink(_) => Residency::Either,
            Self::Sink(prepared) => prepared.residency(),
            Self::Crossing(crossing) => crossing.to_side(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_summary_is_the_first_line_trimmed() {
        assert_eq!(summary_of("  Removes noise.  \n\nMore."), "Removes noise.");
    }

    #[test]
    fn the_details_are_everything_after_the_first_blank_line() {
        let description = "Removes noise.\n\nFirst paragraph.\n\nSecond paragraph.\n";

        assert_eq!(
            details_of(description),
            Some("First paragraph.\n\nSecond paragraph.")
        );
    }

    #[test]
    fn a_one_line_description_has_no_details() {
        assert_eq!(details_of("Removes noise."), None);
    }

    #[test]
    fn blank_details_are_none() {
        assert_eq!(details_of("Removes noise.\n\n   \n"), None);
    }
}
