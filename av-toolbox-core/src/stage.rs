use std::fmt;

use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::{
    Artefact,
    ArtefactData,
    ArtefactDecl,
    ArtefactPiece,
    ArtefactSink,
    BoundInputs,
    Filter,
    InputDemand,
    InputProbe,
    InputSignature,
    KeyFilter,
    Mask,
    MetaSink,
    Parallelism,
    Probed,
    Seeker,
    Sink,
    Source,
    StageError,
    StageSide,
    StreamType,
    Tag,
    WantedArtefacts,
    merge_single,
};

/// The category a stage belongs to.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum StageKind {
    Source,
    Filter,
    Mask,
    /// Supplies what other stages demand, placed by the graph rather than the pipeline.
    Provider,
    /// Moves a stream between host and device, placed by the graph rather than the pipeline.
    Crossing,
    Sink,
}

impl StageKind {
    /// What a stage of this kind produces, or `None` for a sink.
    pub const fn output(&self) -> Option<StreamType> {
        match self {
            Self::Source | Self::Filter | Self::Provider | Self::Crossing => Some(StreamType::Frames),
            Self::Mask => Some(StreamType::Mask),
            Self::Sink => None,
        }
    }
}

impl fmt::Display for StageKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Source => "source",
            Self::Filter => "filter",
            Self::Mask => "mask",
            Self::Provider => "provider",
            Self::Crossing => "crossing",
            Self::Sink => "sink",
        };

        formatter.write_str(name)
    }
}

/// A constructed stage ready to be placed in a pipeline.
pub trait StageInstance: Send + 'static {
    /// Called once before the first item of a stream.
    ///
    /// A stage holding a temporal window uses this to know its ring holds padding rather than
    /// history. A stage that is given one chunk at a time is called at the start of every
    /// chunk, because each chunk is a stream of its own to it.
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        Ok(())
    }

    /// Called once after the stage is built and before its first stream, with the artefacts some sink will write.
    ///
    /// A stage skips the work behind any artefact not named here, and refuses when it cannot produce one that is.
    fn wants_artefacts(&mut self, _wanted: &WantedArtefacts) -> Result<(), StageError> {
        Ok(())
    }

    /// Hands over the artefacts built since the last call.
    ///
    /// Called each time the stream ends, which for a stage given one chunk at a time is the end of every chunk.
    fn take_artefacts(&mut self, _context: &mut FrameContext) -> Result<Vec<Artefact>, StageError> {
        Ok(Vec::new())
    }
}

/// A type of stage that can be registered.
///
/// The arguments type supplies the JSON schema for a stage's arguments, and the input signature
/// declares which inputs it accepts. This carries only what every kind has in common. What a stage
/// negotiates and how it is built come from the matching per-kind trait.
///
/// # Example
///
/// ```
/// use av_toolbox_core::{
///     BoundInputs,
///     Filter,
///     FilterDefinition,
///     InputParam,
///     InputSignature,
///     Inputs,
///     PipelineRegistry,
///     StageDefinition,
///     StageError,
///     StageInstance,
/// };
/// use av_toolbox_gpu::{
///     BuildContext,
///     Capabilities,
///     DeviceSide,
///     FormatSet,
///     Frame,
///     FrameContext,
///     Offered,
///     OutputFormats,
///     ResolvedFormats,
///     Runtime,
///     StreamFormat,
///     StreamTemplate,
/// };
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(Debug, Deserialize, JsonSchema)]
/// struct DenoiseArgs {
///     sigma_scale: f32,
/// }
///
/// struct DenoiseFilter {
///     sigma_scale: f32,
/// }
///
/// impl StageInstance for DenoiseFilter {}
///
/// impl Filter for DenoiseFilter {
///     fn process(
///         &mut self,
///         _context: &mut FrameContext,
///         mut inputs: Inputs,
///     ) -> Result<Vec<Frame>, StageError> {
///         let frame = inputs
///             .take_frame(0)
///             .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
///
///         Ok(vec![frame])
///     }
/// }
///
/// struct Denoise;
///
/// impl StageDefinition for Denoise {
///     type Args = DenoiseArgs;
///
///     fn package() -> &'static str {
///         "av-denoise"
///     }
///
///     fn identifier() -> &'static str {
///         "nl4d"
///     }
///
///     fn description() -> &'static str {
///         "Removes noise by filtering matching blocks across frames."
///     }
///
///     fn inputs() -> &'static InputSignature {
///         const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
///         SIGNATURE
///     }
/// }
///
/// impl FilterDefinition for Denoise {
///     type Side = DeviceSide;
///
///     fn accepts_input(
///         _capabilities: &Capabilities,
///         _args: &Self::Args,
///         _index: usize,
///         offered: StreamFormat,
///         _inputs: &[Offered],
///     ) -> Result<FormatSet, StageError> {
///         Ok(FormatSet::of([offered.sample()]))
///     }
///
///     fn produces(
///         _capabilities: &Capabilities,
///         _args: &Self::Args,
///         chosen: &[Option<StreamFormat>],
///     ) -> Result<OutputFormats, StageError> {
///         let format = chosen.iter().flatten().next().copied();
///         let format = format.ok_or_else(|| StageError::msg("no input was bound"))?;
///         Ok(OutputFormats {
///             template: StreamTemplate::from_format(&format),
///             samples: FormatSet::of([format.sample()]),
///         })
///     }
///
///     fn build<R: Runtime>(
///         _context: &BuildContext<R>,
///         args: Self::Args,
///         _inputs: BoundInputs,
///         _formats: &ResolvedFormats,
///     ) -> Result<Box<dyn Filter<DeviceSide>>, StageError> {
///         Ok(Box::new(DenoiseFilter {
///             sigma_scale: args.sigma_scale,
///         }))
///     }
/// }
///
/// let mut registry = PipelineRegistry::new();
/// registry.register_filter::<Denoise>().expect("should register");
/// ```
pub trait StageDefinition: 'static {
    /// The arguments this stage takes, supplied by the pipeline as JSON.
    type Args: DeserializeOwned + JsonSchema + Send + Sync;

    fn package() -> &'static str;

    fn identifier() -> &'static str;

    /// What this stage does, for someone choosing stages for a pipeline.
    ///
    /// The first line is a short summary. A longer explanation may follow after a blank line.
    fn description() -> &'static str;

    fn tags() -> &'static [Tag] {
        &[]
    }

    /// The smallest unit of a stream this stage can be given on its own.
    ///
    /// The whole stream is always safe, so that is the default. A stage that can take less
    /// says so, and more than one of it may then be run at once.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Stream
    }

    /// The most frames this stage may hold back at once, given what its inputs and output resolved to.
    ///
    /// It sizes how far a source may run ahead of the stage. The stage may hold fewer at any
    /// point, and holding more only risks the source waiting on its budget.
    fn held_frames(_args: &Self::Args, _formats: &ResolvedFormats) -> usize {
        0
    }

    /// What this stage asks of the frames reaching its inputs.
    ///
    /// A chunk stage's own demand is added on its chunk authority without being listed here.
    fn demands(_args: &Self::Args) -> Vec<InputDemand> {
        Vec::new()
    }

    /// What this stage needs to learn from its sources before formats are solved.
    fn probes(_args: &Self::Args) -> Vec<InputProbe> {
        Vec::new()
    }

    /// Settles the args with what this stage's probes found.
    ///
    /// It is called each time the stage is prepared once its probes have run.
    fn settle(_args: &mut Self::Args, _probed: &Probed) -> Result<(), StageError> {
        Ok(())
    }

    /// How many workers this stage would like when the pipeline does not say.
    fn workers(_args: &Self::Args) -> Option<usize> {
        None
    }

    /// The artefacts this stage can hand over at the end of a stream.
    fn artefacts() -> &'static [ArtefactDecl] {
        &[]
    }

    /// Merges the pieces of the artefact `name` into one, given in chunk order.
    ///
    /// A stage run as more than one instance, or given one chunk at a time, hands over a piece per instance and
    /// chunk, as does a stage sharing a region routed by chunk. The result must hold the kind of data the artefact
    /// is declared with. The default accepts a lone piece and refuses more.
    fn merge_artefact(name: &'static str, pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
        merge_single(name, pieces)
    }

    fn inputs() -> &'static InputSignature;
}

/// A stage that produces frames from something outside the pipeline.
pub trait SourceDefinition: StageDefinition {
    /// The side this stage runs on.
    type Side: StageSide;

    /// What opening the input yields, handed from `produces` to `build`.
    type Opened: Send + 'static;

    /// Opens the input, which happens once per graph.
    fn open(args: &Self::Args) -> Result<Self::Opened, StageError>;

    /// Whether opening the input waits on a live stream, such as standard input or a pipe.
    fn reads_stream(_args: &Self::Args) -> bool {
        false
    }

    /// A reader that can jump to any frame, for sources that can seek.
    ///
    /// Reading through it never moves the stream the built source produces.
    fn seeker(_opened: &Self::Opened) -> Result<Option<Box<dyn Seeker>>, StageError> {
        Ok(None)
    }

    /// What it could produce, given what opening the input found.
    fn produces(capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError>;

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        opened: Self::Opened,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<Self::Side>>, StageError>;
}

/// A stage that changes the frames passing through it.
pub trait FilterDefinition: StageDefinition {
    /// The side this stage runs on.
    type Side: StageSide;

    /// Which sample formats this stage will take on one input, most preferred first.
    ///
    /// The offered layout, dimensions and colour information are the stage's to validate, and an
    /// unusable one is an error rather than an empty set. Raise one only for those, because the
    /// stage is asked once per sample its producer could emit and an error stops the whole edge
    /// rather than the one sample. A sample the stage cannot work in is left out of the set.
    ///
    /// `inputs` holds every input in signature order, including this one at `index`. A stage whose
    /// answer depends on more than one input reads it, and what it returns is binding only when
    /// every other input is [Offered::Resolved](av_toolbox_gpu::Offered::Resolved). While any of them is
    /// [Offered::Proposed](av_toolbox_gpu::Offered::Proposed) or
    /// [Offered::Unknown](av_toolbox_gpu::Offered::Unknown), leaving a sample out reads as a preference
    /// rather than a refusal, so the solver may still settle on it when nothing else suits every
    /// reader. An optional input nobody bound never settles, so it does not hold an answer back.
    /// The stage is asked again once everything is decided, and that answer is the one that holds.
    /// Apply the same check from whichever input is asked about, so the answer does not depend on
    /// which came first.
    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    /// What it could produce, given what it will be given.
    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter<Self::Side>>, StageError>;
}

/// A stage that selects pixels from two or more inputs.
pub trait MaskDefinition: StageDefinition {
    /// The side this stage runs on.
    type Side: StageSide;

    /// Which sample formats this stage will take on one input, most preferred first.
    ///
    /// The offered layout, dimensions and colour information are the stage's to validate, and an
    /// unusable one is an error rather than an empty set. Raise one only for those, because the
    /// stage is asked once per sample its producer could emit and an error stops the whole edge
    /// rather than the one sample. A sample the stage cannot work in is left out of the set.
    ///
    /// `inputs` holds every input in signature order, including this one at `index`. A stage whose
    /// answer depends on more than one input reads it, and what it returns is binding only when
    /// every other input is [Offered::Resolved](av_toolbox_gpu::Offered::Resolved). While any of them is
    /// [Offered::Proposed](av_toolbox_gpu::Offered::Proposed) or
    /// [Offered::Unknown](av_toolbox_gpu::Offered::Unknown), leaving a sample out reads as a preference
    /// rather than a refusal, so the solver may still settle on it when nothing else suits every
    /// reader. An optional input nobody bound never settles, so it does not hold an answer back.
    /// The stage is asked again once everything is decided, and that answer is the one that holds.
    /// Apply the same check from whichever input is asked about, so the answer does not depend on
    /// which came first.
    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    /// What it could produce, given what it will be given.
    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask<Self::Side>>, StageError>;
}

/// A sink that only reads where items sit in the stream, and so runs on either side.
pub trait MetaSinkDefinition: StageDefinition {
    /// Which sample formats this stage will take on one input, most preferred first.
    ///
    /// The same contract as [SinkDefinition::accepts_input](crate::SinkDefinition::accepts_input).
    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    /// Builds the instance. A metadata-only sink has no side to build for.
    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaSink>, StageError>;
}

/// A sink that writes the artefacts stages hand over, rather than frames.
///
/// It names the stages it takes artefacts from as its inputs, declaring
/// [InputSignature::ARTEFACTS](crate::InputSignature::ARTEFACTS), and is handed its artefacts once
/// the run has ended.
pub trait ArtefactSinkDefinition: StageDefinition {
    /// The artefacts this sink takes from its inputs, matched against artefact names such as `grain_table`.
    fn selection(args: &Self::Args) -> KeyFilter;

    fn build(args: Self::Args) -> Result<Box<dyn ArtefactSink>, StageError>;
}

/// A stage that consumes the finished stream.
pub trait SinkDefinition: StageDefinition {
    /// The side this stage runs on.
    type Side: StageSide;

    /// Which sample formats this stage will take on one input, most preferred first.
    ///
    /// The offered layout, dimensions and colour information are the stage's to validate, and an
    /// unusable one is an error rather than an empty set. Raise one only for those, because the
    /// stage is asked once per sample its producer could emit and an error stops the whole edge
    /// rather than the one sample. A sample the stage cannot work in is left out of the set.
    ///
    /// `inputs` holds every input in signature order, including this one at `index`. A stage whose
    /// answer depends on more than one input reads it, and what it returns is binding only when
    /// every other input is [Offered::Resolved](av_toolbox_gpu::Offered::Resolved). While any of them is
    /// [Offered::Proposed](av_toolbox_gpu::Offered::Proposed) or
    /// [Offered::Unknown](av_toolbox_gpu::Offered::Unknown), leaving a sample out reads as a preference
    /// rather than a refusal, so the solver may still settle on it when nothing else suits every
    /// reader. An optional input nobody bound never settles, so it does not hold an answer back.
    /// The stage is asked again once everything is decided, and that answer is the one that holds.
    /// Apply the same check from whichever input is asked about, so the answer does not depend on
    /// which came first.
    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink<Self::Side>>, StageError>;
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use av_toolbox_gpu::{DeviceSide, Frame, StreamTemplate};
    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use crate::bind_inputs;
    use crate::{InputParam, Inputs};

    #[derive(Debug, Deserialize, JsonSchema)]
    struct ExampleArgs {
        sigma_scale: f32,
    }

    struct ExampleInstance;

    impl StageInstance for ExampleInstance {}

    impl Filter for ExampleInstance {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs,
        ) -> Result<Vec<Frame>, StageError> {
            let frame = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

            Ok(vec![frame])
        }
    }

    static RECORDED_SIGMA_SCALE: AtomicU32 = AtomicU32::new(0);

    struct ExampleDefinition;

    impl StageDefinition for ExampleDefinition {
        type Args = ExampleArgs;

        fn package() -> &'static str {
            "av-denoise"
        }

        fn identifier() -> &'static str {
            "nl4d"
        }

        fn description() -> &'static str {
            "A stage used by tests."
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
            SIGNATURE
        }
    }

    impl FilterDefinition for ExampleDefinition {
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
            args: Self::Args,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Filter>, StageError> {
            RECORDED_SIGMA_SCALE.store(args.sigma_scale.to_bits(), Ordering::SeqCst);

            Ok(Box::new(ExampleInstance))
        }
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_definition_builds_its_instance_against_a_real_backend() {
        use av_toolbox_gpu::{Backend, BuildTask, test_device};

        struct BuildExample {
            args: ExampleArgs,
            inputs: BoundInputs,
        }

        impl BuildTask for BuildExample {
            type Output = Result<Box<dyn Filter>, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                let formats = ResolvedFormats::new(vec![], None);
                ExampleDefinition::build(context, self.args, self.inputs, &formats)
            }
        }

        let backend = Backend::open_default(test_device()).expect("should open");
        let bound = bind_inputs(&InputSignature::EMPTY, &[]).expect("should bind");

        let task = BuildExample {
            args: ExampleArgs { sigma_scale: 0.8 },
            inputs: bound,
        };

        let instance = backend.build(task).expect("should build");
        drop(instance);

        let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
        assert_eq!(recorded_sigma_scale, 0.8);
    }

    #[test]
    fn tags_default_to_empty() {
        assert!(ExampleDefinition::tags().is_empty());
    }

    #[test]
    fn a_stage_defaults_to_needing_the_whole_stream() {
        let args = ExampleArgs { sigma_scale: 0.8 };

        assert_eq!(ExampleDefinition::parallelism(&args), Parallelism::Stream);
    }

    #[test]
    fn every_kind_but_a_sink_produces_a_stream() {
        assert_eq!(StageKind::Source.output(), Some(StreamType::Frames));
        assert_eq!(StageKind::Filter.output(), Some(StreamType::Frames));
        assert_eq!(StageKind::Provider.output(), Some(StreamType::Frames));
        assert_eq!(StageKind::Crossing.output(), Some(StreamType::Frames));
        assert_eq!(StageKind::Mask.output(), Some(StreamType::Mask));
        assert_eq!(StageKind::Sink.output(), None);
    }

    #[test]
    fn a_crossing_displays_as_crossing() {
        assert_eq!(StageKind::Crossing.to_string(), "crossing");
    }
}
