use std::marker::PhantomData;

use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    Capabilities,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use jsonschema::Validator;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    ArgsError,
    BindInputsError,
    BoundInputs,
    BuildStageError,
    BuiltFilter,
    BuiltMask,
    BuiltSink,
    BuiltSource,
    BuiltTransform,
    Crossing,
    FilterDefinition,
    InputArgument,
    InputSignature,
    MaskDefinition,
    MetaTransform,
    MetaTransformDefinition,
    OpenSlot,
    Parallelism,
    RegistryError,
    Residency,
    SchemaViolation,
    SinkDefinition,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageKind,
    StageSide,
    Tag,
    TransformDefinition,
    bind_inputs,
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
}

/// A source whose arguments are deserialized, ready to negotiate and build.
pub trait PreparedSource: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    fn produces(&self, capabilities: &Capabilities) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltSource, BuildStageError>;
}

/// A filter whose arguments are deserialized and whose inputs are bound.
pub trait PreparedFilter: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltFilter, BuildStageError>;
}

/// A mask whose arguments are deserialized and whose inputs are bound.
pub trait PreparedMask: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltMask, BuildStageError>;
}

/// A transform whose arguments are deserialized and whose inputs are bound.
pub trait PreparedTransform: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltTransform, BuildStageError>;
}

/// A metadata-only transform whose arguments are deserialized and whose inputs are bound.
pub trait PreparedMetaTransform: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    /// Builds the instance against a real backend. A metadata-only transform has no side to build
    /// for.
    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaTransform>, BuildStageError>;
}

/// A sink whose arguments are deserialized and whose inputs are bound.
pub trait PreparedSink: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltSink, BuildStageError>;
}

/// A prepared stage of any kind.
pub enum PreparedStage {
    Source(Box<dyn PreparedSource>),
    Filter(Box<dyn PreparedFilter>),
    Mask(Box<dyn PreparedMask>),
    Transform(Box<dyn PreparedTransform>),
    MetaTransform(Box<dyn PreparedMetaTransform>),
    Sink(Box<dyn PreparedSink>),
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
            Self::Sink(_) => StageKind::Sink,
        }
    }

    pub fn parallelism(&self) -> Parallelism {
        match self {
            Self::Source(prepared) => prepared.parallelism(),
            Self::Filter(prepared) => prepared.parallelism(),
            Self::Mask(prepared) => prepared.parallelism(),
            Self::Transform(prepared) => prepared.parallelism(),
            Self::MetaTransform(prepared) => prepared.parallelism(),
            Self::Sink(prepared) => prepared.parallelism(),
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
            Self::Sink(prepared) => prepared.chunk_authority(),
            // A crossing's one input is the only thing it could line up by.
            Self::Crossing(_) => Some(0),
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
            Self::Sink(prepared) => prepared.accepts_input(capabilities, index, offered, inputs),
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
            Self::Sink(_) => Err(StageError::msg(
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
    /// A metadata-only transform reports [Residency::Either](crate::Residency::Either), and a crossing reports the side it
    /// leads to.
    pub fn residency(&self) -> Residency {
        match self {
            Self::Source(prepared) => prepared.residency(),
            Self::Filter(prepared) => prepared.residency(),
            Self::Mask(prepared) => prepared.residency(),
            Self::Transform(prepared) => prepared.residency(),
            Self::MetaTransform(_) => Residency::Either,
            Self::Sink(prepared) => prepared.residency(),
            Self::Crossing(crossing) => crossing.to_side(),
        }
    }
}

/// The adapter that erases a concrete [StageDefinition](crate::StageDefinition).
pub(crate) struct Definition<D: StageDefinition> {
    kind: StageKind,
    full_identifier: String,
    schema: Value,
    validator: Validator,
    marker: PhantomData<fn() -> D>,
}

impl<D: StageDefinition> Definition<D> {
    /// Generates and compiles the stage's schema, failing if anything about it is malformed.
    pub(crate) fn new(kind: StageKind) -> Result<Self, RegistryError> {
        Self::check_segment(D::package(), "package")?;
        Self::check_segment(D::identifier(), "identifier")?;

        let full_identifier = format!("{}/{}", D::package(), D::identifier());

        let generator = schemars::SchemaGenerator::default();
        let generated = generator.into_root_schema_for::<D::Args>();
        let schema = serde_json::to_value(&generated).map_err(|error| RegistryError::InvalidSchema {
            identifier: full_identifier.clone(),
            reason: error.to_string(),
        })?;

        let validator = jsonschema::validator_for(&schema).map_err(|error| RegistryError::InvalidSchema {
            identifier: full_identifier.clone(),
            reason: error.to_string(),
        })?;

        Ok(Self {
            kind,
            full_identifier,
            schema,
            validator,
            marker: PhantomData,
        })
    }

    pub(crate) fn full_identifier(&self) -> &str {
        &self.full_identifier
    }

    pub(crate) fn kind(&self) -> StageKind {
        self.kind
    }

    fn stage_info(&self) -> StageInfo<'_> {
        StageInfo {
            kind: self.kind,
            package: D::package(),
            identifier: D::identifier(),
            full_identifier: &self.full_identifier,
            tags: D::tags(),
            schema: &self.schema,
            inputs: D::inputs(),
        }
    }

    fn deserialize_args(&self, args: &Value) -> Result<D::Args, BuildStageError> {
        let violations: Vec<SchemaViolation> = self
            .validator
            .iter_errors(args)
            .map(|error| SchemaViolation::new(error.instance_path().to_string(), error.to_string()))
            .collect();
        if !violations.is_empty() {
            return Err(self.invalid_args(ArgsError::SchemaViolations(violations)));
        }

        let deserialized = D::Args::deserialize(args);
        deserialized.map_err(|error| self.invalid_args(ArgsError::Deserialize(error)))
    }

    fn invalid_args(&self, source: ArgsError) -> BuildStageError {
        BuildStageError::InvalidArgs {
            identifier: self.full_identifier.clone(),
            source,
        }
    }

    fn invalid_inputs(&self, source: BindInputsError) -> BuildStageError {
        BuildStageError::InvalidInputs {
            identifier: self.full_identifier.clone(),
            source,
        }
    }

    /// Rejects a segment that would produce a malformed `package/stage` identifier.
    fn check_segment(segment: &str, label: &'static str) -> Result<(), RegistryError> {
        if !segment.is_empty() && !segment.contains('/') {
            return Ok(());
        }

        Err(RegistryError::InvalidIdentifier {
            package: D::package().to_string(),
            identifier: D::identifier().to_string(),
            segment: label,
        })
    }
}

/// A stage with its arguments deserialized and its inputs bound, ready to negotiate and build.
pub(crate) struct Prepared<D: StageDefinition> {
    identifier: String,
    args: D::Args,
    inputs: BoundInputs,
    slot: OpenSlot,
}

macro_rules! impl_stage_factory {
    ($kind_trait:ident, $definition_trait:ident, $prepared_trait:ident, $task:ident, $built:ty, $wrap:ident) => {
        /// Carries a stage's pieces into whichever runtime the backend opened.
        struct $task<'a, D: $definition_trait> {
            args: D::Args,
            inputs: BoundInputs,
            formats: &'a ResolvedFormats,
        }

        impl<'a, D: $definition_trait> BuildTask for $task<'a, D> {
            type Output = Result<$built, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.args, self.inputs, self.formats)
                    .map(<D::Side as StageSide>::$wrap)
            }
        }

        impl<D: $definition_trait> $prepared_trait for Prepared<D> {
            fn parallelism(&self) -> Parallelism {
                D::parallelism(&self.args)
            }

            fn chunk_authority(&self) -> Option<usize> {
                self.inputs.chunk_authority()
            }

            fn accepts_input(
                &self,
                capabilities: &Capabilities,
                index: usize,
                offered: StreamFormat,
                inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                D::accepts_input(capabilities, &self.args, index, offered, inputs)
            }

            fn produces(
                &self,
                capabilities: &Capabilities,
                chosen: &[Option<StreamFormat>],
            ) -> Result<OutputFormats, StageError> {
                D::produces(capabilities, &self.args, chosen)
            }

            fn residency(&self) -> Residency {
                <D::Side as StageSide>::RESIDENCY
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<$built, BuildStageError> {
                let identifier = self.identifier;
                let task = $task::<D> {
                    args: self.args,
                    inputs: self.inputs,
                    formats,
                };

                backend
                    .build(task)
                    .map_err(|source| BuildStageError::Build { identifier, source })
            }
        }

        impl<D: $definition_trait> StageFactory<dyn $prepared_trait> for Definition<D> {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn prepare(
                &self,
                args: &Value,
                inputs: &[InputArgument],
                slot: &OpenSlot,
            ) -> Result<Box<dyn $prepared_trait>, BuildStageError> {
                let args = self.deserialize_args(args)?;
                let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

                Ok(Box::new(Prepared::<D> {
                    identifier: self.full_identifier().to_string(),
                    args,
                    inputs: bound,
                    slot: slot.clone(),
                }))
            }
        }
    };
}

/// Writes a source's build task and factory, since a source has no `accepts_input` to negotiate.
macro_rules! impl_source_factory {
    ($task:ident) => {
        struct $task<'a, D: SourceDefinition> {
            opened: D::Opened,
            inputs: BoundInputs,
            formats: &'a ResolvedFormats,
        }

        impl<'a, D: SourceDefinition> BuildTask for $task<'a, D> {
            type Output = Result<BuiltSource, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.opened, self.inputs, self.formats)
                    .map(<D::Side as StageSide>::source)
            }
        }

        impl<D: SourceDefinition> PreparedSource for Prepared<D> {
            fn parallelism(&self) -> Parallelism {
                D::parallelism(&self.args)
            }

            fn chunk_authority(&self) -> Option<usize> {
                self.inputs.chunk_authority()
            }

            fn produces(&self, capabilities: &Capabilities) -> Result<OutputFormats, StageError> {
                let open = || D::open(&self.args);
                let produced = self
                    .slot
                    .read(open, |opened: &D::Opened| D::produces(capabilities, opened))?;

                produced
            }

            fn residency(&self) -> Residency {
                <D::Side as StageSide>::RESIDENCY
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<BuiltSource, BuildStageError> {
                let identifier = self.identifier;
                let open = || D::open(&self.args);
                let opened = self
                    .slot
                    .take(open)
                    .map_err(|source| BuildStageError::Build {
                        identifier: identifier.clone(),
                        source,
                    })?;

                let task = $task::<D> {
                    opened,
                    inputs: self.inputs,
                    formats,
                };

                backend
                    .build(task)
                    .map_err(|source| BuildStageError::Build { identifier, source })
            }
        }

        impl<D: SourceDefinition> StageFactory<dyn PreparedSource> for Definition<D> {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn prepare(
                &self,
                args: &Value,
                inputs: &[InputArgument],
                slot: &OpenSlot,
            ) -> Result<Box<dyn PreparedSource>, BuildStageError> {
                let args = self.deserialize_args(args)?;
                let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

                Ok(Box::new(Prepared::<D> {
                    identifier: self.full_identifier().to_string(),
                    args,
                    inputs: bound,
                    slot: slot.clone(),
                }))
            }
        }
    };
}

/// Writes a sink's build task and factory, since a sink has no `produces` to negotiate.
macro_rules! impl_sink_factory {
    ($task:ident) => {
        struct $task<'a, D: SinkDefinition> {
            args: D::Args,
            inputs: BoundInputs,
            formats: &'a ResolvedFormats,
        }

        impl<'a, D: SinkDefinition> BuildTask for $task<'a, D> {
            type Output = Result<BuiltSink, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.args, self.inputs, self.formats).map(<D::Side as StageSide>::sink)
            }
        }

        impl<D: SinkDefinition> PreparedSink for Prepared<D> {
            fn parallelism(&self) -> Parallelism {
                D::parallelism(&self.args)
            }

            fn chunk_authority(&self) -> Option<usize> {
                self.inputs.chunk_authority()
            }

            fn accepts_input(
                &self,
                capabilities: &Capabilities,
                index: usize,
                offered: StreamFormat,
                inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                D::accepts_input(capabilities, &self.args, index, offered, inputs)
            }

            fn residency(&self) -> Residency {
                <D::Side as StageSide>::RESIDENCY
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<BuiltSink, BuildStageError> {
                let identifier = self.identifier;
                let task = $task::<D> {
                    args: self.args,
                    inputs: self.inputs,
                    formats,
                };

                backend
                    .build(task)
                    .map_err(|source| BuildStageError::Build { identifier, source })
            }
        }

        impl<D: SinkDefinition> StageFactory<dyn PreparedSink> for Definition<D> {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn prepare(
                &self,
                args: &Value,
                inputs: &[InputArgument],
                slot: &OpenSlot,
            ) -> Result<Box<dyn PreparedSink>, BuildStageError> {
                let args = self.deserialize_args(args)?;
                let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

                Ok(Box::new(Prepared::<D> {
                    identifier: self.full_identifier().to_string(),
                    args,
                    inputs: bound,
                    slot: slot.clone(),
                }))
            }
        }
    };
}

impl_source_factory!(BuildSource);
impl_stage_factory!(
    Filter,
    FilterDefinition,
    PreparedFilter,
    BuildFilter,
    BuiltFilter,
    filter
);
impl_stage_factory!(Mask, MaskDefinition, PreparedMask, BuildMask, BuiltMask, mask);
impl_stage_factory!(
    Transform,
    TransformDefinition,
    PreparedTransform,
    BuildTransform,
    BuiltTransform,
    transform
);
impl_sink_factory!(BuildSink);

struct BuildMetaTransform<'a, D: MetaTransformDefinition> {
    args: D::Args,
    inputs: BoundInputs,
    formats: &'a ResolvedFormats,
}

impl<'a, D: MetaTransformDefinition> BuildTask for BuildMetaTransform<'a, D> {
    type Output = Result<Box<dyn MetaTransform>, StageError>;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        D::build::<R>(context, self.args, self.inputs, self.formats)
    }
}

impl<D: MetaTransformDefinition> PreparedMetaTransform for Prepared<D> {
    fn parallelism(&self) -> Parallelism {
        D::parallelism(&self.args)
    }

    fn chunk_authority(&self) -> Option<usize> {
        self.inputs.chunk_authority()
    }

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        D::accepts_input(capabilities, &self.args, index, offered, inputs)
    }

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        D::produces(capabilities, &self.args, chosen)
    }

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaTransform>, BuildStageError> {
        let identifier = self.identifier;
        let task = BuildMetaTransform::<D> {
            args: self.args,
            inputs: self.inputs,
            formats,
        };

        backend
            .build(task)
            .map_err(|source| BuildStageError::Build { identifier, source })
    }
}

impl<D: MetaTransformDefinition> StageFactory<dyn PreparedMetaTransform> for Definition<D> {
    fn info(&self) -> StageInfo<'_> {
        self.stage_info()
    }

    fn prepare(
        &self,
        args: &Value,
        inputs: &[InputArgument],
        slot: &OpenSlot,
    ) -> Result<Box<dyn PreparedMetaTransform>, BuildStageError> {
        let args = self.deserialize_args(args)?;
        let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

        Ok(Box::new(Prepared::<D> {
            identifier: self.full_identifier().to_string(),
            args,
            inputs: bound,
            slot: slot.clone(),
        }))
    }
}
