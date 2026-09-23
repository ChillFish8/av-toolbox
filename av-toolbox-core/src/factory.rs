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
    fn prepare(&self, args: &Value, inputs: &[InputArgument]) -> Result<Box<P>, BuildStageError>;
}

/// A source whose arguments are deserialized, ready to negotiate and build.
pub trait PreparedSource: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    fn produces(&self, capabilities: &Capabilities) -> Result<OutputFormats, StageError>;

    /// Which side this stage runs on.
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

    /// Which side this stage runs on.
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

    /// Which side this stage runs on.
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

    /// Which side this stage runs on.
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

    /// Which sample formats this stage will take on one input, delegating to the definition's own
    /// answer.
    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    /// What it could produce, given what it will be given, delegating to the definition's own
    /// answer.
    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    /// Builds the instance against a real backend, without naming a side, since a metadata-only
    /// transform has none of its own.
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

    /// Which side this stage runs on.
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
    /// A crossing the graph inserted to move a stream between host and device.
    ///
    /// No definition ever produces one; it is built straight from a [Crossing] rather than through
    /// a [StageFactory], since it has no arguments to validate and no inputs to bind beyond the one
    /// stream it moves.
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
    /// A metadata-only transform is the one stage that reports [Residency::Either], since it never
    /// touches a plane and so can run on whichever side its input already sits on. A crossing
    /// reports the side it leads to, since that is the side its output has to be placed on.
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

            /// Which side this stage runs on.
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
            ) -> Result<Box<dyn $prepared_trait>, BuildStageError> {
                let args = self.deserialize_args(args)?;
                let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

                Ok(Box::new(Prepared::<D> {
                    identifier: self.full_identifier().to_string(),
                    args,
                    inputs: bound,
                }))
            }
        }
    };
}

/// Writes a source's build task and factory, since a source has no `accepts_input` to negotiate.
macro_rules! impl_source_factory {
    ($task:ident) => {
        struct $task<'a, D: SourceDefinition> {
            args: D::Args,
            inputs: BoundInputs,
            formats: &'a ResolvedFormats,
        }

        impl<'a, D: SourceDefinition> BuildTask for $task<'a, D> {
            type Output = Result<BuiltSource, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.args, self.inputs, self.formats)
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
                D::produces(capabilities, &self.args)
            }

            /// Which side this stage runs on.
            fn residency(&self) -> Residency {
                <D::Side as StageSide>::RESIDENCY
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<BuiltSource, BuildStageError> {
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

        impl<D: SourceDefinition> StageFactory<dyn PreparedSource> for Definition<D> {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn prepare(
                &self,
                args: &Value,
                inputs: &[InputArgument],
            ) -> Result<Box<dyn PreparedSource>, BuildStageError> {
                let args = self.deserialize_args(args)?;
                let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

                Ok(Box::new(Prepared::<D> {
                    identifier: self.full_identifier().to_string(),
                    args,
                    inputs: bound,
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

            /// Which side this stage runs on.
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
            ) -> Result<Box<dyn PreparedSink>, BuildStageError> {
                let args = self.deserialize_args(args)?;
                let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

                Ok(Box::new(Prepared::<D> {
                    identifier: self.full_identifier().to_string(),
                    args,
                    inputs: bound,
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
    ) -> Result<Box<dyn PreparedMetaTransform>, BuildStageError> {
        let args = self.deserialize_args(args)?;
        let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

        Ok(Box::new(Prepared::<D> {
            identifier: self.full_identifier().to_string(),
            args,
            inputs: bound,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use av_toolbox_gpu::{
        Capabilities,
        DeviceSide,
        FormatSet,
        Frame,
        FrameContext,
        HostFrame,
        HostSide,
        Offered,
        OutputFormats,
        StreamFormat,
        StreamTemplate,
    };
    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    use crate::{Crossing, Filter, InputParam, Inputs, StageHandle, StageInstance, StreamType};

    #[derive(Debug, Deserialize, JsonSchema)]
    struct ExampleArgs {
        sigma_scale: f32,
    }

    struct ExampleFilter;

    impl StageInstance for ExampleFilter {}

    impl Filter for ExampleFilter {
        fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
            inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
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

        fn tags() -> &'static [Tag] {
            &[Tag::Denoise]
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
            if args.sigma_scale < 0.0 {
                return Err(StageError::msg("sigma_scale must not be negative"));
            }

            RECORDED_SIGMA_SCALE.store(args.sigma_scale.to_bits(), Ordering::SeqCst);

            Ok(Box::new(ExampleFilter))
        }
    }

    fn definition() -> Definition<ExampleDefinition> {
        Definition::new(StageKind::Filter).expect("should build the definition")
    }

    fn reference_input() -> Vec<InputArgument> {
        vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)]
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct TemporalArgs {
        radius: u32,
    }

    struct TemporalExample;

    impl StageDefinition for TemporalExample {
        type Args = TemporalArgs;

        fn package() -> &'static str {
            "av-denoise"
        }

        fn identifier() -> &'static str {
            "temporal"
        }

        fn parallelism(args: &Self::Args) -> Parallelism {
            if args.radius == 0 {
                Parallelism::Frame
            } else {
                Parallelism::Chunk
            }
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
            SIGNATURE
        }
    }

    impl FilterDefinition for TemporalExample {
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
            Ok(Box::new(ExampleFilter))
        }
    }

    #[test]
    fn a_prepared_stage_reports_the_granularity_its_arguments_asked_for() {
        let definition = Definition::<TemporalExample>::new(StageKind::Filter).expect("should build");

        let wide = StageFactory::<dyn PreparedFilter>::prepare(
            &definition,
            &serde_json::json!({ "radius": 0 }),
            &reference_input(),
        )
        .expect("should prepare");

        let narrow = StageFactory::<dyn PreparedFilter>::prepare(
            &definition,
            &serde_json::json!({ "radius": 4 }),
            &reference_input(),
        )
        .expect("should prepare");

        assert_eq!(wide.parallelism(), Parallelism::Frame);
        assert_eq!(narrow.parallelism(), Parallelism::Chunk);

        let erased = PreparedStage::Filter(narrow);
        assert_eq!(erased.parallelism(), Parallelism::Chunk);
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct ChunkAuthorityArgs {}

    struct ChunkAuthorityExample;

    impl StageDefinition for ChunkAuthorityExample {
        type Args = ChunkAuthorityArgs;

        fn package() -> &'static str {
            "av-denoise"
        }

        fn identifier() -> &'static str {
            "chunk-authority"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[
                InputParam::required("primary"),
                InputParam::required("secondary").chunk_authority(),
            ]);
            SIGNATURE
        }
    }

    impl FilterDefinition for ChunkAuthorityExample {
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
            Ok(Box::new(ExampleFilter))
        }
    }

    fn two_inputs() -> Vec<InputArgument> {
        vec![
            InputArgument::positional(StageHandle::new(1), StreamType::Frames),
            InputArgument::positional(StageHandle::new(2), StreamType::Frames),
        ]
    }

    #[test]
    fn chunk_authority_survives_type_erasure_when_a_later_input_is_marked() {
        let definition = Definition::<ChunkAuthorityExample>::new(StageKind::Filter).expect("should build");

        let prepared: Box<dyn PreparedFilter> =
            StageFactory::<dyn PreparedFilter>::prepare(&definition, &serde_json::json!({}), &two_inputs())
                .expect("should prepare");

        assert_eq!(prepared.chunk_authority(), Some(1));
    }

    #[test]
    fn info_reports_the_generated_schema_and_signature() {
        let definition = definition();
        let info = StageFactory::<dyn PreparedFilter>::info(&definition);

        assert_eq!(info.kind(), StageKind::Filter);
        assert_eq!(info.full_identifier(), "av-denoise/nl4d");
        assert_eq!(info.tags(), &[Tag::Denoise]);
        assert_eq!(info.schema()["type"], "object");
        assert_eq!(info.inputs().params().len(), 1);
    }

    #[test]
    fn arguments_are_rejected_without_a_backend() {
        let definition = Definition::<ExampleDefinition>::new(StageKind::Filter).expect("should build");
        let factory: &dyn StageFactory<dyn PreparedFilter> = &definition;

        let result = factory.prepare(&serde_json::json!({ "sigma_scale": "wide" }), &reference_input());
        let Err(error) = result else {
            panic!("the stage should not have been prepared");
        };

        assert!(matches!(error, BuildStageError::InvalidArgs { .. }));
    }

    #[test]
    fn arguments_that_break_the_schema_are_rejected_before_deserialization() {
        let definition = definition();
        let args = serde_json::json!({ "sigma_scale": "not a number" });

        let result = StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input());
        let Err(error) = result else {
            panic!("the stage should not have been prepared");
        };

        assert!(error.to_string().contains("av-denoise/nl4d"));

        let BuildStageError::InvalidArgs { source, .. } = error else {
            panic!("expected invalid arguments, got {error:?}");
        };

        let ArgsError::SchemaViolations(violations) = source else {
            panic!("expected schema violations");
        };

        assert_eq!(violations[0].path(), "/sigma_scale");
    }

    #[test]
    fn inputs_that_break_the_signature_are_rejected() {
        let definition = definition();
        let args = serde_json::json!({ "sigma_scale": 0.8 });

        let result = StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &[]);
        let Err(error) = result else {
            panic!("the stage should not have been prepared");
        };

        assert!(error.to_string().contains("av-denoise/nl4d"));
        assert!(matches!(error, BuildStageError::InvalidInputs { .. }));
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    mod device_tests {
        use av_toolbox_gpu::Device;

        use super::*;

        fn backend() -> Backend {
            Backend::open_default(Device::Default).expect("should open")
        }

        fn formats() -> ResolvedFormats {
            ResolvedFormats::new(vec![], None)
        }

        #[test]
        fn valid_arguments_build_a_boxed_filter() {
            let definition = definition();
            let args = serde_json::json!({ "sigma_scale": 0.8 });
            let backend = backend();
            let formats = formats();

            let prepared =
                StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input())
                    .expect("should prepare");
            let filter = prepared.build(&backend, &formats).expect("should build");
            let filter = filter.into_device().expect("a device filter");
            drop(filter);

            let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
            assert_eq!(recorded_sigma_scale, 0.8);
        }

        #[test]
        fn a_stage_failure_is_wrapped_with_its_identifier() {
            let definition = definition();
            let args = serde_json::json!({ "sigma_scale": -1.0 });
            let backend = backend();
            let formats = formats();

            let prepared =
                StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input())
                    .expect("should prepare");
            let result = prepared.build(&backend, &formats);
            let Err(error) = result else {
                panic!("the stage should not have been built");
            };

            assert!(matches!(error, BuildStageError::Build { .. }));
            assert!(error.to_string().contains("av-denoise/nl4d"));

            let source = std::error::Error::source(&error).expect("should carry the stage's error");
            assert!(source.to_string().contains("sigma_scale must not be negative"));
        }
    }

    struct HostExample;

    impl StageInstance for HostExample {}

    impl Filter<HostSide> for HostExample {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs<HostSide>,
        ) -> Result<HostFrame, StageError> {
            inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
        }
    }

    struct HostDefinition;

    impl StageDefinition for HostDefinition {
        type Args = ExampleArgs;

        fn package() -> &'static str {
            "av-denoise"
        }

        fn identifier() -> &'static str {
            "host"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
            SIGNATURE
        }
    }

    impl FilterDefinition for HostDefinition {
        type Side = HostSide;

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
        ) -> Result<Box<dyn Filter<HostSide>>, StageError> {
            Ok(Box::new(HostExample))
        }
    }

    #[test]
    fn a_host_definition_prepares_as_a_host_stage() {
        let definition = Definition::<HostDefinition>::new(StageKind::Filter).expect("should build");
        let args = serde_json::json!({ "sigma_scale": 1.0 });

        let prepared = StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input())
            .expect("should prepare");

        assert_eq!(prepared.residency(), Residency::Host);
        assert_eq!(PreparedStage::Filter(prepared).residency(), Residency::Host);
    }

    #[test]
    fn a_definition_on_the_device_side_prepares_as_a_device_stage() {
        let args = serde_json::json!({ "sigma_scale": 1.0 });

        let prepared = StageFactory::<dyn PreparedFilter>::prepare(&definition(), &args, &reference_input())
            .expect("should prepare");

        assert_eq!(prepared.residency(), Residency::Device);
    }

    #[test]
    fn an_identifier_segment_containing_a_slash_is_rejected() {
        struct BadDefinition;

        impl StageDefinition for BadDefinition {
            type Args = ExampleArgs;

            fn package() -> &'static str {
                "av/denoise"
            }

            fn identifier() -> &'static str {
                "nl4d"
            }

            fn inputs() -> &'static InputSignature {
                &InputSignature::EMPTY
            }
        }

        let result = Definition::<BadDefinition>::new(StageKind::Filter);
        let Err(error) = result else {
            panic!("the definition should not have been built");
        };

        assert!(matches!(
            error,
            RegistryError::InvalidIdentifier {
                segment: "package",
                ..
            },
        ));
    }

    /// A capability table that can do anything, matching the one `av-toolbox-graph`'s solver tests
    /// build, so a crossing's `produces` has something real to be asked against.
    fn capabilities() -> Capabilities {
        let every = [
            av_toolbox_gpu::ElementType::F32,
            av_toolbox_gpu::ElementType::F16,
            av_toolbox_gpu::ElementType::BF16,
            av_toolbox_gpu::ElementType::U8,
            av_toolbox_gpu::ElementType::U16,
            av_toolbox_gpu::ElementType::U32,
            av_toolbox_gpu::ElementType::I8,
            av_toolbox_gpu::ElementType::I16,
            av_toolbox_gpu::ElementType::I32,
        ];

        Capabilities::builder(av_toolbox_gpu::default_accelerators()[0])
            .storable(every)
            .computable(every)
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build()
    }

    #[test]
    fn a_crossing_passes_its_one_input_through_and_runs_frame_by_frame() {
        let crossing = PreparedStage::Crossing(Crossing::Unload);
        let format = StreamFormat::Mask(av_toolbox_gpu::MaskFormat {
            width: 8,
            height: 8,
            sample: av_toolbox_gpu::SampleFormat::U8,
        });
        let capabilities = capabilities();

        let produced = crossing
            .produces(&capabilities, &[Some(format)])
            .expect("passes through");

        assert_eq!(produced.template, StreamTemplate::from_format(&format));
        assert_eq!(crossing.parallelism(), Parallelism::Frame);
        assert_eq!(crossing.chunk_authority(), Some(0));
        assert_eq!(crossing.residency(), Residency::Host);
        assert_eq!(crossing.kind(), StageKind::Transform);
    }
}
