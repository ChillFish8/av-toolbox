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
};
use jsonschema::Validator;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    ArgsError,
    BindInputsError,
    BoundInputs,
    BuildStageError,
    Filter,
    FilterDefinition,
    InputArgument,
    InputSignature,
    Mask,
    MaskDefinition,
    RegistryError,
    SchemaViolation,
    Sink,
    SinkDefinition,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageKind,
    Tag,
    Transform,
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
    fn produces(&self, capabilities: &Capabilities) -> Result<OutputFormats, StageError>;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, BuildStageError>;
}

/// A filter whose arguments are deserialized and whose inputs are bound.
pub trait PreparedFilter: Send + Sync {
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

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, BuildStageError>;
}

/// A mask whose arguments are deserialized and whose inputs are bound.
pub trait PreparedMask: Send + Sync {
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

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, BuildStageError>;
}

/// A transform whose arguments are deserialized and whose inputs are bound.
pub trait PreparedTransform: Send + Sync {
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

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Transform>, BuildStageError>;
}

/// A sink whose arguments are deserialized and whose inputs are bound.
pub trait PreparedSink: Send + Sync {
    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, BuildStageError>;
}

/// A prepared stage of any kind.
pub enum PreparedStage {
    Source(Box<dyn PreparedSource>),
    Filter(Box<dyn PreparedFilter>),
    Mask(Box<dyn PreparedMask>),
    Transform(Box<dyn PreparedTransform>),
    Sink(Box<dyn PreparedSink>),
}

impl PreparedStage {
    pub fn kind(&self) -> StageKind {
        match self {
            Self::Source(_) => StageKind::Source,
            Self::Filter(_) => StageKind::Filter,
            Self::Mask(_) => StageKind::Mask,
            Self::Transform(_) => StageKind::Transform,
            Self::Sink(_) => StageKind::Sink,
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
            Self::Sink(prepared) => prepared.accepts_input(capabilities, index, offered, inputs),
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
            Self::Sink(_) => Err(StageError::msg(
                "produces was asked of a sink, which produces no stream, so reaching this is a solver bug",
            )),
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
    ($kind_trait:ident, $definition_trait:ident, $prepared_trait:ident, $task:ident) => {
        /// Carries a stage's pieces into whichever runtime the backend opened.
        struct $task<'a, D: $definition_trait> {
            args: D::Args,
            inputs: BoundInputs,
            formats: &'a ResolvedFormats,
        }

        impl<'a, D: $definition_trait> BuildTask for $task<'a, D> {
            type Output = Result<Box<dyn $kind_trait>, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.args, self.inputs, self.formats)
            }
        }

        impl<D: $definition_trait> $prepared_trait for Prepared<D> {
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
            ) -> Result<Box<dyn $kind_trait>, BuildStageError> {
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
            type Output = Result<Box<dyn Source>, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.args, self.inputs, self.formats)
            }
        }

        impl<D: SourceDefinition> PreparedSource for Prepared<D> {
            fn produces(&self, capabilities: &Capabilities) -> Result<OutputFormats, StageError> {
                D::produces(capabilities, &self.args)
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<Box<dyn Source>, BuildStageError> {
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
            type Output = Result<Box<dyn Sink>, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.args, self.inputs, self.formats)
            }
        }

        impl<D: SinkDefinition> PreparedSink for Prepared<D> {
            fn accepts_input(
                &self,
                capabilities: &Capabilities,
                index: usize,
                offered: StreamFormat,
                inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                D::accepts_input(capabilities, &self.args, index, offered, inputs)
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<Box<dyn Sink>, BuildStageError> {
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
impl_stage_factory!(Filter, FilterDefinition, PreparedFilter, BuildFilter);
impl_stage_factory!(Mask, MaskDefinition, PreparedMask, BuildMask);
impl_stage_factory!(Transform, TransformDefinition, PreparedTransform, BuildTransform);
impl_sink_factory!(BuildSink);

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use av_toolbox_gpu::{
        Capabilities,
        FormatSet,
        Frame,
        FrameContext,
        Offered,
        OutputFormats,
        StreamFormat,
        StreamTemplate,
    };
    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    use crate::{InputParam, Inputs, StageHandle, StageInstance, StreamType};

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
}
