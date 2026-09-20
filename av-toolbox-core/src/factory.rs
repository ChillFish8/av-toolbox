use std::marker::PhantomData;

use av_toolbox_gpu::{Backend, BuildContext, BuildTask, ResolvedFormats, Runtime};
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

/// Builds stages of one kind without naming their concrete definition.
pub trait StageFactory<K: ?Sized>: Send + Sync {
    fn info(&self) -> StageInfo<'_>;

    fn build(
        &self,
        backend: &Backend,
        args: &Value,
        inputs: &[InputArgument],
        formats: &ResolvedFormats,
    ) -> Result<Box<K>, BuildStageError>;
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

macro_rules! impl_stage_factory {
    ($kind_trait:ident, $definition_trait:ident, $task:ident) => {
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

        impl<D: $definition_trait> StageFactory<dyn $kind_trait> for Definition<D> {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn build(
                &self,
                backend: &Backend,
                args: &Value,
                inputs: &[InputArgument],
                formats: &ResolvedFormats,
            ) -> Result<Box<dyn $kind_trait>, BuildStageError> {
                let args = self.deserialize_args(args)?;
                let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

                let task = $task::<D> {
                    args,
                    inputs: bound,
                    formats,
                };

                backend
                    .build(task)
                    .map_err(|source| BuildStageError::Build {
                        identifier: self.full_identifier().to_string(),
                        source,
                    })
            }
        }
    };
}

impl_stage_factory!(Source, SourceDefinition, BuildSource);
impl_stage_factory!(Filter, FilterDefinition, BuildFilter);
impl_stage_factory!(Mask, MaskDefinition, BuildMask);
impl_stage_factory!(Transform, TransformDefinition, BuildTransform);
impl_stage_factory!(Sink, SinkDefinition, BuildSink);

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use av_toolbox_gpu::{Capabilities, FormatSet, OutputFormats, StreamFormat, StreamTemplate};
    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    use crate::{InputParam, StageInstance};
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use crate::{StageHandle, StreamType};

    #[derive(Debug, Deserialize, JsonSchema)]
    struct ExampleArgs {
        sigma_scale: f32,
    }

    struct ExampleFilter;

    impl StageInstance for ExampleFilter {}
    impl Filter for ExampleFilter {}

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
        fn accepts(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            offered: &[StreamFormat],
        ) -> Result<Vec<FormatSet>, StageError> {
            Ok(offered
                .iter()
                .map(|format| FormatSet::of([format.sample()]))
                .collect())
        }

        fn produces(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            chosen: &[StreamFormat],
        ) -> Result<OutputFormats, StageError> {
            let format = chosen[0];
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

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    fn reference_input() -> Vec<InputArgument> {
        vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)]
    }

    #[test]
    fn info_reports_the_generated_schema_and_signature() {
        let definition = definition();
        let info = StageFactory::<dyn Filter>::info(&definition);

        assert_eq!(info.kind(), StageKind::Filter);
        assert_eq!(info.full_identifier(), "av-denoise/nl4d");
        assert_eq!(info.tags(), &[Tag::Denoise]);
        assert_eq!(info.schema()["type"], "object");
        assert_eq!(info.inputs().params().len(), 1);
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

            let filter: Box<dyn Filter> = definition
                .build(&backend, &args, &reference_input(), &formats)
                .expect("should build");
            drop(filter);

            let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
            assert_eq!(recorded_sigma_scale, 0.8);
        }

        #[test]
        fn arguments_that_break_the_schema_are_rejected_before_deserialization() {
            let definition = definition();
            let args = serde_json::json!({ "sigma_scale": "not a number" });
            let backend = backend();
            let formats = formats();

            let result =
                StageFactory::<dyn Filter>::build(&definition, &backend, &args, &reference_input(), &formats);
            let Err(error) = result else {
                panic!("the stage should not have been built");
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
            let backend = backend();
            let formats = formats();

            let result = StageFactory::<dyn Filter>::build(&definition, &backend, &args, &[], &formats);
            let Err(error) = result else {
                panic!("the stage should not have been built");
            };

            assert!(error.to_string().contains("av-denoise/nl4d"));
            assert!(matches!(error, BuildStageError::InvalidInputs { .. }));
        }

        #[test]
        fn a_stage_failure_is_wrapped_with_its_identifier() {
            let definition = definition();
            let args = serde_json::json!({ "sigma_scale": -1.0 });
            let backend = backend();
            let formats = formats();

            let result =
                StageFactory::<dyn Filter>::build(&definition, &backend, &args, &reference_input(), &formats);
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
