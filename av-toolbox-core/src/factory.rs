use std::marker::PhantomData;

use jsonschema::Validator;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    ArgsError,
    BindInputsError,
    BuildStageError,
    Filter,
    InputArgument,
    InputSignature,
    Mask,
    RegistryError,
    SchemaViolation,
    Sink,
    Source,
    StageDefinition,
    StageKind,
    Tag,
    Transform,
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

    fn build(&self, args: &Value, inputs: &[InputArgument]) -> Result<Box<K>, BuildStageError>;
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

    /// Validates, deserializes and binds, then hands the pieces to the stage.
    ///
    /// The order matters. Schema validation runs before deserialization, so the arguments are
    /// reported against the schema the stage published.
    fn build_instance(&self, args: &Value, inputs: &[InputArgument]) -> Result<D::Instance, BuildStageError> {
        let args = self.deserialize_args(args)?;

        let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

        D::build(args, bound).map_err(|source| BuildStageError::Build {
            identifier: self.full_identifier.clone(),
            source,
        })
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
    ($kind_trait:ident) => {
        impl<D> StageFactory<dyn $kind_trait> for Definition<D>
        where
            D: StageDefinition,
            D::Instance: $kind_trait,
        {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn build(
                &self,
                args: &Value,
                inputs: &[InputArgument],
            ) -> Result<Box<dyn $kind_trait>, BuildStageError> {
                let instance = self.build_instance(args, inputs)?;
                Ok(Box::new(instance))
            }
        }
    };
}

impl_stage_factory!(Source);
impl_stage_factory!(Filter);
impl_stage_factory!(Mask);
impl_stage_factory!(Transform);
impl_stage_factory!(Sink);

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    use crate::{BoundInputs, InputParam, StageError, StageHandle, StageInstance, StreamType};

    #[derive(Debug, Deserialize, JsonSchema)]
    struct ExampleArgs {
        sigma_scale: f32,
    }

    struct ExampleFilter {
        sigma_scale: f32,
    }

    impl StageInstance for ExampleFilter {}
    impl Filter for ExampleFilter {}

    static RECORDED_SIGMA_SCALE: AtomicU32 = AtomicU32::new(0);

    struct ExampleDefinition;

    impl StageDefinition for ExampleDefinition {
        type Args = ExampleArgs;
        type Instance = ExampleFilter;

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

        fn build(args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
            if args.sigma_scale < 0.0 {
                return Err(StageError::msg("sigma_scale must not be negative"));
            }

            let instance = ExampleFilter {
                sigma_scale: args.sigma_scale,
            };
            RECORDED_SIGMA_SCALE.store(instance.sigma_scale.to_bits(), Ordering::SeqCst);

            Ok(instance)
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
        let info = StageFactory::<dyn Filter>::info(&definition);

        assert_eq!(info.kind(), StageKind::Filter);
        assert_eq!(info.full_identifier(), "av-denoise/nl4d");
        assert_eq!(info.tags(), &[Tag::Denoise]);
        assert_eq!(info.schema()["type"], "object");
        assert_eq!(info.inputs().params().len(), 1);
    }

    #[test]
    fn valid_arguments_build_a_boxed_filter() {
        let definition = definition();
        let args = serde_json::json!({ "sigma_scale": 0.8 });

        let filter: Box<dyn Filter> = definition.build(&args, &reference_input()).expect("should build");

        let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
        assert_eq!(recorded_sigma_scale, 0.8);

        drop(filter);
    }

    #[test]
    fn arguments_that_break_the_schema_are_rejected_before_deserialization() {
        let definition = definition();
        let args = serde_json::json!({ "sigma_scale": "not a number" });

        let result = StageFactory::<dyn Filter>::build(&definition, &args, &reference_input());
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

        let result = StageFactory::<dyn Filter>::build(&definition, &args, &[]);
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

        let result = StageFactory::<dyn Filter>::build(&definition, &args, &reference_input());
        let Err(error) = result else {
            panic!("the stage should not have been built");
        };

        assert!(matches!(error, BuildStageError::Build { .. }));
        assert!(error.to_string().contains("av-denoise/nl4d"));

        let source = std::error::Error::source(&error).expect("should carry the stage's error");
        assert!(source.to_string().contains("sigma_scale must not be negative"));
    }

    #[test]
    fn an_identifier_segment_containing_a_slash_is_rejected() {
        struct BadDefinition;

        impl StageDefinition for BadDefinition {
            type Args = ExampleArgs;
            type Instance = ExampleFilter;

            fn package() -> &'static str {
                "av/denoise"
            }

            fn identifier() -> &'static str {
                "nl4d"
            }

            fn inputs() -> &'static InputSignature {
                &InputSignature::EMPTY
            }

            fn build(args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
                Ok(ExampleFilter {
                    sigma_scale: args.sigma_scale,
                })
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
