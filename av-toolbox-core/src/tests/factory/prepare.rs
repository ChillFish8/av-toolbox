use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    HostFrame,
    HostSide,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};

use super::{ExampleArgs, ExampleDefinition, definition, reference_input};
use crate::factory::Definition;
use crate::{
    ArgsError,
    BoundInputs,
    BuildStageError,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    OpenSlot,
    PreparedFilter,
    PreparedStage,
    RegistryError,
    Residency,
    StageDefinition,
    StageError,
    StageFactory,
    StageInstance,
    StageKind,
    Tag,
};

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

    let result = factory.prepare(
        &serde_json::json!({ "sigma_scale": "wide" }),
        &reference_input(),
        &OpenSlot::new(),
    );
    let Err(error) = result else {
        panic!("the stage should not have been prepared");
    };

    assert!(matches!(error, BuildStageError::InvalidArgs { .. }));
}

#[test]
fn arguments_that_break_the_schema_are_rejected_before_deserialization() {
    let definition = definition();
    let args = serde_json::json!({ "sigma_scale": "not a number" });

    let result =
        StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input(), &OpenSlot::new());
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

    let result = StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &[], &OpenSlot::new());
    let Err(error) = result else {
        panic!("the stage should not have been prepared");
    };

    assert!(error.to_string().contains("av-denoise/nl4d"));
    assert!(matches!(error, BuildStageError::InvalidInputs { .. }));
}

#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod device_tests {
    use std::sync::atomic::Ordering;

    use av_toolbox_gpu::{Backend, test_device};

    use super::*;
    use crate::tests::factory::RECORDED_SIGMA_SCALE;

    fn backend() -> Backend {
        Backend::open_default(test_device()).expect("should open")
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

        let prepared = StageFactory::<dyn PreparedFilter>::prepare(
            &definition,
            &args,
            &reference_input(),
            &OpenSlot::new(),
        )
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

        let prepared = StageFactory::<dyn PreparedFilter>::prepare(
            &definition,
            &args,
            &reference_input(),
            &OpenSlot::new(),
        )
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
    ) -> Result<Vec<HostFrame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
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

    fn description() -> &'static str {
        "A stage used by tests."
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

    let prepared =
        StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input(), &OpenSlot::new())
            .expect("should prepare");

    assert_eq!(prepared.residency(), Residency::Host);
    assert_eq!(PreparedStage::Filter(prepared).residency(), Residency::Host);
}

#[test]
fn a_definition_on_the_device_side_prepares_as_a_device_stage() {
    let args = serde_json::json!({ "sigma_scale": 1.0 });

    let definition = definition();
    let prepared =
        StageFactory::<dyn PreparedFilter>::prepare(&definition, &args, &reference_input(), &OpenSlot::new())
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

        fn description() -> &'static str {
            "A stage used by tests."
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
