use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::ExampleFilter;
use crate::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    PipelineRegistry,
    RegistryError,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};

#[derive(Debug, Deserialize, JsonSchema)]
struct PlanesOfArgs {}

struct PlanesOfMask;

impl StageInstance for PlanesOfMask {}

impl Mask for PlanesOfMask {
    fn select(
        &mut self,
        _context: &mut FrameContext,
        _inputs: Inputs,
    ) -> Result<av_toolbox_gpu::Mask, StageError> {
        Err(StageError::msg("not built in this test"))
    }
}

fn planes_of_mask_accepts_input(
    _capabilities: &Capabilities,
    _args: &PlanesOfArgs,
    _index: usize,
    offered: StreamFormat,
    _inputs: &[Offered],
) -> Result<FormatSet, StageError> {
    Ok(FormatSet::of([offered.sample()]))
}

fn planes_of_mask_produces(
    _capabilities: &Capabilities,
    _args: &PlanesOfArgs,
    chosen: &[Option<StreamFormat>],
) -> Result<OutputFormats, StageError> {
    let format = chosen.iter().flatten().next().copied().expect("an input");
    Ok(OutputFormats {
        template: StreamTemplate::from_format(&format),
        samples: FormatSet::of([format.sample()]),
    })
}

fn planes_of_mask_build(
    _context: &BuildContext,
    _args: PlanesOfArgs,
    _inputs: BoundInputs,
    _formats: &ResolvedFormats,
) -> Result<Box<dyn Mask>, StageError> {
    Ok(Box::new(PlanesOfMask))
}

struct ValidPlanesOf;

impl StageDefinition for ValidPlanesOf {
    type Args = PlanesOfArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "valid-planes-of"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base"),
            InputParam::required("mask")
                .accepting(StreamType::Mask)
                .planes_of("base"),
        ]);
        SIGNATURE
    }
}

impl MaskDefinition for ValidPlanesOf {
    type Side = DeviceSide;

    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        planes_of_mask_accepts_input(capabilities, args, index, offered, inputs)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        planes_of_mask_produces(capabilities, args, chosen)
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        planes_of_mask_build(context, args, inputs, formats)
    }
}

struct MaskNamingAMissingInput;

impl StageDefinition for MaskNamingAMissingInput {
    type Args = PlanesOfArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "missing-planes-of"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("mask")
            .accepting(StreamType::Mask)
            .planes_of("missing")]);
        SIGNATURE
    }
}

impl MaskDefinition for MaskNamingAMissingInput {
    type Side = DeviceSide;

    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        planes_of_mask_accepts_input(capabilities, args, index, offered, inputs)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        planes_of_mask_produces(capabilities, args, chosen)
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        planes_of_mask_build(context, args, inputs, formats)
    }
}

struct PlanesOfOnAFrameInput;

impl StageDefinition for PlanesOfOnAFrameInput {
    type Args = PlanesOfArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "planes-of-on-a-frame-input"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base"),
            InputParam::required("overlay").planes_of("base"),
        ]);
        SIGNATURE
    }
}

impl MaskDefinition for PlanesOfOnAFrameInput {
    type Side = DeviceSide;

    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        planes_of_mask_accepts_input(capabilities, args, index, offered, inputs)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        planes_of_mask_produces(capabilities, args, chosen)
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        planes_of_mask_build(context, args, inputs, formats)
    }
}

struct FilterNamingAMissingInput;

impl StageDefinition for FilterNamingAMissingInput {
    type Args = PlanesOfArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "filter-missing-planes-of"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("mask")
            .accepting(StreamType::Mask)
            .planes_of("missing")]);
        SIGNATURE
    }
}

impl FilterDefinition for FilterNamingAMissingInput {
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

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(ExampleFilter))
    }
}

#[test]
fn a_mask_input_with_a_valid_planes_of_registers() {
    let mut registry = PipelineRegistry::new();

    let result = registry.register_mask::<ValidPlanesOf>();

    assert!(result.is_ok());
}

#[test]
fn a_planes_of_naming_a_missing_input_is_rejected() {
    let mut registry = PipelineRegistry::new();

    let error = registry
        .register_mask::<MaskNamingAMissingInput>()
        .expect_err("should not register");

    assert!(matches!(
        error,
        RegistryError::InvalidPlanesOf {
            param: "mask",
            reason: "it names no input of this stage",
            ..
        }
    ));
}

#[test]
fn a_planes_of_on_an_input_that_does_not_accept_masks_is_rejected() {
    let mut registry = PipelineRegistry::new();

    let error = registry
        .register_mask::<PlanesOfOnAFrameInput>()
        .expect_err("should not register");

    assert!(matches!(
        error,
        RegistryError::InvalidPlanesOf {
            param: "overlay",
            reason: "it is only allowed on an input that takes masks",
            ..
        }
    ));
}

#[test]
fn the_planes_of_check_applies_to_every_kind() {
    let mut registry = PipelineRegistry::new();

    let error = registry
        .register_filter::<FilterNamingAMissingInput>()
        .expect_err("should not register");

    assert!(matches!(
        error,
        RegistryError::InvalidPlanesOf {
            param: "mask",
            reason: "it names no input of this stage",
            ..
        }
    ));
}
