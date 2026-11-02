use av_toolbox_core::{
    ArtefactDecl,
    ArtefactSink,
    ArtefactSinkDefinition,
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    KeyFilter,
    MergedArtefact,
    Pattern,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::shared::{NoArgs, first_bound};

struct GrainStage;

impl StageInstance for GrainStage {}

impl Filter for GrainStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// A pass through filter declaring an opaque `table` and a structured `cuts`.
pub(super) struct Grain;

impl StageDefinition for Grain {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "grain"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn artefacts() -> &'static [ArtefactDecl] {
        const DECLARED: &[ArtefactDecl] = &[
            ArtefactDecl::opaque("table", "tbl", "A grain table."),
            ArtefactDecl::structured("cuts", "Where each scene starts."),
        ];
        DECLARED
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl FilterDefinition for Grain {
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
        let format = first_bound(chosen)?;

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
        Ok(Box::new(GrainStage))
    }
}

/// A pass through filter declaring a structured `shots`.
pub(super) struct Scenes;

impl StageDefinition for Scenes {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "scenes"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn artefacts() -> &'static [ArtefactDecl] {
        const DECLARED: &[ArtefactDecl] = &[ArtefactDecl::structured("shots", "Where each shot starts.")];
        DECLARED
    }

    fn inputs() -> &'static InputSignature {
        Grain::inputs()
    }
}

impl FilterDefinition for Scenes {
    type Side = DeviceSide;

    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Grain::accepts_input(capabilities, args, index, offered, inputs)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        Grain::produces(capabilities, args, chosen)
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Grain::build(context, args, inputs, formats)
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(super) struct FilesArgs {
    include: Option<Vec<Pattern>>,
    exclude: Option<Vec<Pattern>>,
}

struct FilesStage;

impl ArtefactSink for FilesStage {
    fn write(&mut self, artefacts: &[&MergedArtefact]) -> Result<usize, StageError> {
        Ok(artefacts.len())
    }
}

/// An artefact sink selecting by its `include` and `exclude`.
pub(super) struct Files;

impl StageDefinition for Files {
    type Args = FilesArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "files"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::ARTEFACTS
    }
}

impl ArtefactSinkDefinition for Files {
    fn selection(args: &Self::Args) -> KeyFilter {
        let include = args.include.clone();
        let exclude = args.exclude.clone();

        KeyFilter::new(include, exclude)
    }

    fn build(_args: Self::Args) -> Result<Box<dyn ArtefactSink>, StageError> {
        Ok(Box::new(FilesStage))
    }
}
