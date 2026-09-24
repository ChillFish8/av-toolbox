use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    MetaInputs,
    MetaItem,
    MetaTransform,
    MetaTransformDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
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
    StreamTemplate,
};

use super::limit::SplitLimit;
use super::scenes::Scenes;

/// Splits the chunks of a scored stream so none is longer than a limit.
pub(crate) struct SceneSplit;

impl StageDefinition for SceneSplit {
    type Args = SplitLimit;

    fn package() -> &'static str {
        "scene"
    }

    fn identifier() -> &'static str {
        "split"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MetaTransformDefinition for SceneSplit {
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
        let format = chosen.iter().flatten().next().copied();
        let format = format.ok_or_else(|| StageError::msg("scene/split needs its input bound"))?;

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
    ) -> Result<Box<dyn MetaTransform>, StageError> {
        let split = Split::new(args);

        Ok(Box::new(split))
    }
}

/// Holds each incoming chunk as a scene and splits it to the limit.
pub(crate) struct Split {
    scenes: Scenes<MetaItem>,
    open_chunk: Option<u64>,
}

impl Split {
    pub(crate) fn new(limit: SplitLimit) -> Self {
        Self {
            scenes: Scenes::new(limit),
            open_chunk: None,
        }
    }
}

impl StageInstance for Split {}

impl MetaTransform for Split {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: MetaInputs,
    ) -> Result<Vec<MetaItem>, StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("scene/split expected an item in slot 0"))?;

        let chunk = item.meta().chunk;
        let starts_scene = self.open_chunk != Some(chunk);
        self.open_chunk = Some(chunk);

        self.scenes.push(item, starts_scene)
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<MetaItem>, StageError> {
        self.scenes.finish()
    }
}
