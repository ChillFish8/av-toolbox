pub(crate) mod args;
mod detect;
#[cfg(test)]
mod tests;
pub(crate) mod window;

use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    StageDefinition,
    StageError,
    StreamType,
    Transform,
    TransformDefinition,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    HostSide,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};

use self::args::DetectArgs;
use self::detect::Detect;
use self::window::check_layout;

/// Detects scene changes in a stream and divides it into chunks at them.
pub(crate) struct SceneDetect;

impl StageDefinition for SceneDetect {
    type Args = DetectArgs;

    fn package() -> &'static str {
        "scene"
    }

    fn identifier() -> &'static str {
        "detect"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl TransformDefinition for SceneDetect {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(format) = offered.as_frames() else {
            return Err(StageError::msg("scene/detect reads frames, not masks"));
        };

        check_layout(format)?;

        let accepted = match format.sample {
            SampleFormat::U8 | SampleFormat::U16 { .. } => FormatSet::of([format.sample]),
            SampleFormat::F16 | SampleFormat::F32 => FormatSet::new(),
        };

        Ok(accepted)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied();
        let format = format.ok_or_else(|| StageError::msg("scene/detect needs its input bound"))?;

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Transform<HostSide>>, StageError> {
        let input = formats
            .input(0)
            .ok_or_else(|| StageError::msg("scene/detect was built without its input's format"))?;
        let format = input
            .as_frames()
            .ok_or_else(|| StageError::msg("scene/detect reads frames, not masks"))?;

        let detect = Detect::new(&args, format)?;

        Ok(Box::new(detect))
    }
}
