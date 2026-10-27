use av_toolbox_core::{InputParam, InputSignature, StageError, StreamType};
use av_toolbox_gpu::{FormatSet, FrameFormat, StreamFormat};

/// The one input of a `like` stage, the clip it follows.
pub(super) const REFERENCE_SIGNATURE: &InputSignature =
    &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);

fn not_frames(stage: &str) -> StageError {
    let message = format!("{stage} follows frames, not a mask");

    StageError::msg(message)
}

fn missing_reference(stage: &str) -> StageError {
    let message = format!("{stage} needs its reference bound");

    StageError::msg(message)
}

/// Which samples a `like` stage takes from its reference.
pub(super) fn accepts_reference(offered: StreamFormat, stage: &str) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames(stage));
    };

    Ok(FormatSet::of([format.sample]))
}

/// The reference's format, once it is known to be frames.
pub(super) fn reference_format(
    chosen: &[Option<StreamFormat>],
    stage: &str,
) -> Result<FrameFormat, StageError> {
    let Some(reference) = chosen.first().copied().flatten() else {
        return Err(missing_reference(stage));
    };

    let Some(format) = reference.as_frames() else {
        return Err(not_frames(stage));
    };

    Ok(*format)
}
