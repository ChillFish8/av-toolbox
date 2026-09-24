use av_toolbox_core::StageError;
use av_toolbox_gpu::{
    ColourInfo,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// The frame format every fixture works in, so they all name the same dimensions.
pub(super) fn frame_format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

/// The format of the first bound input, since every fixture passes its first input through.
///
/// A signature with an optional parameter can leave a hole before the input a fixture wants, so it
/// takes the first one that is actually there rather than the first slot.
pub(super) fn first_bound(chosen: &[Option<StreamFormat>]) -> Result<StreamFormat, StageError> {
    let format = chosen.iter().flatten().next().copied();

    format.ok_or_else(|| StageError::msg("a stage with inputs was given none"))
}

/// A fixed frame format used as the starting point for every fixture source.
pub(super) fn fixture_format() -> StreamFormat {
    StreamFormat::Frames(frame_format())
}

/// The mask format every fixture mask allocates into, so a select fixture needs nothing from its
/// inputs to produce one.
///
/// The sample is arbitrary and does not have to agree with whatever `mask_template` negotiated,
/// because nothing reads a fixture's allocated output.
pub(super) fn fixture_mask_format() -> MaskFormat {
    MaskFormat {
        width: 64,
        height: 64,
        sample: SampleFormat::U8,
    }
}

pub(super) fn frames_template(chosen: StreamFormat) -> StreamTemplate {
    StreamTemplate::from_format(&chosen)
}

/// A mask of the fixture dimensions, carrying whatever sample was chosen.
///
/// A mask stage emits a mask, so it cannot pass a frames template straight through.
pub(super) fn mask_template(chosen: StreamFormat) -> StreamTemplate {
    StreamTemplate::Mask(MaskFormat {
        width: 64,
        height: 64,
        sample: chosen.sample(),
    })
}

/// The fixture arguments exist to give the registry a schema to validate against, so nothing ever
/// reads the fields back out.
#[derive(Debug, Deserialize, JsonSchema)]
#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
pub(crate) struct FileArgs {
    pub(crate) path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
pub(crate) struct BlurArgs {
    pub(crate) sigma: f64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
pub(crate) struct SplitArgs {
    pub(crate) seconds: u32,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct NoArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct TemporalArgs {
    #[serde(default)]
    pub(crate) radius: u32,
}
