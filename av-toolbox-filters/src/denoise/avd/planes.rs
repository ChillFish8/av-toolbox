use av_denoise_core::{ChromaLayout, ChromaSiting, Geometry};
use av_toolbox_core::StageError;
use av_toolbox_gpu::{ChromaLocation, FrameFormat, FrameLayout, SampleFormat, Subsampling};

use crate::planes::ChannelName;
use crate::selection::{PlaneDefault, selected_planes};

/// The planes one set of av-denoise options covers, luma or U and V together.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum PlaneGroup {
    Luma,
    Chroma,
}

/// Which plane groups a stage denoises.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct PlaneGroups {
    pub(crate) luma: bool,
    pub(crate) chroma: bool,
}

fn unsupported_layout(stage: &str, layout: FrameLayout) -> StageError {
    let message = format!("{stage} needs YUV or Gray frames, got {layout}");

    StageError::msg(message)
}

/// Refuses a layout that is neither YUV nor Gray.
pub(crate) fn checked_layout(stage: &str, layout: FrameLayout) -> Result<(), StageError> {
    if !matches!(layout, FrameLayout::Yuv { .. } | FrameLayout::Grey) {
        return Err(unsupported_layout(stage, layout));
    }

    Ok(())
}

fn half_chroma(stage: &str) -> StageError {
    let message = format!("{stage} denoises U and V together, so planes must list both or neither");

    StageError::msg(message)
}

/// The plane groups `planes` selects from `input`, every group when not given.
pub(crate) fn plane_groups(
    stage: &str,
    planes: Option<&[ChannelName]>,
    input: &FrameFormat,
) -> Result<PlaneGroups, StageError> {
    checked_layout(stage, input.layout)?;

    let selected = selected_planes(stage, planes, input.layout, PlaneDefault::Every)?;
    let luma = selected[0];

    if input.layout == FrameLayout::Grey {
        return Ok(PlaneGroups { luma, chroma: false });
    }

    let chroma = match (selected[1], selected[2]) {
        (true, true) => true,
        (false, false) => false,
        _ => return Err(half_chroma(stage)),
    };

    Ok(PlaneGroups { luma, chroma })
}

fn chroma_layout(layout: FrameLayout) -> ChromaLayout {
    let FrameLayout::Yuv { subsampling } = layout else {
        return ChromaLayout::Gray;
    };

    match subsampling {
        Subsampling::Yuv420 => ChromaLayout::Yuv420,
        Subsampling::Yuv422 => ChromaLayout::Yuv422,
        Subsampling::Yuv444 => ChromaLayout::Yuv444,
    }
}

/// The av-denoise siting of `location`, [ChromaSiting::Unknown] where av-denoise has no name for it.
fn chroma_siting(location: ChromaLocation) -> ChromaSiting {
    match location {
        ChromaLocation::Left => ChromaSiting::Left,
        ChromaLocation::Center => ChromaSiting::Center,
        ChromaLocation::TopLeft => ChromaSiting::TopLeft,
        ChromaLocation::Top | ChromaLocation::BottomLeft | ChromaLocation::Bottom => ChromaSiting::Unknown,
    }
}

/// The frame an av-denoise engine is built for, reading `input` and writing F32.
pub(crate) fn engine_geometry(input: &FrameFormat) -> Geometry {
    let chroma = chroma_layout(input.layout);
    let chroma_siting = chroma_siting(input.colour.chroma_location);

    Geometry {
        width: input.width,
        height: input.height,
        chroma,
        input: engine_sample(input.sample),
        output: av_denoise_core::SampleFormat::F32,
        chroma_siting,
    }
}

/// Whether integer `input` reaches the engine as `code / max` floats.
///
/// av-denoise copies a plane group it leaves out word for word, which needs the same format in and out, and the
/// engine always writes F32.
pub(crate) fn normalises(input: &FrameFormat, groups: PlaneGroups) -> bool {
    let passes_through = input.layout != FrameLayout::Grey && groups.luma != groups.chroma;

    passes_through && input.sample != SampleFormat::F32
}

/// The frame the engine for `groups` of `input` is built for, reading normalised floats when [normalises] says so.
pub(crate) fn stage_geometry(input: &FrameFormat, groups: PlaneGroups) -> Geometry {
    let geometry = engine_geometry(input);
    if !normalises(input, groups) {
        return geometry;
    }

    Geometry {
        input: av_denoise_core::SampleFormat::F32,
        ..geometry
    }
}

/// The av-denoise format of `sample`.
pub(crate) fn engine_sample(sample: SampleFormat) -> av_denoise_core::SampleFormat {
    match sample {
        SampleFormat::U8 => av_denoise_core::SampleFormat::U8,
        SampleFormat::U16 { depth } => {
            let depth = depth.bits();
            av_denoise_core::SampleFormat::U16 { depth }
        },
        SampleFormat::F32 => av_denoise_core::SampleFormat::F32,
    }
}

/// Whether a plane of these samples falls short of whole 4-byte words, which av-denoise reads in.
pub(crate) fn needs_padding(sample: SampleFormat, width: u32, height: u32) -> bool {
    let samples = width as usize * height as usize;
    let bytes = samples * sample.bytes();

    !bytes.is_multiple_of(4)
}
