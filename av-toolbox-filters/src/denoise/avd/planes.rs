use av_denoise_core::ChannelMode;
use av_toolbox_core::StageError;
use av_toolbox_gpu::{Channel, FrameFormat, FrameLayout, SampleFormat, Subsampling};

use crate::planes::ChannelName;
use crate::selection::{PlaneDefault, selected_planes};

/// One engine's planes, as indices into the frame's channels, and the size it runs at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EngineSpec {
    pub(crate) channels: ChannelMode,
    pub(crate) planes: Vec<usize>,
    pub(crate) width: u32,
    pub(crate) height: u32,
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

fn fused_needs_444(stage: &str, layout: FrameLayout) -> StageError {
    let message = format!("{stage} can only fuse planes of a yuv 4:4:4 clip, got {layout}");

    StageError::msg(message)
}

fn fused_needs_every_plane(stage: &str) -> StageError {
    let message = format!("{stage} fuses every plane, so planes cannot leave one out");

    StageError::msg(message)
}

fn missing_plane(stage: &str, channel: Channel) -> StageError {
    let message = format!("{stage} was handed a format missing {channel}");

    StageError::msg(message)
}

fn spec_of(
    stage: &str,
    input: &FrameFormat,
    channels: ChannelMode,
    planes: Vec<usize>,
) -> Result<EngineSpec, StageError> {
    let channel = input.channels()[planes[0]];
    let Some((width, height)) = input.channel_dimensions(channel) else {
        return Err(missing_plane(stage, channel));
    };

    Ok(EngineSpec {
        channels,
        planes,
        width,
        height,
    })
}

fn fused_spec(stage: &str, input: &FrameFormat, selected: &[bool]) -> Result<EngineSpec, StageError> {
    let is_444 = input.layout
        == FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        };
    if !is_444 {
        return Err(fused_needs_444(stage, input.layout));
    }

    if selected.iter().any(|chosen| !chosen) {
        return Err(fused_needs_every_plane(stage));
    }

    spec_of(stage, input, ChannelMode::Yuv, vec![0, 1, 2])
}

/// The engines that denoise the planes `planes` and `fused` select from `input`.
pub(crate) fn engine_specs(
    stage: &str,
    planes: Option<&[ChannelName]>,
    fused: bool,
    input: &FrameFormat,
) -> Result<Vec<EngineSpec>, StageError> {
    checked_layout(stage, input.layout)?;

    let selected = selected_planes(stage, planes, input.layout, PlaneDefault::Every)?;

    if fused {
        let spec = fused_spec(stage, input, &selected)?;
        return Ok(vec![spec]);
    }

    let mut specs = Vec::with_capacity(2);
    if selected[0] {
        let luma = spec_of(stage, input, ChannelMode::Luma, vec![0])?;
        specs.push(luma);
    }

    if input.layout == FrameLayout::Grey {
        return Ok(specs);
    }

    match (selected[1], selected[2]) {
        (true, true) => {
            let chroma = spec_of(stage, input, ChannelMode::Chroma, vec![1, 2])?;
            specs.push(chroma);
        },
        (false, false) => {},
        _ => return Err(half_chroma(stage)),
    }

    Ok(specs)
}

/// The av-denoise format of `sample`, or `None` when av-denoise cannot read it.
pub(crate) fn engine_sample(sample: SampleFormat) -> Option<av_denoise_core::SampleFormat> {
    match sample {
        SampleFormat::U8 => Some(av_denoise_core::SampleFormat::U8),
        SampleFormat::U16 { depth } => {
            let depth = depth.bits();
            Some(av_denoise_core::SampleFormat::U16 { depth })
        },
        SampleFormat::F32 => Some(av_denoise_core::SampleFormat::F32),
    }
}

/// Whether a plane of these samples falls short of whole 4-byte words, which av-denoise reads in.
pub(crate) fn needs_padding(sample: SampleFormat, width: u32, height: u32) -> bool {
    let samples = width as usize * height as usize;
    let bytes = samples * sample.bytes();

    !bytes.is_multiple_of(4)
}
