use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_gpu::{
    Channel,
    FormatSet,
    Frame,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    OutputFormats,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::sizing::Sizing;

/// Takes the luma of YUV and Grey frames at U8 and U16, leaving float samples out.
pub(crate) fn accepts_luma(stage: &str, offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        let message = format!("{stage} reads frames, not masks");
        return Err(StageError::msg(message));
    };

    if format.layout == FrameLayout::Rgb {
        let message = format!("{stage} reads YUV or Grey frames, not RGB");
        return Err(StageError::msg(message));
    }

    if let FrameLayout::Chroma { .. } = format.layout {
        let message = format!("{stage} reads YUV or Grey frames, not chroma planes");
        return Err(StageError::msg(message));
    }

    let accepted = match format.sample {
        SampleFormat::U8 | SampleFormat::U16 { .. } => FormatSet::of([format.sample]),
        SampleFormat::F16 | SampleFormat::F32 => FormatSet::new(),
    };

    Ok(accepted)
}

fn bound_frames(stage: &str, format: Option<StreamFormat>) -> Result<FrameFormat, StageError> {
    let Some(format) = format else {
        let message = format!("{stage} needs its input bound");
        return Err(StageError::msg(message));
    };

    let Some(frames) = format.as_frames() else {
        let message = format!("{stage} reads frames, not masks");
        return Err(StageError::msg(message));
    };

    Ok(*frames)
}

/// A U8 mask the size of the input's luma.
pub(crate) fn produces_luma_mask(
    stage: &str,
    chosen: &[Option<StreamFormat>],
) -> Result<OutputFormats, StageError> {
    let first = chosen.first().copied().flatten();
    let format = bound_frames(stage, first)?;
    let mask = MaskFormat {
        width: format.width,
        height: format.height,
        sample: SampleFormat::U8,
        chroma: None,
    };

    Ok(OutputFormats {
        template: StreamTemplate::Mask(mask),
        samples: FormatSet::of([SampleFormat::U8]),
    })
}

/// How the luma a stage reads is stored.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Luma {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The largest sample value, which maps to 1.
    largest: f32,
    wide: bool,
}

impl Luma {
    pub(crate) fn from_formats(stage: &str, formats: &ResolvedFormats) -> Result<Self, StageError> {
        let input = formats.input(0);
        let format = bound_frames(stage, input)?;

        let (largest, wide) = match format.sample {
            SampleFormat::U8 => (255.0, false),
            SampleFormat::U16 { depth } => {
                let max_value = depth.max_value();

                (f32::from(max_value), true)
            },
            SampleFormat::F16 | SampleFormat::F32 => {
                let message = format!("{stage} reads integer samples, not floats");
                return Err(StageError::msg(message));
            },
        };

        Ok(Self {
            width: format.width,
            height: format.height,
            largest,
            wide,
        })
    }

    pub(crate) const fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub(crate) const fn mask_format(&self) -> MaskFormat {
        MaskFormat {
            width: self.width,
            height: self.height,
            sample: SampleFormat::U8,
            chroma: None,
        }
    }
}

#[cube(launch_unchecked)]
fn normalise<E: Int>(input: &Array<E>, output: &mut Array<f32>, scale: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = f32::cast_from(input[ABSOLUTE_POS]) * scale;
}

fn no_luma() -> StageError {
    StageError::msg("a line mask was handed a frame without luma")
}

/// Writes the frame's luma into `output` as f32 between 0 and 1.
pub(crate) fn load_luma<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    frame: &Frame,
    luma: &Luma,
    output: DevicePlane<'_>,
) -> Result<(), StageError> {
    let plane = frame.plane(Channel::Y).ok_or_else(no_luma)?;

    let scale = 1.0 / luma.largest;
    let written = output.array::<R>();
    let samples = luma.samples();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: the luma plane holds `samples` values of the declared width, `output` holds
    // `samples` f32 values, and the kernel writes nothing past the output's length.
    unsafe {
        if luma.wide {
            let read = plane.array::<u16, R>();
            normalise::launch_unchecked::<u16, R>(client, cubes, shape, read, written, scale);
        } else {
            let read = plane.array::<u8, R>();
            normalise::launch_unchecked::<u8, R>(client, cubes, shape, read, written, scale);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, Subsampling};

    use super::*;

    fn frames(layout: FrameLayout, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width: 9,
            height: 7,
            layout,
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
        })
    }

    #[test]
    fn rgb_is_refused_naming_the_stage() {
        let offered = frames(FrameLayout::Rgb, SampleFormat::U8);

        let error = accepts_luma("lines/test", offered).expect_err("rgb has no luma");

        assert!(error.to_string().contains("lines/test reads YUV or Grey"));
    }

    #[test]
    fn a_chroma_clip_is_refused() {
        let layout = FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        };
        let offered = frames(layout, SampleFormat::U8);

        let refused = accepts_luma("luma/test", offered);

        assert!(refused.is_err());
    }

    #[test]
    fn float_samples_are_left_out() {
        let offered = frames(FrameLayout::Grey, SampleFormat::F32);

        let accepted = accepts_luma("lines/test", offered).expect("grey is readable");

        assert!(accepted.is_empty());
    }

    #[test]
    fn the_mask_is_the_size_of_the_luma() {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let chosen = [Some(frames(layout, SampleFormat::U8))];

        let produced = produces_luma_mask("lines/test", &chosen).expect("yuv is readable");

        let expected = MaskFormat {
            width: 9,
            height: 7,
            sample: SampleFormat::U8,
            chroma: None,
        };
        assert_eq!(produced.template, StreamTemplate::Mask(expected));
    }
}
