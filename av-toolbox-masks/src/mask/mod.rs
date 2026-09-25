mod threshold;
mod union;
mod vote;

use av_toolbox_core::StageError;
use av_toolbox_gpu::{FormatSet, MaskFormat, OutputFormats, SampleFormat, StreamFormat, StreamTemplate};

pub(crate) use self::threshold::Threshold;
pub(crate) use self::union::Union;
pub(crate) use self::vote::Vote;

/// Takes U8 masks and leaves every other sample out.
pub(crate) fn accepts_mask(stage: &str, offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_mask() else {
        let message = format!("{stage} reads masks, not frames");
        return Err(StageError::msg(message));
    };

    let accepted = match format.sample {
        SampleFormat::U8 => FormatSet::of([SampleFormat::U8]),
        _ => FormatSet::new(),
    };

    Ok(accepted)
}

/// A U8 mask the size of every input, refusing inputs of different sizes.
pub(crate) fn produces_mask(
    stage: &str,
    chosen: &[Option<StreamFormat>],
) -> Result<OutputFormats, StageError> {
    let mut dimensions: Option<(u32, u32)> = None;

    for format in chosen.iter().flatten() {
        let current = format.dimensions();

        let Some(first) = dimensions else {
            dimensions = Some(current);
            continue;
        };

        if first != current {
            let (first_width, first_height) = first;
            let (width, height) = current;
            let message = format!(
                "{stage} needs every mask the same size, got {first_width}x{first_height} and {width}x{height}",
            );
            return Err(StageError::msg(message));
        }
    }

    let Some((width, height)) = dimensions else {
        let message = format!("{stage} needs its input bound");
        return Err(StageError::msg(message));
    };

    let format = MaskFormat {
        width,
        height,
        sample: SampleFormat::U8,
        chroma: None,
    };

    Ok(OutputFormats {
        template: StreamTemplate::Mask(format),
        samples: FormatSet::of([SampleFormat::U8]),
    })
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, Depth, FrameFormat, FrameLayout};

    use super::*;

    fn mask(width: u32, height: u32, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Mask(MaskFormat {
            width,
            height,
            sample,
            chroma: None,
        })
    }

    fn grey_frame() -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width: 8,
            height: 8,
            layout: FrameLayout::Grey,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
        })
    }

    #[test]
    fn a_u8_mask_is_accepted() {
        let offered = mask(8, 8, SampleFormat::U8);

        let accepted = accepts_mask("mask/test", offered).expect("a mask is readable");

        assert!(accepted.accepts(SampleFormat::U8));
    }

    #[test]
    fn a_u16_mask_is_left_out_rather_than_refused() {
        let sample = SampleFormat::U16 { depth: Depth::TEN };
        let offered = mask(8, 8, sample);

        let accepted = accepts_mask("mask/test", offered).expect("a mask is readable");

        assert!(accepted.is_empty());
    }

    #[test]
    fn frames_are_refused_naming_the_stage() {
        let error = accepts_mask("mask/test", grey_frame()).expect_err("frames are not masks");

        assert!(error.to_string().contains("mask/test reads masks"));
    }

    #[test]
    fn the_output_is_a_u8_mask_the_size_of_the_inputs() {
        let chosen = [
            Some(mask(7, 5, SampleFormat::U8)),
            Some(mask(7, 5, SampleFormat::U8)),
        ];

        let produced = produces_mask("mask/test", &chosen).expect("the inputs agree");

        let expected = MaskFormat {
            width: 7,
            height: 5,
            sample: SampleFormat::U8,
            chroma: None,
        };
        assert_eq!(produced.template, StreamTemplate::Mask(expected));
        assert_eq!(produced.samples, FormatSet::of([SampleFormat::U8]));
    }

    #[test]
    fn masks_of_different_sizes_are_refused() {
        let chosen = [
            Some(mask(7, 5, SampleFormat::U8)),
            Some(mask(8, 5, SampleFormat::U8)),
        ];

        let error = produces_mask("mask/test", &chosen).expect_err("the sizes disagree");

        assert!(error.to_string().contains("7x5 and 8x5"));
    }
}
