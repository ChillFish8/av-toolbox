mod rate;
mod target;
mod y4m;

#[cfg(test)]
pub(crate) mod testing;

use std::io::Write;

use av_toolbox_core::{InputParam, InputSignature, Inputs, Sink, StageError, StageInstance, StreamType};
use av_toolbox_gpu::{
    Depth,
    FormatSet,
    FrameContext,
    FrameLayout,
    HostSide,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
};

pub(crate) use self::rate::RateArg;
pub(crate) use self::target::{TargetFile, open_target};
pub(crate) use self::y4m::Y4mWriter;

pub(crate) const INPUTS: &InputSignature =
    &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);

const NO_RATE: &str = "the stream has no frame rate, so the y4m sink needs a frame_rate argument";

/// The samples a y4m sink will take for `offered`.
pub(crate) fn accepts(offered: StreamFormat, frame_rate: Option<RateArg>) -> Result<FormatSet, StageError> {
    let StreamFormat::Frames(format) = offered else {
        return Err(StageError::msg("a y4m sink writes frames, not a mask"));
    };

    let Some(samples) = writable_samples(format.layout) else {
        let message = format!("y4m cannot carry {} frames", format.layout);

        return Err(StageError::msg(message));
    };

    let has_rate = frame_rate.is_some() || format.frame_rate.is_some();
    if !has_rate {
        return Err(StageError::msg(NO_RATE));
    }

    Ok(samples)
}

/// A writer for the frames input in `formats`, at the argument's rate or else the stream's.
pub(crate) fn writer_for<W: Write>(
    output: W,
    formats: &ResolvedFormats,
    frame_rate: Option<RateArg>,
) -> Result<Y4mWriter<W>, StageError> {
    let Some(StreamFormat::Frames(format)) = formats.input(0) else {
        return Err(StageError::msg("a y4m sink was built without a frames input"));
    };

    let argument_rate = frame_rate.map(|argument| argument.0);
    let chosen_rate = argument_rate.or(format.frame_rate);
    let rate = chosen_rate.ok_or_else(|| StageError::msg(NO_RATE))?;

    Y4mWriter::new(output, format, rate)
}

fn writable_samples(layout: FrameLayout) -> Option<FormatSet> {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let twelve_bit = SampleFormat::U16 { depth: Depth::TWELVE };

    match layout {
        FrameLayout::Yuv { .. } => Some(FormatSet::of([SampleFormat::U8, ten_bit, twelve_bit])),
        FrameLayout::Grey => Some(FormatSet::of([SampleFormat::U8, twelve_bit])),
        FrameLayout::Rgb | FrameLayout::Chroma { .. } => None,
    }
}

/// A sink writing every frame it is given to a y4m stream.
pub(crate) struct Y4mSink<W: Write> {
    writer: Y4mWriter<W>,
}

impl<W: Write> Y4mSink<W> {
    pub(crate) fn new(writer: Y4mWriter<W>) -> Self {
        Self { writer }
    }
}

impl<W: Write + Send + 'static> StageInstance for Y4mSink<W> {}

impl<W: Write + Send + 'static> Sink<HostSide> for Y4mSink<W> {
    fn consume(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<(), StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        self.writer.write_frame(&frame)
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.writer.finish()
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{FrameFormat, FrameRate, MaskFormat, Subsampling};

    use super::*;
    use crate::shared::testing::{SharedBuffer, resolved, yuv420};

    fn rate(numerator: u32, denominator: u32) -> FrameRate {
        FrameRate::new(numerator, denominator).expect("a valid rate")
    }

    fn frames(format: FrameFormat) -> StreamFormat {
        StreamFormat::Frames(format)
    }

    #[test]
    fn yuv_takes_eight_ten_and_twelve_bit_samples() {
        let format = yuv420(4, 2, SampleFormat::F32, Some(rate(25, 1)));

        let accepted = accepts(frames(format), None).expect("accepted");

        assert!(accepted.accepts(SampleFormat::U8));
        assert!(accepted.accepts(SampleFormat::U16 { depth: Depth::TEN }));
        assert!(accepted.accepts(SampleFormat::U16 { depth: Depth::TWELVE }));
        assert!(!accepted.accepts(SampleFormat::U16 {
            depth: Depth::SIXTEEN
        }));
        assert!(!accepted.accepts(SampleFormat::F32));
    }

    #[test]
    fn grey_takes_eight_and_twelve_bit_samples() {
        let format = FrameFormat {
            layout: FrameLayout::Grey,
            ..yuv420(4, 2, SampleFormat::U8, Some(rate(25, 1)))
        };

        let accepted = accepts(frames(format), None).expect("accepted");

        assert!(accepted.accepts(SampleFormat::U8));
        assert!(accepted.accepts(SampleFormat::U16 { depth: Depth::TWELVE }));
        assert!(!accepted.accepts(SampleFormat::U16 { depth: Depth::TEN }));
    }

    #[test]
    fn rgb_is_refused() {
        let format = FrameFormat {
            layout: FrameLayout::Rgb,
            ..yuv420(4, 2, SampleFormat::U8, Some(rate(25, 1)))
        };

        let error = accepts(frames(format), None).expect_err("refused");

        assert!(error.to_string().contains("rgb"));
    }

    #[test]
    fn a_chroma_clip_is_not_writable() {
        let layout = FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        };

        assert_eq!(writable_samples(layout), None);
    }

    #[test]
    fn a_mask_is_refused() {
        let mask = StreamFormat::Mask(MaskFormat {
            frame_rate: None,
            width: 4,
            height: 2,
            sample: SampleFormat::U8,
            chroma: None,
        });

        assert!(accepts(mask, None).is_err());
    }

    #[test]
    fn a_stream_without_a_rate_needs_the_argument() {
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let argument = RateArg(rate(24, 1));

        let without = accepts(frames(format), None);
        let with = accepts(frames(format), Some(argument));

        let error = without.expect_err("no rate anywhere");
        assert!(error.to_string().contains("frame_rate"));
        assert!(with.is_ok());
    }

    #[test]
    fn the_argument_rate_overrides_the_stream_rate() {
        let format = yuv420(4, 2, SampleFormat::U8, Some(rate(25, 1)));
        let formats = resolved(format);
        let buffer = SharedBuffer::default();
        let argument = RateArg(rate(24, 1));

        let mut writer = writer_for(buffer.clone(), &formats, Some(argument)).expect("writable");
        writer.finish().expect("finishes");

        assert_eq!(buffer.contents(), b"YUV4MPEG2 W4 H2 F24:1 C420\n");
    }

    #[test]
    fn the_stream_rate_is_used_without_an_argument() {
        let format = yuv420(4, 2, SampleFormat::U8, Some(rate(30000, 1001)));
        let formats = resolved(format);
        let buffer = SharedBuffer::default();

        let mut writer = writer_for(buffer.clone(), &formats, None).expect("writable");
        writer.finish().expect("finishes");

        assert_eq!(buffer.contents(), b"YUV4MPEG2 W4 H2 F30000:1001 C420\n");
    }
}
