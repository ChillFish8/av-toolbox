use std::io;
use std::io::{BufWriter, Write};

use av_toolbox_core::StageError;
use av_toolbox_gpu::{
    Channel,
    ColourRange,
    FrameFormat,
    FrameLayout,
    FrameRate,
    HostFrame,
    SampleFormat,
    Subsampling,
};

const FRAME_MARKER: &[u8] = b"FRAME\n";

/// A y4m stream written frame by frame from host frames.
///
/// The header goes out with the first frame, or on [finish](crate::shared::Y4mWriter::finish) when no frame came.
pub(crate) struct Y4mWriter<W: Write> {
    output: BufWriter<W>,
    format: FrameFormat,
    header: Vec<u8>,
    header_written: bool,
}

impl<W: Write> Y4mWriter<W> {
    /// Settles the header for `format` at `frame_rate`, failing when y4m cannot carry the format.
    pub(crate) fn new(output: W, format: FrameFormat, frame_rate: FrameRate) -> Result<Self, StageError> {
        let header = header_of(&format, frame_rate)?;

        Ok(Self {
            output: BufWriter::new(output),
            format,
            header,
            header_written: false,
        })
    }

    pub(crate) fn write_frame(&mut self, frame: &HostFrame) -> Result<(), StageError> {
        let index = frame.meta().index;
        let position = format!("frame {index}");

        self.write_header()
            .map_err(|error| write_error(&position, &error))?;

        self.output
            .write_all(FRAME_MARKER)
            .map_err(|error| write_error(&position, &error))?;

        for channel in self.format.channels() {
            let bytes = plane_bytes(&self.format, frame, *channel, index)?;
            let written = self.output.write_all(bytes);

            written.map_err(|error| write_error(&position, &error))?;
        }

        Ok(())
    }

    /// Writes the header if no frame has, and flushes.
    pub(crate) fn finish(&mut self) -> Result<(), StageError> {
        let position = "the end of the stream";

        self.write_header()
            .map_err(|error| write_error(position, &error))?;

        let flushed = self.output.flush();

        flushed.map_err(|error| write_error(position, &error))
    }

    fn write_header(&mut self) -> io::Result<()> {
        if self.header_written {
            return Ok(());
        }

        self.output.write_all(&self.header)?;
        self.header_written = true;

        Ok(())
    }
}

/// The y4m colourspace name for `format`, without its leading `C`.
pub(crate) fn colourspace_of(format: &FrameFormat) -> Result<&'static str, StageError> {
    let bits = bits_of(format.sample);
    let name = match (format.layout, bits) {
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            Some(8),
        ) => "420",
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            Some(10),
        ) => "420p10",
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            Some(12),
        ) => "420p12",
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv422,
            },
            Some(8),
        ) => "422",
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv422,
            },
            Some(10),
        ) => "422p10",
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv422,
            },
            Some(12),
        ) => "422p12",
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv444,
            },
            Some(8),
        ) => "444",
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv444,
            },
            Some(10),
        ) => "444p10",
        (
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv444,
            },
            Some(12),
        ) => "444p12",
        (FrameLayout::Grey, Some(8)) => "mono",
        (FrameLayout::Grey, Some(12)) => "mono12",
        _ => {
            let message = format!("y4m cannot carry {format}");

            return Err(StageError::msg(message));
        },
    };

    Ok(name)
}

fn bits_of(sample: SampleFormat) -> Option<u8> {
    match sample {
        SampleFormat::U8 => Some(8),
        SampleFormat::U16 { depth } => Some(depth.bits()),
        SampleFormat::F16 | SampleFormat::F32 => None,
    }
}

fn header_of(format: &FrameFormat, frame_rate: FrameRate) -> Result<Vec<u8>, StageError> {
    let colourspace = colourspace_of(format)?;
    let range = range_field(format.colour.range);
    let header = format!(
        "YUV4MPEG2 W{} H{} F{}:{}{range} C{colourspace}\n",
        format.width,
        format.height,
        frame_rate.numerator(),
        frame_rate.denominator(),
    );

    Ok(header.into_bytes())
}

fn range_field(range: ColourRange) -> &'static str {
    match range {
        ColourRange::Full => " XCOLORRANGE=FULL",
        ColourRange::Limited => " XCOLORRANGE=LIMITED",
        ColourRange::Unspecified => "",
    }
}

/// One plane's bytes, cut to the size the format gives it.
fn plane_bytes<'frame>(
    format: &FrameFormat,
    frame: &'frame HostFrame,
    channel: Channel,
    index: u64,
) -> Result<&'frame [u8], StageError> {
    let expected = format
        .channel_bytes(channel)
        .expect("the layout carries every channel it lists");
    let plane = frame.plane(channel);
    let bytes = plane.map(|plane| plane.bytes()).unwrap_or_default();

    if bytes.len() < expected {
        let message = format!("frame {index} is missing samples in its {channel} plane");

        return Err(StageError::msg(message));
    }

    Ok(&bytes[..expected])
}

fn write_error(position: &str, error: &io::Error) -> StageError {
    let message = match error.kind() {
        io::ErrorKind::BrokenPipe => {
            format!("the reader closed the y4m stream before {position} was written")
        },
        _ => format!("writing {position} of the y4m stream failed: {error}"),
    };

    StageError::msg(message)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use av_toolbox_gpu::{ColourRange, Depth, FrameMeta, testing_context};

    use super::*;
    use crate::shared::testing::{SharedBuffer, filled_frame, yuv420};

    fn rate(numerator: u32, denominator: u32) -> FrameRate {
        FrameRate::new(numerator, denominator).expect("a valid rate")
    }

    fn ten_bit() -> SampleFormat {
        SampleFormat::U16 { depth: Depth::TEN }
    }

    fn twelve_bit() -> SampleFormat {
        SampleFormat::U16 { depth: Depth::TWELVE }
    }

    fn format_of(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
        FrameFormat {
            layout,
            ..yuv420(4, 2, sample, None)
        }
    }

    fn written(format: FrameFormat, frames: &[HostFrame]) -> Vec<u8> {
        let buffer = SharedBuffer::default();
        let mut writer = Y4mWriter::new(buffer.clone(), format, rate(25, 1)).expect("writable");

        for frame in frames {
            writer.write_frame(frame).expect("writes");
        }

        writer.finish().expect("finishes");

        buffer.contents()
    }

    #[test]
    fn every_writable_format_maps_to_its_colourspace() {
        let yuv = |subsampling| FrameLayout::Yuv { subsampling };
        let table = [
            (yuv(Subsampling::Yuv420), SampleFormat::U8, "420"),
            (yuv(Subsampling::Yuv420), ten_bit(), "420p10"),
            (yuv(Subsampling::Yuv420), twelve_bit(), "420p12"),
            (yuv(Subsampling::Yuv422), SampleFormat::U8, "422"),
            (yuv(Subsampling::Yuv422), ten_bit(), "422p10"),
            (yuv(Subsampling::Yuv422), twelve_bit(), "422p12"),
            (yuv(Subsampling::Yuv444), SampleFormat::U8, "444"),
            (yuv(Subsampling::Yuv444), ten_bit(), "444p10"),
            (yuv(Subsampling::Yuv444), twelve_bit(), "444p12"),
            (FrameLayout::Grey, SampleFormat::U8, "mono"),
            (FrameLayout::Grey, twelve_bit(), "mono12"),
        ];

        for (layout, sample, expected) in table {
            let format = format_of(layout, sample);

            assert_eq!(colourspace_of(&format).expect("writable"), expected, "{format}");
        }
    }

    #[test]
    fn formats_y4m_cannot_carry_have_no_colourspace() {
        let sixteen_bit = SampleFormat::U16 {
            depth: Depth::SIXTEEN,
        };
        let yuv = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let refused = [
            format_of(yuv, SampleFormat::F32),
            format_of(yuv, SampleFormat::F16),
            format_of(yuv, sixteen_bit),
            format_of(FrameLayout::Grey, ten_bit()),
            format_of(FrameLayout::Rgb, SampleFormat::U8),
        ];

        for format in refused {
            assert!(colourspace_of(&format).is_err(), "{format}");
        }
    }

    #[test]
    fn the_header_carries_size_rate_range_and_colourspace() {
        let mut format = yuv420(4, 2, SampleFormat::U8, None);
        format.colour.range = ColourRange::Full;
        let buffer = SharedBuffer::default();
        let mut writer = Y4mWriter::new(buffer.clone(), format, rate(30000, 1001)).expect("writable");

        writer.finish().expect("finishes");

        assert_eq!(
            buffer.contents(),
            b"YUV4MPEG2 W4 H2 F30000:1001 XCOLORRANGE=FULL C420\n"
        );
    }

    #[test]
    fn a_limited_range_is_written_and_an_unspecified_one_is_left_out() {
        let mut limited = yuv420(4, 2, SampleFormat::U8, None);
        limited.colour.range = ColourRange::Limited;
        let unspecified = yuv420(4, 2, SampleFormat::U8, None);

        let limited_bytes = written(limited, &[]);
        let unspecified_bytes = written(unspecified, &[]);

        assert_eq!(limited_bytes, b"YUV4MPEG2 W4 H2 F25:1 XCOLORRANGE=LIMITED C420\n");
        assert_eq!(unspecified_bytes, b"YUV4MPEG2 W4 H2 F25:1 C420\n");
    }

    #[test]
    fn an_empty_stream_still_writes_a_header() {
        let format = yuv420(4, 2, SampleFormat::U8, None);

        let bytes = written(format, &[]);

        let decoder = y4m::Decoder::new(Cursor::new(bytes)).expect("a valid y4m header");
        assert_eq!(decoder.get_width(), 4);
    }

    #[test]
    fn finishing_twice_writes_the_header_once() {
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let buffer = SharedBuffer::default();
        let mut writer = Y4mWriter::new(buffer.clone(), format, rate(25, 1)).expect("writable");

        writer.finish().expect("finishes");
        writer.finish().expect("finishes again");

        assert_eq!(buffer.contents(), b"YUV4MPEG2 W4 H2 F25:1 C420\n");
    }

    #[test]
    fn eight_bit_frames_read_back_through_a_y4m_decoder() {
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let frames = [filled_frame(&format, 0, 7), filled_frame(&format, 1, 9)];

        let bytes = written(format, &frames);

        let mut decoder = y4m::Decoder::new(Cursor::new(bytes)).expect("decodes");
        let rate = decoder.get_framerate();
        assert_eq!((rate.num, rate.den), (25, 1));
        assert_eq!(decoder.get_bit_depth(), 8);

        for value in [7u8, 9] {
            let frame = decoder.read_frame().expect("a frame");

            assert_eq!(frame.get_y_plane(), vec![value; 8].as_slice());
            assert_eq!(frame.get_u_plane(), vec![value; 2].as_slice());
            assert_eq!(frame.get_v_plane(), vec![value; 2].as_slice());
        }
    }

    #[test]
    fn ten_bit_frames_read_back_as_little_endian_samples() {
        let format = yuv420(4, 2, ten_bit(), None);
        let context = testing_context();
        let meta = FrameMeta::at(0);
        let mut frame = context.alloc_host_frame(&format, meta);
        for channel in format.channels() {
            let plane = frame.plane_mut(*channel).expect("carried");
            for sample in plane.chunks_mut(2) {
                sample.copy_from_slice(&0x0301u16.to_le_bytes());
            }
        }

        let frames = [frame.freeze()];

        let bytes = written(format, &frames);

        let mut decoder = y4m::Decoder::new(Cursor::new(bytes)).expect("decodes");
        assert_eq!(decoder.get_bit_depth(), 10);

        let decoded = decoder.read_frame().expect("a frame");
        assert_eq!(&decoded.get_y_plane()[..2], &[0x01, 0x03]);
        assert_eq!(decoded.get_y_plane().len(), 4 * 2 * 2);
    }

    #[test]
    fn an_odd_sized_frame_writes_rounded_up_chroma() {
        let format = yuv420(5, 3, SampleFormat::U8, None);
        let frames = [filled_frame(&format, 0, 1)];

        let bytes = written(format, &frames);

        let mut decoder = y4m::Decoder::new(Cursor::new(bytes)).expect("decodes");
        let decoded = decoder.read_frame().expect("a whole frame");
        assert_eq!(decoded.get_y_plane().len(), 15);
        assert_eq!(decoded.get_u_plane().len(), 6);
    }

    #[test]
    fn a_closed_reader_is_reported_as_closed() {
        struct ClosedReader;

        impl Write for ClosedReader {
            fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }

            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }

        let format = yuv420(4, 2, SampleFormat::U8, None);
        let frame = filled_frame(&format, 0, 1);
        let mut writer = Y4mWriter::new(ClosedReader, format, rate(25, 1)).expect("writable");

        let outcome = writer.write_frame(&frame).and_then(|()| writer.finish());

        let error = outcome.expect_err("the reader is gone");
        assert!(error.to_string().contains("reader closed"), "{error}");
    }
}
