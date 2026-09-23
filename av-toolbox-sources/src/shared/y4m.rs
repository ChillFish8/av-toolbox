use std::io;
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use av_toolbox_core::StageError;
use av_toolbox_gpu::{
    Channel,
    ColourInfo,
    ColourRange,
    Depth,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostFrame,
    SampleFormat,
    Subsampling,
};
use y4m::{Colorspace, Decoder};

/// A y4m stream read frame by frame into host frames.
pub(crate) struct Y4mReader<R: Read> {
    decoder: Decoder<CountingReader<R>>,
    format: FrameFormat,
    frame_rate: (usize, usize),
    next_index: u64,
    bytes_read: Arc<AtomicU64>,
    /// How many bytes had been read when the last whole frame ended.
    frame_end: u64,
}

/// Counts the bytes read through it, so a stream ending mid-frame can be told from one ending cleanly.
struct CountingReader<R> {
    inner: R,
    bytes_read: Arc<AtomicU64>,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.bytes_read.fetch_add(count as u64, Ordering::Relaxed);

        Ok(count)
    }
}

impl<R: Read> Y4mReader<R> {
    /// Reads the stream header and settles the format.
    pub(crate) fn open(reader: R) -> Result<Self, StageError> {
        let counter = AtomicU64::new(0);
        let bytes_read = Arc::new(counter);
        let counting = CountingReader {
            inner: reader,
            bytes_read: bytes_read.clone(),
        };
        let decoder = Decoder::new(counting).map_err(|error| header_error(&error))?;

        let colorspace = decoder.get_colorspace();
        let (layout, sample) = layout_of(colorspace)?;
        let raw_params = decoder.get_raw_params();
        let range = range_of(raw_params);
        let colour = ColourInfo {
            range,
            ..ColourInfo::default()
        };
        let format = FrameFormat {
            width: decoder.get_width() as u32,
            height: decoder.get_height() as u32,
            layout,
            sample,
            colour,
        };

        let frame_rate = decoder.get_framerate();
        let frame_end = bytes_read.load(Ordering::Relaxed);

        Ok(Self {
            decoder,
            format,
            frame_rate: (frame_rate.num, frame_rate.den),
            next_index: 0,
            bytes_read,
            frame_end,
        })
    }

    pub(crate) fn format(&self) -> &FrameFormat {
        &self.format
    }

    /// The next frame, or `None` at a clean end of the stream.
    pub(crate) fn read_frame(&mut self, context: &FrameContext) -> Result<Option<HostFrame>, StageError> {
        let index = self.next_index;
        let outcome = self.decoder.read_frame();
        let frame = match outcome {
            Ok(frame) => frame,
            Err(y4m::Error::EOF) => {
                if self.at_frame_boundary() {
                    return Ok(None);
                }

                return Err(frame_error(index, &y4m::Error::EOF));
            },
            Err(error) => return Err(frame_error(index, &error)),
        };

        let meta = FrameMeta {
            timestamp: timestamp_of(index, self.frame_rate),
            ..FrameMeta::at(index)
        };
        let mut host_frame = context.alloc_host_frame(&self.format, meta);
        let planes = [
            (Channel::Y, frame.get_y_plane()),
            (Channel::U, frame.get_u_plane()),
            (Channel::V, frame.get_v_plane()),
        ];
        for (channel, bytes) in planes {
            if let Some(plane) = host_frame.plane_mut(channel) {
                plane.copy_from_slice(bytes);
            }
        }

        self.next_index += 1;
        self.frame_end = self.bytes_read.load(Ordering::Relaxed);

        Ok(Some(host_frame.freeze()))
    }

    /// Whether nothing has been read since the last whole frame.
    fn at_frame_boundary(&self) -> bool {
        self.bytes_read.load(Ordering::Relaxed) == self.frame_end
    }
}

/// The layout and sample a y4m colourspace carries.
fn layout_of(colorspace: Colorspace) -> Result<(FrameLayout, SampleFormat), StageError> {
    let subsampling = match colorspace {
        Colorspace::Cmono | Colorspace::Cmono12 => None,
        Colorspace::C422 | Colorspace::C422p10 | Colorspace::C422p12 => Some(Subsampling::Yuv422),
        Colorspace::C444 | Colorspace::C444p10 | Colorspace::C444p12 => Some(Subsampling::Yuv444),
        Colorspace::C420
        | Colorspace::C420p10
        | Colorspace::C420p12
        | Colorspace::C420jpeg
        | Colorspace::C420paldv
        | Colorspace::C420mpeg2 => Some(Subsampling::Yuv420),
        unknown => {
            let message = format!("the y4m colourspace {unknown:?} is not supported");

            return Err(StageError::msg(message));
        },
    };

    let layout = match subsampling {
        Some(subsampling) => FrameLayout::Yuv { subsampling },
        None => FrameLayout::Grey,
    };

    let bits = colorspace.get_bit_depth();
    let sample = match bits {
        8 => SampleFormat::U8,
        bits => {
            let depth = Depth::new(bits as u8).map_err(|_| {
                let message = format!("a y4m stream of {bits} bits is not supported");

                StageError::msg(message)
            })?;

            SampleFormat::U16 { depth }
        },
    };

    Ok((layout, sample))
}

/// The range the `XCOLORRANGE` header extension names, if any.
fn range_of(raw_params: &[u8]) -> ColourRange {
    let fields = raw_params.split(|&byte| byte == b' ');

    for field in fields {
        match field {
            b"XCOLORRANGE=FULL" => return ColourRange::Full,
            b"XCOLORRANGE=LIMITED" => return ColourRange::Limited,
            _ => {},
        }
    }

    ColourRange::Unspecified
}

fn timestamp_of(index: u64, (numerator, denominator): (usize, usize)) -> Option<Duration> {
    if numerator == 0 {
        return None;
    }

    let nanos = index as u128 * denominator as u128 * 1_000_000_000 / numerator as u128;
    let nanos = u64::try_from(nanos).ok()?;

    Some(Duration::from_nanos(nanos))
}

fn header_error(error: &y4m::Error) -> StageError {
    let message = format!("the y4m header could not be read: {error}");

    StageError::msg(message)
}

fn frame_error(index: u64, error: &y4m::Error) -> StageError {
    let message = format!("y4m frame {index} could not be read: {error}");

    StageError::msg(message)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::time::Duration;

    use av_toolbox_gpu::{
        Channel,
        ColourRange,
        Depth,
        FrameLayout,
        SampleFormat,
        Subsampling,
        testing_context,
    };
    use y4m::{Colorspace, Ratio, VendorExtensionString};

    use super::*;

    fn stream(colorspace: Colorspace, frames: usize, range: Option<&str>) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut builder = y4m::encode(4, 2, Ratio::new(25, 1)).with_colorspace(colorspace);
        if let Some(range) = range {
            let field = format!("COLORRANGE={range}");
            let extension = VendorExtensionString::new(field.into_bytes()).expect("valid");
            builder = builder.append_vendor_extension(extension);
        }

        let mut encoder = builder.write_header(&mut bytes).expect("writes the header");

        let sample_bytes = colorspace.get_bytes_per_sample();
        for index in 0..frames {
            let (luma, chroma) = plane_sizes(colorspace, sample_bytes);
            let y_plane = vec![index as u8; luma];
            let u_plane = vec![100 + index as u8; chroma];
            let v_plane = vec![200 + index as u8; chroma];
            let frame = y4m::Frame::new([&y_plane, &u_plane, &v_plane], None);
            encoder.write_frame(&frame).expect("writes a frame");
        }

        bytes
    }

    /// Plane sizes for a 4 by 2 frame.
    fn plane_sizes(colorspace: Colorspace, sample_bytes: usize) -> (usize, usize) {
        let luma = 8 * sample_bytes;
        let chroma = match colorspace {
            Colorspace::Cmono | Colorspace::Cmono12 => 0,
            Colorspace::C422 | Colorspace::C422p10 | Colorspace::C422p12 => 4 * sample_bytes,
            Colorspace::C444 | Colorspace::C444p10 | Colorspace::C444p12 => 8 * sample_bytes,
            _ => 2 * sample_bytes,
        };

        (luma, chroma)
    }

    #[test]
    fn every_colourspace_maps_to_its_layout_and_sample() {
        let cases = [
            (Colorspace::Cmono, FrameLayout::Grey, SampleFormat::U8),
            (
                Colorspace::Cmono12,
                FrameLayout::Grey,
                SampleFormat::U16 { depth: Depth::TWELVE },
            ),
            (
                Colorspace::C420,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv420,
                },
                SampleFormat::U8,
            ),
            (
                Colorspace::C420jpeg,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv420,
                },
                SampleFormat::U8,
            ),
            (
                Colorspace::C420paldv,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv420,
                },
                SampleFormat::U8,
            ),
            (
                Colorspace::C420mpeg2,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv420,
                },
                SampleFormat::U8,
            ),
            (
                Colorspace::C420p10,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv420,
                },
                SampleFormat::U16 { depth: Depth::TEN },
            ),
            (
                Colorspace::C420p12,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv420,
                },
                SampleFormat::U16 { depth: Depth::TWELVE },
            ),
            (
                Colorspace::C422,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv422,
                },
                SampleFormat::U8,
            ),
            (
                Colorspace::C422p10,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv422,
                },
                SampleFormat::U16 { depth: Depth::TEN },
            ),
            (
                Colorspace::C422p12,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv422,
                },
                SampleFormat::U16 { depth: Depth::TWELVE },
            ),
            (
                Colorspace::C444,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv444,
                },
                SampleFormat::U8,
            ),
            (
                Colorspace::C444p10,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv444,
                },
                SampleFormat::U16 { depth: Depth::TEN },
            ),
            (
                Colorspace::C444p12,
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv444,
                },
                SampleFormat::U16 { depth: Depth::TWELVE },
            ),
        ];

        for (colorspace, layout, sample) in cases {
            let bytes = stream(colorspace, 0, None);
            let reader = Y4mReader::open(Cursor::new(bytes)).expect("opens");

            assert_eq!(reader.format().layout, layout, "{colorspace:?}");
            assert_eq!(reader.format().sample, sample, "{colorspace:?}");
            assert_eq!((reader.format().width, reader.format().height), (4, 2));
        }
    }

    #[test]
    fn the_colour_range_extension_sets_the_range() {
        let full_stream = stream(Colorspace::C420, 0, Some("FULL"));
        let limited_stream = stream(Colorspace::C420, 0, Some("LIMITED"));
        let unset_stream = stream(Colorspace::C420, 0, None);

        let full = Y4mReader::open(Cursor::new(full_stream)).expect("opens");
        let limited = Y4mReader::open(Cursor::new(limited_stream)).expect("opens");
        let unset = Y4mReader::open(Cursor::new(unset_stream)).expect("opens");

        assert_eq!(full.format().colour.range, ColourRange::Full);
        assert_eq!(limited.format().colour.range, ColourRange::Limited);
        assert_eq!(unset.format().colour.range, ColourRange::Unspecified);
    }

    #[test]
    fn frames_carry_their_bytes_index_and_timestamp() {
        let context = testing_context();
        let bytes = stream(Colorspace::C420, 3, None);
        let mut reader = Y4mReader::open(Cursor::new(bytes)).expect("opens");

        let mut frames = Vec::new();
        while let Some(frame) = reader.read_frame(&context).expect("reads") {
            frames.push(frame);
        }

        assert_eq!(frames.len(), 3);
        let last = &frames[2];
        assert_eq!(last.meta().index, 2);
        assert_eq!(last.meta().timestamp, Some(Duration::from_millis(80)));
        assert!(
            last.plane(Channel::Y)
                .expect("luma")
                .bytes()
                .iter()
                .all(|&byte| byte == 2)
        );
        assert!(
            last.plane(Channel::V)
                .expect("chroma")
                .bytes()
                .iter()
                .all(|&byte| byte == 202)
        );
    }

    #[test]
    fn sixteen_bit_samples_keep_both_bytes() {
        let context = testing_context();
        let mut bytes = Vec::new();
        let mut encoder = y4m::encode(4, 2, Ratio::new(25, 1))
            .with_colorspace(Colorspace::C420p10)
            .write_header(&mut bytes)
            .expect("writes the header");
        let y_plane = [0x34u8, 0x02].repeat(8);
        let u_plane = [0x78u8, 0x01].repeat(2);
        let v_plane = [0xbcu8, 0x03].repeat(2);
        let frame = y4m::Frame::new([&y_plane, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
        let mut reader = Y4mReader::open(Cursor::new(bytes)).expect("opens");

        let frame = reader.read_frame(&context).expect("reads").expect("a frame");

        assert_eq!(frame.plane(Channel::Y).expect("luma").bytes(), y_plane.as_slice());
        assert_eq!(
            frame.plane(Channel::U).expect("chroma").bytes(),
            u_plane.as_slice()
        );
        assert_eq!(
            frame.plane(Channel::V).expect("chroma").bytes(),
            v_plane.as_slice()
        );
    }

    #[test]
    fn a_zero_frame_rate_gives_no_timestamps() {
        let context = testing_context();
        let mut bytes = Vec::new();
        let mut encoder = y4m::encode(4, 2, Ratio::new(0, 1))
            .write_header(&mut bytes)
            .expect("writes the header");
        let planes = [vec![0u8; 8], vec![0u8; 2], vec![0u8; 2]];
        let frame = y4m::Frame::new([&planes[0], &planes[1], &planes[2]], None);
        encoder.write_frame(&frame).expect("writes");
        let mut reader = Y4mReader::open(Cursor::new(bytes)).expect("opens");

        let frame = reader.read_frame(&context).expect("reads").expect("a frame");

        assert_eq!(frame.meta().timestamp, None);
    }

    #[test]
    fn a_truncated_frame_is_an_error_naming_its_index() {
        let context = testing_context();
        let mut bytes = stream(Colorspace::C420, 2, None);
        bytes.truncate(bytes.len() - 3);
        let mut reader = Y4mReader::open(Cursor::new(bytes)).expect("opens");

        reader.read_frame(&context).expect("the first frame is whole");
        let error = reader.read_frame(&context).expect_err("the second is cut short");

        assert!(error.to_string().contains("frame 1"));
    }

    #[test]
    fn a_stream_cut_short_inside_the_frame_marker_is_an_error_naming_its_index() {
        let context = testing_context();
        let mut bytes = stream(Colorspace::C420, 2, None);
        bytes.extend_from_slice(b"FRA");
        let mut reader = Y4mReader::open(Cursor::new(bytes)).expect("opens");

        reader.read_frame(&context).expect("the first frame is whole");
        reader.read_frame(&context).expect("the second frame is whole");
        let error = reader
            .read_frame(&context)
            .expect_err("the third frame's marker is cut short");

        assert!(error.to_string().contains("frame 2"));
    }

    #[test]
    fn a_clean_end_of_stream_ends_the_reader() {
        let context = testing_context();
        let bytes = stream(Colorspace::C420, 1, None);
        let mut reader = Y4mReader::open(Cursor::new(bytes)).expect("opens");

        reader.read_frame(&context).expect("reads").expect("a frame");

        assert!(reader.read_frame(&context).expect("ends cleanly").is_none());
    }
}
