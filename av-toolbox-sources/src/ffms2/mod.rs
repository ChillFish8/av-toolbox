mod format;
mod index;
mod sys;

use std::path::{Path, PathBuf};
use std::time::Duration;

use av_toolbox_core::{
    BoundInputs,
    InputSignature,
    Seeker,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
    HostFrame,
    HostFrameMut,
    HostPool,
    HostSide,
    OutputFormats,
    ResolvedFormats,
    Runtime,
};
use schemars::JsonSchema;
use serde::Deserialize;

use self::format::Choice;
use self::sys::{Ffms2Error, Index, StridedPlane, VideoSource};
use crate::shared::{Carried, RawColour, colour_of, field_order_of, produced_by};

/// Arguments for `ffms2/file`.
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct FileArgs {
    /// The media file to decode.
    pub(crate) path: PathBuf,
    /// The video track to decode, the first video track by default.
    pub(crate) track: Option<usize>,
    /// A folder to keep the index in, beside the source file by default.
    pub(crate) index_dir: Option<PathBuf>,
    /// A carried format to force instead of the one ffms2 chooses.
    pub(crate) format: Option<Carried>,
    /// Decoder threads, left to ffms2 by default.
    pub(crate) threads: Option<usize>,
}

/// The size and pixel format of a decoded frame.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct FrameShape {
    width: i32,
    height: i32,
    format_id: i32,
}

/// A video track opened and ready to decode.
pub(crate) struct Opened {
    video: VideoSource,
    format: FrameFormat,
    /// The shape every decoded frame is expected to convert into.
    expected: FrameShape,
    path: PathBuf,
    next: i32,
    /// The index the stream was opened from, kept to open seekers.
    index: Index,
    track: i32,
    threads: i32,
}

impl Opened {
    pub(crate) fn format(&self) -> &FrameFormat {
        &self.format
    }
}

pub(crate) struct File;

impl StageDefinition for File {
    type Args = FileArgs;

    fn package() -> &'static str {
        "ffms2"
    }

    fn identifier() -> &'static str {
        "file"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for File {
    type Side = HostSide;
    type Opened = Opened;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        let track = to_i32("track", args.track)?;
        let threads = to_i32("threads", args.threads)?;

        let index_dir = args.index_dir.as_deref();
        let index = index::load_or_build(&args.path, index_dir)?;

        let track_index = match track {
            Some(track) => track,
            None => {
                let found = index.first_video_track();

                found.map_err(|error| track_error(&args.path, &error))?
            },
        };

        let threads = threads.unwrap_or(0);
        let created = VideoSource::open(&args.path, track_index, &index, threads);
        let mut video = created.map_err(|error| open_error(&args.path, &error))?;

        let (fps_numerator, fps_denominator) = video.frame_rate();
        let frame_rate = rate_from(fps_numerator, fps_denominator);

        let probed = video.frame(0);
        let probe = probed.map_err(|error| frame_error(&args.path, 0, &error))?;
        let encoded_id = probe.encoded_pixel_format();
        let width = probe.width();
        let height = probe.height();
        let range = probe.colour_range();
        let matrix = probe.colour_space();
        let primaries = probe.colour_primaries();
        let transfer = probe.transfer();
        let chroma_location = probe.chroma_location();
        let interlaced = probe.interlaced();
        let top_field_first = probe.top_field_first();

        let resolved_name = pixel_format_name(encoded_id);
        let choice = resolve_choice(&args.path, encoded_id, resolved_name, args.format)?;

        let output_name = choice.output.ffmpeg_name();
        let output_id = sys::pixel_format_id(output_name);
        let configured = video.set_output_format(output_id, width, height);
        configured.map_err(|error| open_error(&args.path, &error))?;

        let (layout, sample) = choice.output.layout();
        let source_layout = resolved_name.and_then(format::mapped_layout);
        let raw_colour = RawColour {
            range,
            matrix,
            primaries,
            transfer,
            chroma_location,
        };
        let colour = colour_of(raw_colour, choice.full_range, source_layout, layout);
        let field_order = field_order_of(interlaced, top_field_first);
        let format = FrameFormat {
            width: width as u32,
            height: height as u32,
            layout,
            sample,
            colour,
            frame_rate,
            field_order: Some(field_order),
        };
        let expected = FrameShape {
            width,
            height,
            format_id: output_id,
        };

        Ok(Opened {
            video,
            format,
            expected,
            path: args.path.clone(),
            next: 0,
            index,
            track: track_index,
            threads,
        })
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let format = opened.format();
        let produced = produced_by(format);

        Ok(produced)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<HostSide>>, StageError> {
        let source = FileSource::new(opened);

        Ok(Box::new(source))
    }

    fn seeker(opened: &Self::Opened) -> Result<Option<Box<dyn Seeker>>, StageError> {
        let created = VideoSource::open(&opened.path, opened.track, &opened.index, opened.threads);
        let mut video = created.map_err(|error| open_error(&opened.path, &error))?;

        let shape = opened.expected;
        let configured = video.set_output_format(shape.format_id, shape.width, shape.height);
        configured.map_err(|error| open_error(&opened.path, &error))?;

        let seeker = FileSeeker {
            video,
            format: opened.format,
            expected: shape,
            path: opened.path.clone(),
            pool: HostPool::new(),
        };

        Ok(Some(Box::new(seeker)))
    }
}

/// A source decoding an ffms2 video track in order.
pub(crate) struct FileSource {
    opened: Opened,
}

impl FileSource {
    pub(crate) fn new(opened: Opened) -> Self {
        Self { opened }
    }
}

impl StageInstance for FileSource {}

impl Source<HostSide> for FileSource {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<HostFrame>, StageError> {
        let opened = &mut self.opened;
        let total_frames = opened.video.num_frames();

        if opened.next >= total_frames {
            return Ok(None);
        }

        let current = opened.next;
        let meta = meta_of(&opened.video, current);
        let mut host_frame = context.alloc_host_frame(&opened.format, meta);
        decode_into(
            &mut opened.video,
            &opened.format,
            opened.expected,
            &opened.path,
            current,
            &mut host_frame,
        )?;

        opened.next += 1;

        Ok(Some(host_frame.freeze()))
    }
}

/// Decodes frame `current` of `video` into `host_frame`.
fn decode_into(
    video: &mut VideoSource,
    format: &FrameFormat,
    expected: FrameShape,
    path: &Path,
    current: i32,
    host_frame: &mut HostFrameMut,
) -> Result<(), StageError> {
    let decoded = video.frame(current);
    let frame = decoded.map_err(|error| frame_error(path, current, &error))?;
    let actual = FrameShape {
        width: frame.width(),
        height: frame.height(),
        format_id: frame.converted_pixel_format(),
    };
    validate_decoded_frame(actual, expected, path, current)?;

    let plane_channels = ffmpeg_plane_channels(format.layout);
    for (plane_index, channel) in plane_channels.iter().enumerate() {
        let (row_bytes, rows) = plane_extent(format, *channel, path, current)?;

        // SAFETY: `validate_decoded_frame` confirmed the frame holds the output format at the
        // output resolution, and `plane_extent` sized this plane from that same format.
        let plane = unsafe { frame.plane(plane_index, row_bytes, rows) };
        let plane = plane.ok_or_else(|| missing_plane_error(path, current, *channel))?;

        copy_plane(host_frame, *channel, &plane, path, current)?;
    }

    Ok(())
}

/// The metadata of frame `current`, with its timestamp when the track has one.
fn meta_of(video: &VideoSource, current: i32) -> FrameMeta {
    let pts = video.pts(current);
    let (numerator, denominator) = video.time_base();
    let timestamp = pts.and_then(|pts| timestamp_of(pts, numerator, denominator));

    FrameMeta {
        timestamp,
        ..FrameMeta::at(current as u64)
    }
}

/// Reads any frame of a file through a decoder of its own.
pub(crate) struct FileSeeker {
    video: VideoSource,
    format: FrameFormat,
    expected: FrameShape,
    path: PathBuf,
    pool: HostPool,
}

fn past_end(path: &Path, index: usize, count: usize) -> StageError {
    let message = format!(
        "{} has {count} frames, so frame {index} cannot be read",
        path.display()
    );

    StageError::msg(message)
}

impl Seeker for FileSeeker {
    fn format(&self) -> &FrameFormat {
        &self.format
    }

    fn frame_count(&self) -> usize {
        let count = self.video.num_frames();

        usize::try_from(count).unwrap_or(0)
    }

    fn read(&mut self, index: usize) -> Result<HostFrame, StageError> {
        let count = self.frame_count();
        if index >= count {
            return Err(past_end(&self.path, index, count));
        }

        let current = index as i32;
        let meta = meta_of(&self.video, current);
        let mut host_frame = self.pool.alloc_frame(&self.format, meta);
        decode_into(
            &mut self.video,
            &self.format,
            self.expected,
            &self.path,
            current,
            &mut host_frame,
        )?;

        Ok(host_frame.freeze())
    }
}

/// Fails unless a decoded frame's own shape matches the expected one.
fn validate_decoded_frame(
    actual: FrameShape,
    expected: FrameShape,
    path: &Path,
    frame_index: i32,
) -> Result<(), StageError> {
    if actual == expected {
        return Ok(());
    }

    Err(unexpected_frame_error(path, frame_index))
}

/// The bytes in each row of a channel's plane, and its row count.
fn plane_extent(
    format: &FrameFormat,
    channel: Channel,
    path: &Path,
    frame_index: i32,
) -> Result<(usize, usize), StageError> {
    let dimensions = format.channel_dimensions(channel);
    let (width, height) = dimensions.ok_or_else(|| missing_plane_error(path, frame_index, channel))?;
    let row_bytes = width as usize * format.sample.bytes();
    let rows = height as usize;

    if rows == 0 || row_bytes == 0 {
        return Err(missing_plane_error(path, frame_index, channel));
    }

    Ok((row_bytes, rows))
}

/// Copies one plane into a host frame's matching channel, row by row.
fn copy_plane(
    host_frame: &mut HostFrameMut,
    channel: Channel,
    plane: &StridedPlane,
    path: &Path,
    frame_index: i32,
) -> Result<(), StageError> {
    let destination = host_frame
        .plane_mut(channel)
        .ok_or_else(|| missing_plane_error(path, frame_index, channel))?;

    let row_bytes = plane.row_bytes();
    let destination_rows = destination.chunks_mut(row_bytes);
    for (source_row, destination_row) in plane.rows().zip(destination_rows) {
        destination_row.copy_from_slice(source_row);
    }

    Ok(())
}

/// The channel each ffmpeg plane maps to, in ffmpeg's own plane order.
fn ffmpeg_plane_channels(layout: FrameLayout) -> &'static [Channel] {
    match layout {
        FrameLayout::Yuv { .. } => &[Channel::Y, Channel::U, Channel::V],
        FrameLayout::Grey => &[Channel::Y],
        FrameLayout::Rgb => &[Channel::G, Channel::B, Channel::R],
        FrameLayout::Chroma { .. } => unreachable!("ffms2 never opens a source as chroma planes"),
    }
}

/// The ffmpeg pixel format name a frame was decoded from, matched by id.
fn pixel_format_name(id: i32) -> Option<&'static str> {
    let names = format::known_names();
    let matched = names.iter().find(|name| {
        let candidate = sys::pixel_format_id(name);

        candidate == id
    });

    matched.copied()
}

/// Resolves the format to decode into, from the source's pixel format id and any forced format.
///
/// A forced format overrides an id ffms2/file cannot name at all, since naming it is only needed
/// to look up its own carried equivalent and range.
fn resolve_choice(
    path: &Path,
    id: i32,
    resolved_name: Option<&str>,
    forced: Option<Carried>,
) -> Result<Choice, StageError> {
    match (resolved_name, forced) {
        (Some(name), _) => format::choose(name, forced),
        (None, Some(forced)) => Ok(Choice {
            output: forced,
            full_range: false,
        }),
        (None, None) => Err(unresolved_pixel_format_error(path, id)),
    }
}

/// A frame's timestamp from its PTS and the track's time base.
///
/// The time base's `Num` and `Den` give the frame's place in milliseconds, so the nanosecond
/// count scales that by another `1_000_000`. A negative PTS or an overflow anywhere in the
/// scaling gives no timestamp rather than wrapping or panicking.
fn timestamp_of(pts: i64, numerator: i64, denominator: i64) -> Option<Duration> {
    if denominator <= 0 {
        return None;
    }

    let pts = u64::try_from(pts).ok()?;
    let numerator = u64::try_from(numerator).ok()?;
    let denominator = u64::try_from(denominator).ok()?;

    let pts = u128::from(pts);
    let numerator = u128::from(numerator);
    let denominator = u128::from(denominator);

    let scaled = pts.checked_mul(numerator)?;
    let scaled = scaled.checked_mul(1_000_000)?;
    let nanos = scaled / denominator;
    let nanos = u64::try_from(nanos).ok()?;

    Some(Duration::from_nanos(nanos))
}

/// The frame rate FFMS2 reports, or `None` when either part is zero or negative.
fn rate_from(numerator: i32, denominator: i32) -> Option<FrameRate> {
    let numerator = u32::try_from(numerator).ok()?;
    let denominator = u32::try_from(denominator).ok()?;

    FrameRate::new(numerator, denominator)
}

/// Converts an optional argument to the `i32` FFMS2 takes, failing above `i32::MAX`.
fn to_i32(name: &str, value: Option<usize>) -> Result<Option<i32>, StageError> {
    let Some(value) = value else {
        return Ok(None);
    };

    let converted = i32::try_from(value);

    converted.map(Some).map_err(|_| {
        let limit = i32::MAX;
        let message = format!("{name} is {value}, above the most ffms2 accepts of {limit}");

        StageError::msg(message)
    })
}

fn open_error(path: &Path, error: &Ffms2Error) -> StageError {
    let display = path.display();
    let message = format!("ffms2 could not open {display}: {error}");

    StageError::msg(message)
}

fn track_error(path: &Path, error: &Ffms2Error) -> StageError {
    let display = path.display();
    let message = format!("ffms2 could not find a video track in {display}: {error}");

    StageError::msg(message)
}

fn frame_error(path: &Path, frame: i32, error: &Ffms2Error) -> StageError {
    let display = path.display();
    let message = format!("ffms2 could not decode frame {frame} of {display}: {error}");

    StageError::msg(message)
}

fn unresolved_pixel_format_error(path: &Path, id: i32) -> StageError {
    let display = path.display();
    let message = format!(
        "ffms2 decodes {display} as pixel format {id}, which has no carried equivalent. Pass format to choose one."
    );

    StageError::msg(message)
}

fn unexpected_frame_error(path: &Path, frame: i32) -> StageError {
    let display = path.display();
    let message = format!(
        "ffms2 frame {frame} of {display} does not match the resolution or pixel format SetOutputFormatV2 was asked for"
    );

    StageError::msg(message)
}

fn missing_plane_error(path: &Path, frame: i32, channel: Channel) -> StageError {
    let display = path.display();
    let message = format!("ffms2 frame {frame} of {display} is missing its {channel} plane");

    StageError::msg(message)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use av_toolbox_core::Seeker;
    use av_toolbox_gpu::{
        Channel,
        ColourInfo,
        ColourRange,
        FieldOrder,
        FrameLayout,
        FrameRate,
        Matrix,
        SampleFormat,
        Subsampling,
        testing_context,
    };

    use super::*;

    fn copied(name: &str) -> (tempfile::TempDir, PathBuf) {
        let folder = tempfile::tempdir().expect("a temp dir");
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let media = manifest.join("tests").join("media").join(name);
        let source = folder.path().join(name);
        std::fs::copy(media, &source).expect("copies");

        (folder, source)
    }

    fn opened(name: &str, format: Option<Carried>) -> (tempfile::TempDir, Opened) {
        let (folder, source) = copied(name);
        let args = FileArgs {
            path: source,
            track: None,
            index_dir: None,
            format,
            threads: None,
        };

        let opened = File::open(&args).expect("opens");

        (folder, opened)
    }

    fn decode_all(opened: Opened) -> Vec<av_toolbox_gpu::HostFrame> {
        let mut context = testing_context();
        let mut source = FileSource::new(opened);
        let mut frames = Vec::new();
        while let Some(frame) = source.next(&mut context).expect("decodes") {
            frames.push(frame);
        }

        frames
    }

    fn seeker_over(opened: &Opened) -> Box<dyn Seeker> {
        let found = File::seeker(opened).expect("opens a second source");

        found.expect("ffms2 can seek")
    }

    #[test]
    fn the_seeker_reads_what_the_stream_decodes() {
        let (_folder, opened) = opened("yuv420p.mkv", None);
        let mut seeker = seeker_over(&opened);
        let (_second_folder, streamed) = self::opened("yuv420p.mkv", None);
        let frames = decode_all(streamed);

        assert_eq!(seeker.frame_count(), frames.len());
        for index in [0, 4, 7] {
            let read = seeker.read(index).expect("reads");

            assert_eq!(read.meta().index, frames[index].meta().index);
            for channel in [Channel::Y, Channel::U, Channel::V] {
                let read_plane = read.plane(channel).expect("carried");
                let streamed_plane = frames[index].plane(channel).expect("carried");

                assert_eq!(read_plane.bytes(), streamed_plane.bytes(), "{index} {channel}");
            }
        }
    }

    #[test]
    fn reading_through_the_seeker_leaves_the_stream_at_its_start() {
        let (_folder, opened) = opened("yuv420p.mkv", None);
        let mut seeker = seeker_over(&opened);
        seeker.read(5).expect("reads");

        let frames = decode_all(opened);

        assert_eq!(frames.len(), 8);
        assert_eq!(frames[0].meta().index, 0);
    }

    #[test]
    fn reading_past_the_end_is_refused() {
        let (_folder, opened) = opened("yuv420p.mkv", None);
        let mut seeker = seeker_over(&opened);

        let refused = seeker.read(8);

        assert!(refused.is_err());
    }

    #[test]
    fn a_planar_file_decodes_every_frame_in_order() {
        let (_folder, opened) = opened("yuv420p.mkv", None);

        assert_eq!(
            opened.format().layout,
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420
            }
        );
        let frames = decode_all(opened);

        assert_eq!(frames.len(), 8);
        let indices: Vec<u64> = frames.iter().map(|frame| frame.meta().index).collect();
        assert_eq!(indices, (0..8).collect::<Vec<_>>());
        assert_eq!(
            frames[1].meta().timestamp,
            Some(std::time::Duration::from_millis(40))
        );
    }

    #[test]
    fn a_file_reports_its_frame_rate() {
        let (_folder, opened) = opened("yuv420p.mkv", None);

        assert_eq!(opened.format().frame_rate, FrameRate::new(25, 1));
    }

    #[test]
    fn a_zero_or_negative_ffms2_rate_is_no_rate() {
        assert_eq!(rate_from(0, 1), None);
        assert_eq!(rate_from(25, 0), None);
        assert_eq!(rate_from(-25, 1), None);
    }

    #[test]
    fn an_ffms2_rate_is_reduced() {
        assert_eq!(rate_from(50, 2), FrameRate::new(25, 1));
    }

    #[test]
    fn a_ten_bit_file_keeps_its_depth() {
        let (_folder, opened) = opened("yuv420p10.mkv", None);

        assert_eq!(
            opened.format().sample,
            SampleFormat::U16 {
                depth: av_toolbox_gpu::Depth::TEN
            }
        );
    }

    #[test]
    fn a_ten_bit_file_decodes_every_frame_with_two_byte_samples() {
        let (_folder, opened) = opened("yuv420p10.mkv", None);

        let frames = decode_all(opened);

        assert_eq!(frames.len(), 8);
        for frame in &frames {
            let luma = frame.plane(Channel::Y).expect("luma");
            let u_plane = frame.plane(Channel::U).expect("chroma");
            let v_plane = frame.plane(Channel::V).expect("chroma");

            assert_eq!(luma.bytes().len(), 64 * 64 * 2);
            assert_eq!(u_plane.bytes().len(), 32 * 32 * 2);
            assert_eq!(v_plane.bytes().len(), 32 * 32 * 2);
            assert!(luma.bytes().chunks(2).any(|sample| sample[1] != 0));

            let mut samples = luma.bytes().chunks(2);
            assert!(samples.all(|sample| u16::from_le_bytes([sample[0], sample[1]]) < 1024));
        }
    }

    #[test]
    fn nv12_decodes_as_planar_yuv() {
        let (_folder, opened) = opened("nv12.nut", None);

        assert_eq!(
            opened.format().layout,
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420
            }
        );
        assert_eq!(decode_all(opened).len(), 8);
    }

    #[test]
    fn rgb_decodes_as_planar_rgb() {
        let (_folder, opened) = opened("rgb24.nut", None);

        assert_eq!(opened.format().layout, FrameLayout::Rgb);
        let frames = decode_all(opened);
        assert_eq!(frames[0].plane(Channel::G).expect("green").bytes().len(), 64 * 64);
    }

    #[test]
    fn nv12_decodes_to_the_same_pixels_as_its_planar_twin() {
        let (_planar_folder, planar) = opened("yuv420p.mkv", None);
        let (_nv12_folder, nv12) = opened("nv12.nut", None);

        let planar_frames = decode_all(planar);
        let nv12_frames = decode_all(nv12);

        for channel in [Channel::Y, Channel::U, Channel::V] {
            let planar_plane = planar_frames[3].plane(channel).expect("carried");
            let nv12_plane = nv12_frames[3].plane(channel).expect("carried");

            assert_eq!(planar_plane.bytes(), nv12_plane.bytes(), "{channel}");
        }
    }

    #[test]
    fn a_forced_format_is_what_decodes() {
        let (_folder, opened) = opened("yuv420p10.mkv", Some(Carried::Yuv420p));

        assert_eq!(opened.format().sample, SampleFormat::U8);
    }

    #[test]
    fn the_range_is_read_from_the_file() {
        let (_folder, opened) = opened("yuv420p.mkv", None);

        assert_ne!(opened.format().colour.range, ColourRange::Full);
    }

    #[test]
    fn forcing_less_chroma_than_the_source_fills_every_chroma_row() {
        let (_folder, opened) = opened("rgb24.nut", Some(Carried::Yuv420p));
        let frames = decode_all(opened);

        for channel in [Channel::U, Channel::V] {
            let plane = frames[0].plane(channel).expect("chroma");

            assert!(plane.bytes().iter().any(|&byte| byte != 0), "{channel}");
        }
    }

    #[test]
    fn forcing_more_chroma_than_the_source_fills_every_chroma_row() {
        let (_folder, opened) = opened("yuv420p.mkv", Some(Carried::Yuv444p));
        let frames = decode_all(opened);
        let format = *frames[0].format();

        for channel in [Channel::U, Channel::V] {
            let plane = frames[0].plane(channel).expect("chroma");
            let (_, height) = format.channel_dimensions(channel).expect("dims");
            let row_bytes = plane.bytes().len() / height as usize;
            let last_row = &plane.bytes()[plane.bytes().len() - row_bytes..];

            assert!(last_row.iter().any(|&byte| byte != 0), "{channel}");
        }
    }

    #[test]
    fn a_forced_format_overrides_an_unrecognised_pixel_format_id() {
        let path = PathBuf::from("/nonexistent");

        let choice = resolve_choice(&path, 9999, None, Some(Carried::Yuv420p)).expect("forced wins");

        assert_eq!(choice.output, Carried::Yuv420p);
    }

    #[test]
    fn an_unrecognised_pixel_format_id_with_no_forced_format_is_an_error() {
        let path = PathBuf::from("/nonexistent");

        let error = resolve_choice(&path, 9999, None, None).expect_err("nothing to fall back on");

        assert!(error.to_string().contains("9999"));
    }

    #[cfg(unix)]
    #[test]
    fn a_non_utf8_path_is_refused() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let bytes = [0x66, 0x6f, 0x80, 0x6f];
        let os_str = OsStr::from_bytes(&bytes);
        let args = FileArgs {
            path: PathBuf::from(os_str),
            track: None,
            index_dir: None,
            format: None,
            threads: None,
        };

        let opened = File::open(&args);
        let error = opened.err().expect("not valid utf-8");

        assert!(error.to_string().contains("utf-8"));
    }

    #[cfg(unix)]
    #[test]
    fn a_path_with_an_interior_nul_byte_is_refused() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let bytes = [0x66, 0x6f, 0x00, 0x6f];
        let os_str = OsStr::from_bytes(&bytes);
        let args = FileArgs {
            path: PathBuf::from(os_str),
            track: None,
            index_dir: None,
            format: None,
            threads: None,
        };

        let opened = File::open(&args);
        let error = opened.err().expect("a nul byte is not a valid path");

        assert!(error.to_string().contains("nul"));
    }

    #[test]
    fn a_forced_rgb_output_from_a_yuv_source_gets_identity_matrix_and_full_range() {
        let (_folder, opened) = opened("yuv420p.mkv", Some(Carried::Gbrp));

        assert_eq!(opened.format().colour.matrix, Matrix::Identity);
        assert_eq!(opened.format().colour.range, ColourRange::Full);
    }

    #[test]
    fn a_forced_yuv_output_from_an_rgb_source_gets_unspecified_matrix_and_range() {
        let (_folder, opened) = opened("rgb24.nut", Some(Carried::Yuv420p));

        assert_eq!(opened.format().colour.matrix, Matrix::Unspecified);
        assert_eq!(opened.format().colour.range, ColourRange::Unspecified);
    }

    #[test]
    fn an_error_opening_a_missing_file_carries_the_whole_ffms2_message() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let args = FileArgs {
            path: folder.path().join("missing.mkv"),
            track: None,
            index_dir: None,
            format: None,
            threads: None,
        };

        let opened = File::open(&args);
        let error = opened.err().expect("the file is missing");
        let message = error.to_string();

        assert_eq!(message.matches("missing.mkv").count(), 2, "{message:?}");
        assert!(!message.contains('\0'), "{message:?}");
    }

    #[cfg(unix)]
    #[test]
    fn a_non_utf8_index_dir_still_opens_the_file() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let (folder, source) = copied("yuv420p.mkv");
        let bytes = [0x66, 0x6f, 0x80, 0x6f];
        let index_dir = folder.path().join(OsStr::from_bytes(&bytes));
        std::fs::create_dir(&index_dir).expect("creates the index dir");
        let args = FileArgs {
            path: source,
            track: None,
            index_dir: Some(index_dir),
            format: None,
            threads: None,
        };

        let opened = File::open(&args);

        assert!(opened.is_ok());
    }

    #[test]
    fn a_frame_whose_resolution_does_not_match_the_output_is_an_error() {
        let expected = FrameShape {
            width: 64,
            height: 64,
            format_id: 7,
        };
        let actual = FrameShape {
            width: 32,
            height: 32,
            format_id: 7,
        };
        let path = PathBuf::from("/nonexistent");

        let error = validate_decoded_frame(actual, expected, &path, 3).expect_err("mismatched resolution");

        assert!(error.to_string().contains('3'));
    }

    #[test]
    fn a_frame_whose_converted_format_does_not_match_the_output_is_an_error() {
        let expected = FrameShape {
            width: 64,
            height: 64,
            format_id: 7,
        };
        let actual = FrameShape {
            width: 64,
            height: 64,
            format_id: 1,
        };
        let path = PathBuf::from("/nonexistent");

        let error = validate_decoded_frame(actual, expected, &path, 3).expect_err("mismatched pixel format");

        assert!(error.to_string().contains('3'));
    }

    #[test]
    fn a_frame_matching_the_output_passes_validation() {
        let expected = FrameShape {
            width: 64,
            height: 64,
            format_id: 7,
        };
        let path = PathBuf::from("/nonexistent");

        assert!(validate_decoded_frame(expected, expected, &path, 3).is_ok());
    }

    #[test]
    fn a_plane_with_zero_row_bytes_is_an_error_not_a_silent_no_op() {
        let format = FrameFormat {
            width: 0,
            height: 64,
            layout: FrameLayout::Grey,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        };
        let path = PathBuf::from("/nonexistent");

        let error = plane_extent(&format, Channel::Y, &path, 0).expect_err("zero row bytes");

        assert!(error.to_string().contains('0'));
    }

    #[test]
    fn a_track_or_thread_count_above_i32_max_is_refused() {
        let too_large = i32::MAX as usize + 1;
        let (_folder, source) = copied("yuv420p.mkv");
        let track_args = FileArgs {
            path: source.clone(),
            track: Some(too_large),
            index_dir: None,
            format: None,
            threads: None,
        };
        let threads_args = FileArgs {
            path: source,
            track: None,
            index_dir: None,
            format: None,
            threads: Some(too_large),
        };

        let track_error = File::open(&track_args).err().expect("the track is too large");
        let threads_error = File::open(&threads_args)
            .err()
            .expect("the thread count is too large");

        assert!(track_error.to_string().contains("track"));
        assert!(threads_error.to_string().contains("threads"));
    }

    #[test]
    fn a_negative_pts_gives_no_timestamp() {
        assert_eq!(timestamp_of(-1, 1000, 1), None);
    }

    #[test]
    fn an_overflowing_pts_gives_no_timestamp_instead_of_panicking() {
        assert_eq!(timestamp_of(i64::MAX, i64::MAX, 1), None);
    }

    #[test]
    fn a_file_reports_its_field_order() {
        let (_progressive_folder, progressive) = opened("yuv420p.mkv", None);
        let (_top_folder, top_first) = opened("interlaced_tff.mkv", None);
        let (_bottom_folder, bottom_first) = opened("interlaced_bff.mkv", None);

        assert_eq!(progressive.format().field_order, Some(FieldOrder::Progressive));
        assert_eq!(top_first.format().field_order, Some(FieldOrder::TopFirst));
        assert_eq!(bottom_first.format().field_order, Some(FieldOrder::BottomFirst));
    }
}
