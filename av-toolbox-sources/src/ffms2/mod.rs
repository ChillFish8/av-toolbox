mod format;
mod index;

use std::path::{Path, PathBuf};
use std::slice;
use std::time::Duration;

use av_toolbox_core::{
    BoundInputs,
    InputSignature,
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
    ColourInfo,
    ColourRange,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostFrame,
    HostFrameMut,
    HostSide,
    Matrix,
    OutputFormats,
    Primaries,
    ResolvedFormats,
    Runtime,
    Transfer,
};
use ffms2::frame::{Frame, Resizers};
use ffms2::track::{Track, TrackType};
use ffms2::video::{SeekMode, VideoSource};
use schemars::JsonSchema;
use serde::Deserialize;

use self::format::{Carried, Choice};
use crate::shared::produced_by;

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

/// A video track opened and ready to decode.
pub(crate) struct Opened {
    video: VideoSource,
    track: Track,
    format: FrameFormat,
    /// The ffmpeg pixel format id every decoded frame is expected to convert into.
    output_id: i32,
    path: PathBuf,
    next: usize,
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
        require_utf8_path(&args.path)?;

        let index_path = index::index_path(&args.path, args.index_dir.as_deref())?;
        require_utf8_path(&index_path)?;

        let index = index::load_or_build(&args.path, args.index_dir.as_deref())?;

        let track_index = match args.track {
            Some(track) => track,
            None => {
                let found = index.FirstTrackOfType(TrackType::TYPE_VIDEO);

                found.map_err(|error| track_error(&args.path, error))?
            },
        };

        let threads = args.threads.unwrap_or(0);
        // SEEK_LINEAR_NO_RW refuses to seek at all, and ffms2 needs to seek to reach even frame
        // 0 of a regular seekable container, so this needs SEEK_NORMAL instead.
        let created = VideoSource::new(&args.path, track_index, &index, threads, SeekMode::SEEK_NORMAL);
        let mut video = created.map_err(|error| open_error(&args.path, error))?;
        let track = Track::TrackFromVideo(&mut video);

        let probed = Frame::GetFrame(&mut video, 0);
        let probe = probed.map_err(|error| frame_error(&args.path, 0, error))?;

        let resolved_name = pixel_format_name(probe.EncodedPixelFormat);
        let choice = resolve_choice(&args.path, probe.EncodedPixelFormat, resolved_name, args.format)?;

        let resolution = probe.get_frame_resolution();
        let width = resolution.width as usize;
        let height = resolution.height as usize;
        let output_name = choice.output.ffmpeg_name();
        let output_id = Frame::GetPixFmt(output_name);
        let mut targets = vec![output_id];
        let configured = video.SetOutputFormatV2(&mut targets, width, height, Resizers::RESIZER_BICUBIC);
        configured.map_err(|error| open_error(&args.path, error))?;

        let (layout, sample) = choice.output.layout();
        let source_layout = resolved_name.and_then(format::mapped_layout);
        let colour = colour_of(
            probe.ColorRange,
            probe.ColorSpace,
            probe.ColorPrimaries,
            probe.TransferCharateristics,
            choice.full_range,
            source_layout,
            layout,
        );
        let format = FrameFormat {
            width: resolution.width as u32,
            height: resolution.height as u32,
            layout,
            sample,
            colour,
        };

        Ok(Opened {
            video,
            track,
            format,
            output_id,
            path: args.path.clone(),
            next: 0,
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
        let total_frames = self.opened.track.NumFrames();

        if self.opened.next == total_frames {
            return Ok(None);
        }

        let current = self.opened.next;
        let decoded = Frame::GetFrame(&mut self.opened.video, current);
        let frame = decoded.map_err(|error| frame_error(&self.opened.path, current, error))?;
        validate_decoded_frame(
            &frame,
            self.opened.format.width,
            self.opened.format.height,
            self.opened.output_id,
            &self.opened.path,
            current,
        )?;

        let frame_info = self.opened.track.FrameInfo(current);
        let time_base = self.opened.track.TimeBase();
        let timestamp = timestamp_of(frame_info.PTS, time_base.Num, time_base.Den);

        let meta = FrameMeta {
            timestamp,
            ..FrameMeta::at(current as u64)
        };
        let mut host_frame = context.alloc_host_frame(&self.opened.format, meta);

        let plane_channels = ffmpeg_plane_channels(self.opened.format.layout);
        for (plane_index, channel) in plane_channels.iter().enumerate() {
            copy_plane(
                &mut host_frame,
                &self.opened.format,
                &frame,
                plane_index,
                *channel,
                &self.opened.path,
                current,
            )?;
        }

        self.opened.next += 1;

        Ok(Some(host_frame.freeze()))
    }
}

/// Fails unless a decoded frame's own resolution and converted pixel format match what
/// `SetOutputFormatV2` was asked for.
///
/// `next` calls this before any plane is sliced, so `copy_plane`'s row and stride arithmetic can
/// trust the frame actually holds the output format it was built from.
fn validate_decoded_frame(
    frame: &Frame,
    expected_width: u32,
    expected_height: u32,
    expected_format_id: i32,
    path: &Path,
    frame_index: usize,
) -> Result<(), StageError> {
    let resolution = frame.get_frame_resolution();
    let width_matches = resolution.width as u32 == expected_width;
    let height_matches = resolution.height as u32 == expected_height;
    let format_matches = frame.ConvertedPixelFormat == expected_format_id;

    if width_matches && height_matches && format_matches {
        return Ok(());
    }

    Err(unexpected_frame_error(path, frame_index))
}

/// Copies one ffmpeg plane into a host frame's matching channel, row by row.
///
/// Reads straight from the frame's own `Data` pointer and `Linesize`, sized from our own output
/// format rather than ffms2's `get_pixel_data`, which sizes every plane from the encoded format's
/// chroma subsampling and not the output format `SetOutputFormatV2` actually wrote.
fn copy_plane(
    host_frame: &mut HostFrameMut,
    format: &FrameFormat,
    frame: &Frame,
    plane_index: usize,
    channel: Channel,
    path: &Path,
    frame_index: usize,
) -> Result<(), StageError> {
    let dimensions = format.channel_dimensions(channel);
    let (width, height) = dimensions.ok_or_else(|| missing_plane_error(path, frame_index, channel))?;
    let row_bytes = width as usize * format.sample.bytes();
    let rows = height as usize;

    if rows == 0 || row_bytes == 0 {
        return Err(missing_plane_error(path, frame_index, channel));
    }

    let pointer = frame.Data[plane_index];
    let linesize = frame.Linesize[plane_index];
    let large_enough_stride = linesize >= 0 && (linesize as usize) >= row_bytes;

    if pointer.is_null() || !large_enough_stride {
        return Err(missing_plane_error(path, frame_index, channel));
    }

    let linesize = linesize as usize;
    let span = linesize * (rows - 1) + row_bytes;

    // SAFETY: `next` has already checked this frame's resolution and converted pixel format
    // against the output `SetOutputFormatV2` was asked for, so ffms2 holds `rows` strided rows of
    // exactly this output format here, valid until the video source's next `GetFrame` call.
    let source = unsafe { slice::from_raw_parts(pointer, span) };

    let destination = host_frame
        .plane_mut(channel)
        .ok_or_else(|| missing_plane_error(path, frame_index, channel))?;

    let source_rows = source.chunks(linesize);
    let destination_rows = destination.chunks_mut(row_bytes);
    for (source_row, destination_row) in source_rows.zip(destination_rows).take(rows) {
        destination_row.copy_from_slice(&source_row[..row_bytes]);
    }

    Ok(())
}

/// The channel each ffmpeg plane maps to, in ffmpeg's own plane order.
fn ffmpeg_plane_channels(layout: FrameLayout) -> &'static [Channel] {
    match layout {
        FrameLayout::Yuv { .. } => &[Channel::Y, Channel::U, Channel::V],
        FrameLayout::Grey => &[Channel::Y],
        FrameLayout::Rgb => &[Channel::G, Channel::B, Channel::R],
    }
}

/// The ffmpeg pixel format name a frame was decoded from, matched by id.
fn pixel_format_name(id: i32) -> Option<&'static str> {
    let names = format::known_names();
    let matched = names.iter().find(|name| {
        let candidate = Frame::GetPixFmt(name);

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

/// The colour a frame carries, mapped from its ffms2 properties.
///
/// A forced format that crosses the RGB/YUV family boundary overrides the source's matrix and
/// range, which no longer describe the samples once ffms2 has recoloured them: an RGB output made
/// from a YUV source is identity matrix at full range, and a YUV output made from an RGB source
/// carries no matrix or range at all.
fn colour_of(
    range: i32,
    matrix: i32,
    primaries: i32,
    transfer: i32,
    full_range: bool,
    source_layout: Option<FrameLayout>,
    output_layout: FrameLayout,
) -> ColourInfo {
    let mapped = ColourInfo {
        range: range_of(range, full_range),
        matrix: matrix_of(matrix),
        primaries: primaries_of(primaries),
        transfer: transfer_of(transfer),
    };

    match (source_layout, output_layout) {
        (Some(FrameLayout::Yuv { .. }), FrameLayout::Rgb) => ColourInfo {
            matrix: Matrix::Identity,
            range: ColourRange::Full,
            ..mapped
        },
        (Some(FrameLayout::Rgb), FrameLayout::Yuv { .. }) => ColourInfo {
            matrix: Matrix::Unspecified,
            range: ColourRange::Unspecified,
            ..mapped
        },
        _ => mapped,
    }
}

fn range_of(value: i32, full_range: bool) -> ColourRange {
    if full_range {
        return ColourRange::Full;
    }

    match value {
        1 => ColourRange::Limited,
        2 => ColourRange::Full,
        _ => ColourRange::Unspecified,
    }
}

fn matrix_of(value: i32) -> Matrix {
    match value {
        0 => Matrix::Identity,
        1 => Matrix::Bt709,
        5 | 6 => Matrix::Bt601,
        9 => Matrix::Bt2020Ncl,
        _ => Matrix::Unspecified,
    }
}

fn primaries_of(value: i32) -> Primaries {
    match value {
        1 => Primaries::Bt709,
        5 | 6 => Primaries::Bt601,
        9 => Primaries::Bt2020,
        _ => Primaries::Unspecified,
    }
}

fn transfer_of(value: i32) -> Transfer {
    match value {
        1 => Transfer::Bt709,
        13 => Transfer::Srgb,
        16 => Transfer::Pq,
        18 => Transfer::Hlg,
        _ => Transfer::Unspecified,
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

    let scaled = u128::from(pts).checked_mul(u128::from(numerator))?;
    let scaled = scaled.checked_mul(1_000_000)?;
    let nanos = scaled / u128::from(denominator);
    let nanos = u64::try_from(nanos).ok()?;

    Some(Duration::from_nanos(nanos))
}

/// Fails when `path` is not a UTF-8 string ffms2 can turn into a `CString`.
///
/// ffms2 converts every path it is given with `CString::new(path.to_str().unwrap())`, so a path
/// that is not valid UTF-8, or one that is but carries an interior NUL byte, panics inside it.
fn require_utf8_path(path: &Path) -> Result<(), StageError> {
    let Some(text) = path.to_str() else {
        let display = path.display();
        let message = format!("{display} is not valid utf-8, which ffms2 requires");

        return Err(StageError::msg(message));
    };

    if text.contains('\0') {
        let message = format!("{text} carries a nul byte, which ffms2 cannot turn into a c string");

        return Err(StageError::msg(message));
    }

    Ok(())
}

fn open_error(path: &Path, error: ffms2::Error) -> StageError {
    let display = path.display();
    let message = format!("ffms2 could not open {display}: {error:?}");

    StageError::msg(message)
}

fn track_error(path: &Path, error: ffms2::Error) -> StageError {
    let display = path.display();
    let message = format!("ffms2 could not find a video track in {display}: {error:?}");

    StageError::msg(message)
}

fn frame_error(path: &Path, frame: usize, error: ffms2::Error) -> StageError {
    let display = path.display();
    let message = format!("ffms2 could not decode frame {frame} of {display}: {error:?}");

    StageError::msg(message)
}

fn unresolved_pixel_format_error(path: &Path, id: i32) -> StageError {
    let display = path.display();
    let message = format!(
        "ffms2 decodes {display} as pixel format {id}, which has no carried equivalent. Pass format to choose one."
    );

    StageError::msg(message)
}

fn unexpected_frame_error(path: &Path, frame: usize) -> StageError {
    let display = path.display();
    let message = format!(
        "ffms2 frame {frame} of {display} does not match the resolution or pixel format SetOutputFormatV2 was asked for"
    );

    StageError::msg(message)
}

fn missing_plane_error(path: &Path, frame: usize, channel: Channel) -> StageError {
    let display = path.display();
    let message = format!("ffms2 frame {frame} of {display} is missing its {channel} plane");

    StageError::msg(message)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use av_toolbox_gpu::{Channel, ColourRange, FrameLayout, SampleFormat, Subsampling, testing_context};

    use super::*;

    fn opened(name: &str, format: Option<Carried>) -> (tempfile::TempDir, Opened) {
        let folder = tempfile::tempdir().expect("a temp dir");
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let source = folder.path().join(name);
        std::fs::copy(manifest.join("tests").join("media").join(name), &source).expect("copies");
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
    fn a_non_utf8_path_is_refused_before_touching_ffms2() {
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
    fn a_path_with_an_interior_nul_byte_is_refused_before_touching_ffms2() {
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
    fn a_frame_whose_resolution_does_not_match_the_output_is_an_error() {
        let mut frame = Frame::default();
        frame.EncodedWidth = 32;
        frame.EncodedHeight = 32;
        frame.ScaledWidth = -1;
        frame.ScaledHeight = -1;
        frame.ConvertedPixelFormat = 7;
        let path = PathBuf::from("/nonexistent");

        let error = validate_decoded_frame(&frame, 64, 64, 7, &path, 3).expect_err("mismatched resolution");

        assert!(error.to_string().contains('3'));
    }

    #[test]
    fn a_frame_whose_converted_format_does_not_match_the_output_is_an_error() {
        let mut frame = Frame::default();
        frame.EncodedWidth = 64;
        frame.EncodedHeight = 64;
        frame.ScaledWidth = -1;
        frame.ScaledHeight = -1;
        frame.ConvertedPixelFormat = 1;
        let path = PathBuf::from("/nonexistent");

        let error = validate_decoded_frame(&frame, 64, 64, 7, &path, 3).expect_err("mismatched pixel format");

        assert!(error.to_string().contains('3'));
    }

    #[test]
    fn a_frame_matching_the_output_passes_validation() {
        let mut frame = Frame::default();
        frame.EncodedWidth = 64;
        frame.EncodedHeight = 64;
        frame.ScaledWidth = -1;
        frame.ScaledHeight = -1;
        frame.ConvertedPixelFormat = 7;
        let path = PathBuf::from("/nonexistent");

        assert!(validate_decoded_frame(&frame, 64, 64, 7, &path, 3).is_ok());
    }

    #[test]
    fn a_plane_with_zero_row_bytes_is_an_error_not_a_silent_no_op() {
        let context = testing_context();
        let format = FrameFormat {
            width: 0,
            height: 64,
            layout: FrameLayout::Grey,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
        };
        let mut host_frame = context.alloc_host_frame(&format, FrameMeta::at(0));
        let frame = Frame::default();
        let path = PathBuf::from("/nonexistent");

        let error = copy_plane(&mut host_frame, &format, &frame, 0, Channel::Y, &path, 0)
            .expect_err("zero row bytes");

        assert!(error.to_string().contains('0'));
    }

    #[test]
    fn a_negative_pts_gives_no_timestamp() {
        assert_eq!(timestamp_of(-1, 1000, 1), None);
    }

    #[test]
    fn an_overflowing_pts_gives_no_timestamp_instead_of_panicking() {
        assert_eq!(timestamp_of(i64::MAX, i64::MAX, 1), None);
    }
}
