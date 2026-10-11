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
use avtbx_bestsource::{
    BestSourceError,
    Frame,
    IndexCache,
    OpenOptions,
    PlanarTarget,
    PlaneTarget,
    VideoFormat,
    VideoProperties,
    VideoSource,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::format::{describe_format, mapped_format, plane_channels};
use crate::shared::{RawColour, colour_of, field_order_of, produced_by};

/// Arguments for `bestsource/file`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileArgs {
    /// The media file to decode.
    pub(crate) path: PathBuf,
    /// The stream index of the video track to decode, the first video track by default.
    pub(crate) track: Option<usize>,
    /// A folder to keep the index in, beside the source file by default.
    pub(crate) index_dir: Option<PathBuf>,
    /// Decoder threads, left to BestSource by default.
    pub(crate) threads: Option<usize>,
}

/// A video track opened and ready to decode.
pub(crate) struct Opened {
    video: VideoSource,
    format: FrameFormat,
    /// The format of the first frame, which every later frame must match.
    video_format: VideoFormat,
    /// The options the track was opened with, kept to open seekers.
    options: OpenOptions,
    path: PathBuf,
    time_base: (i64, i64),
    next: i64,
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
        "bestsource"
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
        let options = open_options(args)?;
        let created = VideoSource::open(&args.path, &options);
        let mut video = created.map_err(|error| open_error(&args.path, &error))?;

        let properties = *video.properties();
        if properties.num_frames < 1 {
            return Err(no_frames_error(&args.path));
        }

        let probed = video.frame(0);
        let probe = probed.map_err(|error| frame_error(&args.path, 0, &error))?;
        let format = format_of(&args.path, &properties, &probe)?;
        let video_format = probe.format();
        log_trimmed_size(&args.path, &probe);
        drop(probe);

        Ok(Opened {
            video,
            format,
            video_format,
            options,
            path: args.path.clone(),
            time_base: properties.time_base,
            next: 0,
        })
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let produced = produced_by(opened.format());

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
        let created = VideoSource::open(&opened.path, &opened.options);
        let video = created.map_err(|error| open_error(&opened.path, &error))?;

        let seeker = FileSeeker {
            video,
            format: opened.format,
            video_format: opened.video_format,
            path: opened.path.clone(),
            time_base: opened.time_base,
            pool: HostPool::new(),
        };

        Ok(Some(Box::new(seeker)))
    }
}

/// A source decoding a BestSource video track in order.
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
        let total_frames = opened.video.properties().num_frames;

        if opened.next >= total_frames {
            return Ok(None);
        }

        let current = opened.next;
        let decoded = opened.video.frame(current);
        let frame = decoded.map_err(|error| frame_error(&opened.path, current, &error))?;
        let meta = meta_of(&frame, current, opened.time_base);
        let mut host_frame = context.alloc_host_frame(&opened.format, meta);
        let expected = Expected {
            format: &opened.format,
            video_format: &opened.video_format,
        };
        export_into(&frame, expected, &opened.path, current, &mut host_frame)?;

        opened.next += 1;

        Ok(Some(host_frame.freeze()))
    }
}

/// Reads any frame of a file through a decoder of its own.
pub(crate) struct FileSeeker {
    video: VideoSource,
    format: FrameFormat,
    video_format: VideoFormat,
    path: PathBuf,
    time_base: (i64, i64),
    pool: HostPool,
}

impl Seeker for FileSeeker {
    fn format(&self) -> &FrameFormat {
        &self.format
    }

    fn frame_count(&self) -> usize {
        let count = self.video.properties().num_frames;

        usize::try_from(count).unwrap_or(0)
    }

    fn read(&mut self, index: usize) -> Result<HostFrame, StageError> {
        let count = self.frame_count();
        if index >= count {
            return Err(past_end_error(&self.path, index, count));
        }

        let Ok(current) = i64::try_from(index) else {
            return Err(past_end_error(&self.path, index, count));
        };

        let decoded = self.video.frame(current);
        let frame = decoded.map_err(|error| frame_error(&self.path, current, &error))?;
        let meta = meta_of(&frame, current, self.time_base);
        let mut host_frame = self.pool.alloc_frame(&self.format, meta);
        let expected = Expected {
            format: &self.format,
            video_format: &self.video_format,
        };
        export_into(&frame, expected, &self.path, current, &mut host_frame)?;

        Ok(host_frame.freeze())
    }
}

fn open_options(args: &FileArgs) -> Result<OpenOptions, StageError> {
    let track = match args.track {
        Some(track) => to_i32("track", track)?,
        None => -1,
    };
    let threads = match args.threads {
        Some(threads) => to_i32("threads", threads)?,
        None => 0,
    };
    let cache = match &args.index_dir {
        Some(index_dir) => IndexCache::Directory(index_dir.clone()),
        None => IndexCache::BesideSource,
    };

    Ok(OpenOptions {
        track,
        threads,
        cache,
    })
}

/// The format every frame decodes as, from the track's properties and its first frame.
fn format_of(path: &Path, properties: &VideoProperties, probe: &Frame) -> Result<FrameFormat, StageError> {
    let video_format = probe.format();
    let Some((layout, sample)) = mapped_format(&video_format) else {
        return Err(unsupported_format_error(path, &video_format));
    };

    let width = dimension(path, probe.ss_mod_width())?;
    let height = dimension(path, probe.ss_mod_height())?;
    let raw_colour = RawColour {
        range: probe.range(),
        matrix: probe.matrix(),
        primaries: probe.primaries(),
        transfer: probe.transfer(),
        chroma_location: probe.chroma_location(),
    };
    let colour = colour_of(raw_colour, false, Some(layout), layout);
    let field_order = field_order_of(probe.interlaced(), probe.top_field_first());
    let frame_rate = frame_rate_of(properties.fps);

    Ok(FrameFormat {
        width,
        height,
        layout,
        sample,
        colour,
        frame_rate,
        field_order: Some(field_order),
    })
}

/// What every decoded frame of a track must look like.
#[derive(Clone, Copy)]
struct Expected<'a> {
    format: &'a FrameFormat,
    video_format: &'a VideoFormat,
}

/// Copies a decoded frame into a host frame's planes, channel by channel.
fn export_into(
    frame: &Frame,
    expected: Expected<'_>,
    path: &Path,
    current: i64,
    host_frame: &mut HostFrameMut,
) -> Result<(), StageError> {
    let format = expected.format;
    let video_format = expected.video_format;
    if !same_shape(video_format, format.width, format.height, frame) {
        let decoded = frame.format();
        let decoded_size = (frame.ss_mod_width(), frame.ss_mod_height());
        let expected_size = (format.width, format.height);

        return Err(changed_format_error(
            path,
            current,
            (&decoded, decoded_size),
            (video_format, expected_size),
        ));
    }

    let channels = plane_channels(format.layout);
    let mut strides = Vec::with_capacity(channels.len());
    for channel in channels {
        let stride = row_bytes(format, *channel, path, current)?;
        strides.push(stride);
    }

    let mut host_planes = host_frame.planes_mut();
    let mut planes = Vec::with_capacity(channels.len());
    for (channel, stride) in channels.iter().zip(strides) {
        let position = host_planes.iter().position(|(carried, _)| carried == channel);
        let Some(position) = position else {
            return Err(missing_plane_error(path, current, *channel));
        };

        let (_, data) = host_planes.swap_remove(position);
        planes.push(PlaneTarget { data, stride });
    }

    let mut target = PlanarTarget { planes, alpha: None };
    let exported = frame.export_planar(&mut target);

    exported.map_err(|error| frame_error(path, current, &error))
}

/// Whether a decoded frame has the format and size the track opened with.
fn same_shape(expected: &VideoFormat, width: u32, height: u32, frame: &Frame) -> bool {
    let actual = frame.format();

    shapes_match(
        expected,
        (width, height),
        &actual,
        (frame.ss_mod_width(), frame.ss_mod_height()),
    )
}

fn shapes_match(
    expected: &VideoFormat,
    expected_size: (u32, u32),
    actual: &VideoFormat,
    actual_size: (i32, i32),
) -> bool {
    let same_width = i64::from(expected_size.0) == i64::from(actual_size.0);
    let same_height = i64::from(expected_size.1) == i64::from(actual_size.1);

    expected == actual && same_width && same_height
}

/// The bytes in each row of a channel's plane.
fn row_bytes(format: &FrameFormat, channel: Channel, path: &Path, current: i64) -> Result<usize, StageError> {
    let dimensions = format.channel_dimensions(channel);
    let Some((width, _)) = dimensions else {
        return Err(missing_plane_error(path, current, channel));
    };

    Ok(width as usize * format.sample.bytes())
}

/// The metadata of frame `current`, with its timestamp when the track has one.
fn meta_of(frame: &Frame, current: i64, time_base: (i64, i64)) -> FrameMeta {
    let index = current.unsigned_abs();
    let timestamp = timestamp_of(frame.pts(), time_base);

    FrameMeta {
        timestamp,
        ..FrameMeta::at(index)
    }
}

/// A frame's timestamp from its PTS and the track's time base in seconds.
///
/// A negative PTS or an overflow anywhere in the scaling gives no timestamp rather than wrapping or
/// panicking.
fn timestamp_of(pts: i64, (numerator, denominator): (i64, i64)) -> Option<Duration> {
    if denominator <= 0 {
        return None;
    }

    let pts = u128::try_from(pts).ok()?;
    let numerator = u128::try_from(numerator).ok()?;
    let denominator = u128::try_from(denominator).ok()?;

    let scaled = pts.checked_mul(numerator)?;
    let scaled = scaled.checked_mul(1_000_000_000)?;
    let nanos = scaled / denominator;
    let nanos = u64::try_from(nanos).ok()?;

    Some(Duration::from_nanos(nanos))
}

/// The frame rate BestSource reports, or `None` when either part is zero, negative or too large.
fn frame_rate_of((numerator, denominator): (i64, i64)) -> Option<FrameRate> {
    let numerator = u32::try_from(numerator).ok()?;
    let denominator = u32::try_from(denominator).ok()?;

    FrameRate::new(numerator, denominator)
}

/// Says once when the exported size trims the coded size to a whole multiple of the subsampling.
fn log_trimmed_size(path: &Path, probe: &Frame) {
    let trimmed = probe.ss_mod_width() != probe.width() || probe.ss_mod_height() != probe.height();
    if !trimmed {
        return;
    }

    tracing::info!(
        path = %path.display(),
        coded_width = probe.width(),
        coded_height = probe.height(),
        width = probe.ss_mod_width(),
        height = probe.ss_mod_height(),
        "bestsource trims the frame to a whole multiple of its chroma subsampling"
    );
}

fn dimension(path: &Path, value: i32) -> Result<u32, StageError> {
    match u32::try_from(value) {
        Ok(converted) => Ok(converted),
        Err(_) => Err(negative_size_error(path, value)),
    }
}

/// Converts an argument to the `i32` BestSource takes, failing above `i32::MAX`.
fn to_i32(name: &str, value: usize) -> Result<i32, StageError> {
    match i32::try_from(value) {
        Ok(converted) => Ok(converted),
        Err(_) => Err(argument_too_large_error(name, value)),
    }
}

fn argument_too_large_error(name: &str, value: usize) -> StageError {
    let limit = i32::MAX;
    let message = format!("{name} is {value}, above the most bestsource accepts of {limit}");

    StageError::msg(message)
}

fn open_error(path: &Path, error: &BestSourceError) -> StageError {
    let display = path.display();
    let message = format!("bestsource could not open {display}: {error}");

    StageError::msg(message)
}

fn frame_error(path: &Path, frame: i64, error: &BestSourceError) -> StageError {
    let display = path.display();
    let message = format!("bestsource could not decode frame {frame} of {display}: {error}");

    StageError::msg(message)
}

fn no_frames_error(path: &Path) -> StageError {
    let display = path.display();
    let message = format!("bestsource found no frames in {display}");

    StageError::msg(message)
}

fn unsupported_format_error(path: &Path, format: &VideoFormat) -> StageError {
    let display = path.display();
    let described = describe_format(format);
    let message = format!(
        "bestsource decodes {display} as {described}, which the pipeline does not carry. \
         Use ffms2/file to convert it on decode."
    );

    StageError::msg(message)
}

fn changed_format_error(
    path: &Path,
    frame: i64,
    (decoded, decoded_size): (&VideoFormat, (i32, i32)),
    (opened, opened_size): (&VideoFormat, (u32, u32)),
) -> StageError {
    let display = path.display();
    let decoded_text = describe_format(decoded);
    let opened_text = describe_format(opened);
    let (decoded_width, decoded_height) = decoded_size;
    let (opened_width, opened_height) = opened_size;
    let message = format!(
        "frame {frame} of {display} decodes as {decoded_text} {decoded_width}x{decoded_height}, \
         but the track opened as {opened_text} {opened_width}x{opened_height}, \
         which bestsource/file cannot carry. Use ffms2/file to convert it on decode."
    );

    StageError::msg(message)
}

fn negative_size_error(path: &Path, value: i32) -> StageError {
    let display = path.display();
    let message = format!("bestsource reported a size of {value} for {display}");

    StageError::msg(message)
}

fn missing_plane_error(path: &Path, frame: i64, channel: Channel) -> StageError {
    let display = path.display();
    let message = format!("bestsource frame {frame} of {display} is missing its {channel} plane");

    StageError::msg(message)
}

fn past_end_error(path: &Path, index: usize, count: usize) -> StageError {
    let display = path.display();
    let message = format!("{display} has {count} frames, so frame {index} cannot be read");

    StageError::msg(message)
}

#[cfg(test)]
mod tests {
    use avtbx_bestsource::ColorFamily;

    use super::*;

    fn yuv420(bits: i32) -> VideoFormat {
        VideoFormat {
            color_family: ColorFamily::Yuv,
            alpha: false,
            float: false,
            bits,
            sub_sampling_w: 1,
            sub_sampling_h: 1,
        }
    }

    #[test]
    fn an_identical_shape_passes() {
        let format = yuv420(8);

        assert!(shapes_match(&format, (64, 48), &format, (64, 48)));
    }

    #[test]
    fn a_bit_depth_change_is_refused() {
        let opened = yuv420(8);
        let decoded = yuv420(10);

        assert!(!shapes_match(&opened, (64, 48), &decoded, (64, 48)));
    }

    #[test]
    fn a_size_change_is_refused() {
        let format = yuv420(8);

        assert!(!shapes_match(&format, (64, 48), &format, (32, 48)));
        assert!(!shapes_match(&format, (64, 48), &format, (64, 24)));
    }

    #[test]
    fn a_pts_scales_by_the_time_base_in_seconds() {
        let timestamp = timestamp_of(3, (1, 25));

        assert_eq!(timestamp, Some(Duration::from_millis(120)));
    }

    #[test]
    fn a_negative_pts_gives_no_timestamp() {
        assert_eq!(timestamp_of(-1, (1, 25)), None);
    }

    #[test]
    fn an_overflowing_pts_gives_no_timestamp_instead_of_panicking() {
        assert_eq!(timestamp_of(i64::MAX, (i64::MAX, 1)), None);
    }

    #[test]
    fn a_zero_or_negative_rate_is_no_rate() {
        assert_eq!(frame_rate_of((0, 1)), None);
        assert_eq!(frame_rate_of((25, 0)), None);
        assert_eq!(frame_rate_of((-25, 1)), None);
    }
}
