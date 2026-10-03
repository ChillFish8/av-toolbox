use std::path::Path;
use std::ptr::NonNull;
use std::slice;

use ffms2_sys::{FFMS_Frame, FFMS_Resizers, FFMS_SeekMode, FFMS_Track, FFMS_VideoSource};

use super::error::{ErrorBuffer, Ffms2Error};
use super::index::Index;
use super::{c_path, init};

/// A decoder for one video track of a source file.
pub(crate) struct VideoSource {
    handle: NonNull<FFMS_VideoSource>,
    track: NonNull<FFMS_Track>,
}

// SAFETY: FFMS2 handles are not tied to the thread that created them, and the track is owned by
// the video source it moves with.
unsafe impl Send for VideoSource {}

impl VideoSource {
    /// Opens a video track for decoding in any order.
    ///
    /// The seek mode is always `SEEK_NORMAL`. `SEEK_LINEAR_NO_RW` fails the first request for
    /// frame 0, since FFMS2 has already decoded past it while opening the track.
    pub(crate) fn open(source: &Path, track: i32, index: &Index, threads: i32) -> Result<Self, Ffms2Error> {
        init();

        let c_source = c_path(source)?;
        let seek_mode = FFMS_SeekMode::FFMS_SEEK_NORMAL as i32;
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: `c_source`, `index` and `info` outlive the call, and FFMS2 copies what it needs
        // from the index.
        let raw = unsafe {
            ffms2_sys::FFMS_CreateVideoSource(
                c_source.as_ptr(),
                track,
                index.as_ptr(),
                threads,
                seek_mode,
                info,
            )
        };
        let handle = NonNull::new(raw).ok_or_else(|| errors.into_error())?;

        // SAFETY: `handle` is a live video source, which owns the track it returns.
        let raw_track = unsafe { ffms2_sys::FFMS_GetTrackFromVideo(handle.as_ptr()) };

        let Some(track) = NonNull::new(raw_track) else {
            // SAFETY: `handle` is a live video source that nothing else holds.
            unsafe { ffms2_sys::FFMS_DestroyVideoSource(handle.as_ptr()) };

            let message = "ffms2 gave no track for the video source".to_string();

            return Err(Ffms2Error::local(message));
        };

        Ok(Self { handle, track })
    }

    /// Converts every later frame to `format_id` at the given size.
    pub(crate) fn set_output_format(
        &mut self,
        format_id: i32,
        width: i32,
        height: i32,
    ) -> Result<(), Ffms2Error> {
        let targets = [format_id, -1];
        let resizer = FFMS_Resizers::FFMS_RESIZER_BICUBIC as i32;
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: `self.handle` is a live video source, `targets` is `-1` terminated, and both
        // `targets` and `info` outlive the call.
        let status = unsafe {
            ffms2_sys::FFMS_SetOutputFormatV2(
                self.handle.as_ptr(),
                targets.as_ptr(),
                width,
                height,
                resizer,
                info,
            )
        };

        errors.check(status)
    }

    pub(crate) fn num_frames(&self) -> i32 {
        // SAFETY: `self.track` is owned by the live video source.
        unsafe { ffms2_sys::FFMS_GetNumFrames(self.track.as_ptr()) }
    }

    /// A frame's presentation timestamp in time base units, or `None` outside the track.
    pub(crate) fn pts(&self, frame: i32) -> Option<i64> {
        let total_frames = self.num_frames();

        if frame < 0 || frame >= total_frames {
            return None;
        }

        // SAFETY: `self.track` is owned by the live video source, and `frame` is inside the track.
        let raw_info = unsafe { ffms2_sys::FFMS_GetFrameInfo(self.track.as_ptr(), frame) };

        // SAFETY: a non-null `raw_info` is info the track owns for the lifetime of `self`.
        let frame_info = unsafe { raw_info.as_ref() }?;

        Some(frame_info.PTS)
    }

    /// The track's time base as a numerator and denominator.
    ///
    /// Gives `(0, 0)` when FFMS2 has no time base for the track.
    pub(crate) fn time_base(&self) -> (i64, i64) {
        // SAFETY: `self.track` is owned by the live video source, which owns the time base too.
        let raw_time_base = unsafe { ffms2_sys::FFMS_GetTimeBase(self.track.as_ptr()) };

        // SAFETY: a non-null `raw_time_base` is the time base the track owns for the lifetime of
        // `self`.
        let time_base = unsafe { raw_time_base.as_ref() };

        let Some(time_base) = time_base else {
            return (0, 0);
        };

        (time_base.Num, time_base.Den)
    }

    /// The track's frame rate as a numerator and denominator.
    ///
    /// Gives `(0, 0)` when FFMS2 has no properties for the track.
    pub(crate) fn frame_rate(&self) -> (i32, i32) {
        // SAFETY: `self.handle` is a live video source, which owns the properties it returns.
        let raw_properties = unsafe { ffms2_sys::FFMS_GetVideoProperties(self.handle.as_ptr()) };

        // SAFETY: a non-null `raw_properties` is owned by the video source for the lifetime of
        // `self`.
        let properties = unsafe { raw_properties.as_ref() };

        let Some(properties) = properties else {
            return (0, 0);
        };

        (properties.FPSNumerator, properties.FPSDenominator)
    }

    /// Decodes a frame, which stays readable until this source is used again.
    pub(crate) fn frame(&mut self, index: i32) -> Result<Frame<'_>, Ffms2Error> {
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: `self.handle` is a live video source, and `info` outlives the call.
        let raw = unsafe { ffms2_sys::FFMS_GetFrame(self.handle.as_ptr(), index, info) };

        if raw.is_null() {
            return Err(errors.into_error());
        }

        // SAFETY: FFMS2 keeps the frame alive until the next call on this source, which the
        // mutable borrow of `self` rules out.
        let frame = unsafe { &*raw };

        Ok(Frame { frame })
    }
}

impl Drop for VideoSource {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is a live video source that nothing else frees.
        unsafe { ffms2_sys::FFMS_DestroyVideoSource(self.handle.as_ptr()) };
    }
}

/// A decoded frame borrowed from its [VideoSource](crate::ffms2::sys::VideoSource).
pub(crate) struct Frame<'source> {
    frame: &'source FFMS_Frame,
}

impl Frame<'_> {
    pub(crate) fn encoded_pixel_format(&self) -> i32 {
        self.frame.EncodedPixelFormat
    }

    pub(crate) fn converted_pixel_format(&self) -> i32 {
        self.frame.ConvertedPixelFormat
    }

    /// The width after scaling, or the encoded width when FFMS2 did not scale.
    pub(crate) fn width(&self) -> i32 {
        if self.frame.ScaledWidth == -1 {
            return self.frame.EncodedWidth;
        }

        self.frame.ScaledWidth
    }

    /// The height after scaling, or the encoded height when FFMS2 did not scale.
    pub(crate) fn height(&self) -> i32 {
        if self.frame.ScaledHeight == -1 {
            return self.frame.EncodedHeight;
        }

        self.frame.ScaledHeight
    }

    pub(crate) fn colour_range(&self) -> i32 {
        self.frame.ColorRange
    }

    pub(crate) fn colour_space(&self) -> i32 {
        self.frame.ColorSpace
    }

    pub(crate) fn colour_primaries(&self) -> i32 {
        self.frame.ColorPrimaries
    }

    pub(crate) fn transfer(&self) -> i32 {
        self.frame.TransferCharateristics
    }

    /// The frame's chroma location, as ffms2's raw `FFMS_ChromaLocations` value.
    pub(crate) fn chroma_location(&self) -> i32 {
        self.frame.ChromaLocation
    }

    pub(crate) fn interlaced(&self) -> bool {
        self.frame.InterlacedFrame != 0
    }

    pub(crate) fn top_field_first(&self) -> bool {
        self.frame.TopFieldFirst != 0
    }

    /// One plane of the frame as `rows` rows of `row_bytes`.
    ///
    /// Gives `None` when the plane is missing or its stride cannot hold `rows` rows of
    /// `row_bytes`.
    ///
    /// # Safety
    ///
    /// `rows` rows of `row_bytes` must fit this frame's converted format at its resolution.
    pub(crate) unsafe fn plane(
        &self,
        index: usize,
        row_bytes: usize,
        rows: usize,
    ) -> Option<StridedPlane<'_>> {
        let pointer = *self.frame.Data.get(index)?;
        let linesize = *self.frame.Linesize.get(index)?;
        let stride = usize::try_from(linesize).ok()?;

        if pointer.is_null() || stride < row_bytes || rows == 0 || row_bytes == 0 {
            return None;
        }

        let full_rows = rows - 1;
        let padded_span = stride.checked_mul(full_rows)?;
        let span = padded_span.checked_add(row_bytes)?;

        // SAFETY: the caller guarantees the plane holds `rows` rows at this stride.
        let bytes = unsafe { slice::from_raw_parts(pointer, span) };

        Some(StridedPlane {
            bytes,
            stride,
            row_bytes,
        })
    }
}

/// A plane's rows, each followed by stride padding.
pub(crate) struct StridedPlane<'frame> {
    bytes: &'frame [u8],
    stride: usize,
    row_bytes: usize,
}

impl<'frame> StridedPlane<'frame> {
    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Each row without its padding.
    pub(crate) fn rows(&self) -> impl Iterator<Item = &'frame [u8]> + use<'frame> {
        let bytes = self.bytes;
        let row_bytes = self.row_bytes;
        let chunks = bytes.chunks(self.stride);

        chunks.map(move |row| &row[..row_bytes])
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::ffms2::sys::{Indexer, pixel_format_id};

    fn media(name: &str) -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        manifest.join("tests").join("media").join(name)
    }

    fn opened(name: &str) -> VideoSource {
        let source = media(name);
        let indexer = Indexer::new(&source).expect("creates an indexer");
        let index = indexer.index().expect("indexes");

        VideoSource::open(&source, 0, &index, 1).expect("opens")
    }

    #[test]
    fn a_video_track_reports_its_frame_count() {
        let video = opened("yuv420p.mkv");

        assert_eq!(video.num_frames(), 8);
    }

    #[test]
    fn frames_outside_the_track_have_no_pts() {
        let video = opened("yuv420p.mkv");

        assert_eq!(video.pts(-1), None);
        assert_eq!(video.pts(8), None);
    }

    #[test]
    fn later_frames_have_later_pts() {
        let video = opened("yuv420p.mkv");

        let first = video.pts(0).expect("a pts");
        let second = video.pts(1).expect("a pts");

        assert!(second > first);
    }

    #[test]
    fn a_decoded_frame_reports_its_size() {
        let mut video = opened("yuv420p.mkv");

        let frame = video.frame(0).expect("decodes");

        assert_eq!(frame.width(), 64);
        assert_eq!(frame.height(), 64);
        assert_eq!(frame.encoded_pixel_format(), pixel_format_id("yuv420p"));
    }

    #[test]
    fn frames_convert_to_the_output_format() {
        let mut video = opened("yuv420p.mkv");
        let grey_id = pixel_format_id("gray");

        video.set_output_format(grey_id, 64, 64).expect("sets the format");

        let frame = video.frame(3).expect("decodes");

        assert_eq!(frame.converted_pixel_format(), grey_id);
    }

    #[test]
    fn a_plane_yields_every_row_at_the_requested_width() {
        let mut video = opened("yuv420p.mkv");
        let grey_id = pixel_format_id("gray");
        video.set_output_format(grey_id, 64, 64).expect("sets the format");

        let frame = video.frame(0).expect("decodes");

        // SAFETY: a 64x64 gray frame holds 64 rows of 64 bytes in plane 0.
        let plane = unsafe { frame.plane(0, 64, 64) }.expect("a luma plane");
        let rows: Vec<&[u8]> = plane.rows().collect();

        assert_eq!(rows.len(), 64);
        assert!(rows.iter().all(|row| row.len() == 64));
    }

    #[test]
    fn a_plane_with_zero_row_bytes_is_none() {
        let mut video = opened("yuv420p.mkv");
        let frame = video.frame(0).expect("decodes");

        // SAFETY: zero bytes are asked for, so nothing is read.
        let plane = unsafe { frame.plane(0, 0, 64) };

        assert!(plane.is_none());
    }

    #[test]
    fn a_plane_index_past_the_frame_is_none() {
        let mut video = opened("yuv420p.mkv");
        let frame = video.frame(0).expect("decodes");

        // SAFETY: plane 4 does not exist, so nothing is read.
        let plane = unsafe { frame.plane(4, 64, 64) };

        assert!(plane.is_none());
    }

    #[test]
    fn a_strided_plane_skips_its_padding() {
        let bytes = [1, 2, 0, 0, 3, 4, 0, 0, 5, 6];
        let plane = StridedPlane {
            bytes: &bytes,
            stride: 4,
            row_bytes: 2,
        };

        let rows: Vec<&[u8]> = plane.rows().collect();

        assert_eq!(rows, vec![&[1, 2][..], &[3, 4][..], &[5, 6][..]]);
    }

    #[test]
    fn a_video_track_reports_its_frame_rate() {
        let video = opened("yuv420p.mkv");

        let (numerator, denominator) = video.frame_rate();

        assert!(numerator > 0);
        assert_eq!(numerator, denominator * 25);
    }

    #[test]
    fn a_track_that_does_not_exist_is_an_error() {
        let source = media("yuv420p.mkv");
        let indexer = Indexer::new(&source).expect("creates an indexer");
        let index = indexer.index().expect("indexes");

        let opened = VideoSource::open(&source, 5, &index, 1);
        let error = opened.err().expect("track 5 does not exist");

        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn a_frame_reports_its_field_flags() {
        let mut progressive = opened("yuv420p.mkv");
        let mut top_first = opened("interlaced_tff.mkv");
        let mut bottom_first = opened("interlaced_bff.mkv");

        let progressive_frame = progressive.frame(0).expect("decodes");
        assert!(!progressive_frame.interlaced());

        let top_first_frame = top_first.frame(0).expect("decodes");
        assert!(top_first_frame.interlaced());
        assert!(top_first_frame.top_field_first());

        let bottom_first_frame = bottom_first.frame(0).expect("decodes");
        assert!(bottom_first_frame.interlaced());
        assert!(!bottom_first_frame.top_field_first());
    }
}
