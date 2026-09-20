use std::sync::Arc;
use std::time::Duration;

use cubecl::server::Handle;

use crate::frame::format::{ColourInfo, FrameFormat, MaskFormat};
use crate::frame::layout::Channel;

/// The device memory behind one plane.
#[derive(Debug)]
pub struct PlaneBuffer {
    handle: Handle,
    bytes: usize,
}

impl PlaneBuffer {
    #[cfg_attr(not(test), expect(dead_code, reason = "used only by tests in this module"))]
    pub(crate) fn new(handle: Handle, bytes: usize) -> Self {
        Self { handle, bytes }
    }

    /// The device memory this plane occupies.
    ///
    /// Borrow it for a kernel launch. Cloning it out and keeping it holds the memory without the
    /// frame knowing, which is what [Frame::try_into_mut] relies on to decide a frame is writable.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// How many bytes of device memory this plane occupies.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

/// One channel's samples on the device.
///
/// Cloning shares the buffer rather than copying it, which is what makes handing a frame to two
/// consumers cheap.
#[derive(Debug, Clone)]
pub struct Plane {
    buffer: Arc<PlaneBuffer>,
    width: u32,
    height: u32,
}

impl Plane {
    #[cfg_attr(not(test), expect(dead_code, reason = "used only by tests in this module"))]
    pub(crate) fn new(buffer: PlaneBuffer, width: u32, height: u32) -> Self {
        Self {
            buffer: Arc::new(buffer),
            width,
            height,
        }
    }

    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// How many samples this plane holds.
    pub const fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn buffer(&self) -> &PlaneBuffer {
        &self.buffer
    }

    /// Whether this is the only holder of the buffer.
    fn is_exclusive(&self) -> bool {
        Arc::strong_count(&self.buffer) == 1
    }
}

/// Where a frame sits in its stream.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FrameMeta {
    pub index: u64,
    pub timestamp: Option<Duration>,
}

impl FrameMeta {
    /// Metadata for the frame at `index`, with no timestamp.
    pub const fn at(index: u64) -> Self {
        Self {
            index,
            timestamp: None,
        }
    }
}

/// A frame on the device.
#[derive(Debug, Clone)]
pub struct Frame {
    format: FrameFormat,
    planes: Vec<Plane>,
    meta: FrameMeta,
}

impl Frame {
    #[cfg_attr(not(test), expect(dead_code, reason = "used only by tests in this module"))]
    pub(crate) fn new(format: FrameFormat, planes: Vec<Plane>, meta: FrameMeta) -> Self {
        debug_assert_eq!(
            planes.len(),
            format.channels().len(),
            "a frame needs one plane per channel its format carries",
        );

        Self { format, planes, meta }
    }

    pub const fn format(&self) -> &FrameFormat {
        &self.format
    }

    pub const fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    pub fn plane(&self, channel: Channel) -> Option<&Plane> {
        let index = self
            .format
            .channels()
            .iter()
            .position(|carried| *carried == channel)?;

        self.planes.get(index)
    }

    /// Takes this frame for writing, or hands it back when somebody else holds a plane.
    ///
    /// A stage that gets the frame back has to allocate its own output instead, which is the one
    /// place a frame is ever duplicated.
    pub fn try_into_mut(self) -> Result<FrameMut, Self> {
        if self.planes.iter().all(Plane::is_exclusive) {
            return Ok(FrameMut { frame: self });
        }

        Err(self)
    }
}

/// A frame no one else holds, so its planes can be written.
#[derive(Debug)]
pub struct FrameMut {
    frame: Frame,
}

impl FrameMut {
    pub const fn format(&self) -> &FrameFormat {
        self.frame.format()
    }

    pub const fn meta(&self) -> &FrameMeta {
        self.frame.meta()
    }

    pub fn plane(&self, channel: Channel) -> Option<&Plane> {
        self.frame.plane(channel)
    }

    /// Declares what the samples now mean, for a filter that converted them in place.
    ///
    /// This is the only part of the format a held frame can change, because it is the only part
    /// that does not decide how large a plane is.
    pub fn set_colour(&mut self, colour: ColourInfo) {
        self.frame.format.colour = colour;
    }

    /// Hands the frame back, ready to pass downstream.
    pub fn freeze(self) -> Frame {
        self.frame
    }
}

/// A mask on the device.
#[derive(Debug, Clone)]
pub struct Mask {
    format: MaskFormat,
    plane: Plane,
    meta: FrameMeta,
}

impl Mask {
    #[cfg_attr(not(test), expect(dead_code, reason = "used only by tests in this module"))]
    pub(crate) fn new(format: MaskFormat, plane: Plane, meta: FrameMeta) -> Self {
        Self { format, plane, meta }
    }

    pub const fn format(&self) -> &MaskFormat {
        &self.format
    }

    pub const fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    pub const fn plane(&self) -> &Plane {
        &self.plane
    }
}

#[cfg(test)]
mod tests {
    use cubecl::server::Handle;
    use cubecl::stream_id::StreamId;

    use super::*;
    use crate::frame::format::ColourRange;
    use crate::frame::layout::{FrameLayout, Subsampling};
    use crate::frame::sample::SampleFormat;
    use crate::{HostPlanes, StreamFormat};

    fn format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
        }
    }

    /// A handle with no device behind it, which is all these tests need: nothing here reads or
    /// writes the memory it names.
    fn plane(width: u32, height: u32) -> Plane {
        let bytes = width as usize * height as usize;
        let handle = Handle::new(StreamId::current(), bytes as u64);

        Plane::new(PlaneBuffer::new(handle, bytes), width, height)
    }

    fn frame() -> Frame {
        let format = format();
        let planes = vec![plane(64, 64), plane(32, 32), plane(32, 32)];

        Frame::new(format, planes, FrameMeta::at(7))
    }

    #[test]
    fn a_frame_reports_the_plane_for_each_channel_it_carries() {
        let frame = frame();

        assert_eq!(
            frame.plane(Channel::Y).expect("should carry luma").dimensions(),
            (64, 64)
        );
        assert_eq!(
            frame.plane(Channel::U).expect("should carry chroma").dimensions(),
            (32, 32)
        );
        assert!(frame.plane(Channel::R).is_none());
    }

    #[test]
    fn a_frame_carries_its_format_and_metadata() {
        let frame = frame();

        assert_eq!(frame.format().width, 64);
        assert_eq!(frame.meta().index, 7);
    }

    #[test]
    fn a_frame_nobody_else_holds_can_be_taken_for_writing() {
        let frame = frame();

        assert!(frame.try_into_mut().is_ok());
    }

    /// Handing the same frame to two consumers is a refcount bump, so neither may write through it.
    #[test]
    fn a_shared_frame_cannot_be_taken_for_writing() {
        let frame = frame();
        let shared = frame.clone();

        let returned = frame
            .try_into_mut()
            .expect_err("a shared frame must not be writable");
        assert_eq!(returned.meta().index, 7);

        drop(shared);
        assert!(
            returned.try_into_mut().is_ok(),
            "the last holder should be able to write"
        );
    }

    #[test]
    fn a_frame_taken_for_writing_freezes_back_into_a_frame() {
        let frame = frame();
        let writable = frame.try_into_mut().expect("should be writable");

        assert_eq!(
            writable
                .plane(Channel::Y)
                .expect("should carry luma")
                .dimensions(),
            (64, 64)
        );

        let frozen = writable.freeze();
        assert_eq!(frozen.meta().index, 7);
    }

    #[test]
    fn a_frame_written_in_place_can_restate_what_its_samples_mean() {
        let frame = frame();
        let mut writable = frame.try_into_mut().expect("should be writable");

        assert_eq!(writable.format().colour.range, ColourRange::Unspecified);

        writable.set_colour(ColourInfo {
            range: ColourRange::Full,
            ..ColourInfo::default()
        });

        let frozen = writable.freeze();
        assert_eq!(frozen.format().colour.range, ColourRange::Full);
    }

    #[test]
    fn a_mask_is_one_plane_and_reports_its_dimensions() {
        let format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
        };

        let mask = Mask::new(format, plane(64, 64), FrameMeta::at(3));
        assert_eq!(mask.plane().dimensions(), (64, 64));
        assert_eq!(mask.meta().index, 3);
    }

    #[test]
    fn a_plane_buffer_reports_the_bytes_it_holds() {
        let plane = plane(8, 4);

        assert_eq!(plane.buffer().bytes(), 32);
    }

    #[test]
    fn cloning_a_plane_shares_one_buffer() {
        let plane = plane(8, 8);
        let shared = plane.clone();

        assert!(std::ptr::eq(plane.buffer(), shared.buffer()));
    }

    /// The geometry a format describes and the planes a frame carries have to agree, or a kernel
    /// launched from the format's dimensions would run off the end of the buffer.
    #[test]
    fn every_plane_matches_the_geometry_its_format_describes() {
        let frame = frame();

        for channel in frame.format().channels() {
            let expected = frame
                .format()
                .channel_dimensions(*channel)
                .expect("the format carries this channel");
            let plane = frame.plane(*channel).expect("the frame carries this channel");

            assert_eq!(plane.dimensions(), expected, "{channel}");
        }
    }

    #[test]
    fn a_frame_and_its_host_planes_agree_on_every_plane_size() {
        let frame = frame();
        let format = StreamFormat::Frames(*frame.format());
        let host = HostPlanes::zeroed(&format);

        for channel in frame.format().channels() {
            let plane = frame.plane(*channel).expect("the frame carries this channel");
            let bytes = frame
                .format()
                .channel_bytes(*channel)
                .expect("the format carries this channel");

            assert_eq!(
                plane.samples() * frame.format().sample.bytes(),
                bytes,
                "{channel}"
            );
            assert_eq!(
                host.plane(*channel)
                    .expect("the host planes carry this channel")
                    .len(),
                bytes,
                "{channel}",
            );
        }
    }
}
