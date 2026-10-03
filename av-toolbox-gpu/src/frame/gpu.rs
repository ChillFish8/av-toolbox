use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};
use std::time::Duration;

use cubecl::prelude::{ArrayArg, CubePrimitive, Runtime};
use cubecl::server::Handle;

use crate::frame::attachments::{Attachments, MetaValue};
use crate::frame::format::{ColourInfo, FrameFormat, MaskChroma, MaskFormat, StreamFormat};
use crate::frame::layout::{Channel, FrameLayout};
use crate::memory::FrameBudget;

/// Where a plane's memory goes when its last holder drops it.
pub(crate) trait Recycle: Send + Sync {
    fn recycle(&self, bytes: usize, handle: Handle);
}

/// The device memory behind one plane.
#[derive(Debug)]
pub struct PlaneBuffer {
    /// Taken by [PlaneBuffer::drop], which is the only thing that ever clears it.
    handle: Option<Handle>,
    bytes: usize,
    recycle: Option<Weak<dyn Recycle>>,
    /// The budget this buffer's bytes were charged to, credited back on drop.
    budget: Option<Weak<FrameBudget>>,
    /// The allocation this is a window into, kept alive for as long as the window is.
    owner: Option<Arc<PlaneBuffer>>,
}

impl PlaneBuffer {
    /// A window into `owner`, or a standalone buffer when there is none.
    ///
    /// A window releases nothing of its own, because the allocation it points into is released as
    /// one piece. Holding the owner is what stops that happening while the window is still alive.
    pub(crate) fn view(owner: Option<Arc<PlaneBuffer>>, handle: Handle, bytes: usize) -> Self {
        Self {
            handle: Some(handle),
            bytes,
            recycle: None,
            budget: None,
            owner,
        }
    }

    /// Memory that goes back to `recycle` when its last holder drops it, crediting `budget` back
    /// at the same time when there is one.
    pub(crate) fn pooled(
        handle: Handle,
        bytes: usize,
        recycle: Weak<dyn Recycle>,
        budget: Option<Weak<FrameBudget>>,
    ) -> Self {
        Self {
            handle: Some(handle),
            bytes,
            recycle: Some(recycle),
            budget,
            owner: None,
        }
    }

    /// The device memory this plane occupies.
    ///
    /// Borrow it for a kernel launch. A clone must not outlive the plane. When the last holder
    /// drops, the memory goes back to the pool and is handed to the next caller wanting that size.
    pub fn handle(&self) -> &Handle {
        self.handle
            .as_ref()
            .expect("a plane buffer holds its handle until it is dropped")
    }

    /// How many bytes of device memory this plane occupies.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether this buffer is a window into somebody else's allocation.
    const fn is_view(&self) -> bool {
        self.owner.is_some()
    }
}

impl Drop for PlaneBuffer {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };

        if let Some(budget) = self.budget.as_ref().and_then(Weak::upgrade) {
            budget.credit(self.bytes);
        }

        let Some(recycle) = self.recycle.as_ref().and_then(Weak::upgrade) else {
            return;
        };

        recycle.recycle(self.bytes, handle);
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

    /// This plane as a kernel argument of `C` elements.
    ///
    /// The length is this plane's byte count divided by `C`'s own size, so whichever element type
    /// a kernel declares gets exactly the count of `C` values this window holds, not a count
    /// carried over from the plane's own sample format. One `array` serves both a kernel that
    /// reads the plane and one that writes it, because the binding only carries the handle, and
    /// it is the kernel that decides what to do with it.
    pub fn array<C: CubePrimitive, R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.buffer.handle().clone();
        let elements = self.buffer.bytes() / C::type_size();

        // Safety: `elements` is this window's own byte count divided by `C`'s size, so it names
        // exactly as many `C` values as the window holds, matching what a kernel declaring `C`
        // computes from the same handle at launch.
        unsafe { ArrayArg::from_raw_parts(handle, elements) }
    }

    /// Whether this is the only holder of the buffer.
    ///
    /// A window into somebody else's allocation is never exclusive, however few holders the window
    /// itself has. Two windows onto one ring slot are separate buffers pointing at the same memory,
    /// so counting holders of either one says nothing about the other.
    fn is_exclusive(&self) -> bool {
        if self.buffer.is_view() {
            return false;
        }

        Arc::strong_count(&self.buffer) == 1
    }
}

/// How strongly scene detection favoured a cut just before a frame.
///
/// Higher is a better place to cut. Two scores are equal when their bits are, so metadata carrying
/// one can still be compared and hashed.
#[derive(Debug, Copy, Clone)]
pub struct SceneScore(f32);

impl SceneScore {
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for SceneScore {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for SceneScore {}

impl Hash for SceneScore {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Where a frame sits in its stream, and what stages attached to it.
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash)]
pub struct FrameMeta {
    pub index: u64,
    /// The chunk this frame belongs to.
    ///
    /// A stream nothing has split is one chunk, so this stays zero until a stage sets it.
    pub chunk: u64,
    pub timestamp: Option<Duration>,
    /// How strongly scene detection favoured a cut just before this frame, when it was scored.
    pub scene_score: Option<SceneScore>,
    pub attachments: Attachments,
}

impl FrameMeta {
    /// Metadata for the frame at `index`, in the first chunk, with no timestamp.
    pub const fn at(index: u64) -> Self {
        Self {
            index,
            chunk: 0,
            timestamp: None,
            scene_score: None,
            attachments: Attachments::new(),
        }
    }

    /// The same metadata in `chunk`.
    pub fn in_chunk(mut self, chunk: u64) -> Self {
        self.chunk = chunk;
        self
    }

    /// Attaches `value` under `key` for the stage that is running.
    pub fn attach(&mut self, key: impl Into<String>, value: MetaValue) {
        self.attachments.attach(key, value);
    }
}

/// Planes that do not fit the format a frame was to be built with.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("a {layout} frame needs {expected} planes, got {actual}")]
    PlaneCount {
        layout: FrameLayout,
        expected: usize,
        actual: usize,
    },
    #[error(
        "the {channel} plane needs to be {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
    )]
    PlaneSize {
        channel: Channel,
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error("the {channel} plane needs {expected} bytes, got {actual}")]
    PlaneBytes {
        channel: Channel,
        expected: usize,
        actual: usize,
    },
}

fn check_plane(format: &FrameFormat, channel: Channel, plane: &Plane) -> Result<(), FrameError> {
    let (expected_width, expected_height) = format
        .channel_dimensions(channel)
        .expect("the layout carries every channel it lists");
    let (actual_width, actual_height) = plane.dimensions();

    if (actual_width, actual_height) != (expected_width, expected_height) {
        return Err(FrameError::PlaneSize {
            channel,
            expected_width,
            expected_height,
            actual_width,
            actual_height,
        });
    }

    let expected = format
        .channel_bytes(channel)
        .expect("the layout carries every channel it lists");
    let actual = plane.buffer().bytes();
    if actual != expected {
        return Err(FrameError::PlaneBytes {
            channel,
            expected,
            actual,
        });
    }

    Ok(())
}

/// A frame on the device.
#[derive(Debug, Clone)]
pub struct Frame {
    format: FrameFormat,
    planes: Vec<Plane>,
    meta: FrameMeta,
}

impl Frame {
    pub(crate) fn new(format: FrameFormat, planes: Vec<Plane>, meta: FrameMeta) -> Self {
        debug_assert_eq!(
            planes.len(),
            format.channels().len(),
            "a frame needs one plane per channel its format carries",
        );

        Self { format, planes, meta }
    }

    /// A frame of `format` holding `planes`, one per channel in plane order.
    ///
    /// The planes are shared rather than copied, so a stage can hand on planes it was given.
    pub fn from_planes(format: FrameFormat, planes: Vec<Plane>, meta: FrameMeta) -> Result<Self, FrameError> {
        let channels = format.channels();
        if planes.len() != channels.len() {
            return Err(FrameError::PlaneCount {
                layout: format.layout,
                expected: channels.len(),
                actual: planes.len(),
            });
        }

        for (channel, plane) in channels.iter().zip(&planes) {
            check_plane(&format, *channel, plane)?;
        }

        let frame = Self::new(format, planes, meta);

        Ok(frame)
    }

    pub const fn format(&self) -> &FrameFormat {
        &self.format
    }

    pub const fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        &mut self.meta
    }

    /// Moves this to `index` in the stream.
    ///
    /// The chunk it belongs to and its timestamp are unchanged.
    pub fn set_index(&mut self, index: u64) {
        self.meta.index = index;
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

    /// The plane for `channel`.
    ///
    /// A kernel may write through the binding this yields, because holding a `FrameMut` at all
    /// means nothing else holds this frame.
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
    chroma: Option<Plane>,
}

impl Mask {
    pub(crate) fn new(format: MaskFormat, plane: Plane, meta: FrameMeta, chroma: Option<Plane>) -> Self {
        debug_assert_eq!(
            chroma.is_some(),
            format.chroma.is_some(),
            "a mask carries a chroma plane exactly when its format has one",
        );

        Self {
            format,
            plane,
            meta,
            chroma,
        }
    }

    /// Takes this mask's chroma plane away, leaving its format without chroma.
    pub(crate) fn take_chroma(&mut self) -> Option<Plane> {
        self.format.chroma = None;
        self.chroma.take()
    }

    /// This mask carrying `plane` as its chroma plane, with `chroma` as its format's chroma.
    pub(crate) fn with_chroma(mut self, chroma: MaskChroma, plane: Plane) -> Self {
        self.format.chroma = Some(chroma);
        self.chroma = Some(plane);
        self
    }

    pub const fn format(&self) -> &MaskFormat {
        &self.format
    }

    pub const fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        &mut self.meta
    }

    /// Moves this to `index` in the stream.
    ///
    /// The chunk it belongs to is unchanged.
    pub fn set_index(&mut self, index: u64) {
        self.meta.index = index;
    }

    pub const fn plane(&self) -> &Plane {
        &self.plane
    }

    pub const fn chroma_plane(&self) -> Option<&Plane> {
        self.chroma.as_ref()
    }

    /// The plane `channel` of a frame in `frame`'s format is blended with.
    ///
    /// A channel at the frame's full size takes the full plane, and a subsampled one takes the
    /// chroma plane. It is `None` when this mask carries no chroma plane of the channel's size.
    pub fn plane_for(&self, channel: Channel, frame: &FrameFormat) -> Option<&Plane> {
        let dimensions = frame.channel_dimensions(channel)?;
        let full = (frame.width, frame.height);
        if dimensions == full {
            return Some(&self.plane);
        }

        let chroma = self.chroma.as_ref()?;
        if chroma.dimensions() != dimensions {
            return None;
        }

        Some(chroma)
    }
}

/// One item flowing along an edge of the pipeline.
#[derive(Debug, Clone)]
pub enum StreamItem {
    Frame(Frame),
    Mask(Mask),
}

impl StreamItem {
    pub const fn is_frame(&self) -> bool {
        matches!(self, Self::Frame(_))
    }

    pub const fn is_mask(&self) -> bool {
        matches!(self, Self::Mask(_))
    }

    pub fn as_frame(&self) -> Option<&Frame> {
        match self {
            Self::Frame(frame) => Some(frame),
            Self::Mask(_) => None,
        }
    }

    pub fn as_mask(&self) -> Option<&Mask> {
        match self {
            Self::Mask(mask) => Some(mask),
            Self::Frame(_) => None,
        }
    }

    pub fn into_frame(self) -> Result<Frame, Self> {
        match self {
            Self::Frame(frame) => Ok(frame),
            Self::Mask(_) => Err(self),
        }
    }

    pub fn into_mask(self) -> Result<Mask, Self> {
        match self {
            Self::Mask(mask) => Ok(mask),
            Self::Frame(_) => Err(self),
        }
    }

    /// This item's format, as [StreamFormat](crate::frame::StreamFormat) describes it.
    pub fn format(&self) -> StreamFormat {
        match self {
            Self::Frame(frame) => StreamFormat::Frames(*frame.format()),
            Self::Mask(mask) => StreamFormat::Mask(*mask.format()),
        }
    }

    /// Where this item sits in its stream.
    pub const fn meta(&self) -> &FrameMeta {
        match self {
            Self::Frame(frame) => frame.meta(),
            Self::Mask(mask) => mask.meta(),
        }
    }

    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        match self {
            Self::Frame(frame) => frame.meta_mut(),
            Self::Mask(mask) => mask.meta_mut(),
        }
    }

    pub fn set_index(&mut self, index: u64) {
        match self {
            Self::Frame(frame) => frame.set_index(index),
            Self::Mask(mask) => mask.set_index(index),
        }
    }

    /// The plane carrying `channel`, if this item has one.
    ///
    /// A mask's full plane answers to [Channel::Y](crate::frame::Channel::Y), and its chroma plane
    /// to both [Channel::U](crate::frame::Channel::U) and [Channel::V](crate::frame::Channel::V).
    pub(crate) fn plane(&self, channel: Channel) -> Option<&Plane> {
        match (self, channel) {
            (Self::Frame(frame), _) => frame.plane(channel),
            (Self::Mask(mask), Channel::Y) => Some(mask.plane()),
            (Self::Mask(mask), Channel::U | Channel::V) => mask.chroma_plane(),
            (Self::Mask(_), _) => None,
        }
    }
}

impl From<Frame> for StreamItem {
    fn from(frame: Frame) -> Self {
        Self::Frame(frame)
    }
}

impl From<Mask> for StreamItem {
    fn from(mask: Mask) -> Self {
        Self::Mask(mask)
    }
}

#[cfg(test)]
mod tests {
    // Whichever backend this build enables, picked in the same order `Backend` tries them, so the
    // array tests below need no device and run under every feature set.
    use std::hash::DefaultHasher;

    #[cfg(all(feature = "cuda", not(any(feature = "vulkan", feature = "metal"))))]
    use cubecl::cuda::CudaRuntime as TestRuntime;
    #[cfg(all(
        feature = "rocm",
        not(any(feature = "vulkan", feature = "metal", feature = "cuda"))
    ))]
    use cubecl::hip::HipRuntime as TestRuntime;
    use cubecl::server::{Binding, Handle};
    use cubecl::stream_id::StreamId;
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    use cubecl::wgpu::WgpuRuntime as TestRuntime;

    use super::*;
    use crate::frame::format::{ChromaLocation, ColourRange};
    use crate::frame::layout::{FrameLayout, Subsampling};
    use crate::frame::sample::SampleFormat;
    use crate::testing::CountingAllocator;
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use crate::{Backend, test_device};
    use crate::{BufferPool, HostPlanes, RingSpec, StreamFormat, StreamPolicy};

    fn format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        }
    }

    /// A handle with no device behind it, which is all these tests need. Nothing here reads or
    /// writes the memory it names.
    fn plane(width: u32, height: u32) -> Plane {
        let bytes = width as usize * height as usize;
        let handle = Handle::new(StreamId::current(), bytes as u64);

        Plane::new(PlaneBuffer::view(None, handle, bytes), width, height)
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
            chroma: None,
        };

        let mask = Mask::new(format, plane(64, 64), FrameMeta::at(3), None);
        assert_eq!(mask.plane().dimensions(), (64, 64));
        assert_eq!(mask.meta().index, 3);
    }

    fn chroma_mask(chroma: Option<Plane>) -> Mask {
        let format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: Some(MaskChroma {
                subsampling: Subsampling::Yuv420,
                location: ChromaLocation::Left,
            }),
        };
        let full = plane(64, 64);
        let meta = FrameMeta::at(0);

        Mask::new(format, full, meta, chroma)
    }

    fn plain_mask() -> Mask {
        let format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: None,
        };
        let full = plane(64, 64);
        let meta = FrameMeta::at(0);

        Mask::new(format, full, meta, None)
    }

    fn same_plane(actual: Option<&Plane>, expected: &Plane) -> bool {
        actual.is_some_and(|plane| std::ptr::eq(plane.buffer(), expected.buffer()))
    }

    #[test]
    fn a_mask_without_chroma_has_no_chroma_plane() {
        let mask = plain_mask();

        assert!(mask.chroma_plane().is_none());
    }

    #[test]
    fn a_420_frame_blends_luma_with_the_full_plane_and_chroma_with_the_chroma_plane() {
        let chroma = plane(32, 32);
        let shared = chroma.clone();
        let mask = chroma_mask(Some(shared));
        let frame = format();

        let luma_plane = mask.plane_for(Channel::Y, &frame);
        let blue_plane = mask.plane_for(Channel::U, &frame);
        let red_plane = mask.plane_for(Channel::V, &frame);

        assert!(same_plane(luma_plane, mask.plane()));
        assert!(same_plane(blue_plane, &chroma));
        assert!(same_plane(red_plane, &chroma));
    }

    #[test]
    fn a_444_frame_blends_every_channel_with_the_full_plane() {
        let chroma = plane(32, 32);
        let mask = chroma_mask(Some(chroma));
        let base = format();
        let frame = FrameFormat {
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv444,
            },
            ..base
        };

        for channel in [Channel::Y, Channel::U, Channel::V] {
            let blended = mask.plane_for(channel, &frame);

            assert!(same_plane(blended, mask.plane()), "{channel}");
        }
    }

    #[test]
    fn a_subsampled_channel_has_no_plane_when_the_mask_carries_no_chroma() {
        let mask = plain_mask();
        let frame = format();

        assert!(mask.plane_for(Channel::U, &frame).is_none());
    }

    #[test]
    fn a_channel_the_frame_does_not_carry_has_no_plane() {
        let chroma = plane(32, 32);
        let mask = chroma_mask(Some(chroma));
        let frame = format();

        assert!(mask.plane_for(Channel::R, &frame).is_none());
    }

    fn mask_with_chroma_at(subsampling: Subsampling, chroma: Plane) -> Mask {
        let format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: Some(MaskChroma {
                subsampling,
                location: ChromaLocation::Left,
            }),
        };

        let full = plane(64, 64);
        let meta = FrameMeta::at(0);

        Mask::new(format, full, meta, Some(chroma))
    }

    fn format_at(subsampling: Subsampling) -> FrameFormat {
        let base = format();

        FrameFormat {
            layout: FrameLayout::Yuv { subsampling },
            ..base
        }
    }

    #[test]
    fn a_422_frame_blends_chroma_with_a_422_chroma_plane() {
        let chroma = plane(32, 64);
        let shared = chroma.clone();
        let mask = mask_with_chroma_at(Subsampling::Yuv422, shared);
        let frame = format_at(Subsampling::Yuv422);

        let luma_plane = mask.plane_for(Channel::Y, &frame);
        let blue_plane = mask.plane_for(Channel::U, &frame);
        let red_plane = mask.plane_for(Channel::V, &frame);

        assert!(same_plane(luma_plane, mask.plane()));
        assert!(same_plane(blue_plane, &chroma));
        assert!(same_plane(red_plane, &chroma));
    }

    #[test]
    fn a_chroma_plane_of_another_shape_is_not_blended_with() {
        let chroma = plane(32, 32);
        let mask = mask_with_chroma_at(Subsampling::Yuv420, chroma);
        let frame = format_at(Subsampling::Yuv422);

        assert!(mask.plane_for(Channel::U, &frame).is_none());
        assert!(mask.plane_for(Channel::V, &frame).is_none());
    }

    #[test]
    fn an_rgb_frame_blends_every_channel_with_the_full_plane() {
        let chroma = plane(32, 32);
        let mask = mask_with_chroma_at(Subsampling::Yuv420, chroma);
        let base = format();
        let frame = FrameFormat {
            layout: FrameLayout::Rgb,
            ..base
        };

        for channel in [Channel::R, Channel::G, Channel::B] {
            let blended = mask.plane_for(channel, &frame);

            assert!(same_plane(blended, mask.plane()), "{channel}");
        }

        assert!(mask.plane_for(Channel::U, &frame).is_none());
    }

    #[test]
    fn a_mask_stream_item_answers_chroma_channels_with_its_chroma_plane() {
        let chroma = plane(32, 32);
        let mask = chroma_mask(Some(chroma));
        let item = StreamItem::Mask(mask);

        let blue_plane = item.plane(Channel::U).expect("the mask carries chroma");

        assert_eq!(blue_plane.dimensions(), (32, 32));
        assert!(item.plane(Channel::V).is_some());
        assert!(item.plane(Channel::R).is_none());
    }

    #[test]
    fn a_frame_and_a_mask_convert_into_stream_items() {
        let format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: None,
        };
        let mask = Mask::new(format, plane(64, 64), FrameMeta::at(3), None);

        let from_frame: StreamItem = frame().into();
        let from_mask: StreamItem = mask.into();

        assert!(from_frame.is_frame());
        assert!(from_mask.is_mask());
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

    /// The raw handle an `array` binding carries, or a failure if it somehow aliased an input.
    ///
    /// A plane never launches through an input array of its own, so `array` always produces the
    /// `Handle` variant. This just unwraps it once for the offset test below to read from.
    fn binding<C: CubePrimitive>(plane: &Plane) -> Binding {
        match plane.array::<C, TestRuntime>() {
            ArrayArg::Handle { handle } => handle.handle,
            ArrayArg::Alias { .. } => panic!("a plane binds its own buffer, never an alias"),
        }
    }

    #[test]
    fn array_declares_one_element_per_sample_for_a_matching_element_type() {
        let plane = plane(8, 4);

        assert_eq!(plane.array::<u8, TestRuntime>().size(), plane.samples());
    }

    /// The regression guard for the fix: a length taken from `samples()` instead of from the
    /// element type would claim twice as many elements as this window holds.
    #[test]
    fn array_does_not_overclaim_for_an_element_type_wider_than_the_plane() {
        let plane = plane(8, 4);

        assert_eq!(plane.array::<u16, TestRuntime>().size(), plane.samples() / 2);
    }

    /// The regression guard for the ring case: a binding built from the wrong handle would carry
    /// no offset, or the ring's own offset instead of its slot's.
    #[test]
    fn a_ring_slot_planes_binding_carries_its_own_offset_not_the_wholes_ring() {
        let allocator = Arc::new(CountingAllocator::new(64));
        let pool = BufferPool::new(allocator, StreamPolicy::SharedStream);
        let ring = pool.ring(RingSpec {
            slots: 2,
            format: format(),
        });

        let expected = ring.geometry().plane_offset(1, 0).expect("the plane exists");
        let luma = ring.slot_plane(1, Channel::Y).expect("the ring carries luma");

        assert_eq!(binding::<u8>(&luma).offset_start, Some(expected as u64));
    }

    /// Data round-tripping through the binding is only meaningful against a real client, so that
    /// is checked where it is launched at a kernel, not here.
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_plane_binds_as_an_array_of_its_samples() {
        let backend = Backend::open_default(test_device()).expect("a backend should open");
        let context = backend.frame_context();
        let frame = context.alloc_frame(&format(), FrameMeta::at(0));

        let luma = frame.plane(Channel::Y).expect("a yuv frame has luma");

        assert_eq!(luma.samples(), 64 * 64);
        assert_eq!(luma.buffer().bytes(), 64 * 64);
    }

    #[test]
    fn metadata_starts_in_the_first_chunk() {
        let meta = FrameMeta::at(7);

        assert_eq!(meta.index, 7);
        assert_eq!(meta.chunk, 0);
    }

    #[test]
    fn metadata_can_be_moved_into_another_chunk() {
        let meta = FrameMeta::at(7).in_chunk(3);

        assert_eq!(meta.index, 7);
        assert_eq!(meta.chunk, 3);
    }

    #[test]
    fn renumbering_a_frame_leaves_its_chunk_and_timestamp_alone() {
        let format = format();
        let planes = vec![plane(64, 64), plane(32, 32), plane(32, 32)];
        let meta = FrameMeta {
            index: 7,
            chunk: 3,
            timestamp: Some(Duration::from_millis(500)),
            scene_score: None,
            attachments: Attachments::new(),
        };
        let mut frame = Frame::new(format, planes, meta);

        frame.set_index(11);

        assert_eq!(frame.meta().index, 11);
        assert_eq!(frame.meta().chunk, 3);
        assert_eq!(frame.meta().timestamp, Some(Duration::from_millis(500)));
    }

    #[test]
    fn renumbering_a_mask_leaves_its_chunk_and_timestamp_alone() {
        let format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: None,
        };
        let meta = FrameMeta {
            index: 7,
            chunk: 3,
            timestamp: Some(Duration::from_millis(500)),
            scene_score: None,
            attachments: Attachments::new(),
        };
        let mut mask = Mask::new(format, plane(64, 64), meta, None);

        mask.set_index(11);

        assert_eq!(mask.meta().index, 11);
        assert_eq!(mask.meta().chunk, 3);
        assert_eq!(mask.meta().timestamp, Some(Duration::from_millis(500)));
    }

    #[test]
    fn renumbering_a_stream_item_holding_a_frame_reaches_it() {
        let format = format();
        let planes = vec![plane(64, 64), plane(32, 32), plane(32, 32)];
        let meta = FrameMeta {
            index: 7,
            chunk: 3,
            timestamp: Some(Duration::from_millis(500)),
            scene_score: None,
            attachments: Attachments::new(),
        };
        let frame = Frame::new(format, planes, meta);
        let mut item = StreamItem::Frame(frame);

        item.set_index(11);

        assert_eq!(item.meta().index, 11);
        assert_eq!(item.meta().chunk, 3);
        assert_eq!(item.meta().timestamp, Some(Duration::from_millis(500)));
    }

    #[test]
    fn renumbering_a_stream_item_holding_a_mask_reaches_it() {
        let format = MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: None,
        };
        let meta = FrameMeta {
            index: 7,
            chunk: 3,
            timestamp: Some(Duration::from_millis(500)),
            scene_score: None,
            attachments: Attachments::new(),
        };
        let mask = Mask::new(format, plane(64, 64), meta, None);
        let mut item = StreamItem::Mask(mask);

        item.set_index(11);

        assert_eq!(item.meta().index, 11);
        assert_eq!(item.meta().chunk, 3);
        assert_eq!(item.meta().timestamp, Some(Duration::from_millis(500)));
    }

    fn hash_of(score: SceneScore) -> u64 {
        let mut hasher = DefaultHasher::new();
        score.hash(&mut hasher);

        hasher.finish()
    }

    #[test]
    fn scene_scores_with_the_same_bits_are_equal_and_hash_equal() {
        let first = SceneScore::new(2.5);
        let second = SceneScore::new(2.5);

        assert_eq!(first, second);
        assert_eq!(hash_of(first), hash_of(second));
    }

    #[test]
    fn scene_scores_compare_by_their_bits() {
        let positive_zero = SceneScore::new(0.0);
        let negative_zero = SceneScore::new(-0.0);

        assert_ne!(positive_zero, negative_zero);
    }

    #[test]
    fn a_nan_scene_score_equals_itself() {
        let score = SceneScore::new(f32::NAN);

        assert_eq!(score, score);
    }

    #[test]
    fn new_metadata_carries_no_scene_score() {
        let meta = FrameMeta::at(3);

        assert_eq!(meta.scene_score, None);
    }

    #[test]
    fn metadata_attach_goes_to_its_pending_values() {
        let mut meta = FrameMeta::at(0);

        meta.attach("cut", MetaValue::Bool(true));
        meta.attachments.stamp("transform.detect");

        let stages: Vec<&str> = meta.attachments.stages().map(|(path, _)| path).collect();
        assert_eq!(stages, vec!["transform.detect"]);
    }
}
