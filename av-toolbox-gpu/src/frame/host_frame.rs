use std::sync::Arc;

use crate::frame::format::{ColourInfo, FrameFormat, MaskFormat, StreamFormat};
use crate::frame::gpu::FrameMeta;
use crate::frame::layout::Channel;
use crate::memory::HostBuffer;

/// One channel's samples on the host.
///
/// Cloning shares the buffer rather than copying it.
#[derive(Debug, Clone)]
pub struct HostPlane {
    buffer: Arc<HostBuffer>,
    width: u32,
    height: u32,
}

impl HostPlane {
    pub(crate) fn new(buffer: HostBuffer, width: u32, height: u32) -> Self {
        Self {
            buffer: Arc::new(buffer),
            width,
            height,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        self.buffer.bytes()
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    fn is_exclusive(&self) -> bool {
        Arc::strong_count(&self.buffer) == 1
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        Arc::get_mut(&mut self.buffer)
            .expect("a writable host item is the only holder of its planes")
            .bytes_mut()
    }
}

/// A frame on the host.
#[derive(Debug, Clone)]
pub struct HostFrame {
    format: FrameFormat,
    planes: Vec<HostPlane>,
    meta: FrameMeta,
}

impl HostFrame {
    pub(crate) fn new(format: FrameFormat, planes: Vec<HostPlane>, meta: FrameMeta) -> Self {
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

    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        &mut self.meta
    }

    /// Moves this to `index` in the stream, leaving its chunk and timestamp alone.
    pub fn set_index(&mut self, index: u64) {
        self.meta.index = index;
    }

    pub fn plane(&self, channel: Channel) -> Option<&HostPlane> {
        let index = self.position_of(channel)?;

        self.planes.get(index)
    }

    /// Takes this frame for writing, or hands it back when somebody else holds a plane.
    pub fn try_into_mut(self) -> Result<HostFrameMut, Self> {
        if self.planes.iter().all(HostPlane::is_exclusive) {
            return Ok(HostFrameMut { frame: self });
        }

        Err(self)
    }

    fn position_of(&self, channel: Channel) -> Option<usize> {
        self.format
            .channels()
            .iter()
            .position(|carried| *carried == channel)
    }
}

/// A host frame no one else holds, so its planes can be written.
#[derive(Debug)]
pub struct HostFrameMut {
    frame: HostFrame,
}

impl HostFrameMut {
    pub const fn format(&self) -> &FrameFormat {
        self.frame.format()
    }

    pub const fn meta(&self) -> &FrameMeta {
        self.frame.meta()
    }

    pub fn plane(&self, channel: Channel) -> Option<&HostPlane> {
        self.frame.plane(channel)
    }

    pub fn plane_mut(&mut self, channel: Channel) -> Option<&mut [u8]> {
        let index = self.frame.position_of(channel)?;

        Some(self.frame.planes[index].bytes_mut())
    }

    /// Declares what the samples now mean, for a stage that converted them in place.
    pub fn set_colour(&mut self, colour: ColourInfo) {
        self.frame.format.colour = colour;
    }

    /// Hands the frame back, ready to pass downstream.
    pub fn freeze(self) -> HostFrame {
        self.frame
    }
}

/// A mask on the host.
#[derive(Debug, Clone)]
pub struct HostMask {
    format: MaskFormat,
    plane: HostPlane,
    meta: FrameMeta,
    chroma: Option<HostPlane>,
}

impl HostMask {
    pub(crate) fn new(
        format: MaskFormat,
        plane: HostPlane,
        meta: FrameMeta,
        chroma: Option<HostPlane>,
    ) -> Self {
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

    pub const fn format(&self) -> &MaskFormat {
        &self.format
    }

    pub const fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        &mut self.meta
    }

    /// Moves this to `index` in the stream, leaving its chunk and timestamp alone.
    pub fn set_index(&mut self, index: u64) {
        self.meta.index = index;
    }

    pub const fn plane(&self) -> &HostPlane {
        &self.plane
    }

    pub const fn chroma_plane(&self) -> Option<&HostPlane> {
        self.chroma.as_ref()
    }

    /// Takes this mask for writing, or hands it back when somebody else holds a plane.
    pub fn try_into_mut(self) -> Result<HostMaskMut, Self> {
        let chroma_is_exclusive = self.chroma.as_ref().is_none_or(HostPlane::is_exclusive);
        if self.plane.is_exclusive() && chroma_is_exclusive {
            return Ok(HostMaskMut { mask: self });
        }

        Err(self)
    }
}

/// A host mask no one else holds, so its planes can be written.
#[derive(Debug)]
pub struct HostMaskMut {
    mask: HostMask,
}

impl HostMaskMut {
    pub const fn format(&self) -> &MaskFormat {
        self.mask.format()
    }

    pub const fn meta(&self) -> &FrameMeta {
        self.mask.meta()
    }

    pub fn plane_mut(&mut self) -> &mut [u8] {
        self.mask.plane.bytes_mut()
    }

    pub fn chroma_plane_mut(&mut self) -> Option<&mut [u8]> {
        let chroma = self.mask.chroma.as_mut()?;

        Some(chroma.bytes_mut())
    }

    /// Hands the mask back, ready to pass downstream.
    pub fn freeze(self) -> HostMask {
        self.mask
    }
}

/// One item flowing along a host edge of the pipeline.
#[derive(Debug, Clone)]
pub enum HostStreamItem {
    Frame(HostFrame),
    Mask(HostMask),
}

impl HostStreamItem {
    pub const fn is_frame(&self) -> bool {
        matches!(self, Self::Frame(_))
    }

    pub const fn is_mask(&self) -> bool {
        matches!(self, Self::Mask(_))
    }

    pub fn as_frame(&self) -> Option<&HostFrame> {
        match self {
            Self::Frame(frame) => Some(frame),
            Self::Mask(_) => None,
        }
    }

    pub fn as_mask(&self) -> Option<&HostMask> {
        match self {
            Self::Mask(mask) => Some(mask),
            Self::Frame(_) => None,
        }
    }

    pub fn into_frame(self) -> Result<HostFrame, Self> {
        match self {
            Self::Frame(frame) => Ok(frame),
            Self::Mask(_) => Err(self),
        }
    }

    pub fn into_mask(self) -> Result<HostMask, Self> {
        match self {
            Self::Mask(mask) => Ok(mask),
            Self::Frame(_) => Err(self),
        }
    }

    pub fn format(&self) -> StreamFormat {
        match self {
            Self::Frame(frame) => StreamFormat::Frames(*frame.format()),
            Self::Mask(mask) => StreamFormat::Mask(*mask.format()),
        }
    }

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
        self.meta_mut().index = index;
    }

    /// The bytes carrying `channel`.
    ///
    /// A mask's full plane answers to [Channel::Y](crate::Channel::Y), and its chroma plane to both
    /// [Channel::U](crate::Channel::U) and [Channel::V](crate::Channel::V).
    pub(crate) fn plane_bytes(&self, channel: Channel) -> Option<&[u8]> {
        match (self, channel) {
            (Self::Frame(frame), _) => frame.plane(channel).map(HostPlane::bytes),
            (Self::Mask(mask), Channel::Y) => Some(mask.plane().bytes()),
            (Self::Mask(mask), Channel::U | Channel::V) => mask.chroma_plane().map(HostPlane::bytes),
            (Self::Mask(_), _) => None,
        }
    }
}

impl From<HostFrame> for HostStreamItem {
    fn from(frame: HostFrame) -> Self {
        Self::Frame(frame)
    }
}

impl From<HostMask> for HostStreamItem {
    fn from(mask: HostMask) -> Self {
        Self::Mask(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{
        Channel,
        ChromaLocation,
        ColourInfo,
        FrameLayout,
        FrameMeta,
        MaskChroma,
        SampleFormat,
        StreamFormat,
        Subsampling,
    };
    use crate::{FrameBudget, HostPool};

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

    fn mask_format() -> MaskFormat {
        MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: None,
        }
    }

    #[test]
    fn a_fresh_frame_has_one_plane_per_channel_at_its_own_size() {
        let frame = HostPool::new().alloc_frame(&format(), FrameMeta::at(3)).freeze();

        assert_eq!(frame.plane(Channel::Y).expect("luma").bytes().len(), 64 * 64);
        assert_eq!(frame.plane(Channel::U).expect("chroma").bytes().len(), 32 * 32);
        assert_eq!(frame.plane(Channel::V).expect("chroma").width(), 32);
        assert!(frame.plane(Channel::R).is_none());
        assert_eq!(frame.meta().index, 3);
    }

    #[test]
    fn cloning_a_host_frame_shares_its_planes() {
        let frame = HostPool::new().alloc_frame(&format(), FrameMeta::at(0)).freeze();
        let clone = frame.clone();

        let original = frame.plane(Channel::Y).expect("luma").bytes().as_ptr();
        let shared = clone.plane(Channel::Y).expect("luma").bytes().as_ptr();

        assert_eq!(original, shared);
    }

    #[test]
    fn a_frame_somebody_else_holds_cannot_be_written() {
        let frame = HostPool::new().alloc_frame(&format(), FrameMeta::at(0)).freeze();
        let held = frame.clone();

        let frame = frame.try_into_mut().expect_err("a clone is still alive");
        drop(held);

        assert!(frame.try_into_mut().is_ok());
    }

    #[test]
    fn what_is_written_through_a_writable_frame_is_what_it_freezes_with() {
        let mut frame = HostPool::new().alloc_frame(&format(), FrameMeta::at(0));
        frame.plane_mut(Channel::U).expect("chroma").fill(9);

        let frame = frame.freeze();

        assert!(
            frame
                .plane(Channel::U)
                .expect("chroma")
                .bytes()
                .iter()
                .all(|&byte| byte == 9)
        );
    }

    #[test]
    fn a_host_mask_is_one_plane_that_answers_to_luma() {
        let mut mask = HostPool::new().alloc_mask(&mask_format(), FrameMeta::at(0));
        mask.plane_mut().fill(4);
        let item = HostStreamItem::from(mask.freeze());

        assert_eq!(
            item.plane_bytes(Channel::Y).expect("the one plane").len(),
            64 * 64
        );
        assert!(item.plane_bytes(Channel::U).is_none());
        assert!(item.is_mask());
    }

    fn chroma_mask_format() -> MaskFormat {
        let base = mask_format();

        MaskFormat {
            chroma: Some(MaskChroma {
                subsampling: Subsampling::Yuv420,
                location: ChromaLocation::Left,
            }),
            ..base
        }
    }

    #[test]
    fn a_host_mask_without_chroma_has_no_chroma_plane() {
        let format = mask_format();
        let meta = FrameMeta::at(0);
        let mut mask = HostPool::new().alloc_mask(&format, meta);

        assert!(mask.chroma_plane_mut().is_none());
        assert!(mask.freeze().chroma_plane().is_none());
    }

    #[test]
    fn a_host_mask_with_chroma_carries_a_chroma_plane_that_answers_to_u_and_v() {
        let format = chroma_mask_format();
        let meta = FrameMeta::at(0);
        let mut mask = HostPool::new().alloc_mask(&format, meta);
        mask.plane_mut().fill(4);
        mask.chroma_plane_mut().expect("the mask carries chroma").fill(6);

        let frozen = mask.freeze();
        let chroma = frozen.chroma_plane().expect("the mask carries chroma");
        assert_eq!((chroma.width(), chroma.height()), (32, 32));

        let item = HostStreamItem::from(frozen);
        let luma = item.plane_bytes(Channel::Y).expect("the full plane");
        let blue = item.plane_bytes(Channel::U).expect("the chroma plane");
        let red = item.plane_bytes(Channel::V).expect("the chroma plane");

        assert!(luma.iter().all(|&byte| byte == 4));
        assert_eq!(blue.len(), 32 * 32);
        assert!(blue.iter().all(|&byte| byte == 6));
        assert_eq!(blue, red);
    }

    #[test]
    fn a_host_mask_whose_chroma_plane_is_shared_cannot_be_written() {
        let format = chroma_mask_format();
        let meta = FrameMeta::at(0);
        let mask = HostPool::new().alloc_mask(&format, meta).freeze();
        let held = mask.chroma_plane().expect("the mask carries chroma").clone();

        let mask = mask.try_into_mut().expect_err("the chroma plane is still held");
        drop(held);

        assert!(mask.try_into_mut().is_ok());
    }

    #[test]
    fn a_host_mask_with_chroma_is_charged_for_both_planes() {
        let budget = FrameBudget::new(1 << 20);
        let shared_budget = budget.clone();
        let pool = HostPool::new().with_budget(shared_budget);
        let format = chroma_mask_format();
        let meta = FrameMeta::at(0);

        let mask = pool.alloc_mask(&format, meta);
        assert_eq!(budget.outstanding(), 64 * 64 + 32 * 32);

        drop(mask);
        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn restamping_an_item_changes_its_place_but_not_its_samples() {
        let frame = HostPool::new().alloc_frame(&format(), FrameMeta::at(0)).freeze();
        let mut item = HostStreamItem::from(frame);

        item.meta_mut().chunk = 5;
        item.set_index(9);

        assert_eq!(item.meta().chunk, 5);
        assert_eq!(item.meta().index, 9);
        assert_eq!(item.format(), StreamFormat::Frames(format()));
    }

    #[test]
    fn a_whole_frame_is_charged_to_the_budget_until_its_last_holder_drops() {
        let budget = FrameBudget::new(1 << 20);
        let pool = HostPool::new().with_budget(budget.clone());

        let frame = pool.alloc_frame(&format(), FrameMeta::at(0)).freeze();
        let clone = frame.clone();
        assert_eq!(budget.outstanding(), format().total_bytes());

        drop(frame);
        assert_eq!(budget.outstanding(), format().total_bytes());

        drop(clone);
        assert_eq!(budget.outstanding(), 0);
    }
}
