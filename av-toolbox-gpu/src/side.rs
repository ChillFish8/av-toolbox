use std::fmt;

use crate::frame::{Frame, FrameMeta, HostFrame, HostMask, HostStreamItem, Mask, StreamFormat, StreamItem};

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::DeviceSide {}
    impl Sealed for super::HostSide {}
}

/// Where a stage's items live, as a type, so a stage is only ever handed items from its own side.
pub trait Side: sealed::Sealed + fmt::Debug + Send + Sync + 'static {
    type Frame: fmt::Debug + Send + 'static;
    type Mask: fmt::Debug + Send + 'static;
    type Item: fmt::Debug + Clone + Send + 'static;

    /// What this side is called in a message.
    const NAME: &'static str;

    fn frame_item(frame: Self::Frame) -> Self::Item;

    fn mask_item(mask: Self::Mask) -> Self::Item;

    fn into_frame(item: Self::Item) -> Result<Self::Frame, Self::Item>;

    fn into_mask(item: Self::Item) -> Result<Self::Mask, Self::Item>;

    fn into_any(item: Self::Item) -> AnyItem;

    /// The item underneath, or `item` handed back when it is from the other side.
    fn from_any(item: AnyItem) -> Result<Self::Item, AnyItem>;
}

/// The accelerator's memory.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct DeviceSide;

/// The CPU's memory.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct HostSide;

impl Side for DeviceSide {
    type Frame = Frame;
    type Mask = Mask;
    type Item = StreamItem;

    const NAME: &'static str = "device";

    fn frame_item(frame: Frame) -> StreamItem {
        StreamItem::Frame(frame)
    }

    fn mask_item(mask: Mask) -> StreamItem {
        StreamItem::Mask(mask)
    }

    fn into_frame(item: StreamItem) -> Result<Frame, StreamItem> {
        item.into_frame()
    }

    fn into_mask(item: StreamItem) -> Result<Mask, StreamItem> {
        item.into_mask()
    }

    fn into_any(item: StreamItem) -> AnyItem {
        AnyItem::Device(item)
    }

    fn from_any(item: AnyItem) -> Result<StreamItem, AnyItem> {
        match item {
            AnyItem::Device(item) => Ok(item),
            other => Err(other),
        }
    }
}

impl Side for HostSide {
    type Frame = HostFrame;
    type Mask = HostMask;
    type Item = HostStreamItem;

    const NAME: &'static str = "host";

    fn frame_item(frame: HostFrame) -> HostStreamItem {
        HostStreamItem::Frame(frame)
    }

    fn mask_item(mask: HostMask) -> HostStreamItem {
        HostStreamItem::Mask(mask)
    }

    fn into_frame(item: HostStreamItem) -> Result<HostFrame, HostStreamItem> {
        item.into_frame()
    }

    fn into_mask(item: HostStreamItem) -> Result<HostMask, HostStreamItem> {
        item.into_mask()
    }

    fn into_any(item: HostStreamItem) -> AnyItem {
        AnyItem::Host(item)
    }

    fn from_any(item: AnyItem) -> Result<HostStreamItem, AnyItem> {
        match item {
            AnyItem::Host(item) => Ok(item),
            other => Err(other),
        }
    }
}

/// An item from either side.
#[derive(Debug, Clone)]
pub enum AnyItem {
    Device(StreamItem),
    Host(HostStreamItem),
}

impl AnyItem {
    pub const fn meta(&self) -> &FrameMeta {
        match self {
            Self::Device(item) => item.meta(),
            Self::Host(item) => item.meta(),
        }
    }

    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        match self {
            Self::Device(item) => item.meta_mut(),
            Self::Host(item) => item.meta_mut(),
        }
    }

    pub fn set_index(&mut self, index: u64) {
        self.meta_mut().index = index;
    }

    pub fn format(&self) -> StreamFormat {
        match self {
            Self::Device(item) => item.format(),
            Self::Host(item) => item.format(),
        }
    }

    pub const fn is_frame(&self) -> bool {
        match self {
            Self::Device(item) => item.is_frame(),
            Self::Host(item) => item.is_frame(),
        }
    }

    pub const fn is_mask(&self) -> bool {
        !self.is_frame()
    }

    pub const fn is_host(&self) -> bool {
        matches!(self, Self::Host(_))
    }

    pub fn into_device(self) -> Option<StreamItem> {
        match self {
            Self::Device(item) => Some(item),
            Self::Host(_) => None,
        }
    }

    pub fn into_host(self) -> Option<HostStreamItem> {
        match self {
            Self::Host(item) => Some(item),
            Self::Device(_) => None,
        }
    }
}

impl From<StreamItem> for AnyItem {
    fn from(item: StreamItem) -> Self {
        Self::Device(item)
    }
}

impl From<HostStreamItem> for AnyItem {
    fn from(item: HostStreamItem) -> Self {
        Self::Host(item)
    }
}

impl From<Frame> for AnyItem {
    fn from(frame: Frame) -> Self {
        Self::Device(StreamItem::Frame(frame))
    }
}

impl From<Mask> for AnyItem {
    fn from(mask: Mask) -> Self {
        Self::Device(StreamItem::Mask(mask))
    }
}

impl From<HostFrame> for AnyItem {
    fn from(frame: HostFrame) -> Self {
        Self::Host(HostStreamItem::Frame(frame))
    }
}

impl From<HostMask> for AnyItem {
    fn from(mask: HostMask) -> Self {
        Self::Host(HostStreamItem::Mask(mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{ColourInfo, FrameFormat, FrameLayout, MaskFormat, SampleFormat, Subsampling};
    use crate::{HostPool, testing_context};

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
        }
    }

    fn device_item(index: u64) -> StreamItem {
        StreamItem::Frame(testing_context().alloc_frame(&format(), FrameMeta::at(index)))
    }

    fn host_item(index: u64) -> HostStreamItem {
        HostStreamItem::Frame(
            HostPool::new()
                .alloc_frame(&format(), FrameMeta::at(index))
                .freeze(),
        )
    }

    #[test]
    fn an_item_comes_back_out_of_the_side_it_went_in_on() {
        let any = DeviceSide::into_any(device_item(2));
        let item = DeviceSide::from_any(any).expect("a device item");

        assert_eq!(item.meta().index, 2);
    }

    #[test]
    fn an_item_from_the_other_side_is_handed_back_untouched() {
        let any = AnyItem::from(host_item(4));

        let refused = DeviceSide::from_any(any).expect_err("a host item is not a device item");

        assert!(refused.is_host());
        assert_eq!(refused.meta().index, 4);
    }

    #[test]
    fn a_mask_is_not_taken_as_a_frame() {
        let mask = testing_context().alloc_mask(
            &MaskFormat {
                width: 8,
                height: 8,
                sample: SampleFormat::U8,
                chroma: None,
            },
            FrameMeta::at(0),
        );
        let item = DeviceSide::mask_item(mask);

        let back = DeviceSide::into_frame(item).expect_err("a mask is not a frame");

        assert!(back.as_mask().is_some());
    }

    #[test]
    fn restamping_any_item_reaches_the_item_underneath() {
        let mut device = AnyItem::from(device_item(0));
        let mut host = AnyItem::from(host_item(0));

        device.meta_mut().chunk = 3;
        host.set_index(7);

        assert_eq!(device.into_device().expect("device").meta().chunk, 3);
        assert_eq!(host.into_host().expect("host").meta().index, 7);
    }

    #[test]
    fn each_side_names_itself() {
        assert_eq!(DeviceSide::NAME, "device");
        assert_eq!(HostSide::NAME, "host");
    }
}
