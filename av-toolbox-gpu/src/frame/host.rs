use crate::frame::format::StreamFormat;
use crate::frame::gpu::FrameMeta;
use crate::frame::layout::Channel;

/// One item that came back from the device, with where it sits in its stream.
///
/// This is what [LoadItem](crate::LoadItem) turns back into, so a sink gets each frame's index and
/// timestamp alongside its samples rather than tracking them in a list of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostItem {
    planes: HostPlanes,
    meta: FrameMeta,
}

impl HostItem {
    pub fn new(planes: HostPlanes, meta: FrameMeta) -> Self {
        Self { planes, meta }
    }

    pub const fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    pub const fn planes(&self) -> &HostPlanes {
        &self.planes
    }

    pub fn into_planes(self) -> HostPlanes {
        self.planes
    }

    /// How many planes this item holds.
    pub fn len(&self) -> usize {
        self.planes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.planes.is_empty()
    }

    pub fn plane(&self, channel: Channel) -> Option<&[u8]> {
        self.planes.plane(channel)
    }

    pub fn plane_mut(&mut self, channel: Channel) -> Option<&mut [u8]> {
        self.planes.plane_mut(channel)
    }

    /// Resizes to hold `format` and takes `meta` as its new place in the stream.
    ///
    /// Whatever allocations already fit are kept, so a caller reading batch after batch into the
    /// same items allocates nothing per batch.
    pub(crate) fn reshape(&mut self, format: &StreamFormat, meta: FrameMeta) {
        self.planes.resize_for(format);
        self.meta = meta;
    }
}

/// Host-side planes, tightly packed, one per channel.
///
/// A frame is loaded from these and unloaded into them. [HostPlanes::resize_for] keeps the
/// allocations it already has, so a caller running frame after frame allocates nothing per frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPlanes {
    format: StreamFormat,
    planes: Vec<Vec<u8>>,
}

impl HostPlanes {
    /// Planes sized for `format`, filled with zeroes.
    pub fn zeroed(format: &StreamFormat) -> Self {
        let mut planes = Self {
            format: *format,
            planes: Vec::new(),
        };
        planes.resize_for(format);

        planes
    }

    /// Resizes to hold `format`, keeping whatever allocations already fit.
    ///
    /// Existing bytes are left as they are and only added ones are zeroed, so a caller that wants
    /// a blank set of planes builds one with [HostPlanes::zeroed].
    pub fn resize_for(&mut self, format: &StreamFormat) {
        let sizes = format.plane_sizes();

        self.planes.truncate(sizes.len());
        while self.planes.len() < sizes.len() {
            self.planes.push(Vec::new());
        }

        for (plane, bytes) in self.planes.iter_mut().zip(sizes) {
            plane.resize(bytes, 0);
        }

        self.format = *format;
    }

    pub const fn format(&self) -> &StreamFormat {
        &self.format
    }

    /// How many planes these hold.
    pub fn len(&self) -> usize {
        self.planes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.planes.is_empty()
    }

    pub fn plane(&self, channel: Channel) -> Option<&[u8]> {
        let index = self.index_of(channel)?;

        Some(&self.planes[index])
    }

    pub fn plane_mut(&mut self, channel: Channel) -> Option<&mut [u8]> {
        let index = self.index_of(channel)?;

        Some(&mut self.planes[index])
    }

    /// Which plane holds `channel`, or `None` when this format does not carry it.
    ///
    /// A mask is a single plane and answers to [Channel::Y], because it has one channel and no
    /// colour meaning to give it a name of its own.
    fn index_of(&self, channel: Channel) -> Option<usize> {
        match &self.format {
            StreamFormat::Frames(format) => format.channels().iter().position(|carried| *carried == channel),
            StreamFormat::Mask(_) => (channel == Channel::Y).then_some(0),
        }
    }

    /// The format and the planes themselves, moved rather than copied.
    pub(crate) fn into_parts(self) -> (StreamFormat, Vec<Vec<u8>>) {
        (self.format, self.planes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::format::{ColourInfo, FrameFormat, MaskFormat};
    use crate::frame::layout::{FrameLayout, Subsampling};
    use crate::frame::sample::SampleFormat;

    fn yuv420(width: u32, height: u32, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width,
            height,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample,
            colour: ColourInfo::default(),
        })
    }

    #[test]
    fn planes_are_sized_from_the_format() {
        let planes = HostPlanes::zeroed(&yuv420(1920, 1080, SampleFormat::U8));

        assert_eq!(planes.len(), 3);
        assert_eq!(
            planes.plane(Channel::Y).expect("should carry luma").len(),
            1920 * 1080
        );
        assert_eq!(
            planes.plane(Channel::U).expect("should carry chroma").len(),
            960 * 540
        );
    }

    #[test]
    fn a_channel_the_format_does_not_carry_is_absent() {
        let planes = HostPlanes::zeroed(&yuv420(64, 64, SampleFormat::U8));

        assert!(planes.plane(Channel::R).is_none());
    }

    #[test]
    fn a_mask_has_exactly_one_plane() {
        let mask = StreamFormat::Mask(MaskFormat {
            width: 64,
            height: 32,
            sample: SampleFormat::U8,
        });

        let planes = HostPlanes::zeroed(&mask);
        assert_eq!(planes.len(), 1);
        assert_eq!(
            planes.plane(Channel::Y).expect("should carry the mask").len(),
            64 * 32
        );
    }

    /// Reusing one buffer across frames is the point, so resizing to the same format must not
    /// throw the allocation away.
    #[test]
    fn resizing_to_the_same_format_keeps_the_allocation() {
        let format = yuv420(1920, 1080, SampleFormat::U8);
        let mut planes = HostPlanes::zeroed(&format);

        let before = planes.plane(Channel::Y).expect("should carry luma").as_ptr();
        planes.resize_for(&format);
        let after = planes.plane(Channel::Y).expect("should carry luma").as_ptr();

        assert_eq!(before, after, "the luma allocation was replaced");
    }

    #[test]
    fn resizing_to_a_larger_format_grows_every_plane() {
        let mut planes = HostPlanes::zeroed(&yuv420(64, 64, SampleFormat::U8));
        planes.resize_for(&yuv420(128, 128, SampleFormat::U8));

        assert_eq!(
            planes.plane(Channel::Y).expect("should carry luma").len(),
            128 * 128
        );
        assert_eq!(
            planes.plane(Channel::U).expect("should carry chroma").len(),
            64 * 64
        );
    }

    #[test]
    fn resizing_between_layouts_changes_the_plane_count() {
        let mut planes = HostPlanes::zeroed(&yuv420(64, 64, SampleFormat::U8));
        assert_eq!(planes.len(), 3);

        let grey = StreamFormat::Frames(FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Grey,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
        });
        planes.resize_for(&grey);

        assert_eq!(planes.len(), 1);
        assert!(planes.plane(Channel::U).is_none());
    }

    #[test]
    fn a_plane_can_be_written_through() {
        let mut planes = HostPlanes::zeroed(&yuv420(4, 4, SampleFormat::U8));

        planes.plane_mut(Channel::Y).expect("should carry luma")[0] = 42;
        assert_eq!(planes.plane(Channel::Y).expect("should carry luma")[0], 42);
    }

    #[test]
    fn resizing_to_the_same_format_keeps_what_the_planes_hold() {
        let format = yuv420(64, 64, SampleFormat::U8);
        let mut planes = HostPlanes::zeroed(&format);

        planes.plane_mut(Channel::Y).expect("should carry luma")[0] = 9;
        planes.resize_for(&format);

        assert_eq!(planes.plane(Channel::Y).expect("should carry luma")[0], 9);
    }
}
