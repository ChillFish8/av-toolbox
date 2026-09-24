use std::fmt;

use crate::frame::layout::{Channel, FrameLayout};
use crate::frame::rate::FrameRate;
use crate::frame::sample::SampleFormat;

/// Whether samples use the full range of their depth or the narrower video range.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ColourRange {
    #[default]
    Unspecified,
    Limited,
    Full,
}

/// The matrix relating a frame's colour channels to each other.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Matrix {
    #[default]
    Unspecified,
    Bt709,
    Bt601,
    Bt2020Ncl,
    Identity,
}

/// The colour primaries a frame's samples are expressed in.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Primaries {
    #[default]
    Unspecified,
    Bt709,
    Bt601,
    Bt2020,
}

/// The transfer function relating a frame's samples to light.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Transfer {
    #[default]
    Unspecified,
    Bt709,
    Srgb,
    Pq,
    Hlg,
}

/// How a frame's samples are to be interpreted as colour.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ColourInfo {
    pub range: ColourRange,
    pub matrix: Matrix,
    pub primaries: Primaries,
    pub transfer: Transfer,
}

/// What a frame is.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FrameFormat {
    pub width: u32,
    pub height: u32,
    pub layout: FrameLayout,
    pub sample: SampleFormat,
    pub colour: ColourInfo,
    /// How many frames the stream shows per second, `None` when it is unknown.
    pub frame_rate: Option<FrameRate>,
}

impl FrameFormat {
    /// The channels this frame carries, in plane order.
    pub const fn channels(&self) -> &'static [Channel] {
        self.layout.channels()
    }

    /// The dimensions of one channel's plane, or `None` when this frame does not carry it.
    pub fn channel_dimensions(&self, channel: Channel) -> Option<(u32, u32)> {
        self.layout.channel_dimensions(channel, self.width, self.height)
    }

    /// How many samples one channel's plane holds.
    pub fn channel_samples(&self, channel: Channel) -> Option<usize> {
        let (width, height) = self.channel_dimensions(channel)?;

        Some(width as usize * height as usize)
    }

    /// How many bytes one channel's plane occupies.
    pub fn channel_bytes(&self, channel: Channel) -> Option<usize> {
        let samples = self.channel_samples(channel)?;

        Some(samples * self.sample.bytes())
    }

    /// The byte size of every plane this frame carries, in plane order.
    pub fn plane_sizes(&self) -> Vec<usize> {
        self.channels()
            .iter()
            .map(|channel| {
                self.channel_bytes(*channel)
                    .expect("the layout carries every channel it lists")
            })
            .collect()
    }

    /// How many bytes every plane of this frame occupies together.
    pub fn total_bytes(&self) -> usize {
        self.plane_sizes().iter().sum()
    }
}

impl fmt::Display for FrameFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}x{} {} {}",
            self.width, self.height, self.layout, self.sample,
        )
    }
}

/// What a mask is, which is a single plane at its own dimensions.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct MaskFormat {
    pub width: u32,
    pub height: u32,
    pub sample: SampleFormat,
}

impl MaskFormat {
    pub const fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub const fn bytes(&self) -> usize {
        self.samples() * self.sample.bytes()
    }
}

impl fmt::Display for MaskFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "mask {}x{} {}", self.width, self.height, self.sample)
    }
}

/// What flows along one edge of a pipeline.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum StreamFormat {
    Frames(FrameFormat),
    Mask(MaskFormat),
}

impl StreamFormat {
    pub const fn sample(&self) -> SampleFormat {
        match self {
            Self::Frames(format) => format.sample,
            Self::Mask(format) => format.sample,
        }
    }

    pub const fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Frames(format) => (format.width, format.height),
            Self::Mask(format) => (format.width, format.height),
        }
    }

    /// The frame format this carries, or `None` when it is a mask.
    pub const fn as_frames(&self) -> Option<&FrameFormat> {
        match self {
            Self::Frames(format) => Some(format),
            Self::Mask(_) => None,
        }
    }

    /// The mask format this carries, or `None` when it is a frame stream.
    pub const fn as_mask(&self) -> Option<&MaskFormat> {
        match self {
            Self::Mask(format) => Some(format),
            Self::Frames(_) => None,
        }
    }

    /// The byte size of every plane, in plane order.
    pub fn plane_sizes(&self) -> Vec<usize> {
        match self {
            Self::Frames(format) => format.plane_sizes(),
            Self::Mask(format) => vec![format.bytes()],
        }
    }

    /// How many bytes every plane occupies together.
    pub fn total_bytes(&self) -> usize {
        match self {
            Self::Frames(format) => format.total_bytes(),
            Self::Mask(format) => format.bytes(),
        }
    }
}

impl fmt::Display for StreamFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Frames(format) => write!(formatter, "{format}"),
            Self::Mask(format) => write!(formatter, "{format}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::layout::Subsampling;
    use crate::frame::sample::Depth;

    fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
        FrameFormat {
            width: 1920,
            height: 1080,
            layout,
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
        }
    }

    fn yuv420(sample: SampleFormat) -> FrameFormat {
        format(
            FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample,
        )
    }

    #[test]
    fn a_chroma_plane_holds_a_quarter_of_the_luma_samples_at_420() {
        let format = yuv420(SampleFormat::U8);

        assert_eq!(format.channel_samples(Channel::Y), Some(1920 * 1080));
        assert_eq!(format.channel_samples(Channel::U), Some(960 * 540));
        assert_eq!(format.channel_samples(Channel::V), Some(960 * 540));
    }

    #[test]
    fn plane_bytes_scale_with_the_sample_format() {
        let eight = yuv420(SampleFormat::U8);
        let float = yuv420(SampleFormat::F32);

        assert_eq!(eight.channel_bytes(Channel::Y), Some(1920 * 1080));
        assert_eq!(float.channel_bytes(Channel::Y), Some(1920 * 1080 * 4));
    }

    /// The figure the design leans on when sizing rings: a 4K 4:2:0 f32 frame is about 50 MB.
    #[test]
    fn a_four_k_float_frame_totals_about_fifty_megabytes() {
        let format = FrameFormat {
            width: 3840,
            height: 2160,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::F32,
            colour: ColourInfo::default(),
            frame_rate: None,
        };

        let expected = (3840 * 2160 + 2 * 1920 * 1080) * 4;
        assert_eq!(format.total_bytes(), expected);
        assert!(
            format.total_bytes() < 52 * 1024 * 1024,
            "{}",
            format.total_bytes()
        );
    }

    #[test]
    fn a_channel_the_layout_does_not_carry_has_no_geometry() {
        let format = yuv420(SampleFormat::U8);

        assert_eq!(format.channel_dimensions(Channel::R), None);
        assert_eq!(format.channel_samples(Channel::R), None);
        assert_eq!(format.channel_bytes(Channel::R), None);
    }

    #[test]
    fn a_grey_frame_totals_one_plane() {
        let format = format(FrameLayout::Grey, SampleFormat::U8);

        assert_eq!(format.channels(), &[Channel::Y]);
        assert_eq!(format.total_bytes(), 1920 * 1080);
    }

    #[test]
    fn a_mask_is_one_plane_at_its_own_dimensions() {
        let mask = MaskFormat {
            width: 1920,
            height: 1080,
            sample: SampleFormat::U8,
        };

        assert_eq!(mask.samples(), 1920 * 1080);
        assert_eq!(mask.bytes(), 1920 * 1080);
    }

    #[test]
    fn plane_sizes_are_listed_in_plane_order() {
        let format = yuv420(SampleFormat::U8);

        assert_eq!(format.plane_sizes(), vec![1920 * 1080, 960 * 540, 960 * 540]);
    }

    #[test]
    fn plane_sizes_sum_to_the_total() {
        let format = yuv420(SampleFormat::F32);
        let summed: usize = format.plane_sizes().iter().sum();

        assert_eq!(summed, (1920 * 1080 + 2 * 960 * 540) * 4);
        assert_eq!(format.total_bytes(), summed);
    }

    #[test]
    fn a_stream_format_sizes_either_kind() {
        let mask = StreamFormat::Mask(MaskFormat {
            width: 64,
            height: 32,
            sample: SampleFormat::U8,
        });

        assert_eq!(mask.plane_sizes(), vec![64 * 32]);
        assert_eq!(mask.total_bytes(), 64 * 32);

        let frames = StreamFormat::Frames(yuv420(SampleFormat::U8));
        assert_eq!(frames.plane_sizes().len(), 3);
    }

    #[test]
    fn a_stream_format_reports_which_kind_it_is() {
        let frames = StreamFormat::Frames(yuv420(SampleFormat::U8));
        let mask = StreamFormat::Mask(MaskFormat {
            width: 8,
            height: 8,
            sample: SampleFormat::U8,
        });

        assert_eq!(frames.sample(), SampleFormat::U8);
        assert_eq!(frames.dimensions(), (1920, 1080));
        assert_eq!(mask.dimensions(), (8, 8));

        let wide = StreamFormat::Mask(MaskFormat {
            width: 8,
            height: 8,
            sample: SampleFormat::U16 { depth: Depth::TEN },
        });
        assert_eq!(wide.sample(), SampleFormat::U16 { depth: Depth::TEN });
    }

    #[test]
    fn a_stream_format_unwraps_to_the_kind_it_holds_and_not_the_other() {
        let inner = yuv420(SampleFormat::U8);
        let frames = StreamFormat::Frames(inner);
        let mask_format = MaskFormat {
            width: 8,
            height: 8,
            sample: SampleFormat::U8,
        };
        let mask = StreamFormat::Mask(mask_format);

        assert_eq!(frames.as_frames(), Some(&inner));
        assert_eq!(frames.as_mask(), None);
        assert_eq!(mask.as_mask(), Some(&mask_format));
        assert_eq!(mask.as_frames(), None);
    }

    /// A negotiation failure names an edge with these, so they have to read as something a user can
    /// act on.
    #[test]
    fn a_format_renders_for_a_reader() {
        let frames = StreamFormat::Frames(yuv420(SampleFormat::U8));
        assert_eq!(frames.to_string(), "1920x1080 yuv 4:2:0 u8");

        let mask = StreamFormat::Mask(MaskFormat {
            width: 64,
            height: 32,
            sample: SampleFormat::U8,
        });
        assert_eq!(mask.to_string(), "mask 64x32 u8");
    }
}
