use std::fmt;

use crate::frame::layout::{Channel, FrameLayout, Subsampling};
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

/// Where chroma samples sit relative to the luma samples they cover.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ChromaLocation {
    #[default]
    Left,
    Center,
    TopLeft,
    Top,
    BottomLeft,
    Bottom,
}

/// Where along an axis a chroma sample sits within the samples it covers.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Siting {
    First,
    Centred,
    Last,
}

impl Siting {
    fn offset(self, factor: u32) -> f32 {
        let span = factor as f32 - 1.0;

        match self {
            Self::First => 0.0,
            Self::Centred => span / 2.0,
            Self::Last => span,
        }
    }
}

impl ChromaLocation {
    /// How far a chroma sample sits past `factor * i` along each axis, in luma samples.
    pub fn offsets(self, factor_x: u32, factor_y: u32) -> (f32, f32) {
        let (horizontal, vertical) = match self {
            Self::Left => (Siting::First, Siting::Centred),
            Self::Center => (Siting::Centred, Siting::Centred),
            Self::TopLeft => (Siting::First, Siting::First),
            Self::Top => (Siting::Centred, Siting::First),
            Self::BottomLeft => (Siting::First, Siting::Last),
            Self::Bottom => (Siting::Centred, Siting::Last),
        };

        let offset_x = horizontal.offset(factor_x);
        let offset_y = vertical.offset(factor_y);

        (offset_x, offset_y)
    }
}

/// How a frame's samples are to be interpreted as colour.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ColourInfo {
    pub range: ColourRange,
    pub matrix: Matrix,
    pub primaries: Primaries,
    pub transfer: Transfer,
    pub chroma_location: ChromaLocation,
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

/// The shape of a mask's chroma plane, as a subsampling and where its samples sit.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct MaskChroma {
    pub subsampling: Subsampling,
    pub location: ChromaLocation,
}

impl MaskChroma {
    /// How many luma samples one chroma sample spans across and down.
    pub const fn factors(&self) -> (u32, u32) {
        self.subsampling.factors()
    }
}

/// What a mask is, which is a full plane at its own dimensions and an optional chroma plane.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct MaskFormat {
    pub width: u32,
    pub height: u32,
    pub sample: SampleFormat,
    /// The chroma plane this mask carries, `None` when it has only the full plane.
    pub chroma: Option<MaskChroma>,
}

impl MaskFormat {
    pub const fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// How many bytes the full plane occupies.
    pub const fn bytes(&self) -> usize {
        self.samples() * self.sample.bytes()
    }

    /// The dimensions of the chroma plane, or `None` when this mask has none.
    pub fn chroma_dimensions(&self) -> Option<(u32, u32)> {
        let chroma = self.chroma?;
        let dimensions = chroma.subsampling.chroma_dimensions(self.width, self.height);

        Some(dimensions)
    }

    /// How many bytes the chroma plane occupies, which is 0 when this mask has none.
    pub fn chroma_bytes(&self) -> usize {
        let Some((width, height)) = self.chroma_dimensions() else {
            return 0;
        };

        width as usize * height as usize * self.sample.bytes()
    }
}

impl fmt::Display for MaskFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "mask {}x{} {}", self.width, self.height, self.sample)?;

        if let Some(chroma) = self.chroma {
            write!(formatter, " with {} chroma", chroma.subsampling)?;
        }

        Ok(())
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
            Self::Mask(format) if format.chroma.is_some() => vec![format.bytes(), format.chroma_bytes()],
            Self::Mask(format) => vec![format.bytes()],
        }
    }

    /// How many bytes every plane occupies together.
    pub fn total_bytes(&self) -> usize {
        match self {
            Self::Frames(format) => format.total_bytes(),
            Self::Mask(format) => format.bytes() + format.chroma_bytes(),
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
            chroma: None,
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
            chroma: None,
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
            chroma: None,
        });

        assert_eq!(frames.sample(), SampleFormat::U8);
        assert_eq!(frames.dimensions(), (1920, 1080));
        assert_eq!(mask.dimensions(), (8, 8));

        let wide = StreamFormat::Mask(MaskFormat {
            width: 8,
            height: 8,
            sample: SampleFormat::U16 { depth: Depth::TEN },
            chroma: None,
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
            chroma: None,
        };
        let mask = StreamFormat::Mask(mask_format);

        assert_eq!(frames.as_frames(), Some(&inner));
        assert_eq!(frames.as_mask(), None);
        assert_eq!(mask.as_mask(), Some(&mask_format));
        assert_eq!(mask.as_frames(), None);
    }

    fn chroma_mask(width: u32, height: u32, subsampling: Subsampling) -> MaskFormat {
        MaskFormat {
            width,
            height,
            sample: SampleFormat::U8,
            chroma: Some(MaskChroma {
                subsampling,
                location: ChromaLocation::Left,
            }),
        }
    }

    #[test]
    fn a_420_mask_carries_a_chroma_plane_of_half_size_rounded_up() {
        let format = chroma_mask(7, 5, Subsampling::Yuv420);
        let stream = StreamFormat::Mask(format);

        assert_eq!(format.chroma_dimensions(), Some((4, 3)));
        assert_eq!(format.chroma_bytes(), 12);
        assert_eq!(stream.plane_sizes(), vec![35, 12]);
        assert_eq!(stream.total_bytes(), 47);
    }

    #[test]
    fn a_mask_without_chroma_has_one_plane() {
        let base = chroma_mask(7, 5, Subsampling::Yuv420);
        let format = MaskFormat { chroma: None, ..base };
        let stream = StreamFormat::Mask(format);

        assert_eq!(format.chroma_dimensions(), None);
        assert_eq!(format.chroma_bytes(), 0);
        assert_eq!(stream.plane_sizes(), vec![35]);
        assert_eq!(stream.total_bytes(), 35);
    }

    #[test]
    fn mask_chroma_factors_follow_the_subsampling() {
        let factors_of = |subsampling| {
            let chroma = MaskChroma {
                subsampling,
                location: ChromaLocation::Left,
            };

            chroma.factors()
        };

        assert_eq!(factors_of(Subsampling::Yuv420), (2, 2));
        assert_eq!(factors_of(Subsampling::Yuv422), (2, 1));
        assert_eq!(factors_of(Subsampling::Yuv444), (1, 1));
    }

    #[test]
    fn a_mask_with_chroma_renders_its_subsampling() {
        let format = chroma_mask(64, 32, Subsampling::Yuv420);
        let mask = StreamFormat::Mask(format);

        assert_eq!(mask.to_string(), "mask 64x32 u8 with 4:2:0 chroma");
    }

    #[test]
    fn chroma_location_defaults_to_left() {
        assert_eq!(ColourInfo::default().chroma_location, ChromaLocation::Left);
    }

    #[test]
    fn each_chroma_location_places_its_samples() {
        assert_eq!(ChromaLocation::Left.offsets(2, 2), (0.0, 0.5));
        assert_eq!(ChromaLocation::Center.offsets(2, 2), (0.5, 0.5));
        assert_eq!(ChromaLocation::TopLeft.offsets(2, 2), (0.0, 0.0));
        assert_eq!(ChromaLocation::Top.offsets(2, 2), (0.5, 0.0));
        assert_eq!(ChromaLocation::BottomLeft.offsets(2, 2), (0.0, 1.0));
        assert_eq!(ChromaLocation::Bottom.offsets(2, 2), (0.5, 1.0));
    }

    #[test]
    fn an_axis_that_is_not_subsampled_has_no_offset() {
        assert_eq!(ChromaLocation::Bottom.offsets(2, 1), (0.5, 0.0));
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
            chroma: None,
        });
        assert_eq!(mask.to_string(), "mask 64x32 u8");
    }
}
