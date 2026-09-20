use std::fmt;

/// One channel of a frame.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Channel {
    Y,
    U,
    V,
    R,
    G,
    B,
}

impl fmt::Display for Channel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Y => "Y",
            Self::U => "U",
            Self::V => "V",
            Self::R => "R",
            Self::G => "G",
            Self::B => "B",
        };

        formatter.write_str(name)
    }
}

/// How far the chroma planes are subsampled from the luma plane.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Subsampling {
    Yuv444,
    Yuv422,
    Yuv420,
}

impl Subsampling {
    /// The dimensions of a chroma plane for a frame of `width` by `height`.
    ///
    /// An odd dimension rounds up, so the chroma plane still covers every luma sample.
    pub const fn chroma_dimensions(self, width: u32, height: u32) -> (u32, u32) {
        match self {
            Self::Yuv444 => (width, height),
            Self::Yuv422 => (width.div_ceil(2), height),
            Self::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
        }
    }
}

impl fmt::Display for Subsampling {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Yuv444 => "4:4:4",
            Self::Yuv422 => "4:2:2",
            Self::Yuv420 => "4:2:0",
        };

        formatter.write_str(name)
    }
}

/// Which channels a frame carries, and how they relate to each other.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum FrameLayout {
    Yuv { subsampling: Subsampling },
    Rgb,
    Grey,
}

impl FrameLayout {
    /// The channels this layout carries, in plane order.
    pub const fn channels(self) -> &'static [Channel] {
        match self {
            Self::Yuv { .. } => &[Channel::Y, Channel::U, Channel::V],
            Self::Rgb => &[Channel::R, Channel::G, Channel::B],
            Self::Grey => &[Channel::Y],
        }
    }

    pub fn carries(self, channel: Channel) -> bool {
        self.channels().contains(&channel)
    }

    /// The dimensions of one channel's plane, or `None` when this layout does not carry it.
    pub fn channel_dimensions(self, channel: Channel, width: u32, height: u32) -> Option<(u32, u32)> {
        if !self.carries(channel) {
            return None;
        }

        let dimensions = match (self, channel) {
            (Self::Yuv { subsampling }, Channel::U | Channel::V) => {
                subsampling.chroma_dimensions(width, height)
            },
            _ => (width, height),
        };

        Some(dimensions)
    }
}

impl fmt::Display for FrameLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Yuv { subsampling } => write!(formatter, "yuv {subsampling}"),
            Self::Rgb => formatter.write_str("rgb"),
            Self::Grey => formatter.write_str("grey"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_is_halved_only_on_the_axes_the_subsampling_names() {
        assert_eq!(Subsampling::Yuv444.chroma_dimensions(1920, 1080), (1920, 1080));
        assert_eq!(Subsampling::Yuv422.chroma_dimensions(1920, 1080), (960, 1080));
        assert_eq!(Subsampling::Yuv420.chroma_dimensions(1920, 1080), (960, 540));
    }

    /// An odd dimension rounds up, so the chroma plane still covers every luma sample.
    #[test]
    fn an_odd_dimension_rounds_the_chroma_plane_up() {
        assert_eq!(Subsampling::Yuv420.chroma_dimensions(1921, 1081), (961, 541));
        assert_eq!(Subsampling::Yuv422.chroma_dimensions(1921, 1081), (961, 1081));
    }

    #[test]
    fn a_layout_lists_the_channels_it_carries() {
        let yuv = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };

        assert_eq!(yuv.channels(), &[Channel::Y, Channel::U, Channel::V]);
        assert_eq!(FrameLayout::Rgb.channels(), &[Channel::R, Channel::G, Channel::B]);
        assert_eq!(FrameLayout::Grey.channels(), &[Channel::Y]);
    }

    #[test]
    fn a_layout_does_not_carry_a_channel_from_another_layout() {
        assert!(!FrameLayout::Rgb.carries(Channel::Y));
        assert!(!FrameLayout::Grey.carries(Channel::U));

        let yuv = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        };
        assert!(!yuv.carries(Channel::R));
    }

    #[test]
    fn only_a_yuv_layout_subsamples_a_chroma_channel() {
        let yuv = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };

        assert_eq!(yuv.channel_dimensions(Channel::Y, 1920, 1080), Some((1920, 1080)));
        assert_eq!(yuv.channel_dimensions(Channel::U, 1920, 1080), Some((960, 540)));
        assert_eq!(
            FrameLayout::Rgb.channel_dimensions(Channel::G, 1920, 1080),
            Some((1920, 1080))
        );
        assert_eq!(FrameLayout::Grey.channel_dimensions(Channel::U, 1920, 1080), None);
    }

    #[test]
    fn a_layout_channel_and_subsampling_render_for_a_reader() {
        assert_eq!(Channel::Y.to_string(), "Y");
        assert_eq!(Subsampling::Yuv420.to_string(), "4:2:0");
        assert_eq!(FrameLayout::Rgb.to_string(), "rgb");
        assert_eq!(FrameLayout::Grey.to_string(), "grey");

        let yuv = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv422,
        };
        assert_eq!(yuv.to_string(), "yuv 4:2:2");
    }
}
