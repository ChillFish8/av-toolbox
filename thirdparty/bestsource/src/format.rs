use crate::sys::BsFormat;

/// How a frame stores colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorFamily {
    Unknown,
    Gray,
    Rgb,
    Yuv,
    Bayer,
}

impl ColorFamily {
    fn from_raw(value: i32) -> Self {
        match value {
            1 => Self::Gray,
            2 => Self::Rgb,
            3 => Self::Yuv,
            4 => Self::Bayer,
            _ => Self::Unknown,
        }
    }
}

/// The sample layout of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoFormat {
    pub color_family: ColorFamily,
    pub alpha: bool,
    pub float: bool,
    pub bits: i32,
    /// Chroma subsampling as a power of two, so `1` halves the chroma width.
    pub sub_sampling_w: i32,
    pub sub_sampling_h: i32,
}

impl VideoFormat {
    pub(crate) fn from_raw(raw: &BsFormat) -> Self {
        Self {
            color_family: ColorFamily::from_raw(raw.color_family),
            alpha: raw.alpha != 0,
            float: raw.is_float != 0,
            bits: raw.bits,
            sub_sampling_w: raw.sub_sampling_w,
            sub_sampling_h: raw.sub_sampling_h,
        }
    }

    /// The number of colour planes, not counting alpha.
    pub fn plane_count(&self) -> usize {
        match self.color_family {
            ColorFamily::Gray => 1,
            ColorFamily::Rgb | ColorFamily::Yuv => 3,
            ColorFamily::Unknown | ColorFamily::Bayer => 0,
        }
    }

    pub fn bytes_per_sample(&self) -> usize {
        match self.bits {
            ..=8 => 1,
            9..=16 => 2,
            _ => 4,
        }
    }

    /// The row size in bytes and the row count of one plane of a `width` by `height` frame.
    ///
    /// Only the chroma planes of YUV are subsampled.
    pub(crate) fn plane_extent(&self, chroma: bool, width: usize, height: usize) -> (usize, usize) {
        let subsampled = chroma && self.color_family == ColorFamily::Yuv;
        let (plane_width, plane_height) = if subsampled {
            (width >> self.sub_sampling_w, height >> self.sub_sampling_h)
        } else {
            (width, height)
        };

        (plane_width * self.bytes_per_sample(), plane_height)
    }
}

#[cfg(test)]
mod tests {
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
    fn yuv_has_three_planes_and_gray_one() {
        let gray = VideoFormat {
            color_family: ColorFamily::Gray,
            ..yuv420(8)
        };

        assert_eq!(yuv420(8).plane_count(), 3);
        assert_eq!(gray.plane_count(), 1);
    }

    #[test]
    fn bayer_and_unknown_have_no_planar_layout() {
        let bayer = VideoFormat {
            color_family: ColorFamily::Bayer,
            ..yuv420(8)
        };

        assert_eq!(bayer.plane_count(), 0);
    }

    #[test]
    fn samples_above_8_bits_take_two_bytes_and_float_four() {
        let float = VideoFormat {
            float: true,
            ..yuv420(32)
        };

        assert_eq!(yuv420(8).bytes_per_sample(), 1);
        assert_eq!(yuv420(10).bytes_per_sample(), 2);
        assert_eq!(float.bytes_per_sample(), 4);
    }

    #[test]
    fn yuv420_chroma_planes_are_half_size() {
        let format = yuv420(10);

        assert_eq!(format.plane_extent(false, 64, 48), (128, 48));
        assert_eq!(format.plane_extent(true, 64, 48), (64, 24));
    }
}
