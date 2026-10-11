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
}
