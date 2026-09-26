use av_toolbox_gpu::{Depth, FrameLayout, SampleFormat, Subsampling};
use schemars::JsonSchema;
use serde::Deserialize;

/// A pixel format a source can produce frames in.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Carried {
    Yuv420p,
    Yuv420p10,
    Yuv420p12,
    Yuv420p16,
    Yuv422p,
    Yuv422p10,
    Yuv422p12,
    Yuv422p16,
    Yuv444p,
    Yuv444p10,
    Yuv444p12,
    Yuv444p16,
    Gray,
    Gray10,
    Gray12,
    Gray16,
    Gbrp,
    Gbrp10,
    Gbrp12,
    Gbrp16,
}

impl Carried {
    /// The layout and sample format a decoded frame carries.
    pub(crate) fn layout(self) -> (FrameLayout, SampleFormat) {
        let yuv420 = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let yuv422 = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv422,
        };
        let yuv444 = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        };

        match self {
            Self::Yuv420p => (yuv420, SampleFormat::U8),
            Self::Yuv420p10 => (yuv420, SampleFormat::U16 { depth: Depth::TEN }),
            Self::Yuv420p12 => (yuv420, SampleFormat::U16 { depth: Depth::TWELVE }),
            Self::Yuv420p16 => (
                yuv420,
                SampleFormat::U16 {
                    depth: Depth::SIXTEEN,
                },
            ),
            Self::Yuv422p => (yuv422, SampleFormat::U8),
            Self::Yuv422p10 => (yuv422, SampleFormat::U16 { depth: Depth::TEN }),
            Self::Yuv422p12 => (yuv422, SampleFormat::U16 { depth: Depth::TWELVE }),
            Self::Yuv422p16 => (
                yuv422,
                SampleFormat::U16 {
                    depth: Depth::SIXTEEN,
                },
            ),
            Self::Yuv444p => (yuv444, SampleFormat::U8),
            Self::Yuv444p10 => (yuv444, SampleFormat::U16 { depth: Depth::TEN }),
            Self::Yuv444p12 => (yuv444, SampleFormat::U16 { depth: Depth::TWELVE }),
            Self::Yuv444p16 => (
                yuv444,
                SampleFormat::U16 {
                    depth: Depth::SIXTEEN,
                },
            ),
            Self::Gray => (FrameLayout::Grey, SampleFormat::U8),
            Self::Gray10 => (FrameLayout::Grey, SampleFormat::U16 { depth: Depth::TEN }),
            Self::Gray12 => (FrameLayout::Grey, SampleFormat::U16 { depth: Depth::TWELVE }),
            Self::Gray16 => (
                FrameLayout::Grey,
                SampleFormat::U16 {
                    depth: Depth::SIXTEEN,
                },
            ),
            Self::Gbrp => (FrameLayout::Rgb, SampleFormat::U8),
            Self::Gbrp10 => (FrameLayout::Rgb, SampleFormat::U16 { depth: Depth::TEN }),
            Self::Gbrp12 => (FrameLayout::Rgb, SampleFormat::U16 { depth: Depth::TWELVE }),
            Self::Gbrp16 => (
                FrameLayout::Rgb,
                SampleFormat::U16 {
                    depth: Depth::SIXTEEN,
                },
            ),
        }
    }
}
