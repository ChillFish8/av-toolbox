use av_toolbox_core::StageError;
use av_toolbox_gpu::{Depth, FrameLayout, SampleFormat, Subsampling};
use schemars::JsonSchema;
use serde::Deserialize;

/// A pixel format ffms2 can decode a track into.
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
    /// The FFmpeg pixel format name ffms2 is asked to decode into.
    pub(crate) fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::Yuv420p => "yuv420p",
            Self::Yuv420p10 => "yuv420p10le",
            Self::Yuv420p12 => "yuv420p12le",
            Self::Yuv420p16 => "yuv420p16le",
            Self::Yuv422p => "yuv422p",
            Self::Yuv422p10 => "yuv422p10le",
            Self::Yuv422p12 => "yuv422p12le",
            Self::Yuv422p16 => "yuv422p16le",
            Self::Yuv444p => "yuv444p",
            Self::Yuv444p10 => "yuv444p10le",
            Self::Yuv444p12 => "yuv444p12le",
            Self::Yuv444p16 => "yuv444p16le",
            Self::Gray => "gray",
            Self::Gray10 => "gray10le",
            Self::Gray12 => "gray12le",
            Self::Gray16 => "gray16le",
            Self::Gbrp => "gbrp",
            Self::Gbrp10 => "gbrp10le",
            Self::Gbrp12 => "gbrp12le",
            Self::Gbrp16 => "gbrp16le",
        }
    }

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

/// What an ffms2 pixel format decodes into, or is converted to, and at what range.
#[derive(Debug)]
pub(crate) struct Choice {
    pub(crate) output: Carried,
    pub(crate) full_range: bool,
}

/// Picks the carried format an ffms2 pixel format name should decode into.
///
/// `forced` overrides the decoded format, but the source's own range still applies.
pub(crate) fn choose(encoded: &str, forced: Option<Carried>) -> Result<Choice, StageError> {
    let mapped = mapped_format(encoded);

    let full_range = mapped.map(|(_, full_range)| full_range).unwrap_or(false);
    let output = match (forced, mapped) {
        (Some(forced), _) => forced,
        (None, Some((carried, _))) => carried,
        (None, None) => {
            let message = format!(
                "ffms2 decodes {encoded}, which has no carried equivalent. Pass format to choose one."
            );

            return Err(StageError::msg(message));
        },
    };

    Ok(Choice { output, full_range })
}

/// Every carried format, in no particular order.
const CARRIED_VARIANTS: [Carried; 20] = [
    Carried::Yuv420p,
    Carried::Yuv420p10,
    Carried::Yuv420p12,
    Carried::Yuv420p16,
    Carried::Yuv422p,
    Carried::Yuv422p10,
    Carried::Yuv422p12,
    Carried::Yuv422p16,
    Carried::Yuv444p,
    Carried::Yuv444p10,
    Carried::Yuv444p12,
    Carried::Yuv444p16,
    Carried::Gray,
    Carried::Gray10,
    Carried::Gray12,
    Carried::Gray16,
    Carried::Gbrp,
    Carried::Gbrp10,
    Carried::Gbrp12,
    Carried::Gbrp16,
];

/// FFmpeg pixel format names that convert to a carried format rather than passing through.
const CONVERTED_NAMES: [&str; 11] = [
    "nv12", "nv21", "p010le", "p016le", "rgb24", "bgr24", "rgb48le", "bgr48le", "yuvj420p", "yuvj422p",
    "yuvj444p",
];

/// Every FFmpeg pixel format name ffms2 might report a frame as encoded in.
pub(crate) fn known_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = CARRIED_VARIANTS
        .iter()
        .map(|carried| carried.ffmpeg_name())
        .collect();

    names.extend(CONVERTED_NAMES);

    names
}

/// The layout family a decoded pixel format name carries, ignoring depth and range.
pub(crate) fn mapped_layout(encoded: &str) -> Option<FrameLayout> {
    let (carried, _) = mapped_format(encoded)?;
    let (layout, _) = carried.layout();

    Some(layout)
}

/// The carried format an ffms2 pixel format name maps to, and whether it carries full range.
fn mapped_format(encoded: &str) -> Option<(Carried, bool)> {
    match encoded {
        "yuv420p" => Some((Carried::Yuv420p, false)),
        "yuv420p10le" => Some((Carried::Yuv420p10, false)),
        "yuv420p12le" => Some((Carried::Yuv420p12, false)),
        "yuv420p16le" => Some((Carried::Yuv420p16, false)),
        "yuv422p" => Some((Carried::Yuv422p, false)),
        "yuv422p10le" => Some((Carried::Yuv422p10, false)),
        "yuv422p12le" => Some((Carried::Yuv422p12, false)),
        "yuv422p16le" => Some((Carried::Yuv422p16, false)),
        "yuv444p" => Some((Carried::Yuv444p, false)),
        "yuv444p10le" => Some((Carried::Yuv444p10, false)),
        "yuv444p12le" => Some((Carried::Yuv444p12, false)),
        "yuv444p16le" => Some((Carried::Yuv444p16, false)),
        "gray" => Some((Carried::Gray, false)),
        "gray10le" => Some((Carried::Gray10, false)),
        "gray12le" => Some((Carried::Gray12, false)),
        "gray16le" => Some((Carried::Gray16, false)),
        "gbrp" => Some((Carried::Gbrp, false)),
        "gbrp10le" => Some((Carried::Gbrp10, false)),
        "gbrp12le" => Some((Carried::Gbrp12, false)),
        "gbrp16le" => Some((Carried::Gbrp16, false)),
        "nv12" | "nv21" => Some((Carried::Yuv420p, false)),
        "p010le" => Some((Carried::Yuv420p10, false)),
        "p016le" => Some((Carried::Yuv420p16, false)),
        "rgb24" | "bgr24" => Some((Carried::Gbrp, false)),
        "rgb48le" | "bgr48le" => Some((Carried::Gbrp16, false)),
        "yuvj420p" => Some((Carried::Yuv420p, true)),
        "yuvj422p" => Some((Carried::Yuv422p, true)),
        "yuvj444p" => Some((Carried::Yuv444p, true)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carried_formats_pass_through() {
        let choice = choose("yuv420p10le", None).expect("carried");

        assert_eq!(choice.output, Carried::Yuv420p10);
        assert!(!choice.full_range);
    }

    #[test]
    fn semi_planar_formats_become_planar_at_the_same_depth() {
        assert_eq!(choose("nv12", None).expect("converts").output, Carried::Yuv420p);
        assert_eq!(choose("nv21", None).expect("converts").output, Carried::Yuv420p);
        assert_eq!(
            choose("p010le", None).expect("converts").output,
            Carried::Yuv420p10
        );
        assert_eq!(
            choose("p016le", None).expect("converts").output,
            Carried::Yuv420p16
        );
    }

    #[test]
    fn packed_rgb_becomes_planar_rgb_at_the_same_depth() {
        assert_eq!(choose("rgb24", None).expect("converts").output, Carried::Gbrp);
        assert_eq!(choose("bgr24", None).expect("converts").output, Carried::Gbrp);
        assert_eq!(choose("rgb48le", None).expect("converts").output, Carried::Gbrp16);
    }

    #[test]
    fn jpeg_range_yuv_becomes_yuv_with_full_range() {
        let choice = choose("yuvj420p", None).expect("converts");

        assert_eq!(choice.output, Carried::Yuv420p);
        assert!(choice.full_range);
    }

    #[test]
    fn a_format_with_alpha_is_refused() {
        let error = choose("yuva420p", None).expect_err("alpha is not carried");

        assert!(error.to_string().contains("yuva420p"));
    }

    #[test]
    fn a_forced_format_wins() {
        let choice = choose("yuv444p12le", Some(Carried::Yuv420p)).expect("forced");

        assert_eq!(choice.output, Carried::Yuv420p);
    }

    #[test]
    fn ffmpeg_name_uses_little_endian_suffixes_above_eight_bits() {
        assert_eq!(Carried::Yuv420p.ffmpeg_name(), "yuv420p");
        assert_eq!(Carried::Yuv420p10.ffmpeg_name(), "yuv420p10le");
        assert_eq!(Carried::Gray16.ffmpeg_name(), "gray16le");
        assert_eq!(Carried::Gbrp12.ffmpeg_name(), "gbrp12le");
    }

    #[test]
    fn known_names_all_resolve_through_choose() {
        for name in known_names() {
            assert!(choose(name, None).is_ok(), "{name}");
        }
    }

    #[test]
    fn layout_maps_subsampling_and_sample_depth() {
        assert_eq!(
            Carried::Yuv422p10.layout(),
            (
                FrameLayout::Yuv {
                    subsampling: Subsampling::Yuv422
                },
                SampleFormat::U16 { depth: Depth::TEN }
            )
        );
        assert_eq!(Carried::Gray.layout(), (FrameLayout::Grey, SampleFormat::U8));
        assert_eq!(
            Carried::Gbrp16.layout(),
            (
                FrameLayout::Rgb,
                SampleFormat::U16 {
                    depth: Depth::SIXTEEN
                }
            )
        );
    }
}
