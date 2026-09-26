use av_toolbox_core::StageError;
use av_toolbox_gpu::{ColourInfo, ColourRange, FrameFormat, FrameLayout, FrameRate, Matrix, Subsampling};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::shared::Carried;

/// The matrix a YUV or grey frame is made with.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MatrixArg {
    Bt709,
    Bt601,
    Bt2020Ncl,
}

impl MatrixArg {
    fn matrix(self) -> Matrix {
        match self {
            Self::Bt709 => Matrix::Bt709,
            Self::Bt601 => Matrix::Bt601,
            Self::Bt2020Ncl => Matrix::Bt2020Ncl,
        }
    }
}

/// The range a frame is made with.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RangeArg {
    Limited,
    Full,
}

/// What a constant source's frames are, as its arguments give them.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Shape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: Carried,
    pub(crate) matrix: Option<MatrixArg>,
    pub(crate) range: Option<RangeArg>,
    pub(crate) frame_rate: Option<[u32; 2]>,
}

fn zero_size(stage: &str) -> StageError {
    let message = format!("{stage} needs a width and height above 0");

    StageError::msg(message)
}

fn bad_rate(stage: &str) -> StageError {
    let message = format!("{stage} needs a frame_rate with a numerator and denominator above 0");

    StageError::msg(message)
}

fn not_divisible(stage: &str, width: u32, height: u32, subsampling: Subsampling) -> StageError {
    let message =
        format!("{stage} needs a size that divides by {subsampling}'s subsampling, got {width}x{height}");

    StageError::msg(message)
}

fn zero_frames(stage: &str) -> StageError {
    let message = format!("{stage} needs frames above 0");

    StageError::msg(message)
}

pub(crate) fn check_frames(frames: u64, stage: &str) -> Result<(), StageError> {
    if frames == 0 {
        return Err(zero_frames(stage));
    }

    Ok(())
}

fn divides(width: u32, height: u32, subsampling: Subsampling) -> bool {
    match subsampling {
        Subsampling::Yuv444 => true,
        Subsampling::Yuv422 => width.is_multiple_of(2),
        Subsampling::Yuv420 => width.is_multiple_of(2) && height.is_multiple_of(2),
    }
}

/// The format a constant source makes, from its arguments.
///
/// RGB frames carry the identity matrix. YUV frames default to limited range and everything else to full.
pub(crate) fn frame_format(shape: &Shape, stage: &str) -> Result<FrameFormat, StageError> {
    if shape.width == 0 || shape.height == 0 {
        return Err(zero_size(stage));
    }

    let (layout, sample) = shape.format.layout();
    if let FrameLayout::Yuv { subsampling } = layout
        && !divides(shape.width, shape.height, subsampling)
    {
        return Err(not_divisible(stage, shape.width, shape.height, subsampling));
    }

    let frame_rate = match shape.frame_rate {
        None => None,
        Some([numerator, denominator]) => {
            let Some(rate) = FrameRate::new(numerator, denominator) else {
                return Err(bad_rate(stage));
            };

            Some(rate)
        },
    };

    let given_matrix = shape.matrix.unwrap_or(MatrixArg::Bt709);
    let matrix = match layout {
        FrameLayout::Rgb => Matrix::Identity,
        FrameLayout::Yuv { .. } | FrameLayout::Grey => given_matrix.matrix(),
    };
    let range = match (shape.range, layout) {
        (Some(RangeArg::Limited), _) => ColourRange::Limited,
        (Some(RangeArg::Full), _) => ColourRange::Full,
        (None, FrameLayout::Yuv { .. }) => ColourRange::Limited,
        (None, FrameLayout::Rgb | FrameLayout::Grey) => ColourRange::Full,
    };
    let colour = ColourInfo {
        range,
        matrix,
        ..ColourInfo::default()
    };

    Ok(FrameFormat {
        width: shape.width,
        height: shape.height,
        layout,
        sample,
        colour,
        frame_rate,
    })
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ChromaLocation, ColourRange, FrameLayout, Matrix, Subsampling};

    use super::*;

    fn shape(format: Carried) -> Shape {
        Shape {
            width: 4,
            height: 2,
            format,
            matrix: None,
            range: None,
            frame_rate: None,
        }
    }

    #[test]
    fn yuv_defaults_to_bt709_limited_left_sited() {
        let found = frame_format(&shape(Carried::Yuv420p), "solid/fill").expect("valid");

        let subsampling = Subsampling::Yuv420;
        assert_eq!(found.layout, FrameLayout::Yuv { subsampling });
        assert_eq!(found.colour.matrix, Matrix::Bt709);
        assert_eq!(found.colour.range, ColourRange::Limited);
        assert_eq!(found.colour.chroma_location, ChromaLocation::Left);
        assert_eq!(found.frame_rate, None);
    }

    #[test]
    fn rgb_defaults_to_identity_full() {
        let found = frame_format(&shape(Carried::Gbrp), "solid/fill").expect("valid");

        assert_eq!(found.colour.matrix, Matrix::Identity);
        assert_eq!(found.colour.range, ColourRange::Full);
    }

    #[test]
    fn grey_defaults_to_full_and_keeps_its_matrix() {
        let bt601 = Shape {
            matrix: Some(MatrixArg::Bt601),
            ..shape(Carried::Gray)
        };

        let found = frame_format(&bt601, "solid/fill").expect("valid");

        assert_eq!(found.colour.matrix, Matrix::Bt601);
        assert_eq!(found.colour.range, ColourRange::Full);
    }

    #[test]
    fn range_and_rate_are_taken_as_given() {
        let given = Shape {
            range: Some(RangeArg::Full),
            frame_rate: Some([24000, 1001]),
            ..shape(Carried::Yuv444p10)
        };

        let found = frame_format(&given, "solid/fill").expect("valid");

        let rate = found.frame_rate.expect("a rate");
        assert_eq!(found.colour.range, ColourRange::Full);
        assert_eq!((rate.numerator(), rate.denominator()), (24000, 1001));
    }

    #[test]
    fn a_zero_rate_is_refused() {
        let given = Shape {
            frame_rate: Some([25, 0]),
            ..shape(Carried::Yuv420p)
        };

        let error = frame_format(&given, "solid/fill").expect_err("no rate");

        assert!(error.to_string().contains("frame_rate"));
    }

    #[test]
    fn a_zero_size_is_refused() {
        let given = Shape {
            width: 0,
            ..shape(Carried::Gray)
        };

        let refused = frame_format(&given, "solid/fill");

        assert!(refused.is_err());
    }

    #[test]
    fn an_odd_width_is_refused_at_420() {
        let given = Shape {
            width: 5,
            ..shape(Carried::Yuv420p)
        };

        let error = frame_format(&given, "image/file").expect_err("5 is odd");

        assert!(error.to_string().contains("divide"));
    }

    #[test]
    fn zero_frames_is_refused() {
        let refused = check_frames(0, "solid/fill");
        let allowed = check_frames(1, "solid/fill");

        assert!(refused.is_err());
        assert!(allowed.is_ok());
    }
}
