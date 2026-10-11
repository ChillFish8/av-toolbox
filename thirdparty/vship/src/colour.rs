use crate::error::VshipError;
use crate::sys;

/// How each sample of a plane is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleType {
    Float,
    Half,
    Uint8,
    Uint9,
    Uint10,
    Uint12,
    Uint14,
    Uint16,
}

impl SampleType {
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Float => 4,
            Self::Uint8 => 1,
            Self::Half | Self::Uint9 | Self::Uint10 | Self::Uint12 | Self::Uint14 | Self::Uint16 => 2,
        }
    }

    fn to_raw(self) -> sys::Vship_Sample_t {
        match self {
            Self::Float => sys::Vship_SampleFLOAT,
            Self::Half => sys::Vship_SampleHALF,
            Self::Uint8 => sys::Vship_SampleUINT8,
            Self::Uint9 => sys::Vship_SampleUINT9,
            Self::Uint10 => sys::Vship_SampleUINT10,
            Self::Uint12 => sys::Vship_SampleUINT12,
            Self::Uint14 => sys::Vship_SampleUINT14,
            Self::Uint16 => sys::Vship_SampleUINT16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Range {
    Limited,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaLocation {
    Left,
    Center,
    TopLeft,
    Top,
}

/// How a frame's planes are stored and what colour they hold.
///
/// Matrix, transfer and primaries use H.273 numbering. A matrix of `0` means RGB, any other value
/// means YUV. The subsampling fields hold the log2 shift, so `0`, `1` or `2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorSpace {
    pub sample: SampleType,
    pub width: i64,
    pub height: i64,
    pub range: Range,
    pub sub_sampling_w: i32,
    pub sub_sampling_h: i32,
    pub chroma_location: ChromaLocation,
    pub matrix: u32,
    pub transfer: u32,
    pub primaries: i32,
}

impl ColorSpace {
    /// Builds Vship's colour space, with no resize and no crop.
    pub(crate) fn to_raw(self) -> sys::Vship_Colorspace_t {
        let range = match self.range {
            Range::Limited => sys::Vship_RangeLimited,
            Range::Full => sys::Vship_RangeFull,
        };
        let chroma_location = match self.chroma_location {
            ChromaLocation::Left => sys::Vship_ChromaLoc_Left,
            ChromaLocation::Center => sys::Vship_ChromaLoc_Center,
            ChromaLocation::TopLeft => sys::Vship_ChromaLoc_TopLeft,
            ChromaLocation::Top => sys::Vship_ChromaLoc_Top,
        };
        let color_family = if self.matrix == 0 {
            sys::Vship_ColorRGB
        } else {
            sys::Vship_ColorYUV
        };

        sys::Vship_Colorspace_t {
            width: self.width,
            height: self.height,
            target_width: -1,
            target_height: -1,
            sample: self.sample.to_raw(),
            range,
            subsampling: sys::Vship_ChromaSubsample_t {
                subw: self.sub_sampling_w,
                subh: self.sub_sampling_h,
            },
            chromaLocation: chroma_location,
            colorFamily: color_family,
            YUVMatrix: self.matrix,
            transferFunction: self.transfer,
            primaries: self.primaries,
            crop: sys::Vship_CropRectangle_t {
                top: 0,
                bottom: 0,
                left: 0,
                right: 0,
            },
        }
    }

    /// The row size in bytes and the row count of plane `index`.
    fn plane_extent(&self, index: usize) -> Result<(usize, usize), VshipError> {
        let shift_w = subsampling_shift(self.sub_sampling_w)?;
        let shift_h = subsampling_shift(self.sub_sampling_h)?;
        let width = dimension(self.width)?;
        let height = dimension(self.height)?;
        let subsampled = index > 0 && self.matrix != 0;
        let (plane_width, plane_height) = if subsampled {
            (size_rounded_up(width, shift_w), size_rounded_up(height, shift_h))
        } else {
            (width, height)
        };
        let Some(row_bytes) = plane_width.checked_mul(self.sample.bytes_per_sample()) else {
            return Err(plane_too_large_error(index));
        };

        Ok((row_bytes, plane_height))
    }
}

/// Three planes of host memory, each with its stride in bytes.
#[derive(Debug, Clone, Copy)]
pub struct Planes<'a> {
    pub data: [&'a [u8]; 3],
    pub strides: [usize; 3],
}

impl Planes<'_> {
    /// Checks every plane holds the rows `color_space` describes, naming the frame as `name`.
    pub(crate) fn check(&self, color_space: &ColorSpace, name: &str) -> Result<(), VshipError> {
        for index in 0..3 {
            let (row_bytes, rows) = color_space.plane_extent(index)?;
            let stride = self.strides[index];
            let length = self.data[index].len();

            if stride < row_bytes {
                return Err(narrow_stride_error(name, index, stride, row_bytes));
            }

            let needed = required_len(stride, row_bytes, rows);
            let Some(needed) = needed else {
                return Err(VshipError::new(format!(
                    "{name} plane {index} is too large to address"
                )));
            };

            if length < needed {
                return Err(short_plane_error(name, index, length, needed));
            }
        }

        Ok(())
    }

    pub(crate) fn pointers(&self) -> [*const u8; 3] {
        [
            self.data[0].as_ptr(),
            self.data[1].as_ptr(),
            self.data[2].as_ptr(),
        ]
    }

    pub(crate) fn line_sizes(&self) -> Result<[i64; 3], VshipError> {
        let mut line_sizes = [0i64; 3];

        for (index, stride) in self.strides.iter().enumerate() {
            line_sizes[index] = match i64::try_from(*stride) {
                Ok(converted) => converted,
                Err(_) => {
                    return Err(VshipError::new(format!(
                        "a stride of {stride} bytes is too large"
                    )));
                },
            };
        }

        Ok(line_sizes)
    }
}

/// The bytes a plane needs, with no padding required after its last row.
fn required_len(stride: usize, row_bytes: usize, rows: usize) -> Option<usize> {
    if rows == 0 {
        return Some(0);
    }

    let leading_rows = stride.checked_mul(rows - 1)?;

    leading_rows.checked_add(row_bytes)
}

/// A chroma size halved `shift` times, rounding up like Vship does.
fn size_rounded_up(size: usize, shift: u32) -> usize {
    (size + (1 << shift) - 1) >> shift
}

fn subsampling_shift(value: i32) -> Result<u32, VshipError> {
    match u32::try_from(value) {
        Ok(shift) if shift <= 2 => Ok(shift),
        _ => Err(unsupported_subsampling_error(value)),
    }
}

fn unsupported_subsampling_error(value: i32) -> VshipError {
    VshipError::new(format!(
        "a subsampling of {value} is not supported. It must be between 0 and 2."
    ))
}

fn plane_too_large_error(index: usize) -> VshipError {
    VshipError::new(format!("plane {index} is too large to address"))
}

fn dimension(value: i64) -> Result<usize, VshipError> {
    match usize::try_from(value) {
        Ok(converted) if converted >= 1 => Ok(converted),
        _ => Err(size_below_one_error(value)),
    }
}

fn size_below_one_error(value: i64) -> VshipError {
    VshipError::new(format!(
        "the colour space has a size of {value}, which is below 1"
    ))
}

fn narrow_stride_error(name: &str, index: usize, stride: usize, row_bytes: usize) -> VshipError {
    VshipError::new(format!(
        "{name} plane {index} has a stride of {stride} bytes, below its {row_bytes} byte rows"
    ))
}

fn short_plane_error(name: &str, index: usize, length: usize, needed: usize) -> VshipError {
    VshipError::new(format!(
        "{name} plane {index} holds {length} bytes but needs {needed}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yuv420(width: i64, height: i64) -> ColorSpace {
        ColorSpace {
            sample: SampleType::Uint8,
            width,
            height,
            range: Range::Limited,
            sub_sampling_w: 1,
            sub_sampling_h: 1,
            chroma_location: ChromaLocation::Left,
            matrix: 1,
            transfer: 1,
            primaries: 1,
        }
    }

    #[test]
    fn the_raw_colour_space_carries_every_field() {
        let color_space = ColorSpace {
            sample: SampleType::Uint10,
            range: Range::Full,
            ..yuv420(64, 48)
        };

        let raw = color_space.to_raw();

        assert_eq!((raw.width, raw.height), (64, 48));
        assert_eq!((raw.target_width, raw.target_height), (-1, -1));
        assert_eq!(raw.sample, sys::Vship_SampleUINT10);
        assert_eq!(raw.range, sys::Vship_RangeFull);
        assert_eq!((raw.subsampling.subw, raw.subsampling.subh), (1, 1));
        assert_eq!(raw.chromaLocation, sys::Vship_ChromaLoc_Left);
        assert_eq!(raw.colorFamily, sys::Vship_ColorYUV);
        assert_eq!(raw.YUVMatrix, 1);
        assert_eq!(raw.transferFunction, 1);
        assert_eq!(raw.primaries, 1);
        assert_eq!(
            (raw.crop.top, raw.crop.bottom, raw.crop.left, raw.crop.right),
            (0, 0, 0, 0)
        );
    }

    #[test]
    fn matrix_zero_is_rgb() {
        let color_space = ColorSpace {
            matrix: 0,
            ..yuv420(64, 48)
        };

        let raw = color_space.to_raw();

        assert_eq!(raw.colorFamily, sys::Vship_ColorRGB);
    }

    #[test]
    fn planes_that_fit_pass() {
        let y_plane = vec![0u8; 64 * 48];
        let u_plane = vec![0u8; 32 * 24];
        let v_plane = vec![0u8; 32 * 24];
        let planes = Planes {
            data: [&y_plane, &u_plane, &v_plane],
            strides: [64, 32, 32],
        };

        let result = planes.check(&yuv420(64, 48), "the reference");

        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn a_short_chroma_plane_is_refused() {
        let y_plane = vec![0u8; 64 * 48];
        let u_plane = vec![0u8; 32 * 24 - 1];
        let v_plane = vec![0u8; 32 * 24];
        let planes = Planes {
            data: [&y_plane, &u_plane, &v_plane],
            strides: [64, 32, 32],
        };

        let error = planes
            .check(&yuv420(64, 48), "the reference")
            .expect_err("a short plane passed");

        assert!(error.message().contains("the reference plane 1"), "{error}");
    }

    #[test]
    fn a_stride_below_the_row_size_is_refused() {
        let y_plane = vec![0u8; 64 * 48];
        let u_plane = vec![0u8; 32 * 24];
        let v_plane = vec![0u8; 32 * 24];
        let planes = Planes {
            data: [&y_plane, &u_plane, &v_plane],
            strides: [63, 32, 32],
        };

        let error = planes
            .check(&yuv420(64, 48), "the distorted frame")
            .expect_err("a narrow stride passed");

        assert!(error.message().contains("the distorted frame plane 0"), "{error}");
    }

    #[test]
    fn rgb_planes_are_all_full_size() {
        let color_space = ColorSpace {
            matrix: 0,
            sample: SampleType::Uint16,
            ..yuv420(8, 4)
        };
        let full_plane = vec![0u8; 16 * 4];
        let planes = Planes {
            data: [&full_plane, &full_plane[..16 * 4 - 1], &full_plane],
            strides: [16, 16, 16],
        };

        let error = planes
            .check(&color_space, "the reference")
            .expect_err("a short rgb plane passed");

        assert!(error.message().contains("plane 1"), "{error}");
    }

    #[test]
    fn odd_sizes_round_chroma_planes_up() {
        let color_space = yuv420(63, 47);
        let y_plane = vec![0u8; 63 * 47];
        let full_u_plane = vec![0u8; 32 * 24];
        let full_v_plane = vec![0u8; 32 * 24];
        let full = Planes {
            data: [&y_plane, &full_u_plane, &full_v_plane],
            strides: [63, 32, 32],
        };
        let short_u_plane = vec![0u8; 31 * 23];
        let short_v_plane = vec![0u8; 31 * 23];
        let short = Planes {
            data: [&y_plane, &short_u_plane, &short_v_plane],
            strides: [63, 31, 31],
        };

        let accepted = full.check(&color_space, "the reference");
        let error = short
            .check(&color_space, "the reference")
            .expect_err("rounded down chroma planes passed");

        assert!(accepted.is_ok(), "{accepted:?}");
        assert!(error.message().contains("plane 1"), "{error}");
    }

    #[test]
    fn an_unsupported_subsampling_is_refused() {
        let color_space = ColorSpace {
            sub_sampling_w: 3,
            ..yuv420(64, 48)
        };
        let y_plane = vec![0u8; 64 * 48];
        let chroma_plane = vec![0u8; 32 * 24];
        let planes = Planes {
            data: [&y_plane, &chroma_plane, &chroma_plane],
            strides: [64, 32, 32],
        };

        let error = planes
            .check(&color_space, "the reference")
            .expect_err("a subsampling of 3 passed");

        assert!(error.message().contains("subsampling"), "{error}");
    }

    #[test]
    fn an_unsupported_subsampling_is_refused_for_rgb() {
        let color_space = ColorSpace {
            matrix: 0,
            sub_sampling_w: 3,
            ..yuv420(64, 48)
        };
        let plane = vec![0u8; 64 * 48];
        let planes = Planes {
            data: [&plane, &plane, &plane],
            strides: [64, 64, 64],
        };

        let error = planes
            .check(&color_space, "the reference")
            .expect_err("a subsampling of 3 passed");

        assert!(error.message().contains("subsampling"), "{error}");
    }

    #[test]
    fn a_row_that_overflows_is_refused() {
        let color_space = ColorSpace {
            sample: SampleType::Float,
            ..yuv420(i64::MAX, 1)
        };

        let error = color_space
            .plane_extent(0)
            .expect_err("an overflowing row passed");

        assert!(error.message().contains("too large"), "{error}");
    }

    #[test]
    fn a_negative_size_is_refused() {
        let empty: &[u8] = &[];
        let planes = Planes {
            data: [empty, empty, empty],
            strides: [0, 0, 0],
        };

        let error = planes
            .check(&yuv420(-1, 4), "the reference")
            .expect_err("a negative size passed");

        assert!(error.message().contains("below 1"), "{error}");
    }

    #[test]
    fn a_zero_size_is_refused() {
        let empty: &[u8] = &[];
        let planes = Planes {
            data: [empty, empty, empty],
            strides: [0, 0, 0],
        };

        let error = planes
            .check(&yuv420(0, 4), "the reference")
            .expect_err("a zero size passed");

        assert!(error.message().contains("below 1"), "{error}");
    }
}
