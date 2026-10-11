use std::mem::size_of;

use crate::error::VshipError;

/// The size Vship writes a metric's distortion maps at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MapSize {
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl MapSize {
    /// Converts the size Vship reports, refusing one that cannot be allocated.
    pub(crate) fn from_raw(width: i64, height: i64) -> Result<Self, VshipError> {
        let converted_width = usize::try_from(width);
        let converted_height = usize::try_from(height);
        let (Ok(map_width), Ok(map_height)) = (converted_width, converted_height) else {
            return Err(unusable_size_error(width, height));
        };

        let pixels = map_width.checked_mul(map_height);
        let bytes = pixels.and_then(|pixel_count| pixel_count.checked_mul(size_of::<f32>()));
        if bytes.is_none() {
            return Err(unusable_size_error(width, height));
        }

        Ok(Self {
            width: map_width,
            height: map_height,
        })
    }
}

/// A per-pixel distortion map from a metric, one `f32` per pixel with rows packed.
#[derive(Debug, Clone, PartialEq)]
pub struct DistortionMap {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl DistortionMap {
    /// A zero filled map of `size`.
    pub(crate) fn new(size: MapSize) -> Self {
        let data = vec![0.0; size.width * size.height];

        Self {
            width: size.width,
            height: size.height,
            data,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The map's values, `width` per row.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The pointer and byte stride Vship writes through, once the map is checked against `size`.
    pub(crate) fn target(&mut self, size: MapSize) -> Result<(*const u8, i64), VshipError> {
        if self.width != size.width || self.height != size.height {
            return Err(wrong_size_error(self.width, self.height, size));
        }

        let expected_len = self.width * self.height;
        if self.data.len() != expected_len {
            return Err(wrong_length_error(self.data.len(), expected_len));
        }

        let stride_bytes = self.width.checked_mul(size_of::<f32>());
        let Some(stride_bytes) = stride_bytes else {
            return Err(stride_too_large_error(self.width));
        };

        let Ok(stride) = i64::try_from(stride_bytes) else {
            return Err(stride_too_large_error(self.width));
        };

        let pointer = self.data.as_mut_ptr().cast::<u8>().cast_const();

        Ok((pointer, stride))
    }
}

fn unusable_size_error(width: i64, height: i64) -> VshipError {
    VshipError::new(format!(
        "vship reported a distortion map size of {width} by {height}, which cannot be used"
    ))
}

fn wrong_size_error(width: usize, height: usize, size: MapSize) -> VshipError {
    VshipError::new(format!(
        "the distortion map is {width} by {height} but the metric writes {} by {}",
        size.width, size.height
    ))
}

fn wrong_length_error(length: usize, expected: usize) -> VshipError {
    VshipError::new(format!(
        "the distortion map holds {length} values but needs {expected}"
    ))
}

fn stride_too_large_error(width: usize) -> VshipError {
    VshipError::new(format!(
        "a distortion map {width} values wide is too large to address"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn size(width: usize, height: usize) -> MapSize {
        MapSize { width, height }
    }

    #[test]
    fn a_new_map_has_its_size_and_is_zero_filled() {
        let map = DistortionMap::new(size(4, 3));

        assert_eq!((map.width(), map.height()), (4, 3));
        assert_eq!(map.data(), &[0.0; 12]);
    }

    #[test]
    fn a_matching_map_gives_a_byte_stride() {
        let mut map = DistortionMap::new(size(4, 3));

        let (_, stride) = map.target(size(4, 3)).expect("a matching map");

        assert_eq!(stride, 16);
    }

    #[test]
    fn a_map_of_another_size_is_refused() {
        let mut map = DistortionMap::new(size(8, 8));

        let error = map.target(size(4, 3)).expect_err("a wrongly sized map passed");

        assert_eq!(error.code(), 0);
        assert!(error.message().contains("distortion map is 8 by 8"), "{error}");
    }

    #[test]
    fn a_negative_reported_size_is_refused() {
        let error = MapSize::from_raw(-1, 4).expect_err("a negative size passed");

        assert!(error.message().contains("-1 by 4"), "{error}");
    }
}
