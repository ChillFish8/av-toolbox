use std::io::{self, Write};
use std::path::Path;

use crate::motion::BlockGrid;

/// The PSNR of `found` against `truth` in decibels, for samples between 0 and `peak`.
///
/// Identical planes give infinity.
pub fn psnr(found: &[f32], truth: &[f32], peak: f32) -> f64 {
    assert_eq!(found.len(), truth.len(), "the planes differ in length");

    let mut total = 0.0f64;
    for (actual, wanted) in found.iter().zip(truth) {
        let difference = f64::from(*actual) - f64::from(*wanted);
        total += difference * difference;
    }

    psnr_of(total, found.len(), peak)
}

/// The PSNR over the samples where `mask` is set.
pub fn psnr_within(found: &[f32], truth: &[f32], mask: &[bool], peak: f32) -> f64 {
    assert_eq!(found.len(), truth.len(), "the planes differ in length");
    assert_eq!(found.len(), mask.len(), "the mask differs in length");

    let mut total = 0.0f64;
    let mut count = 0usize;
    for ((actual, wanted), inside) in found.iter().zip(truth).zip(mask) {
        if *inside {
            let difference = f64::from(*actual) - f64::from(*wanted);
            total += difference * difference;
            count += 1;
        }
    }

    psnr_of(total, count, peak)
}

fn psnr_of(squared_error: f64, count: usize, peak: f32) -> f64 {
    assert!(count > 0, "no samples to compare");

    let mean = squared_error / count as f64;
    let peak = f64::from(peak);

    10.0 * (peak * peak / mean).log10()
}

/// The share of interior blocks whose vector is within `1 / pel` pixel of the true motion.
///
/// Interior blocks lie fully inside the frame. The true motion of a block is the mean of `displacement`
/// over it, and `vectors` holds each block's `(x, y)` in units of `1 / pel` pixel.
pub fn vector_accuracy(
    vectors: &[i32],
    grid: &BlockGrid,
    displacement: &[(f32, f32)],
    pel: u32,
    width: usize,
    height: usize,
) -> f64 {
    assert_eq!(
        vectors.len(),
        2 * grid.blocks(),
        "the vectors do not fit the grid"
    );
    assert_eq!(
        displacement.len(),
        width * height,
        "the displacement does not fit the frame"
    );

    let blksize = grid.blksize as usize;
    let tolerance = 1.0 / f64::from(pel) + 1e-6;

    let mut interior = 0usize;
    let mut accurate = 0usize;
    for block_y in 0..grid.down as usize {
        for block_x in 0..grid.across as usize {
            let left = block_x * grid.step as usize;
            let top = block_y * grid.step as usize;
            if left + blksize > width || top + blksize > height {
                continue;
            }

            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            for row in top..top + blksize {
                for column in left..left + blksize {
                    let (move_x, move_y) = displacement[row * width + column];
                    sum_x += f64::from(move_x);
                    sum_y += f64::from(move_y);
                }
            }

            let samples = (blksize * blksize) as f64;
            let index = block_y * grid.across as usize + block_x;
            let found_x = f64::from(vectors[2 * index]) / f64::from(pel);
            let found_y = f64::from(vectors[2 * index + 1]) / f64::from(pel);

            interior += 1;
            let close_x = (found_x - sum_x / samples).abs() <= tolerance;
            let close_y = (found_y - sum_y / samples).abs() <= tolerance;
            if close_x && close_y {
                accurate += 1;
            }
        }
    }

    assert!(interior > 0, "the frame has no interior blocks");

    accurate as f64 / interior as f64
}

/// How the fields of a y4m's frames are laid out.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Interlace {
    Progressive,
    TopFirst,
}

impl Interlace {
    fn tag(self) -> &'static str {
        match self {
            Self::Progressive => "Ip",
            Self::TopFirst => "It",
        }
    }
}

/// Writes `frames`, each its Y, U and V code values at 4:2:0, as a y4m at `rate` frames per second.
///
/// Samples take one byte when `bits` is 8, and two little-endian bytes above.
pub fn write_y4m(
    path: &Path,
    frames: &[[Vec<i32>; 3]],
    (width, height): (usize, usize),
    bits: u32,
    rate: u32,
    interlace: Interlace,
) -> io::Result<()> {
    let wide = bits > 8;
    let suffix = if wide { format!("p{bits}") } else { String::new() };
    let tag = interlace.tag();
    let header = format!("YUV4MPEG2 W{width} H{height} F{rate}:1 {tag} A1:1 C420{suffix}\n");

    let mut bytes = header.into_bytes();
    for planes in frames {
        bytes.extend_from_slice(b"FRAME\n");
        for code in planes.iter().flatten() {
            if wide {
                bytes.extend_from_slice(&(*code as u16).to_le_bytes());
            } else {
                bytes.push(*code as u8);
            }
        }
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&bytes)
}

/// Writes `frames`, each a luma plane normalised to between 0 and 1, as a 4:2:0 y4m with flat chroma.
///
/// Samples take one byte when `bits` is 8, and two little-endian bytes above.
pub fn write_luma_y4m(
    path: &Path,
    width: usize,
    height: usize,
    bits: u32,
    frames: &[Vec<f32>],
) -> io::Result<()> {
    let peak = ((1u32 << bits) - 1) as f32;
    let flat = 1i32 << (bits - 1);
    let chroma_samples = (width / 2) * (height / 2);

    let mut coded = Vec::with_capacity(frames.len());
    for frame in frames {
        assert_eq!(
            frame.len(),
            width * height,
            "a frame does not fit {width}x{height}"
        );

        let luma = frame
            .iter()
            .map(|sample| (sample * peak).round().clamp(0.0, peak) as i32)
            .collect();
        coded.push([luma, vec![flat; chroma_samples], vec![flat; chroma_samples]]);
    }

    write_y4m(path, &coded, (width, height), bits, 24, Interlace::Progressive)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_planes_have_infinite_psnr() {
        let plane = [0.25, 0.5, 0.75];

        assert!(psnr(&plane, &plane, 1.0).is_infinite());
    }

    #[test]
    fn a_uniform_error_of_a_tenth_is_twenty_decibels() {
        let found = [0.1; 4];
        let truth = [0.0; 4];

        let value = psnr(&found, &truth, 1.0);

        assert!((value - 20.0).abs() < 1e-6, "{value}");
    }

    #[test]
    fn the_mask_picks_the_samples() {
        let found = [0.1, 0.9];
        let truth = [0.0, 0.0];

        let value = psnr_within(&found, &truth, &[true, false], 1.0);

        assert!((value - 20.0).abs() < 1e-6, "{value}");
    }

    #[test]
    fn only_interior_blocks_count_towards_accuracy() {
        let settings = crate::motion::MotionSettings {
            blksize: 4,
            overlap: 0,
            pel: 1,
            ..crate::motion::MotionSettings::default()
        };
        let grid = BlockGrid::new(&settings, 6, 4);
        let displacement = vec![(1.0, 0.0); 24];

        let vectors = [1, 0, 7, 7];
        let found = vector_accuracy(&vectors, &grid, &displacement, 1, 6, 4);
        assert_eq!(found, 1.0);

        let vectors = [3, 0, 1, 0];
        let found = vector_accuracy(&vectors, &grid, &displacement, 1, 6, 4);
        assert_eq!(found, 0.0);
    }
}
