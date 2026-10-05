use super::motion::{covering_windows, half_pel_sample, scaled_vector};
use crate::degrain::{DegrainSettings, degrain_weight};
use crate::motion::{BlockGrid, PlaneScale, SceneChange};

/// A neighbour plane and its field's vectors and normalised SADs, for [reference_degrain].
pub struct ReferenceNeighbour<'a> {
    pub plane: &'a [f32],
    pub vectors: &'a [i32],
    pub sads: &'a [f32],
}

/// Degrains `centre`, `width` by `height`, with `neighbours` on the CPU, mirroring
/// [degrain](crate::degrain::degrain).
///
/// `centre` and the neighbours are planes scaled by `scale`. `flagged` says which neighbours are scene
/// changes. With no neighbours the result is `centre`.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference takes each input of the device call as its own argument"
)]
pub fn reference_degrain(
    centre: &[f32],
    (width, height): (usize, usize),
    neighbours: &[ReferenceNeighbour<'_>],
    flagged: &[bool],
    grid: &BlockGrid,
    settings: &DegrainSettings,
    pel: u32,
    scale: PlaneScale,
    chroma: bool,
) -> Vec<f32> {
    if neighbours.is_empty() {
        return centre.to_vec();
    }

    let threshold = settings.sad_threshold(grid.blksize, chroma);

    let mut output = Vec::with_capacity(width * height);
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            let mut weight_sum = 0.0f32;
            let mut value_sum = 0.0f32;
            for covering in covering_windows(x, y, grid, scale) {
                let window = covering.window;
                let centre_weight = 256.0 * window;
                weight_sum += centre_weight;
                value_sum += centre_weight * centre[y as usize * width + x as usize];

                for (neighbour, is_flagged) in neighbours.iter().zip(flagged) {
                    if *is_flagged {
                        continue;
                    }

                    let weight = window * degrain_weight(neighbour.sads[covering.block], threshold);
                    let (vector_x, vector_y) = scaled_vector(neighbour.vectors, covering.block, pel, scale);
                    let value = half_pel_sample(
                        neighbour.plane,
                        width,
                        height,
                        2 * x as i32 + vector_x,
                        2 * y as i32 + vector_y,
                    );

                    weight_sum += weight;
                    value_sum += weight * value;
                }
            }

            output.push(value_sum / weight_sum);
        }
    }

    output
}

/// Whether a field with the normalised block `sads` on `grid` is a scene change under `rule`.
///
/// It is when more than `thscd2` of every 255 blocks have a SAD above `thscd1`.
pub fn reference_scene_changed(sads: &[f32], grid: &BlockGrid, rule: SceneChange) -> bool {
    let threshold = rule.threshold(grid.blksize);
    let changed = sads.iter().filter(|sad| **sad > threshold).count() as u64;
    let allowed = u64::from(rule.thscd2) * grid.blocks() as u64;

    changed * 255 > allowed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scene_change_needs_more_than_thscd2_of_255_blocks() {
        let grid = BlockGrid {
            blksize: 8,
            step: 8,
            across: 4,
            down: 1,
        };
        let lenient = SceneChange {
            thscd1: 400.0,
            thscd2: 255,
        };
        let strict = SceneChange {
            thscd1: 400.0,
            thscd2: 127,
        };
        let high = 100.0;

        assert!(!reference_scene_changed(&[high; 4], &grid, lenient));
        assert!(reference_scene_changed(&[high, high, 0.0, 0.0], &grid, strict));
        assert!(!reference_scene_changed(&[high, 0.0, 0.0, 0.0], &grid, strict));
    }
}
