use super::motion::{covering_along, half_pel_sample};
use crate::degrain::{DegrainSettings, degrain_weight};
use crate::motion::{BlockGrid, PlaneScale, window_weight};

/// A neighbour plane and its field's vectors and normalised SADs, for [reference_degrain].
pub struct ReferenceNeighbour<'a> {
    pub plane: &'a [f32],
    pub vectors: &'a [i32],
    pub sads: &'a [f32],
}

/// Degrains `centre`, `width` by `height`, with `neighbours` on the CPU, mirroring
/// [degrain](crate::degrain::degrain).
///
/// `centre` and the neighbours are planes scaled by `scale`.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference takes each input of the device call as its own argument"
)]
pub fn reference_degrain(
    centre: &[f32],
    (width, height): (usize, usize),
    neighbours: &[ReferenceNeighbour<'_>],
    weights_on: &[bool],
    grid: &BlockGrid,
    settings: &DegrainSettings,
    pel: u32,
    scale: PlaneScale,
    chroma: bool,
) -> Vec<f32> {
    let overlap = grid.blksize - grid.step;
    let length_x = grid.blksize >> scale.shift_x;
    let length_y = grid.blksize >> scale.shift_y;
    let overlap_x = overlap >> scale.shift_x;
    let overlap_y = overlap >> scale.shift_y;
    let doubling = (2 / pel) as i32;
    let threshold = settings.sad_threshold(grid.blksize, chroma);

    let mut output = Vec::with_capacity(width * height);
    for y in 0..height as u32 {
        let rows = covering_along(y, grid.down, grid.step, grid.blksize, scale.shift_y);
        for x in 0..width as u32 {
            let columns = covering_along(x, grid.across, grid.step, grid.blksize, scale.shift_x);

            let mut weight_sum = 0.0f32;
            let mut value_sum = 0.0f32;
            for row in &rows {
                for column in &columns {
                    let weight_x = window_weight(
                        x - column.start,
                        length_x,
                        overlap_x,
                        column.block == 0,
                        column.block == grid.across - 1,
                    );
                    let weight_y = window_weight(
                        y - row.start,
                        length_y,
                        overlap_y,
                        row.block == 0,
                        row.block == grid.down - 1,
                    );
                    let window = weight_x * weight_y;

                    let centre_weight = 256.0 * window;
                    weight_sum += centre_weight;
                    value_sum += centre_weight * centre[y as usize * width + x as usize];

                    let block = (row.block * grid.across + column.block) as usize;
                    for (neighbour, on) in neighbours.iter().zip(weights_on) {
                        if !on {
                            continue;
                        }

                        let weight = window * degrain_weight(neighbour.sads[block], threshold);
                        let vector_x = (neighbour.vectors[2 * block] >> scale.shift_x) * doubling;
                        let vector_y = (neighbour.vectors[2 * block + 1] >> scale.shift_y) * doubling;
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
            }

            output.push(value_sum / weight_sum);
        }
    }

    output
}
