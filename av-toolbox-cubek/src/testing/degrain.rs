use super::motion::{covering_windows, half_pel_sample, scaled_vector};
use crate::degrain::{DegrainSettings, degrain_weight};
use crate::motion::{BlockGrid, PlaneScale};

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

                for (neighbour, on) in neighbours.iter().zip(weights_on) {
                    if !on {
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
