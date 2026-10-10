// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::{DevicePlane, Dispatch, reflect};

/// The widest radius a blur reaches, in samples either side of its centre.
pub const MAX_RADIUS: u32 = 4096;

/// The widest deviation a blur takes, whose radius is exactly [MAX_RADIUS].
pub const MAX_SIGMA: f32 = MAX_RADIUS as f32 / 4.0;

/// The radius a blur of `sigma` reaches, four standard deviations out.
///
/// `sigma` has to be finite and at most [MAX_SIGMA], so the radius fits
/// the kernels' arithmetic.
pub fn radius_for(sigma: f32) -> u32 {
    let reach = (4.0 * sigma).ceil() as u32;

    reach.max(1)
}

/// Gaussian weights from `-radius` to `radius`, normalised to sum to 1.
pub fn gaussian_weights(sigma: f32, radius: u32) -> Vec<f32> {
    let radius = radius as i32;
    let spread = 2.0 * sigma * sigma;

    let raw: Vec<f32> = (-radius..=radius)
        .map(|offset| {
            let distance = offset as f32;
            (-(distance * distance) / spread).exp()
        })
        .collect();
    let total: f32 = raw.iter().sum();

    raw.iter().map(|weight| weight / total).collect()
}

#[cube(launch_unchecked)]
fn blur_rows(input: &Array<f32>, weights: &Array<f32>, output: &mut Array<f32>, width: u32, radius: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row_start = position - column;

    let mut total = 0.0f32;
    for tap in 0..2 * radius + 1 {
        let offset = tap as i32 - radius as i32;
        let source = reflect(column as i32 + offset, width as i32);
        let index = row_start + source as u32;
        total += input[index as usize] * weights[tap as usize];
    }

    output[ABSOLUTE_POS] = total;
}

#[cube(launch_unchecked)]
fn blur_columns(
    input: &Array<f32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;

    let mut total = 0.0f32;
    for tap in 0..2 * radius + 1 {
        let offset = tap as i32 - radius as i32;
        let source = reflect(row as i32 + offset, height as i32);
        let index = source as u32 * width + column;
        total += input[index as usize] * weights[tap as usize];
    }

    output[ABSOLUTE_POS] = total;
}

/// The planes one blur reads, passes through and writes. All are f32 and of one size.
#[derive(Debug, Clone, Copy)]
pub struct BlurPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub scratch: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// A separable gaussian blur with its weights already on the device.
#[derive(Debug)]
pub struct Gaussian {
    weights: Handle,
    radius: u32,
}

impl Gaussian {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, sigma: f32) -> Self {
        let radius = radius_for(sigma);

        Self::with_radius(client, sigma, radius)
    }

    pub fn with_radius<R: Runtime>(client: &ComputeClient<R>, sigma: f32, radius: u32) -> Self {
        let weights = gaussian_weights(sigma, radius);
        let bytes = f32::as_bytes(&weights);
        let weights = client.create_from_slice(bytes);

        Self { weights, radius }
    }

    pub const fn radius(&self) -> u32 {
        self.radius
    }

    fn weights<R: Runtime>(&self) -> ArrayArg<R> {
        let taps = 2 * self.radius as usize + 1;
        let handle = self.weights.clone();

        // Safety: the weights buffer was made from exactly `taps` f32 values.
        unsafe { ArrayArg::from_raw_parts(handle, taps) }
    }

    pub fn apply<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: BlurPlanes<'_>,
    ) {
        let samples = planes.output.samples();
        let width = planes.output.width();
        let height = planes.output.height();

        let input = planes.input.array::<R>();
        let weights = self.weights::<R>();
        let scratch = planes.scratch.array::<R>();
        let cubes = dispatch.cube_count(samples);
        let shape = dispatch.cube_dim();

        // Safety: every plane holds `samples` f32 values and the kernel writes nothing past the
        // output's length.
        unsafe {
            blur_rows::launch_unchecked::<R>(
                client,
                cubes,
                shape,
                input,
                weights,
                scratch,
                width,
                self.radius,
            );
        }

        let scratch = planes.scratch.array::<R>();
        let weights = self.weights::<R>();
        let output = planes.output.array::<R>();
        let cubes = dispatch.cube_count(samples);
        let shape = dispatch.cube_dim();

        // Safety: every plane holds `samples` f32 values and the kernel writes nothing past the
        // output's length.
        unsafe {
            blur_columns::launch_unchecked::<R>(
                client,
                cubes,
                shape,
                scratch,
                weights,
                output,
                width,
                height,
                self.radius,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weights_sum_to_one_and_are_symmetric() {
        let weights = gaussian_weights(1.0, 4);

        let total: f32 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert_eq!(weights.len(), 9);
        assert!((weights[0] - weights[8]).abs() < 1e-9);
    }

    #[test]
    fn the_radius_reaches_four_deviations() {
        assert_eq!(radius_for(1.0), 4);
        assert_eq!(radius_for(1.6), 7);
        assert_eq!(radius_for(0.1), 1);
    }

    #[test]
    fn the_widest_sigma_reaches_the_widest_radius() {
        assert_eq!(radius_for(MAX_SIGMA), MAX_RADIUS);
    }
}
