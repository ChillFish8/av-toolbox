// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::{DevicePlane, Dispatch, reflect};

/// Bilinear, weighing a sample by how close it is.
pub fn bilinear(distance: f32) -> f32 {
    (1.0 - distance.abs()).max(0.0)
}

pub const BILINEAR_SUPPORT: f32 = 1.0;

/// Which input samples each output sample of one axis reads, and how much each weighs.
///
/// Output sample `i` reads `count` input samples from `starts[i]`, weighted by
/// `weights[i * count..(i + 1) * count]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Taps {
    pub starts: Vec<i32>,
    pub weights: Vec<f32>,
    pub count: u32,
}

/// The taps of one axis, with output sample `i` at input position `scale * i + offset`.
///
/// When `scale` is above 1 the filter is widened by it, so a downscale averages every sample it
/// covers. Each output sample's weights sum to 1.
pub fn taps(output_len: u32, scale: f32, offset: f32, filter: impl Fn(f32) -> f32, support: f32) -> Taps {
    let stretch = scale.max(1.0);
    let reach = support * stretch;
    let count = (2.0 * reach).ceil() as u32 + 1;

    let mut starts = Vec::with_capacity(output_len as usize);
    let mut weights = Vec::with_capacity((output_len * count) as usize);

    for index in 0..output_len {
        let position = scale * index as f32 + offset;
        let start = (position - reach).floor() as i32;

        let raw: Vec<f32> = (0..count)
            .map(|step| {
                let distance = (start + step as i32) as f32 - position;
                filter(distance / stretch)
            })
            .collect();
        let total: f32 = raw.iter().sum();

        starts.push(start);
        weights.extend(raw.iter().map(|weight| weight / total));
    }

    Taps {
        starts,
        weights,
        count,
    }
}

/// One axis's taps on the device.
#[derive(Debug)]
pub struct Weights {
    starts: Handle,
    weights: Handle,
    count: u32,
    outputs: u32,
}

pub fn upload<R: Runtime>(client: &ComputeClient<R>, taps: &Taps) -> Weights {
    let start_bytes = i32::as_bytes(&taps.starts);
    let starts = client.create_from_slice(start_bytes);
    let weight_bytes = f32::as_bytes(&taps.weights);
    let weights = client.create_from_slice(weight_bytes);

    Weights {
        starts,
        weights,
        count: taps.count,
        outputs: taps.starts.len() as u32,
    }
}

impl Weights {
    fn starts<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.starts.clone();
        let len = self.outputs as usize;

        // Safety: the buffer was made from exactly `outputs` i32 values.
        unsafe { ArrayArg::from_raw_parts(handle, len) }
    }

    fn weights<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.weights.clone();
        let len = (self.outputs * self.count) as usize;

        // Safety: the buffer was made from exactly `outputs * count` f32 values.
        unsafe { ArrayArg::from_raw_parts(handle, len) }
    }
}

/// The planes one resize reads, passes through and writes.
#[derive(Debug, Clone, Copy)]
pub struct ResizePlanes<'a> {
    pub input: DevicePlane<'a>,
    pub scratch: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn resize_rows(
    input: &Array<u8>,
    starts: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    in_width: u32,
    out_width: u32,
    count: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % out_width;
    let row = position / out_width;
    let start = starts[column as usize];

    let mut total = 0.0f32;
    for step in 0..count {
        let source = reflect(start + step as i32, in_width as i32);
        let sample = f32::cast_from(input[(row * in_width + source as u32) as usize]);
        let weight = weights[(column * count + step) as usize];
        total += sample * weight;
    }

    output[ABSOLUTE_POS] = total;
}

#[cube(launch_unchecked)]
fn resize_columns(
    input: &Array<f32>,
    starts: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<u8>,
    width: u32,
    in_height: u32,
    count: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;
    let start = starts[row as usize];

    let mut total = 0.0f32;
    for step in 0..count {
        let source = reflect(start + step as i32, in_height as i32);
        let sample = input[(source as u32 * width + column) as usize];
        let weight = weights[(row * count + step) as usize];
        total += sample * weight;
    }

    let rounded = f32::round(total);
    output[ABSOLUTE_POS] = u8::cast_from(f32::clamp(rounded, 0.0, 255.0));
}

/// Resizes a U8 plane through one table per axis.
pub fn apply<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ResizePlanes<'_>,
    across: &Weights,
    down: &Weights,
) {
    let in_width = planes.input.width();
    let in_height = planes.input.height();
    let out_width = planes.output.width();

    let input = planes.input.array::<R>();
    let starts = across.starts::<R>();
    let weights = across.weights::<R>();
    let rows = planes.scratch.array::<R>();
    let row_samples = planes.scratch.samples();
    let cubes = dispatch.cube_count(row_samples);
    let unit = dispatch.cube_dim();

    // Safety: the input holds `in_width * in_height` bytes, the scratch `out_width * in_height`
    // f32 values, the tables `out_width` starts and `out_width * count` weights, and every tap is
    // folded back inside its row before it is read.
    unsafe {
        resize_rows::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            starts,
            weights,
            rows,
            in_width,
            out_width,
            across.count,
        );
    }

    let rows = planes.scratch.array::<R>();
    let starts = down.starts::<R>();
    let weights = down.weights::<R>();
    let output = planes.output.array::<R>();
    let output_samples = planes.output.samples();
    let cubes = dispatch.cube_count(output_samples);
    let unit = dispatch.cube_dim();

    // Safety: as above, with every tap folded back inside its column.
    unsafe {
        resize_columns::launch_unchecked::<R>(
            client, cubes, unit, rows, starts, weights, output, out_width, in_height, down.count,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The non-zero taps of output sample `index`, as input index and weight.
    fn nonzero(table: &Taps, index: usize) -> Vec<(i32, f32)> {
        let count = table.count as usize;
        let row = &table.weights[index * count..(index + 1) * count];

        row.iter()
            .enumerate()
            .filter(|(_, weight)| **weight > 0.0)
            .map(|(step, weight)| (table.starts[index] + step as i32, *weight))
            .collect()
    }

    #[test]
    fn a_left_sited_halving_weighs_one_two_one() {
        let table = taps(4, 2.0, 0.0, bilinear, BILINEAR_SUPPORT);

        assert_eq!(nonzero(&table, 2), vec![(3, 0.25), (4, 0.5), (5, 0.25)]);
    }

    #[test]
    fn a_centred_halving_weighs_one_three_three_one() {
        let table = taps(4, 2.0, 0.5, bilinear, BILINEAR_SUPPORT);

        assert_eq!(
            nonzero(&table, 2),
            vec![(3, 0.125), (4, 0.375), (5, 0.375), (6, 0.125)]
        );
    }

    #[test]
    fn a_scale_of_one_copies() {
        let table = taps(5, 1.0, 0.0, bilinear, BILINEAR_SUPPORT);

        assert_eq!(nonzero(&table, 3), vec![(3, 1.0)]);
    }

    #[test]
    fn a_doubling_interpolates_between_neighbours() {
        let table = taps(8, 0.5, 0.0, bilinear, BILINEAR_SUPPORT);

        assert_eq!(nonzero(&table, 3), vec![(1, 0.5), (2, 0.5)]);
    }

    #[test]
    fn every_output_samples_weights_sum_to_one() {
        let table = taps(4, 2.0, 0.5, bilinear, BILINEAR_SUPPORT);

        for row in table.weights.chunks(table.count as usize) {
            let total: f32 = row.iter().sum();
            assert!((total - 1.0).abs() < 1e-6);
        }
    }
}
