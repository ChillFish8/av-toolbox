// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::neighbourhood::Edge;
use crate::{DevicePlane, Dispatch, reflect};

/// How much an input sample weighs by its distance from where an output sample lands.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Kernel {
    /// The nearest input sample only.
    Point,
    Bilinear,
    /// The Mitchell-Netravali cubics, where `b` blurs and `c` sharpens.
    Bicubic {
        b: f32,
        c: f32,
    },
    /// A sinc windowed by a wider sinc, reaching `taps` samples each side.
    Lanczos {
        taps: u32,
    },
    Spline16,
    Spline36,
    Spline64,
    /// Every input sample within half a sample weighs the same, so a downscale averages the samples it
    /// covers.
    Box,
}

impl Kernel {
    /// The weight of an input sample `distance` input samples from the output sample.
    pub fn weight(&self, distance: f32) -> f32 {
        if let Self::Point | Self::Box = self {
            return point(distance);
        }

        let x = distance.abs();
        if x >= self.support() {
            return 0.0;
        }

        match self {
            Self::Point | Self::Box => point(distance),
            Self::Bilinear => 1.0 - x,
            Self::Bicubic { b, c } => bicubic(x, *b, *c),
            Self::Lanczos { taps } => lanczos(x, *taps as f32),
            Self::Spline16 => spline16(x),
            Self::Spline36 => spline36(x),
            Self::Spline64 => spline64(x),
        }
    }

    /// How far this kernel reaches either side, in input samples, before a downscale widens it.
    pub fn support(&self) -> f32 {
        match self {
            Self::Point | Self::Box => 0.5,
            Self::Bilinear => 1.0,
            Self::Bicubic { .. } | Self::Spline16 => 2.0,
            Self::Lanczos { taps } => *taps as f32,
            Self::Spline36 => 3.0,
            Self::Spline64 => 4.0,
        }
    }

    /// Whether a downscale widens this kernel so it averages every input sample it covers.
    ///
    /// Point never widens, so it always picks exactly one sample.
    pub fn widens(&self) -> bool {
        !matches!(self, Self::Point)
    }
}

fn point(distance: f32) -> f32 {
    if (-0.5..0.5).contains(&distance) { 1.0 } else { 0.0 }
}

fn bicubic(x: f32, b: f32, c: f32) -> f32 {
    if x < 1.0 {
        let cubic = (12.0 - 9.0 * b - 6.0 * c) * x * x * x;
        let square = (-18.0 + 12.0 * b + 6.0 * c) * x * x;
        let constant = 6.0 - 2.0 * b;

        return (cubic + square + constant) / 6.0;
    }

    let cubic = (-b - 6.0 * c) * x * x * x;
    let square = (6.0 * b + 30.0 * c) * x * x;
    let linear = (-12.0 * b - 48.0 * c) * x;
    let constant = 8.0 * b + 24.0 * c;

    (cubic + square + linear + constant) / 6.0
}

fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }

    let angle = std::f32::consts::PI * x;

    angle.sin() / angle
}

fn lanczos(x: f32, taps: f32) -> f32 {
    let centre = sinc(x);
    let window = sinc(x / taps);

    centre * window
}

fn spline16(x: f32) -> f32 {
    if x < 1.0 {
        return ((x - 9.0 / 5.0) * x - 1.0 / 5.0) * x + 1.0;
    }

    let t = x - 1.0;

    ((-1.0 / 3.0 * t + 4.0 / 5.0) * t - 7.0 / 15.0) * t
}

fn spline36(x: f32) -> f32 {
    if x < 1.0 {
        return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0;
    }

    if x < 2.0 {
        let t = x - 1.0;

        return ((-6.0 / 11.0 * t + 270.0 / 209.0) * t - 156.0 / 209.0) * t;
    }

    let t = x - 2.0;

    ((1.0 / 11.0 * t - 45.0 / 209.0) * t + 26.0 / 209.0) * t
}

fn spline64(x: f32) -> f32 {
    if x < 1.0 {
        return ((49.0 / 41.0 * x - 6387.0 / 2911.0) * x - 3.0 / 2911.0) * x + 1.0;
    }

    if x < 2.0 {
        let t = x - 1.0;

        return ((-24.0 / 41.0 * t + 4032.0 / 2911.0) * t - 2328.0 / 2911.0) * t;
    }

    if x < 3.0 {
        let t = x - 2.0;

        return ((6.0 / 41.0 * t - 1008.0 / 2911.0) * t + 582.0 / 2911.0) * t;
    }

    let t = x - 3.0;

    ((-1.0 / 41.0 * t + 168.0 / 2911.0) * t - 97.0 / 2911.0) * t
}

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
/// When `scale` is above 1 a kernel that widens is stretched by it, so a downscale averages every
/// sample it covers. Each output sample's weights sum to 1.
pub fn taps(output_len: u32, scale: f32, offset: f32, kernel: &Kernel) -> Taps {
    let stretch = if kernel.widens() { scale.max(1.0) } else { 1.0 };
    let reach = kernel.support() * stretch;
    let count = (2.0 * reach).ceil() as u32 + 1;

    let mut starts = Vec::with_capacity(output_len as usize);
    let mut weights = Vec::with_capacity((output_len * count) as usize);

    for index in 0..output_len {
        let position = scale * index as f32 + offset;
        let start = (position - reach).floor() as i32;

        let raw: Vec<f32> = (0..count)
            .map(|step| {
                let distance = (start + step as i32) as f32 - position;
                kernel.weight(distance / stretch)
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

/// The taps of a centred resize of one axis from `input_len` samples to `output_len`.
pub fn centred_taps(input_len: u32, output_len: u32, kernel: &Kernel) -> Taps {
    let scale = input_len as f32 / output_len as f32;
    let offset = (scale - 1.0) / 2.0;

    taps(output_len, scale, offset, kernel)
}

/// One axis's taps on the device.
#[derive(Debug)]
pub struct Weights {
    starts: Handle,
    weights: Handle,
    pub(crate) count: u32,
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
    pub(crate) fn starts<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.starts.clone();
        let len = self.outputs as usize;

        // Safety: the buffer was made from exactly `outputs` i32 values.
        unsafe { ArrayArg::from_raw_parts(handle, len) }
    }

    pub(crate) fn weights<R: Runtime>(&self) -> ArrayArg<R> {
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

/// How a plane's samples are stored.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Samples {
    U8,
    /// Samples held in `u16`, which never exceed `largest`.
    U16 {
        largest: u16,
    },
}

impl Samples {
    fn largest(self) -> f32 {
        match self {
            Self::U8 => f32::from(u8::MAX),
            Self::U16 { largest } => f32::from(largest),
        }
    }
}

#[cube(launch_unchecked)]
fn resize_rows<E: Int>(
    input: &Array<E>,
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
fn resize_columns<E: Int>(
    input: &Array<f32>,
    starts: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<E>,
    width: u32,
    in_height: u32,
    count: u32,
    largest: f32,
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
    output[ABSOLUTE_POS] = E::cast_from(f32::clamp(rounded, 0.0, largest));
}

/// Resizes a plane through one table per axis.
pub fn apply<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ResizePlanes<'_>,
    across: &Weights,
    down: &Weights,
    samples: Samples,
) {
    let largest = samples.largest();

    match samples {
        Samples::U8 => launch::<u8, R>(client, dispatch, planes, across, down, largest),
        Samples::U16 { .. } => launch::<u16, R>(client, dispatch, planes, across, down, largest),
    }
}

fn launch<E: Int, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ResizePlanes<'_>,
    across: &Weights,
    down: &Weights,
    largest: f32,
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

    // Safety: the input holds `in_width * in_height` samples of `E`, the scratch
    // `out_width * in_height` f32 values, the tables `out_width` starts and `out_width * count`
    // weights, and every tap is folded back inside its row before it is read.
    unsafe {
        resize_rows::launch_unchecked::<E, R>(
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

    // Safety: the scratch holds `out_width * in_height` f32 values, the output
    // `out_width * out_height` samples of `E`, the tables `out_height` starts and
    // `out_height * count` weights, and every tap is folded back inside its column before it is
    // read.
    unsafe {
        resize_columns::launch_unchecked::<E, R>(
            client, cubes, unit, rows, starts, weights, output, out_width, in_height, down.count, largest,
        );
    }
}

/// Brings a tap position inside `length` samples, by clamping when `clamp` is 1 and reflecting otherwise.
#[cube]
fn fold(index: i32, length: i32, clamp: u32) -> i32 {
    let mut folded = reflect(index, length);
    if clamp == 1u32 {
        folded = i32::clamp(index, 0, length - 1);
    }

    folded
}

#[cube(launch_unchecked)]
fn resize_rows_float(
    input: &Array<f32>,
    starts: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    in_width: u32,
    out_width: u32,
    count: u32,
    clamp: u32,
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
        let source = fold(start + step as i32, in_width as i32, clamp);
        let sample = input[(row * in_width + source as u32) as usize];
        let weight = weights[(column * count + step) as usize];
        total += sample * weight;
    }

    output[ABSOLUTE_POS] = total;
}

#[cube(launch_unchecked)]
fn resize_columns_float(
    input: &Array<f32>,
    starts: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    in_height: u32,
    count: u32,
    clamp: u32,
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
        let source = fold(start + step as i32, in_height as i32, clamp);
        let sample = input[(source as u32 * width + column) as usize];
        let weight = weights[(row * count + step) as usize];
        total += sample * weight;
    }

    output[ABSOLUTE_POS] = total;
}

/// Resizes an f32 plane through one table per axis, keeping f32 throughout.
pub fn apply_float<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ResizePlanes<'_>,
    across: &Weights,
    down: &Weights,
) {
    apply_float_with_edge(client, dispatch, planes, across, down, Edge::Reflect);
}

/// The same as [apply_float], reading taps past the plane's edge as `edge` says.
pub fn apply_float_with_edge<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ResizePlanes<'_>,
    across: &Weights,
    down: &Weights,
    edge: Edge,
) {
    let clamp = u32::from(edge == Edge::Clamp);
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

    // Safety: the input holds `in_width * in_height` f32 values, the scratch
    // `out_width * in_height`, the tables `out_width` starts and `out_width * count` weights, and
    // every tap is folded back inside its row before it is read.
    unsafe {
        resize_rows_float::launch_unchecked::<R>(
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
            clamp,
        );
    }

    let rows = planes.scratch.array::<R>();
    let starts = down.starts::<R>();
    let weights = down.weights::<R>();
    let output = planes.output.array::<R>();
    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the scratch holds `out_width * in_height` f32 values, the output
    // `out_width * out_height`, the tables `out_height` starts and `out_height * count` weights, and
    // every tap is folded back inside its column before it is read.
    unsafe {
        resize_columns_float::launch_unchecked::<R>(
            client, cubes, unit, rows, starts, weights, output, out_width, in_height, down.count, clamp,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERPOLATING: [Kernel; 7] = [
        Kernel::Bilinear,
        Kernel::Bicubic { b: 0.0, c: 0.5 },
        Kernel::Lanczos { taps: 2 },
        Kernel::Lanczos { taps: 3 },
        Kernel::Spline16,
        Kernel::Spline36,
        Kernel::Spline64,
    ];

    const EVERY: [Kernel; 10] = [
        Kernel::Point,
        Kernel::Bilinear,
        Kernel::Bicubic {
            b: 1.0 / 3.0,
            c: 1.0 / 3.0,
        },
        Kernel::Bicubic { b: 0.0, c: 0.5 },
        Kernel::Lanczos { taps: 3 },
        Kernel::Lanczos { taps: 4 },
        Kernel::Spline16,
        Kernel::Spline36,
        Kernel::Spline64,
        Kernel::Box,
    ];

    /// The non-zero taps of output sample `index`, as input index and weight.
    fn nonzero(table: &Taps, index: usize) -> Vec<(i32, f32)> {
        let count = table.count as usize;
        let row = &table.weights[index * count..(index + 1) * count];

        row.iter()
            .enumerate()
            .filter(|(_, weight)| **weight != 0.0)
            .map(|(step, weight)| (table.starts[index] + step as i32, *weight))
            .collect()
    }

    fn assert_near(found: f32, expected: f32) {
        assert!((found - expected).abs() < 1e-5, "{found} was not near {expected}");
    }

    #[test]
    fn interpolating_kernels_are_one_at_zero_and_zero_at_whole_distances() {
        for kernel in INTERPOLATING {
            let reach = kernel.support() as i32;

            assert_near(kernel.weight(0.0), 1.0);

            for distance in 1..=reach {
                let ahead = kernel.weight(distance as f32);
                let behind = kernel.weight(-distance as f32);

                assert_near(ahead, 0.0);
                assert_near(behind, 0.0);
            }
        }
    }

    #[test]
    fn every_kernel_is_zero_at_and_past_its_support() {
        for kernel in EVERY {
            let support = kernel.support();
            let at = kernel.weight(support);
            let past = kernel.weight(support + 0.25);
            let before = kernel.weight(-support - 0.25);

            assert_eq!(at, 0.0, "{kernel:?}");
            assert_eq!(past, 0.0, "{kernel:?}");
            assert_eq!(before, 0.0, "{kernel:?}");
        }
    }

    #[test]
    fn known_kernel_values() {
        let mitchell = Kernel::Bicubic {
            b: 1.0 / 3.0,
            c: 1.0 / 3.0,
        };
        let lanczos = Kernel::Lanczos { taps: 3 };

        assert_near(Kernel::Spline36.weight(0.5), 0.598_684);
        assert_near(lanczos.weight(0.5), 0.607_927);
        assert_near(mitchell.weight(0.0), 8.0 / 9.0);
        assert_near(mitchell.weight(1.0), 1.0 / 18.0);
    }

    #[test]
    fn point_picks_the_nearest_sample_on_a_4x_downscale() {
        let table = taps(2, 4.0, 1.5, &Kernel::Point);

        assert_eq!(nonzero(&table, 0), vec![(1, 1.0)]);
        assert_eq!(nonzero(&table, 1), vec![(5, 1.0)]);
    }

    #[test]
    fn every_kernels_weights_sum_to_one_up_and_down() {
        for kernel in EVERY {
            let upscale = taps(16, 0.5, -0.25, &kernel);
            let downscale = taps(4, 2.5, 0.75, &kernel);

            for table in [upscale, downscale] {
                for row in table.weights.chunks(table.count as usize) {
                    let total: f32 = row.iter().sum();
                    assert_near(total, 1.0);
                }
            }
        }
    }

    #[test]
    fn a_left_sited_halving_weighs_one_two_one() {
        let table = taps(4, 2.0, 0.0, &Kernel::Bilinear);

        assert_eq!(nonzero(&table, 2), vec![(3, 0.25), (4, 0.5), (5, 0.25)]);
    }

    #[test]
    fn a_centred_halving_weighs_one_three_three_one() {
        let table = taps(4, 2.0, 0.5, &Kernel::Bilinear);

        assert_eq!(
            nonzero(&table, 2),
            vec![(3, 0.125), (4, 0.375), (5, 0.375), (6, 0.125)]
        );
    }

    #[test]
    fn a_centred_box_halving_averages_each_pair() {
        let table = centred_taps(6, 3, &Kernel::Box);

        for index in 0..3 {
            let first = 2 * index as i32;
            let expected = vec![(first, 0.5), (first + 1, 0.5)];

            assert_eq!(nonzero(&table, index), expected);
        }
    }

    #[test]
    fn a_box_halving_weighs_taps_past_the_edge_at_zero() {
        let table = centred_taps(6, 3, &Kernel::Box);

        assert_eq!(table.starts[0], -1);
        assert_eq!(table.weights[0], 0.0);
    }

    #[test]
    fn a_left_aligned_box_doubling_weighs_one() {
        let table = taps(8, 0.5, 0.0, &Kernel::Box);

        for row in table.weights.chunks(table.count as usize) {
            let total: f32 = row.iter().sum();
            assert_near(total, 1.0);
        }
    }

    #[test]
    fn a_scale_of_one_copies() {
        let table = taps(5, 1.0, 0.0, &Kernel::Bilinear);

        assert_eq!(nonzero(&table, 3), vec![(3, 1.0)]);
    }

    #[test]
    fn a_doubling_interpolates_between_neighbours() {
        let table = taps(8, 0.5, 0.0, &Kernel::Bilinear);

        assert_eq!(nonzero(&table, 3), vec![(1, 0.5), (2, 0.5)]);
    }

    #[test]
    fn centred_taps_keep_a_flat_plane_flat() {
        let bicubic = Kernel::Bicubic {
            b: 1.0 / 3.0,
            c: 1.0 / 3.0,
        };
        let down = centred_taps(40, 16, &bicubic);
        let up = centred_taps(16, 40, &bicubic);
        let unchanged = same_height(3);
        let flat = vec![100.0f32; 40 * 3];

        let shrunk = crate::testing::reference_resize_float(&flat, 40, 3, &down, &unchanged);
        let grown = crate::testing::reference_resize_float(&shrunk, 16, 3, &up, &unchanged);

        assert!(grown.iter().all(|value| (value - 100.0).abs() < 1e-3));
    }

    /// Taps that keep an axis of `length` as it is.
    fn same_height(length: u32) -> Taps {
        centred_taps(length, length, &Kernel::Point)
    }
}
