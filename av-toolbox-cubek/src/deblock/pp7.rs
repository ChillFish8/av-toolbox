// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::h264::peak_of;
use super::rounding::{OPAQUE_ZERO, rounded_product};
use crate::matrix::MatrixSamples;
use crate::plane::are_distinct;
use crate::{DevicePlane, mirror};

/// How a coefficient past its threshold is kept.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Pp7Mode {
    /// Kept as it is.
    Hard,
    /// Moved toward zero by the threshold.
    Soft,
    /// Twice its distance past the threshold, and kept as it is past twice the threshold.
    Medium,
}

impl Pp7Mode {
    /// The mode's code, which the kernel is compiled for.
    const fn code(self) -> u32 {
        match self {
            Self::Hard => HARD,
            Self::Soft => SOFT,
            Self::Medium => MEDIUM,
        }
    }
}

/// The kernel's code for [Pp7Mode::Hard].
const HARD: u32 = 0;

/// The kernel's code for [Pp7Mode::Soft].
const SOFT: u32 = 1;

/// The kernel's code for [Pp7Mode::Medium].
const MEDIUM: u32 = 2;

/// The side of the square of output samples one cube computes.
const TILE: u32 = 16;

/// How far the transform reaches from its centre sample along each axis.
const REACH: u32 = 3;

/// The side of the square of input samples one cube loads, its outputs and [REACH] more on each side.
const LOADED: u32 = TILE + 2 * REACH;

const LOADED_SAMPLES: u32 = LOADED * LOADED;

/// The units of one cube, one for each output sample of its tile.
const UNITS: u32 = TILE * TILE;

/// How many rounds the units take to load their tile.
const LOAD_ROUNDS: u32 = LOADED_SAMPLES.div_ceil(UNITS);

/// The most cubes every device allows along one axis.
const MAX_CUBES_PER_AXIS: u32 = 65535;

/// The most samples a plane can hold, so every index into it fits in an i32.
const MAX_SAMPLES: u64 = i32::MAX as u64;

/// The largest padded side, so twice it still fits in an i32.
const MAX_ALIGNED: u32 = i32::MAX as u32 / 2;

/// The lowest `qp` a threshold table is made for.
const MIN_QP: f64 = 1.0;

/// The highest `qp` a threshold table is made for.
const MAX_QP: f64 = 63.0;

/// Each side of a frame is padded to a multiple of this before its border folds back a second time.
const ALIGNMENT: u32 = 16;

/// Each coefficient's weight in the output sample, by `ky` then `kx`.
///
/// It is `floor(2^18 w_ky w_kx)` with `w` being 1/8, 1/10, 1/8 and 1/20, the weights that rebuild the
/// transform's centre sample.
pub const WEIGHTS: [[i32; 4]; 4] = [
    [4096, 3276, 4096, 1638],
    [3276, 2621, 3276, 1310],
    [4096, 3276, 4096, 1638],
    [1638, 1310, 1638, 655],
];

/// The bits of fraction the weights carry, which integer outputs shift away.
pub const WEIGHT_BITS: u32 = 18;

/// Half of the last weight bit shifted away, which integer outputs add to round half up.
const ROUNDING: i32 = 1 << (WEIGHT_BITS - 1);

/// Float samples are scaled by this into 8-bit units before the transform.
pub const FLOAT_INPUT_SCALE: f32 = 255.0;

/// Float outputs are scaled by this, `1 / (255 * 2^18)`, out of 8-bit units and the weights' fraction.
pub const FLOAT_OUTPUT_SCALE: f32 = (1.0 / (255.0 * (1u32 << WEIGHT_BITS) as f64)) as f32;

/// The factor of coefficient `ky`, `kx`'s threshold, 4 times the norms of its two transform rows.
///
/// It is 16 when both are even, `8√10` when exactly one is odd and 40 when both are odd.
fn threshold_factor(ky: usize, kx: usize) -> f64 {
    let odd = ky % 2 + kx % 2;

    match odd {
        0 => 16.0,
        1 => 8.0 * 10.0f64.sqrt(),
        _ => 40.0,
    }
}

/// Each coefficient's threshold for `qp` on planes of `samples`, by `ky` then `kx`.
///
/// Integer samples take `floor((qp f - 1) peak / 255)`, and float samples `floor(qp f - 1)`, all in f64.
/// The DC coefficient is never thresholded and holds 0.
pub fn thresholds(qp: f64, samples: MatrixSamples) -> [i32; 16] {
    let mut found = [0; 16];

    for (place, threshold) in found.iter_mut().enumerate().skip(1) {
        let factor = threshold_factor(place / 4, place % 4);
        let unscaled = qp * factor - 1.0;
        let scaled = match samples.bits() {
            Some(bits) => unscaled * f64::from((1u32 << bits) - 1) / 255.0,
            None => unscaled,
        };

        *threshold = scaled.floor() as i32;
    }

    found
}

/// The size a `width` by `height` frame's sides are padded to before its border folds back a second time,
/// each rounded up to a multiple of 16.
pub fn aligned_frame(width: u32, height: u32) -> (u32, u32) {
    let aligned_width = width.next_multiple_of(ALIGNMENT);
    let aligned_height = height.next_multiple_of(ALIGNMENT);

    (aligned_width, aligned_height)
}

/// The thresholds of every coefficient, uploaded once for planes of one sample type.
#[derive(Debug)]
pub struct ThresholdTable {
    values: Handle,
    samples: MatrixSamples,
}

impl ThresholdTable {
    fn values<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.values.clone();

        // Safety: the buffer was made from exactly 16 i32 values.
        unsafe { ArrayArg::from_raw_parts(handle, 16) }
    }
}

/// Works out the thresholds for `qp` on planes of `samples` and uploads them.
///
/// # Panics
///
/// Panics when `qp` is not between 1 and 63.
pub fn upload_thresholds<R: Runtime>(
    client: &ComputeClient<R>,
    qp: f64,
    samples: MatrixSamples,
) -> ThresholdTable {
    let in_range = (MIN_QP..=MAX_QP).contains(&qp);
    assert!(in_range, "a pp7 qp must be between 1 and 63, not {qp}");

    let found = thresholds(qp, samples);
    let bytes = i32::as_bytes(&found);
    let values = client.create_from_slice(bytes);

    ThresholdTable { values, samples }
}

/// The planes one pp7 launch reads and writes, both the same size.
#[derive(Debug, Clone, Copy)]
pub struct Pp7Planes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// A plane's sides and the sides they are padded to before the border folds back a second time.
#[derive(CubeType)]
struct Extent {
    width: i32,
    height: i32,
    aligned_width: i32,
    aligned_height: i32,
}

/// Where position `position` of a side of `length` samples, padded to `aligned`, reads.
///
/// Positions from `aligned` on fold back about it first. Then every position mirrors into the side with the
/// edge sample repeated.
#[cube]
fn extended(position: i32, length: i32, aligned: i32) -> i32 {
    let mut folded = position;

    if position >= aligned {
        folded = 2 * aligned - 1 - position;
    }

    mirror(folded, length)
}

/// Loads the input square of the tile at `left`, `top` into `tile`, row by row, through the border.
///
/// Float samples are scaled into 8-bit units as they load.
#[cube]
fn load_tile<E: CubePrimitive, S: Numeric>(
    input: &Array<E>,
    tile: &mut SharedMemory<S>,
    left: u32,
    top: u32,
    extent: &Extent,
    #[comptime] scaled: bool,
) {
    #[unroll]
    for round in 0..LOAD_ROUNDS {
        let slot = UNIT_POS + round * UNITS;

        if slot < LOADED_SAMPLES {
            let row = (top + slot / LOADED) as i32 - REACH as i32;
            let column = (left + slot % LOADED) as i32 - REACH as i32;
            let source_row = extended(row, extent.height, extent.aligned_height);
            let source_column = extended(column, extent.width, extent.aligned_width);
            let index = (source_row * extent.width + source_column) as usize;
            let mut value = S::cast_from(input[index]);

            if scaled {
                value *= S::cast_from(FLOAT_INPUT_SCALE);
            }

            tile[slot as usize] = value;
        }
    }
}

/// The four coefficients of seven taps along one axis, centred on the fourth.
#[derive(CubeType)]
struct Bands<T: Numeric> {
    zero: T,
    one: T,
    two: T,
    three: T,
}

/// Folds `taps` about their centre and runs the folded four through the H.264 core transform, as
/// butterflies.
#[cube]
fn transformed<T: Numeric>(taps: &Array<T>) -> Bands<T> {
    let centre = taps[3] + taps[3];
    let near = taps[2] + taps[4];
    let middle = taps[1] + taps[5];
    let far = taps[0] + taps[6];
    let outer_sum = centre + far;
    let outer_difference = centre - far;
    let inner_sum = near + middle;
    let inner_difference = near - middle;

    Bands::<T> {
        zero: outer_sum + inner_sum,
        one: (outer_difference + outer_difference) + inner_difference,
        two: outer_sum - inner_sum,
        three: outer_difference - (inner_difference + inner_difference),
    }
}

/// The 16 coefficients of the 7 by 7 square of `tile` centred on this unit's sample, by `ky` then `kx`.
///
/// Each of the square's columns is transformed down first, then each row of the results across.
#[cube]
fn coefficients<T: Numeric>(tile: &SharedMemory<T>, found: &mut Array<T>) {
    let mut taps = Array::<T>::new(7usize);
    let mut down = Array::<T>::new(28usize);

    #[unroll]
    for column in 0..7usize {
        #[unroll]
        for row in 0..7usize {
            let slot = (UNIT_POS_Y + row as u32) * LOADED + UNIT_POS_X + column as u32;
            taps[row] = tile[slot as usize];
        }

        let bands = transformed::<T>(&taps);
        down[comptime!(column * 4)] = bands.zero;
        down[comptime!(column * 4 + 1)] = bands.one;
        down[comptime!(column * 4 + 2)] = bands.two;
        down[comptime!(column * 4 + 3)] = bands.three;
    }

    #[unroll]
    for band in 0..4usize {
        #[unroll]
        for column in 0..7usize {
            taps[column] = down[comptime!(column * 4 + band)];
        }

        let bands = transformed::<T>(&taps);
        found[comptime!(band * 4)] = bands.zero;
        found[comptime!(band * 4 + 1)] = bands.one;
        found[comptime!(band * 4 + 2)] = bands.two;
        found[comptime!(band * 4 + 3)] = bands.three;
    }
}

/// `magnitude` with the sign of `value`.
#[cube]
fn signed_like<T: Numeric>(magnitude: T, value: T) -> T {
    let zero = T::from_int(0);
    let mut signed = magnitude;

    if value < zero {
        signed = zero - magnitude;
    }

    signed
}

/// Integer coefficient `value` as `mode` keeps it against `threshold`.
#[cube]
fn kept_integer(value: i32, threshold: i32, #[comptime] mode: u32) -> i32 {
    let magnitude = i32::abs(value);
    let mut kept = 0i32;

    if magnitude > threshold {
        kept = value;

        if comptime!(mode == SOFT) {
            kept = signed_like::<i32>(magnitude - threshold, value);
        }

        if comptime!(mode == MEDIUM) && magnitude <= 2 * threshold {
            kept = signed_like::<i32>(2 * (magnitude - threshold), value);
        }
    }

    kept
}

/// Float coefficient `value` as `mode` keeps it against `threshold`.
///
/// The comparisons use the coefficient's magnitude rounded down, and the kept values the magnitude itself.
#[cube]
fn kept_float(value: f32, threshold: f32, #[comptime] mode: u32) -> f32 {
    let magnitude = f32::abs(value);
    let decided = f32::floor(magnitude);
    let mut kept = 0.0f32;

    if decided > threshold {
        kept = value;

        if comptime!(mode == SOFT) {
            kept = signed_like::<f32>(magnitude - threshold, value);
        }

        if comptime!(mode == MEDIUM) && decided <= threshold + threshold {
            let excess = magnitude - threshold;
            kept = signed_like::<f32>(excess + excess, value);
        }
    }

    kept
}

/// An integer output sample from its 16 `found` coefficients, rounded half up and kept between 0 and `peak`.
///
/// `wide` sums the weighted coefficients in 64 bits, which samples deeper than 8 bits need.
#[cube]
fn integer_output(
    found: &Array<i32>,
    thresholds: &Array<i32>,
    peak: i32,
    #[comptime] mode: u32,
    #[comptime] wide: bool,
) -> i32 {
    let rounded = if wide {
        wide_sum(found, thresholds, mode)
    } else {
        narrow_sum(found, thresholds, mode)
    };

    i32::clamp(rounded, 0, peak)
}

/// The weighted sum of `found` in 64 bits, rounded half up out of the weights' fraction.
#[cube]
fn wide_sum(found: &Array<i32>, thresholds: &Array<i32>, #[comptime] mode: u32) -> i32 {
    let mut total = i64::cast_from(found[0]) * i64::cast_from(WEIGHTS[0][0]);

    #[unroll]
    for place in 1..16usize {
        let weight = comptime!(WEIGHTS[place / 4][place % 4]);
        let kept = kept_integer(found[place], thresholds[place], mode);
        let widened = i64::cast_from(kept);
        total += widened * i64::cast_from(weight);
    }

    let rounding = i64::cast_from(ROUNDING);
    let shift = i64::cast_from(WEIGHT_BITS);
    let shifted = (total + rounding) >> shift;
    i32::cast_from(shifted)
}

/// The weighted sum of `found` in 32 bits, rounded half up out of the weights' fraction.
///
/// It holds 8-bit samples, whose sum stays below `2^30`.
#[cube]
fn narrow_sum(found: &Array<i32>, thresholds: &Array<i32>, #[comptime] mode: u32) -> i32 {
    let mut total = found[0] * WEIGHTS[0][0];

    #[unroll]
    for place in 1..16usize {
        let weight = comptime!(WEIGHTS[place / 4][place % 4]);
        let kept = kept_integer(found[place], thresholds[place], mode);
        total += kept * weight;
    }

    (total + ROUNDING) >> WEIGHT_BITS as i32
}

/// A float output sample from its 16 `found` coefficients, in 8-bit units.
///
/// Each weighted coefficient is rounded to f32 before it is added, in order.
#[cube]
fn float_output(found: &Array<f32>, thresholds: &Array<i32>, opaque_zero: u32, #[comptime] mode: u32) -> f32 {
    let dc_weight = comptime!(WEIGHTS[0][0] as f32);
    let mut total = rounded_product(found[0], dc_weight, opaque_zero);

    #[unroll]
    for place in 1..16usize {
        let weight = comptime!(WEIGHTS[place / 4][place % 4] as f32);
        let threshold = f32::cast_from(thresholds[place]);
        let kept = kept_float(found[place], threshold, mode);
        total += rounded_product(kept, weight, opaque_zero);
    }

    total * FLOAT_OUTPUT_SCALE
}

/// pp7 on one 16 by 16 tile of an integer plane, one unit to each output sample.
///
/// Every unit loads its share of the tile's input square and waits for the rest, so none leaves early.
#[cube(launch_unchecked)]
fn pp7_integer_kernel<E: CubePrimitive>(
    input: &Array<E>,
    output: &mut Array<E>,
    thresholds: &Array<i32>,
    width: u32,
    height: u32,
    aligned_width: u32,
    aligned_height: u32,
    peak: i32,
    #[comptime] mode: u32,
    #[comptime] wide: bool,
) {
    let mut tile = SharedMemory::<i32>::new(comptime!(LOADED_SAMPLES as usize));
    let left = CUBE_POS_X * TILE;
    let top = CUBE_POS_Y * TILE;
    let extent = Extent {
        width: width as i32,
        height: height as i32,
        aligned_width: aligned_width as i32,
        aligned_height: aligned_height as i32,
    };

    load_tile::<E, i32>(input, &mut tile, left, top, &extent, false);
    sync_cube();

    let x = left + UNIT_POS_X;
    let y = top + UNIT_POS_Y;

    if x < width && y < height {
        let mut found = Array::<i32>::new(16usize);
        coefficients::<i32>(&tile, &mut found);

        let value = integer_output(&found, thresholds, peak, mode, wide);
        let index = (y * width + x) as usize;
        output[index] = E::cast_from(value);
    }
}

/// pp7 on one 16 by 16 tile of a float plane, one unit to each output sample.
///
/// Every unit loads its share of the tile's input square and waits for the rest, so none leaves early.
#[cube(launch_unchecked)]
fn pp7_float_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    thresholds: &Array<i32>,
    width: u32,
    height: u32,
    aligned_width: u32,
    aligned_height: u32,
    opaque_zero: u32,
    #[comptime] mode: u32,
) {
    let mut tile = SharedMemory::<f32>::new(comptime!(LOADED_SAMPLES as usize));
    let left = CUBE_POS_X * TILE;
    let top = CUBE_POS_Y * TILE;
    let extent = Extent {
        width: width as i32,
        height: height as i32,
        aligned_width: aligned_width as i32,
        aligned_height: aligned_height as i32,
    };

    load_tile::<f32, f32>(input, &mut tile, left, top, &extent, true);
    sync_cube();

    let x = left + UNIT_POS_X;
    let y = top + UNIT_POS_Y;

    if x < width && y < height {
        let mut found = Array::<f32>::new(16usize);
        coefficients::<f32>(&tile, &mut found);

        let value = float_output(&found, thresholds, opaque_zero, mode);
        let index = (y * width + x) as usize;
        output[index] = value;
    }
}

/// Deblocks `planes.input` into `planes.output` with pp7, in one launch of a cube to each 16 by 16 tile.
///
/// Each output sample keeps the coefficients of its 7 by 7 neighbourhood that `table` and `mode` let
/// through. Reads past the plane's edges mirror back, folding about `aligned_size` first.
///
/// # Panics
///
/// Panics when the planes differ in size or are empty, when the plane holds more than `i32::MAX` samples,
/// when both are given the same handle, when `aligned_size` is smaller than the plane or a side of it is past
/// `i32::MAX / 2`, when the plane needs more cubes along a side than every device allows, or when `table` was
/// made for other samples.
pub fn pp7_plane<R: Runtime>(
    client: &ComputeClient<R>,
    planes: Pp7Planes<'_>,
    aligned_size: (u32, u32),
    table: &ThresholdTable,
    mode: Pp7Mode,
    samples: MatrixSamples,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    let output_width = planes.output.width();
    let output_height = planes.output.height();
    assert!(width > 0 && height > 0, "a pp7 plane must not be empty");

    let samples_in_plane = u64::from(width) * u64::from(height);
    assert!(
        samples_in_plane <= MAX_SAMPLES,
        "a {width}x{height} plane has more samples than pp7 can index"
    );
    assert!(
        output_width == width && output_height == height,
        "the output plane is {output_width}x{output_height}, expected {width}x{height}"
    );

    let handles = [planes.input.handle(), planes.output.handle()];
    let distinct = are_distinct(&handles);
    assert!(distinct, "pp7 cannot write the plane it reads");

    let (aligned_width, aligned_height) = aligned_size;
    assert!(
        aligned_width >= width && aligned_height >= height,
        "a {width}x{height} plane does not fit a {aligned_width}x{aligned_height} alignment"
    );
    assert!(
        aligned_width <= MAX_ALIGNED && aligned_height <= MAX_ALIGNED,
        "a {aligned_width}x{aligned_height} alignment is wider than pp7 can fold"
    );

    let tiles_across = width.div_ceil(TILE);
    let tiles_down = height.div_ceil(TILE);
    assert!(
        tiles_across <= MAX_CUBES_PER_AXIS && tiles_down <= MAX_CUBES_PER_AXIS,
        "a {width}x{height} plane needs more cubes than a device allows"
    );
    assert!(
        table.samples == samples,
        "the threshold table was made for {:?} samples, not {samples:?}",
        table.samples
    );

    let cubes = CubeCount::Static(tiles_across, tiles_down, 1);
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let values = table.values::<R>();
    let unit = CubeDim::new_2d(TILE, TILE);
    let code = mode.code();

    let peak = peak_of(samples) as i32;

    match samples {
        MatrixSamples::U8 => {
            // Safety: every read is mirrored into the plane and every write is a sample inside it, which both
            // arrays hold as `width * height` u8 samples, as `DevicePlane::new` requires. The table holds the
            // 16 thresholds read. 8-bit sums fit in 32 bits.
            unsafe {
                pp7_integer_kernel::launch_unchecked::<u8, R>(
                    client,
                    cubes,
                    unit,
                    input,
                    output,
                    values,
                    width,
                    height,
                    aligned_width,
                    aligned_height,
                    peak,
                    code,
                    false,
                );
            }
        },
        MatrixSamples::U16 { .. } => {
            // Safety: as for 8-bit samples, for arrays of `width * height` u16 samples, summed in 64 bits.
            unsafe {
                pp7_integer_kernel::launch_unchecked::<u16, R>(
                    client,
                    cubes,
                    unit,
                    input,
                    output,
                    values,
                    width,
                    height,
                    aligned_width,
                    aligned_height,
                    peak,
                    code,
                    true,
                );
            }
        },
        MatrixSamples::F32 => {
            // Safety: as for 8-bit samples, for arrays of `width * height` f32 values.
            unsafe {
                pp7_float_kernel::launch_unchecked::<R>(
                    client,
                    cubes,
                    unit,
                    input,
                    output,
                    values,
                    width,
                    height,
                    aligned_width,
                    aligned_height,
                    OPAQUE_ZERO,
                    code,
                );
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EIGHT_BIT_AT_2: [i32; 16] = [0, 49, 31, 49, 49, 79, 49, 79, 31, 49, 31, 49, 49, 79, 49, 79];

    #[test]
    fn eight_bit_thresholds_are_qp_times_the_factor_less_1_rounded_down() {
        let found = thresholds(2.0, MatrixSamples::U8);
        let lowest = thresholds(1.0, MatrixSamples::U8);

        assert_eq!(found, EIGHT_BIT_AT_2);
        assert_eq!(&lowest[..4], &[0, 24, 15, 24]);
        assert_eq!(lowest[5], 39);
    }

    #[test]
    fn deeper_thresholds_scale_by_the_peak_over_255_after_taking_1() {
        let ten_bit = thresholds(10.0, MatrixSamples::U16 { bits: 10 });
        let sixteen_bit = thresholds(1.0, MatrixSamples::U16 { bits: 16 });

        // Scaled by 4 rather than 1023 / 255 they would be 636, 1007 and 1596.
        assert_eq!((ten_bit[2], ten_bit[1], ten_bit[5]), (637, 1010, 1600));
        // Taking 1 after scaling would give 4111, 6500 and 10279.
        assert_eq!(
            (sixteen_bit[2], sixteen_bit[1], sixteen_bit[5]),
            (3855, 6244, 10023)
        );
    }

    #[test]
    fn float_thresholds_are_in_8_bit_units() {
        let found = thresholds(2.5, MatrixSamples::F32);
        let highest = thresholds(63.0, MatrixSamples::F32);

        assert_eq!((found[0], found[2], found[1], found[5]), (0, 39, 62, 99));
        assert_eq!((highest[2], highest[1], highest[5]), (1007, 1592, 2519));
    }

    #[test]
    fn the_weights_are_the_centre_weights_rounded_down_to_18_bits() {
        let centre = [1.0 / 8.0, 1.0 / 10.0, 1.0 / 8.0, 1.0 / 20.0];
        let scale = f64::from(1u32 << WEIGHT_BITS);

        for (ky, row) in WEIGHTS.iter().enumerate() {
            for (kx, weight) in row.iter().enumerate() {
                let exact = scale * centre[ky] * centre[kx];

                let found = f64::from(*weight);

                assert_eq!(found, exact.floor(), "weight {ky},{kx}");
            }
        }
    }

    #[test]
    fn the_float_output_scale_undoes_8_bit_units_and_the_weights() {
        let expected = (1.0 / 66846720.0f64) as f32;

        assert_eq!(FLOAT_OUTPUT_SCALE, expected);
        assert_eq!(FLOAT_INPUT_SCALE, 255.0);
    }

    #[test]
    fn aligned_frames_round_each_side_up_to_16() {
        let exact = aligned_frame(64, 48);
        let wide = aligned_frame(70, 46);
        let odd = aligned_frame(47, 30);
        let tiny = aligned_frame(6, 4);
        let full_hd = aligned_frame(1920, 1080);

        assert_eq!(exact, (64, 48));
        assert_eq!(wide, (80, 48));
        assert_eq!(odd, (48, 32));
        assert_eq!(tiny, (16, 16));
        assert_eq!(full_hd, (1920, 1088));
    }
}
