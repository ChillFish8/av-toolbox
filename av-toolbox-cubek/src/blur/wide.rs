// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::lines::{both_pass_cubes, line_shape, sample_index};
use super::{BlurPlanes, MAX_CUBES_PER_AXIS, MAX_SHARED_BYTES, Storage};
use crate::reflect;

/// The lines a cube blurs side by side, two units each.
const LANES: u32 = 64;

/// The outputs a cube writes along each line.
const SEGMENT: u32 = 32;

/// The samples a unit loads together before stepping its recursions over them.
const BATCH: u32 = 16;

/// The distance between neighbouring line samples in shared memory.
///
/// It is one longer than the lanes, so a cube storing along a line spreads over the banks.
const STRIDE: u32 = LANES + 1;

const _: () = assert!(SEGMENT.is_multiple_of(BATCH));
const _: () = assert!(2 * SEGMENT * STRIDE * size_of::<f32>() as u32 <= MAX_SHARED_BYTES);

/// Deriche's fit of a gaussian as two damped waves, each `(cosine, sine, decay, frequency)`.
///
/// The wave at distance `x` is `(cosine * cos(frequency * x) + sine * sin(frequency * x)) *
/// exp(-decay * x)`, with `x` in standard deviations.
const WAVES: [(f64, f64, f64, f64); 2] = [(1.680, 3.735, 1.783, 0.6318), (-0.6803, -0.2598, 1.723, 1.997)];

/// The standard deviations a recursion runs over the reflected line before its first output.
///
/// The fit has faded to about 3e-5 of its weight by then.
const WARM_UP_SIGMAS: f32 = 6.0;

/// A complex number, the state and coefficients of one wave's recursion.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
struct Complex {
    real: f32,
    imaginary: f32,
}

/// The coefficients of both waves' recursions.
///
/// Each wave's state steps as `state = pole * state + sample`. A causal output weighs the states by
/// the `ahead` weights and an anti-causal one by the `behind` weights.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
struct Recursion {
    first_pole: Complex,
    second_pole: Complex,
    first_ahead: Complex,
    second_ahead: Complex,
    first_behind: Complex,
    second_behind: Complex,
}

/// The host side of [Complex], in f64 while the coefficients are worked out.
#[derive(Debug, Clone, Copy)]
struct HostComplex {
    real: f64,
    imaginary: f64,
}

impl HostComplex {
    fn multiply(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imaginary * other.imaginary,
            imaginary: self.real * other.imaginary + self.imaginary * other.real,
        }
    }

    fn divide(self, other: Self) -> Self {
        let scale = other.real * other.real + other.imaginary * other.imaginary;

        Self {
            real: (self.real * other.real + self.imaginary * other.imaginary) / scale,
            imaginary: (self.imaginary * other.real - self.real * other.imaginary) / scale,
        }
    }

    /// The same value rounded to the f32 the device runs.
    fn rounded(self) -> Self {
        let real = self.real as f32;
        let imaginary = self.imaginary as f32;

        Self {
            real: f64::from(real),
            imaginary: f64::from(imaginary),
        }
    }

    fn launch<R: Runtime>(self) -> ComplexLaunch<R> {
        ComplexLaunch::new(self.real as f32, self.imaginary as f32)
    }
}

/// The recursion for a gaussian of `sigma`, scaled so its whole discrete kernel sums to 1.
///
/// The total is taken over the poles as rounded to f32, so it holds for the recursion the device runs.
fn recursion<R: Runtime>(sigma: f32) -> RecursionLaunch<R> {
    let sigma = f64::from(sigma);
    let one = HostComplex {
        real: 1.0,
        imaginary: 0.0,
    };

    let mut poles = [one; 2];
    let mut weights = [one; 2];
    let mut total = 0.0;
    for (wave, (cosine, sine, decay, frequency)) in WAVES.into_iter().enumerate() {
        let magnitude = (-decay / sigma).exp();
        let angle = frequency / sigma;
        let exact_pole = HostComplex {
            real: magnitude * angle.cos(),
            imaginary: magnitude * angle.sin(),
        };
        let pole = exact_pole.rounded();
        let weight = HostComplex {
            real: cosine,
            imaginary: -sine,
        };

        let both_sides = HostComplex {
            real: 1.0 + pole.real,
            imaginary: pole.imaginary,
        };
        let one_side = HostComplex {
            real: 1.0 - pole.real,
            imaginary: -pole.imaginary,
        };
        let geometric = both_sides.divide(one_side);
        total += weight.multiply(geometric).real;

        poles[wave] = pole;
        weights[wave] = weight;
    }

    let ahead = weights.map(|weight| HostComplex {
        real: weight.real / total,
        imaginary: weight.imaginary / total,
    });
    let first_behind = ahead[0].multiply(poles[0]);
    let second_behind = ahead[1].multiply(poles[1]);

    RecursionLaunch::new(
        poles[0].launch(),
        poles[1].launch(),
        ahead[0].launch(),
        ahead[1].launch(),
        first_behind.launch(),
        second_behind.launch(),
    )
}

/// Steps `state` past one more sample, as `state = pole * state + sample`.
#[cube]
fn advance(state: &mut Complex, pole: Complex, sample: f32) {
    let real = pole.real * state.real - pole.imaginary * state.imaginary + sample;
    let imaginary = pole.real * state.imaginary + pole.imaginary * state.real;

    state.real = real;
    state.imaginary = imaginary;
}

/// The real part of `weight * state`.
#[cube]
fn weigh(state: Complex, weight: Complex) -> f32 {
    weight.real * state.real - weight.imaginary * state.imaginary
}

/// Loads the `BATCH` samples of `line` from `first_position` on into `samples`.
///
/// Positions outside the line are reflected back inside it.
#[cube]
fn load_batch<F: Float>(
    input: &Array<F>,
    samples: &mut Array<f32>,
    line: u32,
    first_position: i32,
    line_length: u32,
    width: u32,
    #[comptime] along_rows: bool,
) {
    let inside = first_position >= 0 && first_position + BATCH as i32 <= line_length as i32;

    if inside {
        #[unroll]
        for slot in 0..BATCH {
            let position = first_position as u32 + slot;
            let index = sample_index(line, position, width, along_rows);
            samples[slot as usize] = f32::cast_from(input[index as usize]);
        }
    } else {
        #[unroll]
        for slot in 0..BATCH {
            let position = reflect(first_position + slot as i32, line_length as i32);
            let index = sample_index(line, position as u32, width, along_rows);
            samples[slot as usize] = f32::cast_from(input[index as usize]);
        }
    }
}

/// Runs the causal recursion of the unit's line into `totals`, from `warm_up` samples before the
/// segment on.
#[cube]
fn run_ahead<F: Float>(
    input: &Array<F>,
    totals: &mut SharedMemory<f32>,
    recursion: Recursion,
    line: u32,
    start: u32,
    batches: u32,
    line_length: u32,
    width: u32,
    warm_up: u32,
    #[comptime] along_rows: bool,
) {
    let first_pole = recursion.first_pole;
    let second_pole = recursion.second_pole;
    let mut samples = Array::<f32>::new(comptime!(BATCH as usize));
    let mut first = Complex {
        real: 0.0,
        imaginary: 0.0,
    };
    let mut second = Complex {
        real: 0.0,
        imaginary: 0.0,
    };

    for batch in 0..warm_up / BATCH {
        let first_position = (start + batch * BATCH) as i32 - warm_up as i32;
        load_batch(
            input,
            &mut samples,
            line,
            first_position,
            line_length,
            width,
            along_rows,
        );

        #[unroll]
        for slot in 0..BATCH {
            let sample = samples[slot as usize];
            advance(&mut first, first_pole, sample);
            advance(&mut second, second_pole, sample);
        }
    }

    for batch in 0..batches {
        let first_position = start + batch * BATCH;
        load_batch(
            input,
            &mut samples,
            line,
            first_position as i32,
            line_length,
            width,
            along_rows,
        );

        #[unroll]
        for slot in 0..BATCH {
            let sample = samples[slot as usize];
            advance(&mut first, first_pole, sample);
            advance(&mut second, second_pole, sample);

            let ahead = weigh(first, recursion.first_ahead) + weigh(second, recursion.second_ahead);
            let local_position = batch * BATCH + slot;
            totals[(local_position * STRIDE + UNIT_POS_X) as usize] = ahead;
        }
    }
}

/// Runs the anti-causal recursion of the unit's line into `totals`, from `warm_up` samples past
/// the segment back.
#[cube]
fn run_behind<F: Float>(
    input: &Array<F>,
    totals: &mut SharedMemory<f32>,
    recursion: Recursion,
    line: u32,
    start: u32,
    batches: u32,
    line_length: u32,
    width: u32,
    warm_up: u32,
    #[comptime] along_rows: bool,
) {
    let first_pole = recursion.first_pole;
    let second_pole = recursion.second_pole;
    let mut samples = Array::<f32>::new(comptime!(BATCH as usize));
    let mut first = Complex {
        real: 0.0,
        imaginary: 0.0,
    };
    let mut second = Complex {
        real: 0.0,
        imaginary: 0.0,
    };

    let past_end = start + batches * BATCH;
    for batch in 0..warm_up / BATCH {
        let first_position = past_end + warm_up - (batch + 1) * BATCH;
        load_batch(
            input,
            &mut samples,
            line,
            first_position as i32,
            line_length,
            width,
            along_rows,
        );

        #[unroll]
        for step in 0..BATCH {
            let sample = samples[comptime!((BATCH - 1 - step) as usize)];
            advance(&mut first, first_pole, sample);
            advance(&mut second, second_pole, sample);
        }
    }

    for batch in 0..batches {
        let first_local = (batches - 1 - batch) * BATCH;
        let first_position = start + first_local;
        load_batch(
            input,
            &mut samples,
            line,
            first_position as i32,
            line_length,
            width,
            along_rows,
        );

        #[unroll]
        for step in 0..BATCH {
            let slot = comptime!(BATCH - 1 - step);
            let behind = weigh(first, recursion.first_behind) + weigh(second, recursion.second_behind);
            let local_position = first_local + slot;
            totals[(local_position * STRIDE + UNIT_POS_X) as usize] = behind;

            let sample = samples[slot as usize];
            advance(&mut first, first_pole, sample);
            advance(&mut second, second_pole, sample);
        }
    }
}

/// Blurs one segment of `SEGMENT` outputs along each of `LANES` lines.
///
/// Each line has two units. One runs the causal recursion and the other the anti-causal one, each
/// from `warm_up` samples outside the segment. The cube then sums both and writes them out together,
/// so neighbouring units write neighbouring samples.
#[cube(launch_unchecked)]
fn recurse_lines<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    recursion: Recursion,
    width: u32,
    height: u32,
    warm_up: u32,
    #[comptime] along_rows: bool,
) {
    let (line_length, line_count) = line_shape(width, height, along_rows);
    let first_line = CUBE_POS_X * LANES;
    let line = min(first_line + UNIT_POS_X, line_count - 1);
    let start = CUBE_POS_Y * SEGMENT;
    let output_count = min(SEGMENT, line_length - start);
    let batches = output_count.div_ceil(BATCH);
    let mut ahead = SharedMemory::<f32>::new(comptime!((SEGMENT * STRIDE) as usize));
    let mut behind = SharedMemory::<f32>::new(comptime!((SEGMENT * STRIDE) as usize));

    if UNIT_POS_Y == 0 {
        run_ahead(
            input,
            &mut ahead,
            recursion,
            line,
            start,
            batches,
            line_length,
            width,
            warm_up,
            along_rows,
        );
    } else {
        run_behind(
            input,
            &mut behind,
            recursion,
            line,
            start,
            batches,
            line_length,
            width,
            warm_up,
            along_rows,
        );
    }

    sync_cube();

    let unit = UNIT_POS_Y * LANES + UNIT_POS_X;
    for step in 0..comptime!(SEGMENT / 2) {
        let flat = step * 2 * LANES + unit;
        let mut local_line = flat % LANES;
        let mut local_position = flat / LANES;

        if along_rows {
            local_line = flat / SEGMENT;
            local_position = flat % SEGMENT;
        }

        let line = first_line + local_line;
        let position = start + local_position;
        if line < line_count && position < line_length {
            let slot = (local_position * STRIDE + local_line) as usize;
            let total = ahead[slot] + behind[slot];
            let index = sample_index(line, position, width, along_rows);
            output[index as usize] = F::cast_from(total);
        }
    }
}

/// The cubes one pass along lines of `line_length` samples takes, `line_count` lines in all.
fn pass_cubes(line_length: u32, line_count: u32) -> Option<CubeCount> {
    let blocks = line_count.div_ceil(LANES);
    let segments = line_length.div_ceil(SEGMENT);
    if blocks > MAX_CUBES_PER_AXIS || segments > MAX_CUBES_PER_AXIS {
        return None;
    }

    Some(CubeCount::Static(blocks, segments, 1))
}

/// The cubes both passes over a `width` by `height` plane take.
///
/// It is `None` when either pass needs more cubes along an axis than every device allows.
pub(super) fn wide_cubes(width: u32, height: u32) -> Option<(CubeCount, CubeCount)> {
    both_pass_cubes(width, height, pass_cubes)
}

/// Blurs `planes.input` across into `planes.scratch`, then down into `planes.output`.
///
/// Each pass runs Deriche's recursive gaussian along a reflected extension of every line. Its cost
/// grows with `sigma` through each segment's warm-up. It panics when the plane needs more cubes than
/// [wide_cubes](crate::blur::wide::wide_cubes) allows.
pub(super) fn blur_wide<R: Runtime>(
    client: &ComputeClient<R>,
    planes: BlurPlanes<'_>,
    sigma: f32,
    storage: Storage,
) {
    let width = planes.output.width();
    let height = planes.output.height();
    let cubes = wide_cubes(width, height);
    let (row_cubes, column_cubes) = cubes.expect("the plane should fit the wide blur's cube count");
    let shape = CubeDim::new_2d(LANES, 2);
    let reach = (WARM_UP_SIGMAS * sigma).ceil() as u32;
    let warm_up = reach.div_ceil(BATCH) * BATCH;

    let row_recursion = recursion::<R>(sigma);
    let input = planes.input.array::<R>();
    let scratch = planes.scratch.array::<R>();

    // Safety: every plane holds `width * height` values of `storage`, every read is reflected into
    // its line, units past the last line read the last line and every write is guarded to the plane.
    unsafe {
        match storage {
            Storage::F32 => recurse_lines::launch_unchecked::<f32, R>(
                client,
                row_cubes,
                shape,
                input,
                scratch,
                row_recursion,
                width,
                height,
                warm_up,
                true,
            ),
            Storage::F16 => recurse_lines::launch_unchecked::<half::f16, R>(
                client,
                row_cubes,
                shape,
                input,
                scratch,
                row_recursion,
                width,
                height,
                warm_up,
                true,
            ),
        }
    }

    let column_recursion = recursion::<R>(sigma);
    let scratch = planes.scratch.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: every plane holds `width * height` values of `storage`, every read is reflected into
    // its line, units past the last line read the last line and every write is guarded to the plane.
    unsafe {
        match storage {
            Storage::F32 => recurse_lines::launch_unchecked::<f32, R>(
                client,
                column_cubes,
                shape,
                scratch,
                output,
                column_recursion,
                width,
                height,
                warm_up,
                false,
            ),
            Storage::F16 => recurse_lines::launch_unchecked::<half::f16, R>(
                client,
                column_cubes,
                shape,
                scratch,
                output,
                column_recursion,
                width,
                height,
                warm_up,
                false,
            ),
        }
    }
}
