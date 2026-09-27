// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::depth::Element;
use crate::{DevicePlane, Dispatch};

/// One step of libplacebo's pcg3d generator, in wrapping arithmetic.
pub fn advance(state: &mut [u32; 3]) {
    for lane in state.iter_mut() {
        *lane = lane.wrapping_mul(1664525).wrapping_add(1013904223);
    }

    mix(state);
    for lane in state.iter_mut() {
        *lane ^= *lane >> 16;
    }

    mix(state);
}

fn mix(state: &mut [u32; 3]) {
    state[0] = state[0].wrapping_add(state[1].wrapping_mul(state[2]));
    state[1] = state[1].wrapping_add(state[2].wrapping_mul(state[0]));
    state[2] = state[2].wrapping_add(state[0].wrapping_mul(state[1]));
}

/// A state lane as a fraction, `value / 2^32`, the way libplacebo reads it.
pub fn unit(value: u32) -> f32 {
    value as f32 * UNIT
}

/// `1 / 2^32`, exact in f32.
const UNIT: f32 = 1.0 / 4_294_967_296.0;

/// The third lane of every pixel's starting state for one plane of one frame.
///
/// Only the low 32 bits of `frame_index` count, and every sum wraps.
pub fn salt(frame_index: u64, seed: u32, plane: u32) -> u32 {
    let frame = frame_index as u32;
    let spread = plane.wrapping_mul(0x9E37_79B9);

    frame.wrapping_add(seed).wrapping_add(spread)
}

/// Everything one plane is debanded with.
///
/// `threshold` and `grain` are in libplacebo's units, thousandths of full scale. `neutral` and
/// `peak` are code values.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DebandParams {
    pub element: Element,
    pub iterations: u32,
    /// How far the first iteration reaches, in samples of the plane.
    pub radius: f32,
    pub threshold: f32,
    pub grain: f32,
    /// The level grain fades toward.
    pub neutral: f32,
    pub peak: f32,
}

impl DebandParams {
    /// How many draws each pixel takes, one per iteration plus one for grain.
    pub fn draws(&self) -> u32 {
        let grain_draws = u32::from(self.grain > 0.0);

        self.iterations + grain_draws
    }
}

/// The plane one deband reads and the plane it writes, both the same size.
#[derive(Debug, Clone, Copy)]
pub struct DebandPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube]
fn nearest<E: Int>(
    input: &Array<E>,
    column: u32,
    row: u32,
    offset_x: f32,
    offset_y: f32,
    width: u32,
    height: u32,
) -> f32 {
    let column_position = f32::cast_from(column);
    let row_position = f32::cast_from(row);
    let x = f32::floor(column_position + 0.5 + offset_x);
    let y = f32::floor(row_position + 0.5 + offset_y);

    let last_column = f32::cast_from(width - 1);
    let last_row = f32::cast_from(height - 1);
    let clamped_x_position = f32::clamp(x, 0.0, last_column);
    let clamped_y_position = f32::clamp(y, 0.0, last_row);
    let clamped_x = u32::cast_from(clamped_x_position);
    let clamped_y = u32::cast_from(clamped_y_position);
    let index = clamped_y * width + clamped_x;

    f32::cast_from(input[index as usize])
}

#[cube(launch_unchecked)]
fn deband_plane<E: Int>(
    input: &Array<E>,
    output: &mut Array<E>,
    width: u32,
    height: u32,
    iterations: u32,
    draws: u32,
    radius: f32,
    threshold: f32,
    grain: f32,
    neutral: f32,
    peak: f32,
    salt: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;

    let mut lane_x = column;
    let mut lane_y = row;
    let mut lane_z = salt;
    let mut result = f32::cast_from(input[ABSOLUTE_POS]) / peak;

    for draw in 0..draws {
        lane_x = lane_x * 1664525u32 + 1013904223u32;
        lane_y = lane_y * 1664525u32 + 1013904223u32;
        lane_z = lane_z * 1664525u32 + 1013904223u32;
        lane_x += lane_y * lane_z;
        lane_y += lane_z * lane_x;
        lane_z += lane_x * lane_y;
        lane_x ^= lane_x >> 16u32;
        lane_y ^= lane_y >> 16u32;
        lane_z ^= lane_z >> 16u32;
        lane_x += lane_y * lane_z;
        lane_y += lane_z * lane_x;
        lane_z += lane_x * lane_y;

        let first = f32::cast_from(lane_x) * 2.3283064e-10f32;
        let second = f32::cast_from(lane_y) * 2.3283064e-10f32;

        if draw < iterations {
            let step = f32::cast_from(draw + 1);
            let distance = first * step * radius;
            let angle = second * 6.2831855f32;
            let offset_x = distance * f32::cos(angle);
            let offset_y = distance * f32::sin(angle);

            let total = nearest::<E>(input, column, row, offset_x, offset_y, width, height)
                + nearest::<E>(input, column, row, -offset_x, offset_y, width, height)
                + nearest::<E>(input, column, row, -offset_x, -offset_y, width, height)
                + nearest::<E>(input, column, row, offset_x, -offset_y, width, height);
            let average = total / (4.0 * peak);
            let bound = threshold / step;
            let distance_from_average = f32::abs(result - average);

            if distance_from_average <= bound {
                result = average;
            }
        } else {
            let distance_from_neutral = f32::abs(result - neutral);
            let strength = f32::min(distance_from_neutral, grain);
            result += strength * (first - 0.5);
        }
    }

    let scaled = f32::floor(result * peak + 0.5);
    let clamped = f32::clamp(scaled, 0.0, peak);
    output[ABSOLUTE_POS] = E::cast_from(clamped);
}

/// Debands `planes.input` into `planes.output`, drawing from a generator seeded with `salt`.
pub fn deband<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: DebandPlanes<'_>,
    params: &DebandParams,
    salt: u32,
) {
    match params.element {
        Element::U8 => launch::<u8, R>(client, dispatch, planes, params, salt),
        Element::U16 => launch::<u16, R>(client, dispatch, planes, params, salt),
    }
}

fn launch<E: Int, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: DebandPlanes<'_>,
    params: &DebandParams,
    salt: u32,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    let threshold = params.threshold / 1000.0;
    let grain = params.grain / 1000.0;
    let neutral = params.neutral / params.peak;
    let draws = params.draws();

    // Safety: the input and output hold `width * height` samples of `E`, and every read is clamped
    // inside the plane.
    unsafe {
        deband_plane::launch_unchecked::<E, R>(
            client,
            cubes,
            unit,
            input,
            output,
            width,
            height,
            params.iterations,
            draws,
            params.radius,
            threshold,
            grain,
            neutral,
            params.peak,
            salt,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::Element;
    use crate::testing::reference_deband;

    fn params(iterations: u32, threshold: f32, grain: f32) -> DebandParams {
        DebandParams {
            element: Element::U8,
            iterations,
            radius: 16.0,
            threshold,
            grain,
            neutral: 0.0,
            peak: 255.0,
        }
    }

    #[test]
    fn advancing_matches_libplacebos_pcg3d() {
        // Worked from libplacebo's `sh_prng` formula with wrapping u32 arithmetic.
        let mut zero = [0, 0, 0];
        advance(&mut zero);
        assert_eq!(zero, [0x9bafd7c6, 0xa8e88a6b, 0x3f15482c]);

        advance(&mut zero);
        assert_eq!(zero, [0x5c99e65c, 0xa7a56930, 0x5492c147]);

        let mut counted = [1, 2, 3];
        advance(&mut counted);
        assert_eq!(counted, [0xfa9f79a6, 0x48f2f44c, 0x596f5ab1]);

        let mut salted = [7, 11, 0x9E3779B9];
        advance(&mut salted);
        assert_eq!(salted, [0x3a735c44, 0x6f4c482f, 0xdac0603b]);
    }

    #[test]
    fn unit_stays_below_one() {
        assert_eq!(unit(0), 0.0);
        assert!(unit(u32::MAX) <= 1.0);
        assert_eq!(unit(1 << 31), 0.5);
    }

    #[test]
    fn salt_wraps_rather_than_overflowing() {
        let wrapped = salt(u64::MAX, u32::MAX, 2);
        let expected = u32::MAX
            .wrapping_add(u32::MAX)
            .wrapping_add(0x9E37_79B9u32.wrapping_mul(2));

        assert_eq!(wrapped, expected);
        assert_eq!(salt(5, 1, 0), 6);
    }

    #[test]
    fn a_flat_plane_stays_flat_without_grain() {
        let flat = vec![120u16; 32 * 16];
        let debanded = reference_deband(&flat, 32, 16, &params(4, 3.0, 0.0), 9);

        assert_eq!(debanded, flat);
    }

    #[test]
    fn no_iterations_and_no_grain_changes_nothing() {
        let values: Vec<u16> = (0..32 * 16).map(|index| (index * 37 % 256) as u16).collect();
        let debanded = reference_deband(&values, 32, 16, &params(0, 3.0, 0.0), 9);

        assert_eq!(debanded, values);
    }

    #[test]
    fn grain_changes_a_mid_grey_plane() {
        let flat = vec![128u16; 32 * 16];
        let grained = reference_deband(&flat, 32, 16, &params(1, 3.0, 20.0), 9);

        assert_ne!(grained, flat);
    }
}
