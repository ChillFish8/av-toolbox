use crate::depth::Element;

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
