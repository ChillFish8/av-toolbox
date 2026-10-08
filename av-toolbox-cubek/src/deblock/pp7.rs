use crate::matrix::MatrixSamples;

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
