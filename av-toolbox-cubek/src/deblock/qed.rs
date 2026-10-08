use super::h264::peak_of;
use crate::matrix::MatrixSamples;

/// The f32 nearest 1.01, which scales how far the strong pass moved each border sample.
pub const CHANGE_SCALE: f32 = 1.01;

/// The weights `cos((2k + 1) π / 16)` for `k` from 0 to 7, as f32.
///
/// With a constant, they pick out the four lowest DCT frequencies of an 8 by 8 block, which its low-pass
/// keeps. Weight `7 - k` is weight `k` negated.
pub const COSINES: [f32; 8] = [
    0.98078525,
    0.8314696,
    0.55557024,
    0.19509032,
    -0.19509032,
    -0.55557024,
    -0.8314696,
    -0.98078525,
];

/// Frames are extended on the right and bottom to a multiple of this before the low-pass.
const LOWPASS_MULTIPLE: u32 = 16;

/// The size a `width` by `height` frame is extended to before the low-pass, each side rounded up to a
/// multiple of 16.
pub fn lowpass_frame(width: u32, height: u32) -> (u32, u32) {
    let lowpass_width = width.next_multiple_of(LOWPASS_MULTIPLE);
    let lowpass_height = height.next_multiple_of(LOWPASS_MULTIPLE);

    (lowpass_width, lowpass_height)
}

/// The change map's value where the strong pass moved nothing, on every plane.
///
/// It is half the code range for integer samples and 0 for float samples.
pub fn neutral_of(samples: MatrixSamples) -> f32 {
    match samples.bits() {
        Some(bits) => (1u32 << (bits - 1)) as f32,
        None => 0.0,
    }
}

/// The change map's value on a border sample whose source sits `difference` above the strong pass.
///
/// Integer samples take `difference * 1.01 + neutral` as one fused multiply-add, rounded half to even and
/// kept between 0 and the peak. Float samples take `difference * 1.01`, neither rounded nor clamped.
pub fn change_value(difference: f32, samples: MatrixSamples) -> f32 {
    if samples == MatrixSamples::F32 {
        return difference * CHANGE_SCALE;
    }

    let neutral = neutral_of(samples);
    let peak = peak_of(samples);
    let fused = difference.mul_add(CHANGE_SCALE, neutral);

    stored_change(fused, peak)
}

/// `fused` rounded half to even, then kept between 0 and `peak`.
fn stored_change(fused: f32, peak: f32) -> f32 {
    let rounded = fused.round_ties_even();

    rounded.clamp(0.0, peak)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cosines_are_the_dct_weights_and_mirror_each_other() {
        for (index, cosine) in COSINES.iter().enumerate() {
            let angle = (2 * index + 1) as f64 * std::f64::consts::PI / 16.0;
            let exact = angle.cos() as f32;
            let mirrored = COSINES[7 - index];

            assert_eq!(*cosine, exact, "weight {index}");
            assert_eq!(mirrored, -*cosine, "weight {index} against its mirror");
        }
    }

    #[test]
    fn the_change_map_rounds_a_half_to_even() {
        // 50 * 1.01 + 128 rounds once to exactly 178.5 in f32, and -50 * 1.01 + 128 to 77.5.
        assert_eq!(change_value(50.0, MatrixSamples::U8), 178.0);
        assert_eq!(change_value(-50.0, MatrixSamples::U8), 78.0);
    }

    #[test]
    fn the_change_maps_multiply_add_rounds_once() {
        let samples = MatrixSamples::U16 { bits: 16 };

        let found = change_value(-32350.0, samples);

        // Rounded after the multiply as well, the sum would be exactly 94.5 and give 94.
        assert_eq!(found, 95.0);
    }

    #[test]
    fn the_change_map_stays_between_0_and_the_peak() {
        assert_eq!(change_value(-255.0, MatrixSamples::U8), 0.0);
        assert_eq!(change_value(255.0, MatrixSamples::U8), 255.0);
        assert_eq!(change_value(0.0, MatrixSamples::U16 { bits: 10 }), 512.0);
    }

    #[test]
    fn float_changes_are_scaled_without_rounding() {
        let found = change_value(0.25, MatrixSamples::F32);
        let below = change_value(-0.125, MatrixSamples::F32);

        assert_eq!(found, 0.25 * CHANGE_SCALE);
        assert_eq!(below, -0.125 * CHANGE_SCALE);
    }

    #[test]
    fn neutral_is_half_the_code_range_or_0_for_float() {
        let neutrals = [
            neutral_of(MatrixSamples::U8),
            neutral_of(MatrixSamples::U16 { bits: 10 }),
            neutral_of(MatrixSamples::U16 { bits: 16 }),
            neutral_of(MatrixSamples::F32),
        ];

        assert_eq!(neutrals, [128.0, 512.0, 32768.0, 0.0]);
    }

    #[test]
    fn lowpass_frames_round_each_side_up_to_16() {
        assert_eq!(lowpass_frame(64, 48), (64, 48));
        assert_eq!(lowpass_frame(70, 46), (80, 48));
        assert_eq!(lowpass_frame(1920, 1080), (1920, 1088));
        assert_eq!(lowpass_frame(6, 4), (16, 16));
    }
}
