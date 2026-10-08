use crate::matrix::MatrixSamples;

/// The largest index of the threshold tables.
pub const MAX_INDEX: i32 = 60;

/// The step across an edge below which a line is filtered, by alpha index.
///
/// Entries up to 51 are H.264's `alpha'`, and the rest extend it for stronger deblocking.
pub const ALPHA_TABLE: [u8; 61] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28,
    32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255,
];

/// How flat each side of an edge must be for a line to be filtered, by beta index.
///
/// Entries up to 51 are H.264's `beta'`, and the rest extend it for stronger deblocking.
pub const BETA_TABLE: [u8; 61] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
];

/// How far the samples next to an edge may move, by alpha index.
///
/// Entries up to 51 are H.264's `tC0'` for a boundary strength of 2, and the rest extend it for stronger
/// deblocking.
pub const TC0_TABLE: [u8; 61] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,
];

/// Frames are padded on the right and bottom to a multiple of this before deblocking.
const PAD_MULTIPLE: u32 = 8;

/// What every line of a plane is filtered against, in the plane's own code values.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Thresholds {
    /// A line is filtered only when the step across its edge is below this.
    pub alpha: f32,
    /// A line is filtered only when the step next to its edge on each side is below this.
    pub beta: f32,
    /// The furthest the second sample from the edge may move.
    pub c0: f32,
    /// How much further the samples next to the edge may move for each flat side.
    pub c1: f32,
    /// The largest code value. Float samples are never clamped, and take 1.
    pub peak: f32,
}

/// The thresholds `quant`, `aoffset` and `boffset` give a plane of `samples`.
///
/// The alpha index is `quant + aoffset` and the beta index `quant + boffset`, each kept between 0 and 60.
/// Integer thresholds scale with the depth, and float thresholds are divided by 255.
pub fn thresholds(quant: i32, aoffset: i32, boffset: i32, samples: MatrixSamples) -> Thresholds {
    let alpha_index = table_index(quant, aoffset);
    let beta_index = table_index(quant, boffset);
    let alpha = f32::from(ALPHA_TABLE[alpha_index]);
    let beta = f32::from(BETA_TABLE[beta_index]);
    let c0 = f32::from(TC0_TABLE[alpha_index]);

    let Some(bits) = samples.bits() else {
        return Thresholds {
            alpha: alpha / 255.0,
            beta: beta / 255.0,
            c0: c0 / 255.0,
            c1: 1.0 / 255.0,
            peak: 1.0,
        };
    };

    let scale = (1u32 << (bits - 8)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;

    Thresholds {
        alpha: alpha * scale,
        beta: beta * scale,
        c0: c0 * scale,
        c1: scale,
        peak,
    }
}

fn table_index(quant: i32, offset: i32) -> usize {
    let index = quant.saturating_add(offset);
    let clamped = index.clamp(0, MAX_INDEX);

    clamped as usize
}

/// The size a `width` by `height` frame is padded to before deblocking, each side rounded up to a
/// multiple of 8.
pub fn padded_frame(width: u32, height: u32) -> (u32, u32) {
    let padded_width = width.next_multiple_of(PAD_MULTIPLE);
    let padded_height = height.next_multiple_of(PAD_MULTIPLE);

    (padded_width, padded_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eight_bit(quant: i32, aoffset: i32, boffset: i32) -> Thresholds {
        thresholds(quant, aoffset, boffset, MatrixSamples::U8)
    }

    #[test]
    fn the_tables_hold_h264s_values_at_known_indices() {
        let alphas = [15, 16, 25, 51, 60].map(|index| ALPHA_TABLE[index]);
        let betas = [15, 16, 25, 51, 52, 60].map(|index| BETA_TABLE[index]);
        let limits = [20, 21, 25, 51, 52, 60].map(|index| TC0_TABLE[index]);

        assert_eq!(alphas, [0, 4, 13, 255, 255]);
        assert_eq!(betas, [0, 2, 4, 18, 19, 27]);
        assert_eq!(limits, [0, 1, 1, 17, 19, 35]);
    }

    #[test]
    fn quant_25_at_eight_bits_gives_alpha_13_beta_4_and_c0_1() {
        let found = eight_bit(25, 0, 0);

        let expected = Thresholds {
            alpha: 13.0,
            beta: 4.0,
            c0: 1.0,
            c1: 1.0,
            peak: 255.0,
        };
        assert_eq!(found, expected);
    }

    #[test]
    fn ten_bit_thresholds_scale_by_four() {
        let found = thresholds(25, 0, 0, MatrixSamples::U16 { bits: 10 });

        let expected = Thresholds {
            alpha: 52.0,
            beta: 16.0,
            c0: 4.0,
            c1: 4.0,
            peak: 1023.0,
        };
        assert_eq!(found, expected);
    }

    #[test]
    fn c0_follows_the_alpha_index() {
        let found = eight_bit(30, 10, -10);

        assert_eq!((found.alpha, found.c0, found.beta), (80.0, 5.0, 3.0));
    }

    #[test]
    fn offsets_clamp_the_indices_into_the_tables() {
        let high = eight_bit(40, 30, -100);
        let extreme = eight_bit(60, i32::MAX, i32::MIN);

        assert_eq!((high.alpha, high.c0, high.beta), (255.0, 35.0, 0.0));
        assert_eq!((extreme.alpha, extreme.c0, extreme.beta), (255.0, 35.0, 0.0));
    }

    #[test]
    fn float_thresholds_are_divided_by_255() {
        let found = thresholds(51, 0, 0, MatrixSamples::F32);

        let expected = Thresholds {
            alpha: 1.0,
            beta: 18.0 / 255.0,
            c0: 17.0 / 255.0,
            c1: 1.0 / 255.0,
            peak: 1.0,
        };
        assert_eq!(found, expected);
    }

    #[test]
    fn padded_frames_round_each_side_up_to_a_multiple_of_eight() {
        assert_eq!(padded_frame(64, 48), (64, 48));
        assert_eq!(padded_frame(70, 46), (72, 48));
        assert_eq!(padded_frame(1920, 1080), (1920, 1080));
        assert_eq!(padded_frame(1, 1), (8, 8));
    }
}
