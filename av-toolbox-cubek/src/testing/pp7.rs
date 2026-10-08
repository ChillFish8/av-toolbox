use std::ops::{Add, Sub};

use super::dpir::reference_mirror;
use super::h264::{number, samples_named, split_planes};
use super::qed::QedLayout;
use super::{assert_plane_within, assert_same_plane};
use crate::deblock::h264::peak_of;
use crate::deblock::pp7::{
    FLOAT_INPUT_SCALE,
    FLOAT_OUTPUT_SCALE,
    Pp7Mode,
    WEIGHT_BITS,
    WEIGHTS,
    aligned_frame,
    thresholds,
};
use crate::matrix::MatrixSamples;

/// How far a float sample may sit from the plugin's when every coefficient decision agrees.
///
/// It measured 3e-7 on the fixtures and 6e-7 on 1080p planes, so 1e-6 keeps headroom.
pub const PP7_FLOAT_TOLERANCE: f32 = 1e-6;

/// How far a float sample may sit from the plugin's when a coefficient's decision differs.
///
/// One decision moves a sample by at most `2621 x 2520 / (2^18 x 255)`, about 0.099 at qp 63.
pub const PP7_FLIP_BOUND: f32 = 0.1;

/// One run of DeblockPP7's `pp7.DeblockPP7`.
///
/// `metadata` holds `<width> <height> <layout> <bits> <qp> <mode>`, bits 32 meaning float. `input` and
/// `output` hold every plane in channel order, each at its own size. `allowed` is how many float samples
/// may sit past [PP7_FLOAT_TOLERANCE](crate::testing::PP7_FLOAT_TOLERANCE) from the plugin's.
pub struct Pp7Fixture {
    pub name: &'static str,
    pub metadata: &'static str,
    pub input: &'static [u8],
    pub output: &'static [u8],
    pub allowed: usize,
}

macro_rules! fixture {
    ($name:literal, $allowed:literal) => {
        Pp7Fixture {
            name: $name,
            metadata: include_str!(concat!("fixtures/pp7/", $name, ".txt")),
            input: include_bytes!(concat!("fixtures/pp7/", $name, ".in")),
            output: include_bytes!(concat!("fixtures/pp7/", $name, ".out")),
            allowed: $allowed,
        }
    };
}

/// Every `deblock/pp7` fixture.
pub const PP7_FIXTURES: &[Pp7Fixture] = &[
    fixture!("420_64x48_8_q2_hard_blocky", 0),
    fixture!("420_70x46_8_q2p5_soft_blocky", 0),
    fixture!("422_46x22_8_q4_medium_blocky", 0),
    fixture!("444_54x40_8_q1_soft_blocky", 0),
    fixture!("420_6x4_8_q2_hard_blocky", 0),
    fixture!("grey_47x30_10_q10_hard_binary", 0),
    fixture!("420_70x46_10_q7p3_soft_blocky", 0),
    fixture!("grey_33x17_12_q17p25_soft_noise", 0),
    fixture!("444_64x48_16_q63_hard_noise", 0),
    fixture!("grey_64x48_16_q1_soft_noise", 0),
    fixture!("grey_5x3_16_q3p3_medium_noise", 0),
    fixture!("420_64x48_32_q2_hard_blocky", 2),
    fixture!("grey_70x46_32_q13p37_medium_noise", 2),
    fixture!("444_46x31_32_q63_soft_blocky", 2),
];

/// One pp7 run's frame and arguments.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Pp7Case {
    pub width: u32,
    pub height: u32,
    pub layout: QedLayout,
    pub samples: MatrixSamples,
    pub qp: f64,
    pub mode: Pp7Mode,
}

impl Pp7Fixture {
    /// The case the fixture was written with.
    pub fn case(&self) -> Pp7Case {
        let words: Vec<&str> = self.metadata.split_whitespace().collect();
        let [width, height, layout, bits, qp, mode] = words[..] else {
            panic!("{}: the metadata is not six words", self.name);
        };

        let layout = QedLayout::named(layout, self.name);
        let samples = samples_named(bits, self.name);
        let mode = match mode {
            "hard" => Pp7Mode::Hard,
            "soft" => Pp7Mode::Soft,
            "medium" => Pp7Mode::Medium,
            other => panic!("{}: {other} is not a mode", self.name),
        };

        let width = number(width, self.name);
        let height = number(height, self.name);
        let qp = number(qp, self.name);

        Pp7Case {
            width,
            height,
            layout,
            samples,
            qp,
            mode,
        }
    }

    pub fn input_planes(&self) -> Vec<Vec<f64>> {
        self.every_plane(self.input)
    }

    /// The plugin's output, as it wrote it.
    pub fn output_planes(&self) -> Vec<Vec<f64>> {
        self.every_plane(self.output)
    }

    fn every_plane(&self, bytes: &[u8]) -> Vec<Vec<f64>> {
        let case = self.case();
        let sizes = case.plane_sizes();

        split_planes(bytes, &sizes, case.samples)
    }
}

impl Pp7Case {
    /// The size of each plane in channel order, chroma rounded up.
    pub fn plane_sizes(&self) -> Vec<(u32, u32)> {
        self.layout.plane_shares(self.width, self.height)
    }

    /// The size each plane's sides are padded to before the border folds back a second time, its share of
    /// the frame padded to multiples of 16, in channel order.
    pub fn aligned_sizes(&self) -> Vec<(u32, u32)> {
        let (aligned_width, aligned_height) = aligned_frame(self.width, self.height);

        self.layout.plane_shares(aligned_width, aligned_height)
    }
}

/// One plane, `width` by `height`, and the size its sides are padded to before the border folds back a
/// second time.
#[derive(Debug, Copy, Clone)]
pub struct Pp7Plane<'a> {
    pub values: &'a [f64],
    pub width: u32,
    pub height: u32,
    pub aligned_width: u32,
    pub aligned_height: u32,
}

/// Every plane of `case` deblocked by pp7 on the CPU, from its `inputs` in channel order.
pub fn reference_pp7(case: &Pp7Case, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let sizes = case.plane_sizes();
    let aligned_sizes = case.aligned_sizes();

    inputs
        .iter()
        .enumerate()
        .map(|(index, values)| {
            let (width, height) = sizes[index];
            let (aligned_width, aligned_height) = aligned_sizes[index];
            let plane = Pp7Plane {
                values,
                width,
                height,
                aligned_width,
                aligned_height,
            };

            reference_pp7_plane(plane, case.samples, case.qp, case.mode)
        })
        .collect()
}

/// Where position `position` of a side of `length` samples, padded to `aligned`, reads, on the CPU.
///
/// Positions from `aligned` on fold back about it first. Then every position mirrors into the side with the
/// edge sample repeated.
pub fn reference_extended_index(position: i64, length: u32, aligned: u32) -> u32 {
    let aligned = i64::from(aligned);
    let folded = if position >= aligned {
        2 * aligned - 1 - position
    } else {
        position
    };
    let side = i64::from(length);
    let mirrored = reference_mirror(folded, side);

    mirrored as u32
}

/// The source index of each position from -3 to `length + 2` along a side, through the border.
fn extended_side(length: u32, aligned: u32) -> Vec<u32> {
    let last = i64::from(length) + 3;

    (-3..last)
        .map(|position| reference_extended_index(position, length, aligned))
        .collect()
}

/// Folds seven taps about the fourth and runs the four folded values through the H.264 core transform, as
/// butterflies.
pub fn reference_transform<T>(taps: [T; 7]) -> [T; 4]
where
    T: Copy + Add<Output = T> + Sub<Output = T>,
{
    let centre = taps[3] + taps[3];
    let near = taps[2] + taps[4];
    let middle = taps[1] + taps[5];
    let far = taps[0] + taps[6];
    let outer_sum = centre + far;
    let outer_difference = centre - far;
    let inner_sum = near + middle;
    let inner_difference = near - middle;

    [
        outer_sum + inner_sum,
        (outer_difference + outer_difference) + inner_difference,
        outer_sum - inner_sum,
        outer_difference - (inner_difference + inner_difference),
    ]
}

/// The 16 coefficients of a 7 by 7 `square`, given row by row, by `ky` then `kx`.
///
/// Each column is transformed down first, then each row of the results across.
pub fn reference_coefficients<T>(square: &[[T; 7]; 7]) -> [T; 16]
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T>,
{
    let mut down = [[T::default(); 4]; 7];
    for (column, bands) in down.iter_mut().enumerate() {
        let taps = std::array::from_fn(|row| square[row][column]);
        *bands = reference_transform(taps);
    }

    let mut found = [T::default(); 16];
    for band in 0..4 {
        let taps = std::array::from_fn(|column| down[column][band]);
        let across = reference_transform(taps);
        found[band * 4..band * 4 + 4].copy_from_slice(&across);
    }

    found
}

/// Integer coefficient `value` as `mode` keeps it against `threshold`.
pub fn reference_kept_integer(value: i64, threshold: i64, mode: Pp7Mode) -> i64 {
    let magnitude = value.abs();
    if magnitude <= threshold {
        return 0;
    }

    let medium_identity = magnitude > 2 * threshold;
    match mode {
        Pp7Mode::Hard => value,
        Pp7Mode::Soft => (magnitude - threshold) * value.signum(),
        Pp7Mode::Medium if medium_identity => value,
        Pp7Mode::Medium => 2 * (magnitude - threshold) * value.signum(),
    }
}

/// Float coefficient `value` as `mode` keeps it against `threshold`, in f32.
///
/// The comparisons use the magnitude rounded down, and the kept values the magnitude itself.
pub fn reference_kept_float(value: f32, threshold: f32, mode: Pp7Mode) -> f32 {
    let magnitude = value.abs();
    let decided = magnitude.floor();
    if decided <= threshold {
        return 0.0;
    }

    let excess = magnitude - threshold;
    let medium_identity = decided > threshold + threshold;
    let moved = match mode {
        Pp7Mode::Hard => return value,
        Pp7Mode::Soft => excess,
        Pp7Mode::Medium if medium_identity => return value,
        Pp7Mode::Medium => excess + excess,
    };

    if value < 0.0 { -moved } else { moved }
}

/// An integer output sample from its 16 `found` coefficients, rounded half up and kept between 0 and
/// `peak`.
pub fn reference_integer_output(found: &[i64; 16], limits: &[i32; 16], mode: Pp7Mode, peak: i64) -> i64 {
    let mut total = found[0] * i64::from(WEIGHTS[0][0]);
    for place in 1..16 {
        let weight = i64::from(WEIGHTS[place / 4][place % 4]);
        let threshold = i64::from(limits[place]);
        let kept = reference_kept_integer(found[place], threshold, mode);
        total += kept * weight;
    }

    let rounded = (total + (1 << (WEIGHT_BITS - 1))) >> WEIGHT_BITS;
    rounded.clamp(0, peak)
}

/// A float output sample from its 16 `found` coefficients, in f32, each weighted coefficient rounded before
/// it is added.
pub fn reference_float_output(found: &[f32; 16], limits: &[i32; 16], mode: Pp7Mode) -> f32 {
    let mut total = found[0] * WEIGHTS[0][0] as f32;
    for place in 1..16 {
        let weight = WEIGHTS[place / 4][place % 4] as f32;
        let threshold = limits[place] as f32;
        let kept = reference_kept_float(found[place], threshold, mode);
        total += kept * weight;
    }

    total * FLOAT_OUTPUT_SCALE
}

/// `plane` deblocked by pp7 on the CPU, with `qp` and `mode`.
///
/// Integer samples are exact. Float samples are transformed columns first, with every sum and product rounded
/// to f32.
pub fn reference_pp7_plane(plane: Pp7Plane<'_>, samples: MatrixSamples, qp: f64, mode: Pp7Mode) -> Vec<f64> {
    let rows = extended_side(plane.height, plane.aligned_height);
    let columns = extended_side(plane.width, plane.aligned_width);
    let limits = thresholds(qp, samples);
    let mut output = Vec::with_capacity(plane.values.len());

    for y in 0..plane.height as usize {
        for x in 0..plane.width as usize {
            let at = |row: usize, column: usize| {
                let index = rows[y + row] * plane.width + columns[x + column];

                plane.values[index as usize]
            };

            let value = match samples {
                MatrixSamples::F32 => {
                    let square = std::array::from_fn(|row| {
                        std::array::from_fn(|column| at(row, column) as f32 * FLOAT_INPUT_SCALE)
                    });
                    let found = reference_coefficients::<f32>(&square);
                    let deblocked = reference_float_output(&found, &limits, mode);

                    f64::from(deblocked)
                },
                _ => {
                    let square =
                        std::array::from_fn(|row| std::array::from_fn(|column| at(row, column) as i64));
                    let found = reference_coefficients::<i64>(&square);
                    let peak = peak_of(samples) as i64;

                    reference_integer_output(&found, &limits, mode, peak) as f64
                },
            };

            output.push(value);
        }
    }

    output
}

/// The plugin's `output` with every sample it wrote as 65535 above a 9 to 15-bit peak brought down to the
/// peak, and how many it brought down.
pub fn plugin_output_at_peak(output: &[f64], samples: MatrixSamples) -> (Vec<f64>, usize) {
    let MatrixSamples::U16 { bits } = samples else {
        return (output.to_vec(), 0);
    };

    let peak = f64::from((1u32 << bits) - 1);
    let mut brought = 0;
    let values = output
        .iter()
        .map(|value| {
            if *value == 65535.0 && peak < 65535.0 {
                brought += 1;
                return peak;
            }

            *value
        })
        .collect();

    (values, brought)
}

/// Panics where `found`, a `width` wide plane, is further from the plugin's `expected` than pp7 may be, and
/// gives how many float samples sit past
/// [PP7_FLOAT_TOLERANCE](crate::testing::PP7_FLOAT_TOLERANCE).
///
/// Integer samples must match exactly, and give 0. Float samples may sit up to
/// [PP7_FLIP_BOUND](crate::testing::PP7_FLIP_BOUND) away.
pub fn count_pp7_differences(
    found: &[f64],
    expected: &[f64],
    width: u32,
    samples: MatrixSamples,
    label: &str,
) -> usize {
    if samples != MatrixSamples::F32 {
        assert_same_plane(found, expected, width, label);
        return 0;
    }

    let found_values: Vec<f32> = found.iter().map(|value| *value as f32).collect();
    let expected_values: Vec<f32> = expected.iter().map(|value| *value as f32).collect();
    assert_plane_within(&found_values, &expected_values, PP7_FLIP_BOUND, width, label);

    found_values
        .iter()
        .zip(&expected_values)
        .filter(|(actual, wanted)| (*actual - *wanted).abs() > PP7_FLOAT_TOLERANCE)
        .count()
}
