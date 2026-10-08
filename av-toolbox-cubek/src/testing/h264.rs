use super::dpir::reference_mirror;
use super::{assert_same_plane, decode_samples, encode_samples, largest_difference};
use crate::deblock::h264::{Thresholds, padded_frame, thresholds};
use crate::matrix::MatrixSamples;

/// How far a float sample may sit from another run of the filter, since float maths may round differently.
///
/// It measured 0 against the plugin, so 1e-6 keeps headroom.
pub const H264_FLOAT_TOLERANCE: f32 = 1e-6;

/// One run of VapourSynth-Deblock's `deblock.Deblock` on three blocky planes.
///
/// `metadata` holds `<width> <height> <subsampling> <bits> <quant> <aoffset> <boffset>`, bits 32 meaning
/// float. `input` and `output` hold the three planes in channel order, each at its own size.
pub struct H264Fixture {
    pub name: &'static str,
    pub metadata: &'static str,
    pub input: &'static [u8],
    pub output: &'static [u8],
}

macro_rules! fixture {
    ($name:literal) => {
        H264Fixture {
            name: $name,
            metadata: include_str!(concat!("fixtures/h264/", $name, ".txt")),
            input: include_bytes!(concat!("fixtures/h264/", $name, ".in")),
            output: include_bytes!(concat!("fixtures/h264/", $name, ".out")),
        }
    };
}

/// Every `deblock/h264` fixture.
pub const H264_FIXTURES: &[H264Fixture] = &[
    fixture!("420_64x48_8_q25_a0_b0"),
    fixture!("420_64x48_8_q51_a0_b0"),
    fixture!("420_64x48_8_q60_a0_b0"),
    fixture!("420_64x48_8_q12_a0_b0"),
    fixture!("420_64x48_10_q25_a0_b0"),
    fixture!("420_64x48_16_q51_a0_b0"),
    fixture!("420_64x48_32_q51_a0_b0"),
    fixture!("420_70x46_8_q51_a0_b0"),
    fixture!("420_70x46_16_q30_a50_bm5"),
    fixture!("422_70x45_10_q40_a5_bm3"),
    fixture!("422_64x48_8_q25_a10_b20"),
    fixture!("444_70x45_16_q40_a30_b30"),
    fixture!("444_64x48_8_q30_am20_b0"),
    fixture!("444_70x45_32_q60_a0_b0"),
    fixture!("444_64x45_8_q30_a0_b0"),
];

/// What one fixture was written with.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct H264Case {
    pub width: u32,
    pub height: u32,
    /// How far each chroma side is shifted down from luma, across and then down.
    pub chroma_shift: (u32, u32),
    pub samples: MatrixSamples,
    pub quant: i32,
    pub aoffset: i32,
    pub boffset: i32,
}

pub(super) fn samples_named(word: &str, name: &str) -> MatrixSamples {
    match word {
        "8" => MatrixSamples::U8,
        "32" => MatrixSamples::F32,
        deeper => {
            let bits = number(deeper, name);

            MatrixSamples::U16 { bits }
        },
    }
}

pub(super) fn number<T: std::str::FromStr>(word: &str, name: &str) -> T {
    let parsed = word.parse();

    match parsed {
        Ok(value) => value,
        Err(_) => panic!("{name}: {word} is not a number"),
    }
}

impl H264Fixture {
    /// The case the fixture was written with.
    pub fn case(&self) -> H264Case {
        let words: Vec<&str> = self.metadata.split_whitespace().collect();
        let [width, height, subsampling, bits, quant, aoffset, boffset] = words[..] else {
            panic!("{}: the metadata is not seven words", self.name);
        };

        let chroma_shift = match subsampling {
            "420" => (1, 1),
            "422" => (1, 0),
            "444" => (0, 0),
            other => panic!("{}: {other} is not a subsampling", self.name),
        };
        let samples = samples_named(bits, self.name);

        H264Case {
            width: number(width, self.name),
            height: number(height, self.name),
            chroma_shift,
            samples,
            quant: number(quant, self.name),
            aoffset: number(aoffset, self.name),
            boffset: number(boffset, self.name),
        }
    }

    pub fn input_planes(&self) -> Vec<Vec<f64>> {
        let case = self.case();
        let sizes = case.plane_sizes();

        split_planes(self.input, &sizes, case.samples)
    }

    pub fn output_planes(&self) -> Vec<Vec<f64>> {
        let case = self.case();
        let sizes = case.plane_sizes();

        split_planes(self.output, &sizes, case.samples)
    }
}

impl H264Case {
    /// The size of each plane in channel order, with chroma rounded down as in the fixture clips.
    pub fn plane_sizes(&self) -> Vec<(u32, u32)> {
        let (shift_x, shift_y) = self.chroma_shift;
        let chroma = (self.width >> shift_x, self.height >> shift_y);

        vec![(self.width, self.height), chroma, chroma]
    }

    /// The size each plane is padded to, its share of the padded frame, in channel order.
    pub fn padded_sizes(&self) -> Vec<(u32, u32)> {
        let (padded_width, padded_height) = padded_frame(self.width, self.height);
        let (shift_x, shift_y) = self.chroma_shift;
        let chroma = (padded_width >> shift_x, padded_height >> shift_y);

        vec![(padded_width, padded_height), chroma, chroma]
    }

    pub fn thresholds(&self) -> Thresholds {
        thresholds(self.quant, self.aoffset, self.boffset, self.samples)
    }

    /// `values` as plane `index` of this case, with its size and padded size.
    pub fn plane<'a>(&self, index: usize, values: &'a [f64]) -> H264Plane<'a> {
        let (width, height) = self.plane_sizes()[index];
        let (padded_width, padded_height) = self.padded_sizes()[index];

        H264Plane {
            values,
            width,
            height,
            padded_width,
            padded_height,
        }
    }
}

/// Reads `bytes` as samples of `samples` and splits them into planes of `sizes`, in order.
pub fn split_planes(bytes: &[u8], sizes: &[(u32, u32)], samples: MatrixSamples) -> Vec<Vec<f64>> {
    let values = decode_samples(bytes, samples);
    let mut planes = Vec::with_capacity(sizes.len());
    let mut start = 0;

    for (width, height) in sizes {
        let end = start + *width as usize * *height as usize;
        planes.push(values[start..end].to_vec());
        start = end;
    }

    assert_eq!(start, values.len(), "the planes do not fill the bytes");

    planes
}

/// Mixes `first`, `second` and `seed` into a value between 0 and 1, the same every time.
fn hashed_unit(first: u32, second: u32, seed: u32) -> f64 {
    let first_mixed = first.wrapping_mul(0x9E37_79B1);
    let second_mixed = second.wrapping_mul(0x85EB_CA77);
    let seed_mixed = seed.wrapping_mul(0xC2B2_AE3D);
    let mixed = first_mixed ^ second_mixed ^ seed_mixed;
    let spread = mixed ^ (mixed >> 15);
    let scrambled = spread.wrapping_mul(0x2C1B_3C6D);

    f64::from(scrambled >> 8) / f64::from(1u32 << 24)
}

/// Keeps the noise of [blocky_plane](crate::testing::blocky_plane) apart from its blocks.
const NOISE_SALT: u32 = 0x5BD1_E995;

/// A plane of flat 8 by 8 blocks over a top to bottom gradient, with a little noise, as code values of
/// `samples`.
///
/// Integer samples are whole code values, and float samples run from 0 to 1. Each `seed` gives other blocks.
pub fn blocky_plane(width: u32, height: u32, samples: MatrixSamples, seed: u32) -> Vec<f64> {
    let peak = match samples {
        MatrixSamples::U8 => 255.0,
        MatrixSamples::U16 { bits } => f64::from((1u32 << bits) - 1),
        MatrixSamples::F32 => 1.0,
    };
    let whole = samples != MatrixSamples::F32;
    let mut values = Vec::with_capacity(width as usize * height as usize);

    for row in 0..height {
        let gradient = 0.05 + 0.6 * f64::from(row) / f64::from(height);

        for column in 0..width {
            let block = hashed_unit(column / 8, row / 8, seed);
            let noise = hashed_unit(column, row, seed ^ NOISE_SALT) - 0.5;
            let unit = (gradient + 0.15 * block + 0.01 * noise).clamp(0.0, 1.0);
            let value = unit * peak;
            let finished = if whole { value.round() } else { value };

            values.push(finished);
        }
    }

    values
}

/// A `width` by `height` plane of noise over the whole range of `samples`, as the device stores it.
///
/// Integer samples run from 0 to the peak, and float samples from -0.1 to 1.1. Each `seed` gives other noise.
pub fn noise_plane(width: u32, height: u32, samples: MatrixSamples, seed: u32) -> Vec<f64> {
    let mut values = Vec::with_capacity(width as usize * height as usize);

    for row in 0..height {
        for column in 0..width {
            let unit = hashed_unit(column, row, seed);
            let value = match samples {
                MatrixSamples::U8 => (unit * 255.0).round(),
                MatrixSamples::U16 { bits } => (unit * f64::from((1u32 << bits) - 1)).round(),
                MatrixSamples::F32 => unit * 1.2 - 0.1,
            };

            values.push(value);
        }
    }

    let stored = encode_samples(&values, samples);

    decode_samples(&stored, samples)
}

/// One plane, `width` by `height`, and the size it is padded to before deblocking.
#[derive(Debug, Copy, Clone)]
pub struct H264Plane<'a> {
    pub values: &'a [f64],
    pub width: u32,
    pub height: u32,
    pub padded_width: u32,
    pub padded_height: u32,
}

/// Deblocks `plane` on the CPU, padding it with its mirror, filtering it in raster order and cropping it
/// back.
pub fn reference_h264(plane: H264Plane<'_>, samples: MatrixSamples, thresholds: &Thresholds) -> Vec<f64> {
    let mut padded = mirror_padded(plane);

    reference_h264_padded(
        &mut padded,
        plane.padded_width,
        plane.padded_height,
        samples,
        thresholds,
    );

    cropped(&padded, plane)
}

/// `plane` grown to its padded size, each new sample mirrored back with the edge sample repeated.
fn mirror_padded(plane: H264Plane<'_>) -> Vec<f64> {
    let width = i64::from(plane.width);
    let height = i64::from(plane.height);
    let padded_samples = plane.padded_width as usize * plane.padded_height as usize;
    let mut padded = Vec::with_capacity(padded_samples);

    for row in 0..i64::from(plane.padded_height) {
        let source_row = reference_mirror(row, height);

        for column in 0..i64::from(plane.padded_width) {
            let source_column = reference_mirror(column, width);
            let index = source_row * width + source_column;

            padded.push(plane.values[index as usize]);
        }
    }

    padded
}

fn cropped(padded: &[f64], plane: H264Plane<'_>) -> Vec<f64> {
    let width = plane.width as usize;
    let rows = padded
        .chunks(plane.padded_width as usize)
        .take(plane.height as usize);

    rows.flat_map(|row| row[..width].iter().copied()).collect()
}

/// Deblocks a plane whose sides are multiples of 4, in place.
///
/// Blocks of 4 by 4 run in raster order, each filtering its top edge and then its left edge, so every edge
/// reads what the edges before it wrote.
pub fn reference_h264_padded(
    values: &mut [f64],
    width: u32,
    height: u32,
    samples: MatrixSamples,
    thresholds: &Thresholds,
) {
    assert!(
        width.is_multiple_of(4) && height.is_multiple_of(4),
        "the sides must be multiples of 4"
    );
    assert_eq!(
        values.len(),
        width as usize * height as usize,
        "the values must fill the plane"
    );

    let across = (width / 4) as usize;
    let down = (height / 4) as usize;
    let stride = width as usize;

    for row in 0..down {
        for column in 0..across {
            if row > 0 {
                for lane in 0..4 {
                    let first_after = row * 4 * stride + column * 4 + lane;
                    reference_line(values, first_after, stride, samples, thresholds);
                }
            }

            if column > 0 {
                for lane in 0..4 {
                    let first_after = (row * 4 + lane) * stride + column * 4;
                    reference_line(values, first_after, 1, samples, thresholds);
                }
            }
        }
    }
}

/// Filters the line whose first sample past the edge is at `first_after`, its taps `stride` apart.
fn reference_line(
    values: &mut [f64],
    first_after: usize,
    stride: usize,
    samples: MatrixSamples,
    thresholds: &Thresholds,
) {
    let indices = [
        first_after - stride,
        first_after - 2 * stride,
        first_after - 3 * stride,
        first_after,
        first_after + stride,
        first_after + 2 * stride,
    ];
    let taps = indices.map(|index| values[index]);

    let filtered = match samples {
        MatrixSamples::F32 => float_line(taps, thresholds),
        _ => integer_line(taps, thresholds),
    };
    let Some(written) = filtered else {
        return;
    };

    let targets = [indices[1], indices[0], indices[3], indices[4]];
    for (index, value) in targets.into_iter().zip(written) {
        values[index] = value;
    }
}

/// The new `p1, p0, q0, q1` of a line of integer taps `p0, p1, p2, q0, q1, q2`, or `None` when it is not
/// filtered.
fn integer_line(taps: [f64; 6], thresholds: &Thresholds) -> Option<[f64; 4]> {
    let [p0, p1, p2, q0, q1, q2] = taps.map(|value| value as i32);
    let alpha = thresholds.alpha as i32;
    let beta = thresholds.beta as i32;
    let c0 = thresholds.c0 as i32;
    let c1 = thresholds.c1 as i32;
    let peak = thresholds.peak as i32;

    let passes = (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta;
    if !passes {
        return None;
    }

    let p_flat = (p2 - p0).abs() < beta;
    let q_flat = (q2 - q0).abs() < beta;
    let mut limit = c0;
    if p_flat {
        limit += c1;
    }

    if q_flat {
        limit += c1;
    }

    let average = (p0 + q0 + 1) >> 1;
    let raw_delta = (((q0 - p0) << 2) + p1 - q1 + 4) >> 3;
    let delta = raw_delta.clamp(-limit, limit);
    let new_p0 = (p0 + delta).clamp(0, peak);
    let new_q0 = (q0 - delta).clamp(0, peak);

    let mut new_p1 = p1;
    if p_flat {
        let raw = (p2 + average - (p1 << 1)) >> 1;
        let moved = raw.clamp(-c0, c0);
        new_p1 = (p1 + moved).clamp(0, peak);
    }

    let mut new_q1 = q1;
    if q_flat {
        let raw = (q2 + average - (q1 << 1)) >> 1;
        let moved = raw.clamp(-c0, c0);
        new_q1 = (q1 + moved).clamp(0, peak);
    }

    let written = [new_p1, new_p0, new_q0, new_q1];

    Some(written.map(f64::from))
}

/// The new `p1, p0, q0, q1` of a line of float taps `p0, p1, p2, q0, q1, q2`, or `None` when it is not
/// filtered.
fn float_line(taps: [f64; 6], thresholds: &Thresholds) -> Option<[f64; 4]> {
    let [p0, p1, p2, q0, q1, q2] = taps.map(|value| value as f32);
    let alpha = thresholds.alpha;
    let beta = thresholds.beta;
    let c0 = thresholds.c0;
    let c1 = thresholds.c1;

    let passes = (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta;
    if !passes {
        return None;
    }

    let p_flat = (p2 - p0).abs() < beta;
    let q_flat = (q2 - q0).abs() < beta;
    let mut limit = c0;
    if p_flat {
        limit += c1;
    }

    if q_flat {
        limit += c1;
    }

    let average = (p0 + q0) * 0.5;
    let raw_delta = ((q0 - p0) * 4.0 + p1 - q1) * 0.125;
    let delta = raw_delta.clamp(-limit, limit);

    let mut new_p1 = p1;
    if p_flat {
        let raw = (p2 + average - p1 * 2.0) * 0.5;
        let moved = raw.clamp(-c0, c0);
        new_p1 = p1 + moved;
    }

    let mut new_q1 = q1;
    if q_flat {
        let raw = (q2 + average - q1 * 2.0) * 0.5;
        let moved = raw.clamp(-c0, c0);
        new_q1 = q1 + moved;
    }

    let written = [new_p1, p0 + delta, q0 - delta, new_q1];

    Some(written.map(f64::from))
}

/// Panics where `found` differs from `expected`, a `width` wide plane, giving the largest float difference.
///
/// Integer samples must match exactly, and give 0. Float samples may differ by up to
/// [H264_FLOAT_TOLERANCE](crate::testing::H264_FLOAT_TOLERANCE).
pub fn assert_h264_plane(
    found: &[f64],
    expected: &[f64],
    width: u32,
    samples: MatrixSamples,
    label: &str,
) -> f32 {
    if samples != MatrixSamples::F32 {
        assert_same_plane(found, expected, width, label);
        return 0.0;
    }

    let found_values: Vec<f32> = found.iter().map(|value| *value as f32).collect();
    let expected_values: Vec<f32> = expected.iter().map(|value| *value as f32).collect();

    largest_difference(
        &found_values,
        &expected_values,
        H264_FLOAT_TOLERANCE,
        width,
        label,
    )
}
