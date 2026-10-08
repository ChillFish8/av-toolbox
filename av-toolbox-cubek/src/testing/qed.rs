use super::assert_plane_within;
use super::h264::{H264Plane, number, reference_h264, samples_named, split_planes};
use crate::deblock::h264::{Thresholds, padded_frame, peak_of, thresholds};
use crate::deblock::qed::{COSINES, change_value, lowpass_frame, neutral_of};
use crate::matrix::MatrixSamples;

/// How far a float sample may sit from the plugin's, since its low-pass rounds in another order.
///
/// It measured 6e-8 against the plugin, so 1e-6 keeps headroom.
pub const QED_FLOAT_TOLERANCE: f32 = 1e-6;

/// One run of vsdenoise's `deblock_qed`, with both of its `deblock.Deblock` passes and its change map.
///
/// `metadata` holds `<width> <height> <layout> <bits> <quant> <quant> <aoffset> <aoffset> <boffset> <boffset>
/// <chroma>`, each pair normal then strong, bits 32 meaning float. `input`, `normal`, `strong` and `output`
/// hold every plane in channel order, and `changes` the change map of each plane that takes the QED output.
/// `allowed` is how many integer output samples may sit 1 from the plugin's.
pub struct QedFixture {
    pub name: &'static str,
    pub metadata: &'static str,
    pub input: &'static [u8],
    pub normal: &'static [u8],
    pub strong: &'static [u8],
    pub changes: &'static [u8],
    pub output: &'static [u8],
    pub allowed: usize,
}

macro_rules! fixture {
    ($name:literal, $allowed:literal) => {
        QedFixture {
            name: $name,
            metadata: include_str!(concat!("fixtures/qed/", $name, ".txt")),
            input: include_bytes!(concat!("fixtures/qed/", $name, ".in")),
            normal: include_bytes!(concat!("fixtures/qed/", $name, ".normal")),
            strong: include_bytes!(concat!("fixtures/qed/", $name, ".strong")),
            changes: include_bytes!(concat!("fixtures/qed/", $name, ".change")),
            output: include_bytes!(concat!("fixtures/qed/", $name, ".out")),
            allowed: $allowed,
        }
    };
}

/// Every `deblock/qed` fixture.
pub const QED_FIXTURES: &[QedFixture] = &[
    fixture!("420_64x48_8_q24-26_a1-1_b2-2_qed", 2),
    fixture!("420_70x46_8_q24-26_a1-1_b2-2_qed", 2),
    fixture!("420_70x46_8_q24-26_a1-1_b2-2_normal", 2),
    fixture!("420_70x46_8_q24-26_a1-1_b2-2_strong", 2),
    fixture!("420_64x48_8_q24-12_a1-1_b2-2_qed", 2),
    fixture!("422_70x46_8_q24-26_a1-1_b2-2_qed", 2),
    fixture!("444_64x48_8_q30-40_a3-m2_bm1-4_qed", 2),
    fixture!("grey_70x46_10_q24-26_a1-1_b2-2_qed", 2),
    fixture!("grey_64x48_16_q24-26_a1-1_b2-2_qed", 8),
    fixture!("420_70x46_16_q30-40_a3-m2_bm1-4_qed", 8),
    fixture!("grey_70x46_32_q24-26_a1-1_b2-2_qed", 0),
    fixture!("420_6x4_8_q24-26_a1-1_b2-2_qed", 2),
    fixture!("420_6x4_32_q24-26_a1-1_b2-2_qed", 0),
    fixture!("grey_16x3_32_q24-26_a1-1_b2-2_qed", 0),
    fixture!("420_16x4_32_q24-26_a1-1_b2-2_qed", 0),
];

/// How a deblocking fixture's planes are laid out.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum QedLayout {
    Grey,
    /// Luma and two chroma planes, each chroma side shifted down from luma's by `chroma_shift`, across then
    /// down.
    Yuv {
        chroma_shift: (u32, u32),
    },
    Rgb,
}

impl QedLayout {
    pub(super) fn named(word: &str, name: &str) -> Self {
        match word {
            "420" => Self::Yuv { chroma_shift: (1, 1) },
            "422" => Self::Yuv { chroma_shift: (1, 0) },
            "444" => Self::Yuv { chroma_shift: (0, 0) },
            "grey" => Self::Grey,
            other => panic!("{name}: {other} is not a layout"),
        }
    }

    /// Each plane's share of a `width` by `height` frame, in channel order, chroma rounded up.
    pub fn plane_shares(self, width: u32, height: u32) -> Vec<(u32, u32)> {
        match self {
            Self::Grey => vec![(width, height)],
            Self::Rgb => vec![(width, height); 3],
            Self::Yuv {
                chroma_shift: (shift_x, shift_y),
            } => {
                let chroma = (width.div_ceil(1 << shift_x), height.div_ceil(1 << shift_y));

                vec![(width, height), chroma, chroma]
            },
        }
    }
}

/// What the chroma planes of a YUV frame take.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum QedChroma {
    Qed,
    Normal,
    Strong,
}

/// One QED run's frame and arguments, each pair normal then strong.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct QedCase {
    pub width: u32,
    pub height: u32,
    pub layout: QedLayout,
    pub samples: MatrixSamples,
    pub quant: [i32; 2],
    pub aoffset: [i32; 2],
    pub boffset: [i32; 2],
    pub chroma: QedChroma,
}

impl QedFixture {
    /// The case the fixture was written with.
    pub fn case(&self) -> QedCase {
        let words: Vec<&str> = self.metadata.split_whitespace().collect();
        let [
            width,
            height,
            layout,
            bits,
            quant_normal,
            quant_strong,
            aoffset_normal,
            aoffset_strong,
            boffset_normal,
            boffset_strong,
            chroma,
        ] = words[..]
        else {
            panic!("{}: the metadata is not eleven words", self.name);
        };

        let layout = QedLayout::named(layout, self.name);
        let samples = samples_named(bits, self.name);
        let chroma = match chroma {
            "qed" => QedChroma::Qed,
            "normal" => QedChroma::Normal,
            "strong" => QedChroma::Strong,
            other => panic!("{}: {other} is not a chroma", self.name),
        };

        QedCase {
            width: number(width, self.name),
            height: number(height, self.name),
            layout,
            samples,
            quant: [number(quant_normal, self.name), number(quant_strong, self.name)],
            aoffset: [
                number(aoffset_normal, self.name),
                number(aoffset_strong, self.name),
            ],
            boffset: [
                number(boffset_normal, self.name),
                number(boffset_strong, self.name),
            ],
            chroma,
        }
    }

    pub fn input_planes(&self) -> Vec<Vec<f64>> {
        self.every_plane(self.input)
    }

    pub fn normal_planes(&self) -> Vec<Vec<f64>> {
        self.every_plane(self.normal)
    }

    pub fn strong_planes(&self) -> Vec<Vec<f64>> {
        self.every_plane(self.strong)
    }

    pub fn output_planes(&self) -> Vec<Vec<f64>> {
        self.every_plane(self.output)
    }

    /// The change map of each plane that takes the QED output, in channel order.
    pub fn change_planes(&self) -> Vec<Vec<f64>> {
        let case = self.case();
        let sizes = case.plane_sizes();
        let taking: Vec<(u32, u32)> = case.qed_planes().iter().map(|index| sizes[*index]).collect();

        split_planes(self.changes, &taking, case.samples)
    }

    fn every_plane(&self, bytes: &[u8]) -> Vec<Vec<f64>> {
        let case = self.case();
        let sizes = case.plane_sizes();

        split_planes(bytes, &sizes, case.samples)
    }
}

impl QedCase {
    /// The size of each plane in channel order, chroma rounded up.
    pub fn plane_sizes(&self) -> Vec<(u32, u32)> {
        self.shares(self.width, self.height)
    }

    /// The size each plane is padded to for a deblocking pass, its share of the padded frame, in channel
    /// order.
    pub fn deblock_sizes(&self) -> Vec<(u32, u32)> {
        let (padded_width, padded_height) = padded_frame(self.width, self.height);

        self.shares(padded_width, padded_height)
    }

    /// The size each plane is extended to before the low-pass, its share of the frame rounded up to 16.
    pub fn lowpass_sizes(&self) -> Vec<(u32, u32)> {
        let (lowpass_width, lowpass_height) = lowpass_frame(self.width, self.height);

        self.shares(lowpass_width, lowpass_height)
    }

    pub fn normal_thresholds(&self) -> Thresholds {
        thresholds(self.quant[0], self.aoffset[0], self.boffset[0], self.samples)
    }

    pub fn strong_thresholds(&self) -> Thresholds {
        thresholds(self.quant[1], self.aoffset[1], self.boffset[1], self.samples)
    }

    /// The planes, in channel order, that take the QED output rather than one pass.
    pub fn qed_planes(&self) -> Vec<usize> {
        let count = self.plane_sizes().len();

        (0..count).filter(|index| self.takes_qed(*index)).collect()
    }

    fn takes_qed(&self, index: usize) -> bool {
        let is_chroma = matches!(self.layout, QedLayout::Yuv { .. }) && index > 0;

        !is_chroma || self.chroma == QedChroma::Qed
    }

    /// Each plane's share of a `width` by `height` frame, in channel order, chroma rounded up.
    fn shares(&self, width: u32, height: u32) -> Vec<(u32, u32)> {
        self.layout.plane_shares(width, height)
    }
}

/// One plane that takes the QED output, with both passes, its size and the size its low-pass extends it to.
#[derive(Debug, Copy, Clone)]
pub struct QedPlane<'a> {
    pub source: &'a [f64],
    pub normal: &'a [f64],
    pub strong: &'a [f64],
    pub width: u32,
    pub height: u32,
    pub lowpass_width: u32,
    pub lowpass_height: u32,
}

/// Every plane of a QED run on the CPU, in channel order, with the change map of each plane that takes the
/// QED output.
pub struct QedReference {
    pub normal: Vec<Vec<f64>>,
    pub strong: Vec<Vec<f64>>,
    pub changes: Vec<Vec<f64>>,
    pub output: Vec<Vec<f64>>,
}

/// Every plane of `case` deblocked by QED on the CPU, from its `inputs` in channel order.
///
/// Both passes run on every plane, as `deblock.Deblock` runs them. Each plane then takes the QED output, or
/// the chroma takes the pass `case.chroma` names.
pub fn reference_qed(case: &QedCase, inputs: &[Vec<f64>]) -> QedReference {
    let sizes = case.plane_sizes();
    let deblock_sizes = case.deblock_sizes();
    let lowpass_sizes = case.lowpass_sizes();
    let normal_thresholds = case.normal_thresholds();
    let strong_thresholds = case.strong_thresholds();
    let mut reference = QedReference {
        normal: Vec::new(),
        strong: Vec::new(),
        changes: Vec::new(),
        output: Vec::new(),
    };

    for (index, source) in inputs.iter().enumerate() {
        let (width, height) = sizes[index];
        let (padded_width, padded_height) = deblock_sizes[index];
        let (lowpass_width, lowpass_height) = lowpass_sizes[index];
        let deblocked = H264Plane {
            values: source,
            width,
            height,
            padded_width,
            padded_height,
        };
        let normal = reference_h264(deblocked, case.samples, &normal_thresholds);
        let strong = reference_h264(deblocked, case.samples, &strong_thresholds);

        let output = if case.takes_qed(index) {
            let change = reference_change(source, &strong, width, case.samples);
            let plane = QedPlane {
                source,
                normal: &normal,
                strong: &strong,
                width,
                height,
                lowpass_width,
                lowpass_height,
            };

            let output = qed_output(plane, &change, case.samples);

            reference.changes.push(change);
            output
        } else if case.chroma == QedChroma::Normal {
            normal.clone()
        } else {
            strong.clone()
        };

        reference.normal.push(normal);
        reference.strong.push(strong);
        reference.output.push(output);
    }

    reference
}

/// `plane`'s QED output on the CPU.
///
/// Border samples keep the normal pass. Interior samples move by the low-pass of the change map toward the
/// strong pass.
pub fn reference_qed_plane(plane: QedPlane<'_>, samples: MatrixSamples) -> Vec<f64> {
    let change = reference_change(plane.source, plane.strong, plane.width, samples);

    qed_output(plane, &change, samples)
}

/// `plane`'s QED output, from its already built `change` map.
fn qed_output(plane: QedPlane<'_>, change: &[f64], samples: MatrixSamples) -> Vec<f64> {
    let lowpass_size = (plane.lowpass_width, plane.lowpass_height);
    let lowpass = reference_lowpass(change, plane.width, plane.height, lowpass_size);
    let neutral = neutral_of(samples);
    let peak = peak_of(samples);
    let mut output = Vec::with_capacity(plane.source.len());

    for (index, original) in plane.source.iter().enumerate() {
        let column = index as u32 % plane.width;
        let row = index as u32 / plane.width;

        if is_border_sample(column, row) {
            output.push(plane.normal[index]);
            continue;
        }

        let source = *original as f32;
        let smooth = lowpass[index] as f32;
        let value = interior_value(source, smooth, samples, neutral, peak);

        output.push(f64::from(value));
    }

    output
}

/// An interior sample's output from its `source` and its low-pass, `smooth`.
///
/// Float samples take `source - smooth`. Integer samples round `smooth` half away from zero into the sample
/// range, then move `source` by its distance from `neutral`, kept in the sample range.
fn interior_value(source: f32, smooth: f32, samples: MatrixSamples, neutral: f32, peak: f32) -> f32 {
    if samples == MatrixSamples::F32 {
        return source - smooth;
    }

    let rounded = smooth.round();
    let stored = rounded.clamp(0.0, peak);
    let moved = source - (stored - neutral);

    moved.clamp(0.0, peak)
}

/// The change map of a `width` wide plane on the CPU.
///
/// A border sample holds [change_value](crate::deblock::qed::change_value) of how far `source` sits above
/// `strong`, and an interior sample the neutral value.
pub fn reference_change(source: &[f64], strong: &[f64], width: u32, samples: MatrixSamples) -> Vec<f64> {
    let neutral = neutral_of(samples);
    let mut change = Vec::with_capacity(source.len());

    for (index, (original, moved)) in source.iter().zip(strong).enumerate() {
        let column = index as u32 % width;
        let row = index as u32 / width;

        if !is_border_sample(column, row) {
            change.push(f64::from(neutral));
            continue;
        }

        let difference = *original as f32 - *moved as f32;
        let value = change_value(difference, samples);

        change.push(f64::from(value));
    }

    change
}

/// Whether the sample at `column`, `row` lies on the border of its 8 by 8 block.
pub fn is_border_sample(column: u32, row: u32) -> bool {
    let column_place = column % 8;
    let row_place = row % 8;

    column_place == 0 || column_place == 7 || row_place == 0 || row_place == 7
}

/// Where the low-pass reads position `index` of a side of `length` samples, on the CPU.
///
/// Positions inside read themselves. The next `length` positions mirror back with the edge sample repeated,
/// and every later one reads 0.
pub fn reference_lowpass_mirror(index: u32, length: u32) -> u32 {
    if index < length {
        return index;
    }

    if index < 2 * length {
        return 2 * length - 1 - index;
    }

    0
}

/// Every sample's low-pass on the CPU, in f32.
///
/// The `width` by `height` change map is extended to `lowpass_size` through its mirror, and each 8 by 8 block
/// keeps its four lowest DCT frequencies. Every step is an f32 operation. Sums add
/// their terms in raster order, and each weighted sum pairs a line with its mirror before adding.
pub fn reference_lowpass(change: &[f64], width: u32, height: u32, lowpass_size: (u32, u32)) -> Vec<f64> {
    let (lowpass_width, lowpass_height) = lowpass_size;
    let mut lowpass = vec![0.0; change.len()];

    for top in (0..lowpass_height).step_by(8) {
        for left in (0..lowpass_width).step_by(8) {
            let block = block_changes(change, width, height, left, top);
            let sums = block_sums(&block);

            write_block(&mut lowpass, &sums, width, height, left, top);
        }
    }

    lowpass
}

/// The 64 changes of the block at `left`, `top`, in raster order, read through the mirror.
fn block_changes(change: &[f64], width: u32, height: u32, left: u32, top: u32) -> [f32; 64] {
    let mut block = [0.0f32; 64];

    for row in 0..8 {
        let source_row = reference_lowpass_mirror(top + row, height);

        for column in 0..8 {
            let source_column = reference_lowpass_mirror(left + column, width);
            let index = source_row * width + source_column;

            block[(row * 8 + column) as usize] = change[index as usize] as f32;
        }
    }

    block
}

/// The four sums a block's low-pass is built from.
struct BlockSums {
    total: f32,
    row_weighted: f32,
    column_weighted: f32,
    cross: f32,
}

/// Sums the 64 changes of a block in raster order, all in f32.
///
/// Rows run left to right and columns top to bottom, and the total adds the rows top to bottom. Each weighted
/// sum pairs line `k` with line `7 - k`, whose weights differ only in sign, and adds the four pairs in order.
fn block_sums(block: &[f32; 64]) -> BlockSums {
    let mut rows = [0.0f32; 8];
    let mut columns = [0.0f32; 8];

    for (line, (row_total, column_total)) in rows.iter_mut().zip(columns.iter_mut()).enumerate() {
        for place in 0..8 {
            *row_total += block[line * 8 + place];
            *column_total += block[place * 8 + line];
        }
    }

    let mut total = 0.0f32;
    for row_total in rows {
        total += row_total;
    }

    let mut row_weighted = 0.0f32;
    let mut column_weighted = 0.0f32;
    let mut cross = 0.0f32;

    for (outer, outer_weight) in COSINES.iter().take(4).enumerate() {
        let mirrored = 7 - outer;
        let mut part = 0.0f32;

        row_weighted += outer_weight * (rows[outer] - rows[mirrored]);
        column_weighted += outer_weight * (columns[outer] - columns[mirrored]);

        for (inner, inner_weight) in COSINES.iter().take(4).enumerate() {
            let inner_mirrored = 7 - inner;
            let near = block[outer * 8 + inner] - block[outer * 8 + inner_mirrored];
            let far = block[mirrored * 8 + inner] - block[mirrored * 8 + inner_mirrored];

            part += inner_weight * (near - far);
        }

        cross += outer_weight * part;
    }

    BlockSums {
        total,
        row_weighted,
        column_weighted,
        cross,
    }
}

/// Writes the low-pass of the block at `left`, `top` into each of its samples inside the plane.
///
/// Each sample starts from the mean, adds the row term, then the column term, then the cross term. The
/// product of its two weights is rounded to f32 before it scales the cross term.
fn write_block(lowpass: &mut [f64], sums: &BlockSums, width: u32, height: u32, left: u32, top: u32) {
    let mean = sums.total * 0.015625;
    let row_part = sums.row_weighted * 0.03125;
    let column_part = sums.column_weighted * 0.03125;
    let cross_part = sums.cross * 0.0625;

    for (row, row_weight) in (0u32..).zip(COSINES) {
        let y = top + row;
        if y >= height {
            continue;
        }

        let row_value = mean + row_weight * row_part;

        for (column, column_weight) in (0u32..).zip(COSINES) {
            let x = left + column;
            if x >= width {
                continue;
            }

            let cross_weight = row_weight * column_weight;
            let with_column = row_value + column_weight * column_part;
            let value = with_column + cross_weight * cross_part;

            lowpass[(y * width + x) as usize] = f64::from(value);
        }
    }
}

/// A 16 by 16 plane at 100, and a strong pass 8 lower on the four corners of every 8 by 8 block, at 8 bits.
///
/// The corners' changes are 136 and every other change is 128, so every block's low-pass is exactly 128.5.
pub fn half_lowpass_planes() -> (Vec<f64>, Vec<f64>) {
    let source = vec![100.0; 256];
    let strong = (0..256u32)
        .map(|index| {
            let column = index % 16 % 8;
            let row = index / 16 % 8;
            let corner = (column == 0 || column == 7) && (row == 0 || row == 7);

            if corner { 92.0 } else { 100.0 }
        })
        .collect();

    (source, strong)
}

/// Panics where `found`, a `width` wide plane, is further from `expected` than a QED output may be, and gives
/// how many samples differ.
///
/// Integer samples may differ by 1. Float samples may differ by up to
/// [QED_FLOAT_TOLERANCE](crate::testing::QED_FLOAT_TOLERANCE), and give 0.
pub fn count_qed_differences(
    found: &[f64],
    expected: &[f64],
    width: u32,
    samples: MatrixSamples,
    label: &str,
) -> usize {
    let found_values: Vec<f32> = found.iter().map(|value| *value as f32).collect();
    let expected_values: Vec<f32> = expected.iter().map(|value| *value as f32).collect();

    if samples == MatrixSamples::F32 {
        assert_plane_within(&found_values, &expected_values, QED_FLOAT_TOLERANCE, width, label);
        return 0;
    }

    assert_plane_within(&found_values, &expected_values, 1.0, width, label);

    found
        .iter()
        .zip(expected)
        .filter(|(actual, wanted)| actual != wanted)
        .count()
}
