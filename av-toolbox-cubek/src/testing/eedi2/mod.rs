mod directions;
mod fixtures;
mod gaps;
mod lattice;
mod mask;
#[cfg(test)]
mod tests;

use self::directions::{clean_directions, expand_directions, filter_directions, raw_directions};
pub use self::fixtures::{EEDI2_FIXTURES, Eedi2Fixture};
use self::gaps::{expand_gaps, fill_gaps, filter_gaps, gap_directions};
use self::lattice::{lattice, post_process};
use self::mask::{dilate, edge_mask, erode, remove_small_gaps};

/// EEDI2's `limlut`, the agreement threshold for a direction of `index` whole pixels.
///
/// Index 31 holds -1, so a threshold read there is negative and every `<=` test against it fails.
const LIMLUT: [i64; 33] = [
    6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, -1, -1,
];

/// The arguments of one EEDI2 field pass, with the plane's bit depth and subsampling.
///
/// `ssw` is the log2 horizontal subsampling of the plane in the direction the pass runs. It shifts `maxd`
/// and the lattice fallback's reach right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eedi2Parameters {
    pub mthresh: u32,
    pub lthresh: u32,
    pub vthresh: u32,
    pub estr: u32,
    pub dstr: u32,
    pub maxd: u32,
    pub nt: u32,
    pub pp: u32,
    pub bits: u32,
    pub ssw: u32,
}

impl Default for Eedi2Parameters {
    fn default() -> Self {
        Self {
            mthresh: 10,
            lthresh: 20,
            vthresh: 20,
            estr: 2,
            dstr: 4,
            maxd: 24,
            nt: 50,
            pp: 1,
            bits: 8,
            ssw: 0,
        }
    }
}

/// Every buffer of one EEDI2 field pass, holding integer values as f32.
///
/// The masks `e0` to `e` hold 0 or 1. Field buffers have one row per kept row, and gap buffers one row
/// fewer, each `width` values wide. `r` holds the rebuilt rows before post-processing, and `y1` and `y2`
/// hold the re-filtered lattice directions whether or not `pp` applies them. `output` is the whole plane.
pub struct Eedi2Stages {
    pub e0: Vec<f32>,
    pub e1: Vec<f32>,
    pub e2: Vec<f32>,
    pub e3: Vec<f32>,
    pub e: Vec<f32>,
    pub d0: Vec<f32>,
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub d3: Vec<f32>,
    pub x0: Vec<f32>,
    pub x1: Vec<f32>,
    pub x2: Vec<f32>,
    pub x3: Vec<f32>,
    pub x4: Vec<f32>,
    pub l: Vec<f32>,
    pub r: Vec<f32>,
    pub y1: Vec<f32>,
    pub y2: Vec<f32>,
    pub output: Vec<f32>,
}

/// Which left neighbour the lattice's isolation test reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeftDirection {
    /// The direction the lattice already wrote at the column before, as the plugin reads it.
    Updated,
    /// The direction before the lattice ran, only to prove a case depends on the updated one.
    #[cfg_attr(not(test), expect(dead_code))]
    Original,
}

/// Which partner the third agreement test of the gap directions compares with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Typo {
    /// The left direction below, the plugin's typo that the port keeps.
    Kept,
    /// The right direction below, only to prove a case depends on the typo.
    #[cfg_attr(not(test), expect(dead_code))]
    Fixed,
}

/// A buffer of rows `width` values wide, holding a field, a mask or a direction map.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Grid {
    values: Vec<i64>,
    width: i64,
}

impl Grid {
    fn filled(rows: i64, width: i64, value: i64) -> Self {
        let length = rows.max(0) * width;

        Self {
            values: vec![value; length as usize],
            width,
        }
    }

    fn rows(&self) -> i64 {
        self.values.len() as i64 / self.width
    }

    /// The value at `column` of `row`.
    ///
    /// Every pass reads through here, so debug builds check that no read leaves its row.
    fn at(&self, row: i64, column: i64) -> i64 {
        debug_assert!(
            (0..self.width).contains(&column),
            "column {column} leaves a row {} wide",
            self.width
        );
        debug_assert!(
            (0..self.rows()).contains(&row),
            "row {row} leaves a buffer of {} rows",
            self.rows()
        );

        let index = row * self.width + column;
        self.values[index as usize]
    }

    fn set(&mut self, row: i64, column: i64, value: i64) {
        let index = row * self.width + column;
        self.values[index as usize] = value;
    }

    fn to_f32(&self) -> Vec<f32> {
        self.values.iter().map(|value| *value as f32).collect()
    }
}

/// The thresholds of one field pass, scaled to the plane's bit depth as reference section 1.2 gives.
struct Scaled {
    /// `bits - 8`, the shift from 8-bit code values.
    shift: i64,
    unit: i64,
    peak: i64,
    neutral: i64,
    ten: i64,
    twelve: i64,
    eight: i64,
    three: i64,
    nine: i64,
    mthresh_squared: i64,
    lthresh: i64,
    vthresh_81: i64,
    estr: i64,
    dstr: i64,
    maxd: i64,
    dt: i64,
    nt4: i64,
    nt7: i64,
    nt8: i64,
    nt13: i64,
    nt19: i64,
}

impl Scaled {
    fn new(parameters: &Eedi2Parameters) -> Self {
        let bits = i64::from(parameters.bits);
        let shift = bits - 8;
        let unit = 1 << shift;
        let nt = i64::from(parameters.nt) * unit;
        let mthresh = i64::from(parameters.mthresh);

        Self {
            shift,
            unit,
            peak: (1 << bits) - 1,
            neutral: 1 << (bits - 1),
            ten: 10 * unit,
            twelve: 12 * unit,
            eight: 8 * unit,
            three: 3 * unit,
            nine: 9 * unit,
            mthresh_squared: mthresh * mthresh,
            lthresh: i64::from(parameters.lthresh),
            vthresh_81: i64::from(parameters.vthresh) * 81,
            estr: i64::from(parameters.estr),
            dstr: i64::from(parameters.dstr),
            maxd: i64::from(parameters.maxd) >> parameters.ssw,
            dt: 4 >> parameters.ssw,
            nt4: 4 * nt,
            nt7: 7 * nt,
            nt8: 8 * nt,
            nt13: 13 * nt,
            nt19: 19 * nt,
        }
    }

    /// `LIM(value)`, the agreement threshold for the direction map value `value`.
    fn lim(&self, value: i64) -> i64 {
        let index = (value - self.neutral).abs() >> (self.shift + 2);

        LIMLUT[index as usize] * self.unit
    }

    /// The map value for a direction of `pixels`.
    fn map_value(&self, pixels: i64) -> i64 {
        self.neutral + pixels * 4 * self.unit
    }
}

/// `QLIM(pixels)`, the agreement threshold for a raw direction in whole pixels.
fn quarter_lim(pixels: i64) -> i64 {
    let index = pixels.unsigned_abs() as usize;

    (LIMLUT[index] / 4).max(2)
}

/// `RA(total, count)`, the plugin's float average rounded half up, in integers.
fn round_average(total: i64, count: i64) -> i64 {
    (2 * total + count) / (2 * count)
}

/// The median of some direction map values and the values that agree with it.
struct Consensus {
    mid: i64,
    lim: i64,
    count: i64,
    sum: i64,
}

impl Consensus {
    /// Sorts `values`, none of them `peak`, and keeps those within `LIM(mid)` of their median.
    fn of(values: &mut [i64], scaled: &Scaled) -> Self {
        values.sort_unstable();

        let length = values.len();
        let mid = if length % 2 == 1 {
            values[length / 2]
        } else {
            (values[length / 2 - 1] + values[length / 2] + 1) >> 1
        };
        let lim = scaled.lim(mid);

        let agreeing = values.iter().filter(|value| (**value - mid).abs() <= lim);
        let count = agreeing.clone().count() as i64;
        let sum = agreeing.sum();

        Self { mid, lim, count, sum }
    }

    /// The average of the agreeing values and the median, as filters and gap passes write it.
    fn average(&self) -> i64 {
        round_average(self.sum + self.mid, self.count + 1)
    }
}

/// The field of parity `offset`, one kept row of the plane per field row.
fn kept_field(samples: &[i64], width: i64, height: i64, offset: i64) -> Grid {
    let field_height = height / 2;

    let mut field = Grid::filled(field_height, width, 0);
    for row in 0..field_height {
        let start = ((2 * row + offset) * width) as usize;
        let end = start + width as usize;
        let destination = (row * width) as usize;
        field.values[destination..destination + width as usize].copy_from_slice(&samples[start..end]);
    }

    field
}

/// The whole plane, with the kept rows copied, the gap rows rebuilt and the leftover edge row copied from
/// its kept neighbour.
fn assemble(samples: &[i64], field: &Grid, rebuilt: &Grid, height: i64, offset: i64) -> Vec<i64> {
    let width = field.width;

    let mut output = vec![0; samples.len()];
    for row in 0..field.rows() {
        let destination = ((2 * row + offset) * width) as usize;
        let source = (row * width) as usize;
        output[destination..destination + width as usize]
            .copy_from_slice(&field.values[source..source + width as usize]);
    }

    for gap in 0..rebuilt.rows() {
        let destination = ((2 * gap + 1 + offset) * width) as usize;
        let source = (gap * width) as usize;
        output[destination..destination + width as usize]
            .copy_from_slice(&rebuilt.values[source..source + width as usize]);
    }

    let (edge_row, edge_source) = if offset == 0 {
        (height - 1, height - 2)
    } else {
        (0, 1)
    };
    let source_start = (edge_source * width) as usize;
    let source_range = source_start..source_start + width as usize;
    output.copy_within(source_range, (edge_row * width) as usize);

    output
}

fn field_stages(
    plane: &[f32],
    width: u32,
    height: u32,
    offset: u32,
    parameters: &Eedi2Parameters,
    left: LeftDirection,
    typo: Typo,
) -> Eedi2Stages {
    assert!(
        height >= 2 && height.is_multiple_of(2),
        "an EEDI2 plane needs an even height, not {height}"
    );
    assert!(width >= 1, "an EEDI2 plane needs at least one column");
    assert!(offset < 2, "an EEDI2 field offset is 0 or 1, not {offset}");
    assert!(
        (8..=16).contains(&parameters.bits),
        "EEDI2 takes 8 to 16 bits per sample, not {}",
        parameters.bits
    );
    assert!(parameters.pp < 2, "EEDI2's pp is 0 or 1, not {}", parameters.pp);
    assert!(
        (1..=29).contains(&parameters.maxd),
        "EEDI2's maxd is between 1 and 29, not {}",
        parameters.maxd
    );

    let samples_expected = width as usize * height as usize;
    assert_eq!(plane.len(), samples_expected, "the plane is not {width}x{height}");

    let width = i64::from(width);
    let height = i64::from(height);
    let offset = i64::from(offset);
    let scaled = Scaled::new(parameters);
    let samples: Vec<i64> = plane.iter().map(|value| *value as i64).collect();
    let field = kept_field(&samples, width, height, offset);

    let e0 = edge_mask(&field, &scaled);
    let e1 = erode(&e0, scaled.estr);
    let e2 = dilate(&e1, scaled.dstr);
    let e3 = erode(&e2, scaled.estr);
    let edges = remove_small_gaps(&e3);

    let d0 = raw_directions(&field, &edges, &scaled);
    let d1 = filter_directions(&d0, &edges, &scaled);
    let d2 = expand_directions(&d1, &edges, &scaled);
    let d3 = clean_directions(&d2, &edges, &scaled);

    let x0 = gap_directions(&d3, &edges, &scaled, typo);
    let x1 = filter_gaps(&x0, &edges, &scaled);
    let x2 = expand_gaps(&x1, &edges, &scaled);
    let x3 = fill_gaps(&x2, &edges, &scaled);
    let x4 = fill_gaps(&x3, &edges, &scaled);

    let (directions, rebuilt) = lattice(&field, &d3, &x4, &scaled, left);
    let y1 = filter_gaps(&directions, &edges, &scaled);
    let y2 = expand_gaps(&y1, &edges, &scaled);
    let processed = if parameters.pp == 1 {
        post_process(&field, &directions, &y2, &rebuilt, &scaled)
    } else {
        rebuilt.clone()
    };

    let output = assemble(&samples, &field, &processed, height, offset);

    Eedi2Stages {
        e0: e0.to_f32(),
        e1: e1.to_f32(),
        e2: e2.to_f32(),
        e3: e3.to_f32(),
        e: edges.to_f32(),
        d0: d0.to_f32(),
        d1: d1.to_f32(),
        d2: d2.to_f32(),
        d3: d3.to_f32(),
        x0: x0.to_f32(),
        x1: x1.to_f32(),
        x2: x2.to_f32(),
        x3: x3.to_f32(),
        x4: x4.to_f32(),
        l: directions.to_f32(),
        r: rebuilt.to_f32(),
        y1: y1.to_f32(),
        y2: y2.to_f32(),
        output: output.iter().map(|value| *value as f32).collect(),
    }
}

/// One EEDI2 field pass on the CPU, keeping the rows of parity `offset` and rebuilding the others.
///
/// The plane holds integer code values. Both direction searches are clamped to the row, and the row with
/// no kept row on one side copies its kept neighbour.
pub fn reference_eedi2_field_stages(
    plane: &[f32],
    width: u32,
    height: u32,
    offset: u32,
    parameters: &Eedi2Parameters,
) -> Eedi2Stages {
    field_stages(
        plane,
        width,
        height,
        offset,
        parameters,
        LeftDirection::Updated,
        Typo::Kept,
    )
}

/// The output of a [reference_eedi2_field_stages](crate::testing::reference_eedi2_field_stages).
pub fn reference_eedi2_field(
    plane: &[f32],
    width: u32,
    height: u32,
    offset: u32,
    parameters: &Eedi2Parameters,
) -> Vec<f32> {
    let stages = reference_eedi2_field_stages(plane, width, height, offset, parameters);

    stages.output
}

/// EEDI2's double-rate anti-alias on the CPU, `floor((pass(0) + pass(1) + 1) / 2)`.
///
/// Each pass is a [reference_eedi2_field](crate::testing::reference_eedi2_field), one keeping the even
/// rows and one the odd rows.
pub fn reference_eedi2_antialias(
    plane: &[f32],
    width: u32,
    height: u32,
    parameters: &Eedi2Parameters,
) -> Vec<f32> {
    let even_kept = reference_eedi2_field(plane, width, height, 0, parameters);
    let odd_kept = reference_eedi2_field(plane, width, height, 1, parameters);

    even_kept
        .iter()
        .zip(&odd_kept)
        .map(|(first, second)| {
            let merged = (*first as i64 + *second as i64 + 1) >> 1;
            merged as f32
        })
        .collect()
}
