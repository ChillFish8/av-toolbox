// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
#[cfg(any(test, feature = "test-support"))]
use cubecl::server::Handle;

use super::directions::touches_edge;
use super::{Eedi2Settings, FieldBuffer, FieldPass, GapBuffer, row_launch};
#[cfg(any(test, feature = "test-support"))]
use crate::Dispatch;

/// Rounds a value cut down to `kept` to nearest, ties to even, from the first dropped bit `guard` and
/// whether any lower dropped bit was set.
#[cube]
fn round_to_even(kept: u32, guard: bool, sticky: bool) -> u32 {
    let odd = kept % 2 == 1;
    let mut rounded = kept;

    if guard && (sticky || odd) {
        rounded += 1;
    }

    rounded
}

/// An f32 value `mantissa * 2^-shift`, its mantissa holding 24 bits or at most `2^24`.
#[derive(CubeType)]
struct Quotient {
    mantissa: u32,
    shift: u32,
}

/// `fl32(magnitude / length)`, by long division until the quotient holds 24 bits.
#[cube]
fn float_quotient(magnitude: u32, length: u32) -> Quotient {
    let mut quotient = magnitude / length;
    let mut remainder = magnitude % length;
    let mut shift = 0u32;

    while quotient < 8388608u32 {
        remainder *= 2;
        quotient *= 2;

        if remainder >= length {
            remainder -= length;
            quotient += 1;
        }

        shift += 1;
    }

    remainder *= 2;
    let guard = remainder >= length;
    if guard {
        remainder -= length;
    }

    let sticky = remainder != 0;
    let mantissa = round_to_even(quotient, guard, sticky);

    Quotient { mantissa, shift }
}

/// Bit `index` of the value `high * 2^16 + low`, where `low` holds 16 bits.
#[cube]
fn bit_at(high: u32, low: u32, index: u32) -> bool {
    let mut word = low;
    let mut position = index;
    if index >= 16 {
        word = high;
        position = index - 16;
    }

    (word >> position) & 1 == 1
}

/// Whether any bit below `index` of the value `high * 2^16 + low` is set, where `low` holds 16 bits.
#[cube]
fn any_below(high: u32, low: u32, index: u32) -> bool {
    let low_bits = u32::min(index, 16);
    let high_bits = index - low_bits;
    let low_mask = (1u32 << low_bits) - 1;
    let high_mask = (1u32 << high_bits) - 1;

    low & low_mask != 0 || high & high_mask != 0
}

/// An f32 value `mantissa * 2^dropped`, its mantissa holding at most 24 bits or `2^24`.
#[derive(CubeType)]
struct Product {
    mantissa: u32,
    dropped: u32,
}

/// `fl32(step * mantissa)`, with the exact product held as `high * 2^16 + low`.
#[cube]
fn float_product(step: u32, mantissa: u32) -> Product {
    let low_product = step * (mantissa & 65535u32);
    let high = step * (mantissa >> 16) + (low_product >> 16);
    let low = low_product & 65535u32;

    let mut bits = 32 - u32::leading_zeros(low);
    if high != 0 {
        bits = 48 - u32::leading_zeros(high);
    }

    let mut rounded = (high << 16) | low;
    let mut dropped = 0u32;
    if bits > 24 {
        dropped = bits - 24;
        let low_shift = u32::min(dropped, 16);
        let high_shift = dropped - low_shift;
        let high_part = (high << (16 - low_shift)) >> high_shift;
        let kept = high_part | (low >> low_shift);
        let guard = bit_at(high, low, dropped - 1);
        let sticky = any_below(high, low, dropped - 1);
        rounded = round_to_even(kept, guard, sticky);
    }

    Product {
        mantissa: rounded,
        dropped,
    }
}

/// `FILLROUND(difference, length, step)`, the plugin's `trunc(step * (difference / length) + 0.5)` with
/// both the quotient and the product rounded to f32, in 32-bit integers.
///
/// `quotient` is `fl32(|difference| / length)` from `float_quotient`, shared by every step of a run. It
/// holds for `|difference|` and `length` below `2^16` other than a zero `difference`, with `step` below
/// `length`.
#[cube]
fn fill_round(difference: i32, quotient: &Quotient, step: u32) -> i32 {
    let mut result = 0i32;

    if step != 0 {
        let product = float_product(step, quotient.mantissa);
        let exponent = product.dropped as i32 - quotient.shift as i32;

        let mut nearest = 0u32;
        if exponent >= 0 {
            nearest = product.mantissa << exponent as u32;
        } else if exponent >= -25 {
            let shift = (-exponent) as u32;
            let half = 1u32 << (shift - 1);
            nearest = (product.mantissa + half) >> shift;
        }

        result = nearest as i32;
        if difference < 0 {
            result = -i32::max(result - 1, 0);
        }
    }

    result
}

/// A neighbouring gap row's backing for the run being walked, gathered one column at a time.
///
/// The row backs the run while every column so far has a direction and touches the edge mask. A row that
/// does not `exist` never backs a run and is never read.
#[derive(CubeType)]
struct Backing {
    row: u32,
    width: u32,
    exists: bool,
    peak: f32,
    supported: bool,
    lowest: f32,
    highest: f32,
}

#[cube]
impl Backing {
    fn new(row: u32, width: u32, exists: bool, peak: f32) -> Backing {
        Backing {
            row,
            width,
            exists,
            peak,
            supported: false,
            lowest: peak,
            highest: 0.0,
        }
    }

    /// Starts gathering a new run from `column`.
    fn start(&mut self, source: &Array<f32>, edges: &Array<f32>, column: u32) {
        self.supported = self.exists;
        self.lowest = self.peak;
        self.highest = 0.0;
        self.take(source, edges, column);
    }

    /// Adds `column`, read only while the row still backs the run.
    fn take(&mut self, source: &Array<f32>, edges: &Array<f32>, column: u32) {
        if self.supported {
            let value = source[(self.row * self.width + column) as usize];
            let on_edge = touches_edge(edges, self.width, self.row, column);

            if value == self.peak || !on_edge {
                self.supported = false;
            }

            self.lowest = f32::min(self.lowest, value);
            self.highest = f32::max(self.highest, value);
        }
    }

    /// The spread of the row's directions over the run, or 0 when it does not back the run.
    fn spread(&self) -> f32 {
        let mut spread = 0.0f32;
        if self.supported {
            spread = self.highest - self.lowest;
        }

        spread
    }
}

/// A run of missing gap pixels, between the columns `first` and `last` that bound it.
///
/// `back` and `forward` are the values at `first` and `last`. Column 0 never holds a direction, so a run
/// starting at column 1 has none behind it.
#[derive(CubeType)]
struct Run {
    first: u32,
    last: u32,
    back: f32,
    forward: f32,
}

/// The values a fill compares directions with, scaled to the plane.
#[derive(CubeType)]
struct FillScale {
    peak: f32,
    neutral: f32,
    eight: f32,
    reduce_quarter: f32,
}

/// Fills a run when the directions bounding it agree, closely enough or backed by a gap row either side.
///
/// Each pixel steps from the direction before the run toward the one after it, never reaching the latter.
#[cube]
fn fill_run(
    destination: &mut Array<f32>,
    row_start: u32,
    run: &Run,
    above: &Backing,
    below: &Backing,
    scale: &FillScale,
) {
    let back_found = run.first >= 1 && run.back != scale.peak;
    let forward_found = run.forward != scale.peak;

    if back_found && forward_found {
        let forward_reach = f32::abs(run.forward - scale.neutral);
        let back_reach = f32::abs(run.back - scale.neutral);
        let reach = f32::max(forward_reach, back_reach);
        let quarter_reach = f32::floor(reach * 0.25);
        let above_spread = above.spread();
        let below_spread = below.spread();
        let mut threshold = f32::max(quarter_reach, scale.eight);
        threshold = f32::max(threshold, above_spread);
        threshold = f32::max(threshold, below_spread);
        let shifted_reach = f32::floor(reach * scale.reduce_quarter);
        let lim = f32::min(shifted_reach, 6.0);
        let length = run.last - run.first;
        let difference = run.forward - run.back;

        let close = f32::abs(difference) <= threshold;
        let short = (length - 1) as f32 <= lim;
        let supported = above.supported || below.supported;

        if close && (short || supported) {
            let whole_difference = difference as i32;
            let magnitude = f32::abs(difference) as u32;

            if magnitude == 0 {
                for column in run.first + 1..run.last {
                    destination[(row_start + column) as usize] = run.back;
                }
            } else {
                let quotient = float_quotient(magnitude, length);

                for column in run.first + 1..run.last {
                    let step = column - run.first - 1;
                    let offset = fill_round(whole_difference, &quotient, step);
                    destination[(row_start + column) as usize] = run.back + offset as f32;
                }
            }
        }
    }
}

/// Passes 13 and 14, `fill`, one unit per gap row, which walks the row left to right and fills each run of
/// missing gap directions between two that agree.
///
/// A missing gap pixel has no direction and touches the edge mask. The gap rows either side are gathered
/// over each run as the walk passes it, so a run is decided at the column that ends it.
/// `reduce_quarter` is `2^(6 - bits)`, which turns a floor multiplication into a right shift by `bits - 6`.
#[cube(launch_unchecked)]
fn fill_gaps_kernel(
    source: &Array<f32>,
    edges: &Array<f32>,
    destination: &mut Array<f32>,
    width: u32,
    gap_rows: u32,
    peak: f32,
    neutral: f32,
    eight: f32,
    reduce_quarter: f32,
) {
    let gap = ABSOLUTE_POS as u32;
    if gap >= gap_rows {
        terminate!();
    }

    let scale = FillScale {
        peak,
        neutral,
        eight,
        reduce_quarter,
    };
    let row_start = gap * width;
    let above_exists = gap >= 1;
    let below_exists = gap + 1 < gap_rows;
    // Row 0 stands in for the missing row above the first gap row, and is never read.
    let above_row = u32::max(gap, 1) - 1;
    let below_row = gap + 1;

    let mut above = Backing::new(above_row, width, above_exists, peak);
    let mut below = Backing::new(below_row, width, below_exists, peak);
    let mut in_run = false;
    let mut first = 0u32;
    let mut back = peak;
    let mut previous = peak;

    for column in 0..width {
        let index = row_start + column;
        let value = source[index as usize];
        destination[index as usize] = value;

        let on_edge = touches_edge(edges, width, gap, column);
        let missing = value == peak && on_edge;
        let can_start = column >= 1 && column + 1 < width;

        if in_run {
            above.take(source, edges, column);
            below.take(source, edges, column);

            if !missing {
                let run = Run {
                    first,
                    last: column,
                    back,
                    forward: value,
                };
                fill_run(destination, row_start, &run, &above, &below, &scale);
                in_run = false;
            }
        } else if missing && can_start {
            in_run = true;
            first = column - 1;
            back = previous;
            above.start(source, edges, first);
            above.take(source, edges, column);
            below.start(source, edges, first);
            below.take(source, edges, column);
        }

        previous = value;
    }
}

/// Launches a fill, from the gap map in `source` into `destination`, with one unit per gap row.
pub(super) fn fill_gaps<R: Runtime>(
    client: &ComputeClient<R>,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
    source: GapBuffer,
    destination: GapBuffer,
) {
    assert_ne!(source, destination, "a fill cannot fill a gap map in place");

    let gap_rows = pass.rows - 1;
    let (cubes, unit) = row_launch(gap_rows);

    // Safety: every scratch buffer holds at least `width * (rows - 1)` values and `edges` `width * rows`,
    // as `FieldPass::new` checked. Each unit owns one gap row, the only row it writes. It reads the gap rows
    // either side only when they exist, the edge mask rows either side of each gap row it reads, and only
    // columns inside the row.
    unsafe {
        let source = pass.gap_array::<R>(source);
        let edges = pass.field_array::<R>(FieldBuffer::Edges);
        let destination = pass.gap_array::<R>(destination);

        fill_gaps_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            edges,
            destination,
            pass.width,
            gap_rows,
            settings.peak,
            settings.neutral,
            settings.eight,
            settings.reduce_quarter,
        );
    }
}

/// `FILLROUND` for each case, as a fill computes it for a run of `length` bounded by `difference`.
#[cfg(any(test, feature = "test-support"))]
#[cube(launch_unchecked)]
fn fill_round_kernel(
    differences: &Array<f32>,
    lengths: &Array<f32>,
    steps: &Array<f32>,
    results: &mut Array<f32>,
) {
    if ABSOLUTE_POS >= results.len() {
        terminate!();
    }

    let difference = differences[ABSOLUTE_POS] as i32;
    let length = lengths[ABSOLUTE_POS] as u32;
    let step = steps[ABSOLUTE_POS] as u32;
    let mut result = 0i32;

    if difference != 0 {
        let mut magnitude = difference;
        if difference < 0 {
            magnitude = -difference;
        }

        let quotient = float_quotient(magnitude as u32, length);
        result = fill_round(difference, &quotient, step);
    }

    results[ABSOLUTE_POS] = result as f32;
}

/// Evaluates the fills' `FILLROUND(difference, length, step)` on the device for each case, handing back a
/// buffer of one f32 result per case.
///
/// Every `|difference|` must be below `2^16`, every `length` between 2 and 65535, and every `step` below its
/// `length`.
#[cfg(any(test, feature = "test-support"))]
pub fn eedi2_fill_round<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    differences: &[i32],
    lengths: &[u32],
    steps: &[u32],
) -> Handle {
    let cases = differences.len();

    assert!(
        lengths.len() == cases && steps.len() == cases,
        "every case needs a difference, a length and a step"
    );

    let mut difference_values = Vec::with_capacity(cases);
    let mut length_values = Vec::with_capacity(cases);
    let mut step_values = Vec::with_capacity(cases);
    for ((difference, length), step) in differences.iter().zip(lengths).zip(steps) {
        assert!(
            difference.unsigned_abs() < 1 << 16,
            "a fill difference is below 2^16, not {difference}"
        );
        assert!(
            (2..=65535).contains(length) && step < length,
            "a fill needs a length between 2 and 65535 and a step below it, not {length} and {step}"
        );

        difference_values.push(*difference as f32);
        length_values.push(*length as f32);
        step_values.push(*step as f32);
    }

    let difference_bytes = f32::as_bytes(&difference_values);
    let length_bytes = f32::as_bytes(&length_values);
    let step_bytes = f32::as_bytes(&step_values);
    let difference_handle = client.create_from_slice(difference_bytes);
    let length_handle = client.create_from_slice(length_bytes);
    let step_handle = client.create_from_slice(step_bytes);
    let results = client.empty(cases * size_of::<f32>());

    let cubes = dispatch.cube_count(cases);
    let unit = dispatch.cube_dim();

    // Safety: the four buffers were created above with `cases` f32 values each. Each unit reads and writes
    // only its own case, and the checks above keep the long division finite.
    unsafe {
        let differences = ArrayArg::from_raw_parts(difference_handle, cases);
        let lengths = ArrayArg::from_raw_parts(length_handle, cases);
        let steps = ArrayArg::from_raw_parts(step_handle, cases);
        let owned = results.clone();
        let outputs = ArrayArg::from_raw_parts(owned, cases);

        fill_round_kernel::launch_unchecked::<R>(client, cubes, unit, differences, lengths, steps, outputs);
    }

    results
}
