// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::{DevicePlane, Dispatch};

/// The directions SangNom weighs, one cost buffer each.
const DIRECTIONS: u32 = 9;

/// The planes one SangNom field pass reads and writes, with its scratch.
///
/// `input` and `output` are f32 code values of one size. `smoothed` holds `9 * (height / 2) * width`
/// f32 values, as [sangnom_scratch](crate::aa::sangnom_scratch) gives.
#[derive(Debug, Clone, Copy)]
pub struct SangnomPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
    pub smoothed: &'a Handle,
}

/// SangNom's direction threshold for `aa` on a plane of `bits` per sample.
///
/// It is `trunc(aa * 21 / 16 * 2^(bits - 8))`, so 48 gives 63 at 8 bits and 252 at 10 bits.
pub fn sangnom_aaf(aa: u32, bits: u32) -> f32 {
    assert!(
        (8..=16).contains(&bits),
        "SangNom takes 8 to 16 bits per sample, not {bits}"
    );
    assert!(aa <= 128, "SangNom's aa is at most 128, not {aa}");

    let scale = 1u32 << (bits - 8);
    let scaled = aa * 21 * scale;
    let threshold = scaled / 16;

    threshold as f32
}

/// The most a SangNom field pass may rebuild a sample to on a plane of `bits` per sample.
///
/// It is 255 at 8 bits and 65535 from 9 to 16 bits, so a rebuilt sample may pass the format's peak.
pub fn sangnom_max(bits: u32) -> f32 {
    assert!(
        (8..=16).contains(&bits),
        "SangNom takes 8 to 16 bits per sample, not {bits}"
    );

    if bits == 8 { 255.0 } else { 65535.0 }
}

/// The byte size of the `smoothed` scratch a `width` by `height` field pass needs.
///
/// Each direction holds one row per gap between kept lines, `height / 2` rows in all.
pub fn sangnom_scratch(width: u32, height: u32) -> usize {
    let directions = DIRECTIONS as usize;
    let field_height = height as usize / 2;
    let samples = width as usize * field_height;

    directions * samples * size_of::<f32>()
}

/// The two values one direction pairs across the gap between two kept lines.
///
/// The direction's cost is their difference and its candidate is their average.
#[derive(CubeType)]
struct DirectionPair {
    first: f32,
    second: f32,
}

/// Kept line `line` at `column + shift`, with the column clamped into the plane.
#[cube]
fn kept_sample(input: &Array<f32>, width: u32, offset: u32, line: u32, column: u32, shift: i32) -> f32 {
    let reached = column as i32 + shift;
    let last_column = width as i32 - 1;
    let at_least_zero = i32::max(reached, 0);
    let clamped = i32::min(at_least_zero, last_column);
    let row = offset + 2 * line;
    let index = row * width + clamped as u32;

    input[index as usize]
}

/// The SangNom predictor, `(4 * first + 5 * second - third) / 8`, capped at `max`.
///
/// A negative sum gives `max`, as the plugin's unsigned shift does.
#[cube]
fn predict(first: f32, second: f32, third: f32, max: f32) -> f32 {
    let sum = 4.0 * first + 5.0 * second - third;
    let mut predicted = max;

    if sum >= 0.0 {
        let shifted = f32::floor(sum * 0.125);
        predicted = f32::min(shifted, max);
    }

    predicted
}

/// `floor((first + second + 1) / 2)`, exact on integer code values.
#[cube]
fn average(first: f32, second: f32) -> f32 {
    let total = first + second + 1.0;

    f32::floor(total * 0.5)
}

/// The pair `direction` compares at `column` across the gap between kept lines `gap - 1` and `gap`.
///
/// Directions 3 and 5 pair the forward and backward SangNom predictions. The others pair a sample of
/// the upper line with the mirrored sample of the lower line, from 3 columns left at 0 through
/// straight down at 4 to 3 columns right at 8.
#[cube]
fn direction_pair(
    input: &Array<f32>,
    width: u32,
    offset: u32,
    gap: u32,
    column: u32,
    direction: u32,
    max: f32,
) -> DirectionPair {
    let upper = gap - 1;

    let mut reach = 0i32;
    if direction < 3 {
        reach = direction as i32 - 3;
    } else if direction > 5 {
        reach = direction as i32 - 5;
    }

    let mut first = kept_sample(input, width, offset, upper, column, reach);
    let mut second = kept_sample(input, width, offset, gap, column, -reach);

    if direction == 3 || direction == 5 {
        // The reach is 0 here, so `first` and `second` hold the centre samples.
        let upper_left = kept_sample(input, width, offset, upper, column, -1);
        let upper_right = kept_sample(input, width, offset, upper, column, 1);
        let lower_left = kept_sample(input, width, offset, gap, column, -1);
        let lower_right = kept_sample(input, width, offset, gap, column, 1);
        let upper_centre = first;
        let lower_centre = second;

        if direction == 3 {
            first = predict(upper_left, upper_centre, upper_right, max);
            second = predict(lower_right, lower_centre, lower_left, max);
        } else {
            first = predict(upper_right, upper_centre, upper_left, max);
            second = predict(lower_left, lower_centre, lower_right, max);
        }
    }

    DirectionPair { first, second }
}

/// The raw cost of `direction` at `column` in gap `gap`, zero for the first and last gaps.
#[cube]
fn raw_cost(
    input: &Array<f32>,
    width: u32,
    offset: u32,
    field_height: u32,
    gap: u32,
    column: u32,
    direction: u32,
    max: f32,
) -> f32 {
    let mut cost = 0.0f32;

    if gap >= 1 && gap < field_height {
        let pair = direction_pair(input, width, offset, gap, column, direction, max);
        cost = f32::abs(pair.first - pair.second);
    }

    cost
}

/// The line sum `above + own + below` of `direction` at `column` in gap `gap`.
///
/// `above` is the smoothed row of the gap before, and `own` and `below` are the raw costs of this gap
/// and the next.
#[cube]
fn line_sum(
    input: &Array<f32>,
    smoothed: &Array<f32>,
    width: u32,
    field_height: u32,
    offset: u32,
    gap: u32,
    column: u32,
    direction: u32,
    max: f32,
) -> f32 {
    let own = raw_cost(input, width, offset, field_height, gap, column, direction, max);
    let next_gap = gap + 1;
    let below = raw_cost(
        input,
        width,
        offset,
        field_height,
        next_gap,
        column,
        direction,
        max,
    );

    let mut above = 0.0f32;
    if gap > 1 {
        let above_index = (direction * field_height + gap - 1) * width + column;
        above = smoothed[above_index as usize];
    }

    above + own + below
}

/// Smooths gap `gap` of every direction in one step, from the smoothed gap before it.
///
/// Each unit sums 7 columns of the line sums around its own, divides by 16 and caps at `max`.
#[cube(launch_unchecked)]
fn sangnom_smooth_row_kernel(
    input: &Array<f32>,
    smoothed: &mut Array<f32>,
    width: u32,
    field_height: u32,
    offset: u32,
    gap: u32,
    max: f32,
) {
    let position = ABSOLUTE_POS as u32;
    if position >= DIRECTIONS * width {
        terminate!();
    }

    let direction = position / width;
    let column = position % width;
    let last_column = width as i32 - 1;

    let mut total = 0.0f32;
    for tap in 0..7u32 {
        let reached = column as i32 + tap as i32 - 3;
        let at_least_zero = i32::max(reached, 0);
        let clamped = i32::min(at_least_zero, last_column);
        let tap_column = clamped as u32;
        total += line_sum(
            input,
            smoothed,
            width,
            field_height,
            offset,
            gap,
            tap_column,
            direction,
            max,
        );
    }

    let scaled = f32::floor(total * 0.0625);
    let smoothed_index = (direction * field_height + gap) * width + column;
    smoothed[smoothed_index as usize] = f32::min(scaled, max);
}

/// The direction to interpolate along, from the nine smoothed costs at `index` of each buffer.
///
/// The vertical direction wins ties and any pixel whose lowest cost is above `aaf`. Otherwise the
/// lowest cost wins, taking ties in the order 5, 3, 6, 2, 7, 1, 8, 0.
#[cube]
fn choose_direction(smoothed: &Array<f32>, index: u32, samples: u32, aaf: f32) -> u32 {
    let cost_0 = smoothed[index as usize];
    let cost_1 = smoothed[(samples + index) as usize];
    let cost_2 = smoothed[(2 * samples + index) as usize];
    let cost_3 = smoothed[(3 * samples + index) as usize];
    let cost_4 = smoothed[(4 * samples + index) as usize];
    let cost_5 = smoothed[(5 * samples + index) as usize];
    let cost_6 = smoothed[(6 * samples + index) as usize];
    let cost_7 = smoothed[(7 * samples + index) as usize];
    let cost_8 = smoothed[(8 * samples + index) as usize];

    let mut lowest = f32::min(cost_0, cost_1);
    lowest = f32::min(lowest, cost_2);
    lowest = f32::min(lowest, cost_3);
    lowest = f32::min(lowest, cost_4);
    lowest = f32::min(lowest, cost_5);
    lowest = f32::min(lowest, cost_6);
    lowest = f32::min(lowest, cost_7);
    lowest = f32::min(lowest, cost_8);

    let mut direction = 0u32;
    if cost_4 == lowest || lowest > aaf {
        direction = 4u32;
    } else if cost_5 == lowest {
        direction = 5u32;
    } else if cost_3 == lowest {
        direction = 3u32;
    } else if cost_6 == lowest {
        direction = 6u32;
    } else if cost_2 == lowest {
        direction = 2u32;
    } else if cost_7 == lowest {
        direction = 7u32;
    } else if cost_1 == lowest {
        direction = 1u32;
    } else if cost_8 == lowest {
        direction = 8u32;
    }

    direction
}

/// Writes one output sample, copying kept rows and rebuilding the others along their chosen direction.
#[cube(launch_unchecked)]
fn sangnom_choose_kernel(
    input: &Array<f32>,
    smoothed: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    offset: u32,
    aaf: f32,
    max: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;
    let top_edge = offset == 1 && row == 0;
    let bottom_edge = offset == 0 && row == height - 1;

    if row % 2 == offset {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    } else if top_edge {
        output[ABSOLUTE_POS] = input[(width + column) as usize];
    } else if bottom_edge {
        output[ABSOLUTE_POS] = input[((height - 2) * width + column) as usize];
    } else {
        let gap = (row + 1 - offset) / 2;
        let samples = width * (height / 2);
        let index = gap * width + column;
        let direction = choose_direction(smoothed, index, samples, aaf);
        let pair = direction_pair(input, width, offset, gap, column, direction, max);

        output[ABSOLUTE_POS] = average(pair.first, pair.second);
    }
}

/// One SangNom field pass, keeping the rows of parity `offset` and rebuilding the others.
///
/// The planes hold integer code values. `aaf` is the direction threshold from
/// [sangnom_aaf](crate::aa::sangnom_aaf), and `max` the cap from [sangnom_max](crate::aa::sangnom_max),
/// so a rebuilt sample may pass the format's peak. The row with no kept row on one side copies its kept
/// neighbour. The plane's height must be even.
pub fn sangnom_field<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: SangnomPlanes<'_>,
    offset: u32,
    aaf: f32,
    max: f32,
) {
    let width = planes.input.width();
    let height = planes.input.height();

    assert!(offset <= 1, "a SangNom field offset is 0 or 1, not {offset}");
    assert!(
        planes.output.width() == width && planes.output.height() == height,
        "the output is {}x{}, expected {width}x{height}",
        planes.output.width(),
        planes.output.height(),
    );
    assert!(
        width >= 1 && height >= 2 && height.is_multiple_of(2),
        "a SangNom plane needs an even height and a width, not {width}x{height}"
    );

    let smoothed_bytes = sangnom_scratch(width, height);
    assert!(
        planes.smoothed.size_in_used() >= smoothed_bytes as u64,
        "the smoothed scratch holds {} bytes, expected {smoothed_bytes}",
        planes.smoothed.size_in_used(),
    );

    let directions = DIRECTIONS as usize;
    let samples = planes.input.samples();
    let field_height = height / 2;
    let smoothed_values = directions * field_height as usize * width as usize;
    let row_units = directions * width as usize;
    let row_cubes = dispatch.cube_count(row_units);
    let row_unit = dispatch.cube_dim();

    // Each gap reads the whole smoothed gap above it, so every gap is its own launch, which the
    // device finishes before the next starts. Barriers inside one cube did not make global writes
    // visible across it on every device.
    for gap in 1..field_height {
        // Safety: the input holds `width * height` f32 values and `smoothed` at least
        // `9 * (height / 2) * width`, both asserted above. Each unit writes only its own sample of gap
        // `gap`, reads only gap `gap - 1`, and clamps every column into the plane.
        unsafe {
            let input = planes.input.array::<R>();
            let smoothed = ArrayArg::from_raw_parts(planes.smoothed.clone(), smoothed_values);

            sangnom_smooth_row_kernel::launch_unchecked::<R>(
                client,
                row_cubes.clone(),
                row_unit,
                input,
                smoothed,
                width,
                field_height,
                offset,
                gap,
                max,
            );
        }
    }

    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and output hold `width * height` f32 values and `smoothed` at least
    // `9 * (height / 2) * width`, all asserted above. Each unit writes only its own output sample, and every
    // read is clamped into the plane.
    unsafe {
        let input = planes.input.array::<R>();
        let smoothed = ArrayArg::from_raw_parts(planes.smoothed.clone(), smoothed_values);
        let output = planes.output.array::<R>();

        sangnom_choose_kernel::launch_unchecked::<R>(
            client, cubes, unit, input, smoothed, output, width, height, offset, aaf, max,
        );
    }
}
