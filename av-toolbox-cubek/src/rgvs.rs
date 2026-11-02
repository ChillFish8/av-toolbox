// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::blur::{self, FilterPlanes, Smooth, Storage};
use crate::{DevicePlane, Dispatch, Edge, reflect};

/// Which RemoveGrain Repair mode runs.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RepairMode {
    /// Clamps the clip between the least and greatest of the reference's 3x3 neighbourhood.
    One,
    /// Clamps the clip between the second least and second greatest of the reference's 3x3
    /// neighbourhood.
    Two,
    /// Clamps the clip between the reference's third and sixth ranked neighbours, each pulled
    /// towards the reference's centre.
    Thirteen,
    /// Clamps the clip between the reference's pair bounds, pulled towards the reference's centre.
    Seventeen,
    /// Clamps the reference's centre to the clip's value widened by the reference's pair bounds.
    TwentyFour,
}

impl RepairMode {
    /// The mode this repository supports for Repair's numbered `mode`, or `None` for every other
    /// number.
    pub fn from_number(mode: u32) -> Option<Self> {
        match mode {
            1 => Some(Self::One),
            2 => Some(Self::Two),
            13 => Some(Self::Thirteen),
            17 => Some(Self::Seventeen),
            24 => Some(Self::TwentyFour),
            _ => None,
        }
    }

    fn number(self) -> u32 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Thirteen => 13,
            Self::Seventeen => 17,
            Self::TwentyFour => 24,
        }
    }
}

/// Which RemoveGrain mode runs.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RemoveGrainMode {
    /// Clamps the centre to whichever pair bound costs least to reach, weighted by the pair's
    /// spread.
    Seven,
    /// Averages the 3x3 window with a binomial weighting of 4 for the centre, 2 for each edge and
    /// 1 for each corner.
    Eleven,
    /// Clamps the centre between the plane's pair bounds.
    Seventeen,
    /// Clamps the centre between the least and greatest of the pair averages.
    TwentyTwo,
}

impl RemoveGrainMode {
    /// The RemoveGrain mode with this number, or `None` if it is unsupported.
    pub fn from_number(mode: u32) -> Option<Self> {
        match mode {
            7 => Some(Self::Seven),
            11 => Some(Self::Eleven),
            17 => Some(Self::Seventeen),
            22 => Some(Self::TwentyTwo),
            _ => None,
        }
    }

    fn number(self) -> u32 {
        match self {
            Self::Seven => 7,
            Self::Eleven => 11,
            Self::Seventeen => 17,
            Self::TwentyTwo => 22,
        }
    }
}

/// Which samples RemoveGrain reads at the first and last row and column of the plane.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GrainBorder {
    /// Keeps the first and last row and column unchanged, as rgvs does.
    Copy,
    /// Filters the first and last row and column too, reading past the edge through
    /// [reflect](fn@crate::reflect).
    Reflect,
}

impl GrainBorder {
    fn code(self) -> u32 {
        match self {
            Self::Copy => 0,
            Self::Reflect => 1,
        }
    }
}

/// The planes one repair reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct RepairPlanes<'a> {
    pub clip: DevicePlane<'a>,
    pub reference: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The plane one RemoveGrain operation reads and the plane it writes, both f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct RemoveGrainPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The value in `slot` of the 3x3 window at `x`, `y`, counting row by row from the top left.
#[cube]
fn window_value(values: &[f32], width: u32, x: u32, y: u32, slot: u32) -> f32 {
    let row = y + slot / 3 - 1;
    let column = x + slot % 3 - 1;
    let index = row * width + column;

    values[index as usize]
}

/// The value in `slot` of the 3x3 window at `x`, `y`, reading past the edge through `reflect`.
#[cube]
fn reflected_window_value(values: &[f32], width: u32, height: u32, x: u32, y: u32, slot: u32) -> f32 {
    let row = reflect((y + slot / 3) as i32 - 1, height as i32);
    let column = reflect((x + slot % 3) as i32 - 1, width as i32);
    let index = row as u32 * width + column as u32;

    values[index as usize]
}

/// One pair's reach outward from `value`, the greater of how far `value` sits below `mi` and how
/// far it sits above `ma`.
#[cube]
fn pair_reach(mi: f32, ma: f32, value: f32) -> f32 {
    let above = f32::max(ma - value, 0.0);
    let below = f32::max(value - mi, 0.0);

    f32::max(above, below)
}

/// The value of rank `rank`, counting from 0, among the 3x3 window at `x`, `y`.
///
/// The centre only counts when `with_centre` is set.
#[cube]
fn window_rank(values: &[f32], width: u32, x: u32, y: u32, rank: u32, with_centre: bool) -> f32 {
    let mut chosen = 0.0f32;
    let mut found = 0u32;

    for candidate_slot in 0..9u32 {
        let candidate_counts = with_centre || candidate_slot != 4u32;

        if candidate_counts {
            let candidate = window_value(values, width, x, y, candidate_slot);
            let mut below = 0u32;
            let mut equal = 0u32;

            for other_slot in 0..9u32 {
                let other_counts = with_centre || other_slot != 4u32;

                if other_counts {
                    let other = window_value(values, width, x, y, other_slot);

                    if other < candidate {
                        below += 1;
                    } else if other == candidate {
                        equal += 1;
                    }
                }
            }

            if found == 0u32 && below <= rank && rank < below + equal {
                chosen = candidate;
                found = 1u32;
            }
        }
    }

    chosen
}

#[cube(launch_unchecked)]
fn repair_kernel(clip: &[f32], reference: &[f32], output: &mut [f32], width: u32, height: u32, mode: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % width;
    let y = position / width;
    let value = clip[ABSOLUTE_POS];

    if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
        output[ABSOLUTE_POS] = value;
    } else if mode == 1u32 {
        let mut lowest = reference[ABSOLUTE_POS];
        let mut highest = lowest;

        for row in 0..3u32 {
            for column in 0..3u32 {
                let index = (y + row - 1) * width + x + column - 1;
                let neighbour = reference[index as usize];
                lowest = f32::min(lowest, neighbour);
                highest = f32::max(highest, neighbour);
            }
        }

        output[ABSOLUTE_POS] = f32::clamp(value, lowest, highest);
    } else if mode == 2u32 {
        let low = window_rank(reference, width, x, y, 1u32, true);
        let high = window_rank(reference, width, x, y, 7u32, true);

        output[ABSOLUTE_POS] = f32::clamp(value, low, high);
    } else if mode == 13u32 {
        let centre = reference[ABSOLUTE_POS];
        let third_rank = window_rank(reference, width, x, y, 2u32, false);
        let sixth_rank = window_rank(reference, width, x, y, 5u32, false);
        let low = f32::min(third_rank, centre);
        let high = f32::max(sixth_rank, centre);

        output[ABSOLUTE_POS] = f32::clamp(value, low, high);
    } else {
        let centre = reference[ABSOLUTE_POS];
        let top_left = window_value(reference, width, x, y, 0u32);
        let top = window_value(reference, width, x, y, 1u32);
        let top_right = window_value(reference, width, x, y, 2u32);
        let left = window_value(reference, width, x, y, 3u32);
        let right = window_value(reference, width, x, y, 5u32);
        let bottom_left = window_value(reference, width, x, y, 6u32);
        let bottom = window_value(reference, width, x, y, 7u32);
        let bottom_right = window_value(reference, width, x, y, 8u32);

        let mi_one = f32::min(top_left, bottom_right);
        let ma_one = f32::max(top_left, bottom_right);
        let mi_two = f32::min(top, bottom);
        let ma_two = f32::max(top, bottom);
        let mi_three = f32::min(top_right, bottom_left);
        let ma_three = f32::max(top_right, bottom_left);
        let mi_four = f32::min(left, right);
        let ma_four = f32::max(left, right);

        if mode == 17u32 {
            let lower_pairs_one_two = f32::max(mi_one, mi_two);
            let lower_pairs_three_four = f32::max(mi_three, mi_four);
            let lower_bound = f32::max(lower_pairs_one_two, lower_pairs_three_four);

            let upper_pairs_one_two = f32::min(ma_one, ma_two);
            let upper_pairs_three_four = f32::min(ma_three, ma_four);
            let upper_bound = f32::min(upper_pairs_one_two, upper_pairs_three_four);

            let bounds_low = f32::min(lower_bound, upper_bound);
            let low = f32::min(bounds_low, centre);

            let bounds_high = f32::max(lower_bound, upper_bound);
            let high = f32::max(bounds_high, centre);

            output[ABSOLUTE_POS] = f32::clamp(value, low, high);
        } else {
            let reach_one = pair_reach(mi_one, ma_one, value);
            let reach_two = pair_reach(mi_two, ma_two, value);
            let reach_three = pair_reach(mi_three, ma_three, value);
            let reach_four = pair_reach(mi_four, ma_four, value);

            let reach_one_two = f32::min(reach_one, reach_two);
            let reach_three_four = f32::min(reach_three, reach_four);
            let reach = f32::min(reach_one_two, reach_three_four);

            output[ABSOLUTE_POS] = f32::clamp(centre, value - reach, value + reach);
        }
    }
}

#[cube(launch_unchecked)]
fn remove_grain_kernel(input: &[f32], output: &mut [f32], width: u32, height: u32, mode: u32, border: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % width;
    let y = position / width;
    let centre = input[ABSOLUTE_POS];
    let at_border = x == 0 || y == 0 || x + 1 == width || y + 1 == height;

    if at_border && border == 0u32 {
        output[ABSOLUTE_POS] = centre;
    } else {
        let top_left = reflected_window_value(input, width, height, x, y, 0u32);
        let top = reflected_window_value(input, width, height, x, y, 1u32);
        let top_right = reflected_window_value(input, width, height, x, y, 2u32);
        let left = reflected_window_value(input, width, height, x, y, 3u32);
        let right = reflected_window_value(input, width, height, x, y, 5u32);
        let bottom_left = reflected_window_value(input, width, height, x, y, 6u32);
        let bottom = reflected_window_value(input, width, height, x, y, 7u32);
        let bottom_right = reflected_window_value(input, width, height, x, y, 8u32);

        let mi_one = f32::min(top_left, bottom_right);
        let ma_one = f32::max(top_left, bottom_right);
        let mi_two = f32::min(top, bottom);
        let ma_two = f32::max(top, bottom);
        let mi_three = f32::min(top_right, bottom_left);
        let ma_three = f32::max(top_right, bottom_left);
        let mi_four = f32::min(left, right);
        let ma_four = f32::max(left, right);

        if mode == 22u32 {
            let average_one = (top_left + bottom_right) / 2.0;
            let average_two = (top + bottom) / 2.0;
            let average_three = (top_right + bottom_left) / 2.0;
            let average_four = (left + right) / 2.0;

            let low_one_two = f32::min(average_one, average_two);
            let low_three_four = f32::min(average_three, average_four);
            let low = f32::min(low_one_two, low_three_four);

            let high_one_two = f32::max(average_one, average_two);
            let high_three_four = f32::max(average_three, average_four);
            let high = f32::max(high_one_two, high_three_four);

            output[ABSOLUTE_POS] = f32::clamp(centre, low, high);
        } else if mode == 17u32 {
            let lower_pairs_one_two = f32::max(mi_one, mi_two);
            let lower_pairs_three_four = f32::max(mi_three, mi_four);
            let lower_bound = f32::max(lower_pairs_one_two, lower_pairs_three_four);

            let upper_pairs_one_two = f32::min(ma_one, ma_two);
            let upper_pairs_three_four = f32::min(ma_three, ma_four);
            let upper_bound = f32::min(upper_pairs_one_two, upper_pairs_three_four);

            let low = f32::min(lower_bound, upper_bound);
            let high = f32::max(lower_bound, upper_bound);

            output[ABSOLUTE_POS] = f32::clamp(centre, low, high);
        } else {
            let distance_four = ma_four - mi_four;
            let clipped_four = f32::clamp(centre, mi_four, ma_four);
            let cost_four = f32::abs(centre - clipped_four) + distance_four;

            let mut chosen = clipped_four;
            let mut lowest_cost = cost_four;

            let distance_two = ma_two - mi_two;
            let clipped_two = f32::clamp(centre, mi_two, ma_two);
            let cost_two = f32::abs(centre - clipped_two) + distance_two;

            if cost_two < lowest_cost {
                chosen = clipped_two;
                lowest_cost = cost_two;
            }

            let distance_three = ma_three - mi_three;
            let clipped_three = f32::clamp(centre, mi_three, ma_three);
            let cost_three = f32::abs(centre - clipped_three) + distance_three;

            if cost_three < lowest_cost {
                chosen = clipped_three;
                lowest_cost = cost_three;
            }

            let distance_one = ma_one - mi_one;
            let clipped_one = f32::clamp(centre, mi_one, ma_one);
            let cost_one = f32::abs(centre - clipped_one) + distance_one;

            if cost_one < lowest_cost {
                chosen = clipped_one;
            }

            output[ABSOLUTE_POS] = chosen;
        }
    }
}

/// Copies each sample of the first and last row and column from `input` into `output`.
///
/// The first `width` units take the top row and the next `width` the bottom row. The rest alternate
/// between the first and last column of each row between them.
#[cube(launch_unchecked)]
fn copy_border_kernel(input: &[f32], output: &mut [f32], width: u32, height: u32, ring: u32) {
    let position = ABSOLUTE_POS as u32;
    if position >= ring {
        terminate!();
    }

    let mut column = position;
    let mut row = 0u32;
    if position >= 2 * width {
        let side = position - 2 * width;
        column = (side % 2) * (width - 1);
        row = 1 + side / 2;
    } else if position >= width {
        column = position - width;
        row = height - 1;
    }

    let index = row * width + column;
    output[index as usize] = input[index as usize];
}

/// The samples in the outer ring of a `width` by `height` plane, counting a one-sample-wide column twice.
fn ring_samples(width: u32, height: u32) -> u32 {
    if height == 1 {
        return width;
    }

    2 * width + 2 * (height - 2)
}

/// Repairs `clip` from `reference` with RemoveGrain's `mode`.
///
/// The first and last row and column of the clip are copied unchanged, as rgvs does.
pub fn repair(client: &Client, dispatch: &impl Dispatch, planes: RepairPlanes<'_>, mode: RepairMode) {
    let width = planes.clip.width();
    let height = planes.clip.height();
    let samples = planes.output.samples();
    let clip = planes.clip.array();
    let reference = planes.reference.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values, and only inner samples read neighbours.
    unsafe {
        repair_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            clip,
            reference,
            output,
            width,
            height,
            mode.number(),
        );
    }
}

/// Runs RemoveGrain's `mode` on `planes.input` into `planes.output`.
///
/// With [GrainBorder::Copy], the first and last row and column of
/// the plane are copied unchanged, as rgvs does. With
/// [GrainBorder::Reflect], they are filtered too, reading past
/// the edge through [reflect](fn@crate::reflect).
pub fn remove_grain(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: RemoveGrainPlanes<'_>,
    mode: RemoveGrainMode,
    border: GrainBorder,
) {
    if mode == RemoveGrainMode::Eleven {
        remove_grain_eleven(client, dispatch, planes, border);
        return;
    }

    let width = planes.input.width();
    let height = planes.input.height();
    let samples = planes.output.samples();
    let input = planes.input.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` f32 values, and with `GrainBorder::Copy` only inner
    // samples read neighbours. `GrainBorder::Reflect` reads every neighbour through `reflect`.
    unsafe {
        remove_grain_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input,
            output,
            width,
            height,
            mode.number(),
            border.code(),
        );
    }
}

/// RemoveGrain's mode 11, the 3x3 binomial blur, with its border kept as `border` says.
fn remove_grain_eleven(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: RemoveGrainPlanes<'_>,
    border: GrainBorder,
) {
    let filter_planes = FilterPlanes {
        input: planes.input,
        output: planes.output,
    };

    blur::smooth(
        client,
        dispatch,
        filter_planes,
        Smooth::Binomial,
        Edge::Reflect,
        Storage::F32,
    );

    if border == GrainBorder::Copy {
        copy_border(client, dispatch, planes);
    }
}

/// Copies the first and last row and column of `planes.input` into `planes.output`.
fn copy_border(client: &Client, dispatch: &impl Dispatch, planes: RemoveGrainPlanes<'_>) {
    let width = planes.input.width();
    let height = planes.input.height();
    let ring = ring_samples(width, height);
    let input = planes.input.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(ring as usize);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `width` by `height` f32 values, and every unit writes one sample of
    // the plane's outer ring.
    unsafe {
        copy_border_kernel::launch_unchecked(client, cubes, unit, input, output, width, height, ring);
    }
}
