use super::reflect;
use crate::rgvs::{GrainBorder, RemoveGrainMode, RepairMode};

/// Repairs `clip` from `reference` on the CPU, mirroring [repair](crate::rgvs::repair).
pub fn reference_repair(
    clip: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
    mode: RepairMode,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(clip.len());

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let border = x == 0 || y == 0 || x + 1 == width || y + 1 == height;

            if border {
                output.push(clip[index]);
                continue;
            }

            let window = reference_window(reference, width, x, y);
            let value = clip[index];
            let centre = window[4];

            let repaired = match mode {
                RepairMode::One => repair_one(value, &window),
                RepairMode::Two => repair_two(value, &window),
                RepairMode::Thirteen => repair_thirteen(value, &window, centre),
                RepairMode::Seventeen => repair_seventeen(value, &window, centre),
                RepairMode::TwentyFour => repair_twenty_four(value, &window, centre),
            };

            output.push(repaired);
        }
    }

    output
}

/// The 3x3 window of `values` centred on `x`, `y`, in row order.
fn reference_window(values: &[f32], width: u32, x: u32, y: u32) -> [f32; 9] {
    let mut window = [0.0f32; 9];

    for row in 0..3u32 {
        for column in 0..3u32 {
            let sample_row = y + row - 1;
            let sample_column = x + column - 1;
            let index = (sample_row * width + sample_column) as usize;

            window[(row * 3 + column) as usize] = values[index];
        }
    }

    window
}

fn sorted_window(window: &[f32; 9]) -> [f32; 9] {
    let mut sorted = *window;
    sorted.sort_by(f32::total_cmp);

    sorted
}

/// The eight neighbours of `window`, sorted, leaving the centre out.
fn sorted_neighbours(window: &[f32; 9]) -> [f32; 8] {
    let mut neighbours = [0.0f32; 8];
    let mut slot = 0;

    for (index, value) in window.iter().enumerate() {
        if index != 4 {
            neighbours[slot] = *value;
            slot += 1;
        }
    }

    neighbours.sort_by(f32::total_cmp);

    neighbours
}

fn repair_one(value: f32, window: &[f32; 9]) -> f32 {
    let mut lowest = window[0];
    let mut highest = window[0];

    for neighbour in window {
        lowest = lowest.min(*neighbour);
        highest = highest.max(*neighbour);
    }

    value.clamp(lowest, highest)
}

fn repair_two(value: f32, window: &[f32; 9]) -> f32 {
    let sorted = sorted_window(window);

    value.clamp(sorted[1], sorted[7])
}

fn repair_thirteen(value: f32, window: &[f32; 9], centre: f32) -> f32 {
    let sorted = sorted_neighbours(window);
    let low = f32::min(sorted[2], centre);
    let high = f32::max(sorted[5], centre);

    value.clamp(low, high)
}

/// The opposite pairs of `window` as `(min, max)`, in the order 1 to 4.
fn window_pairs(window: &[f32; 9]) -> [(f32, f32); 4] {
    let pair_one = (f32::min(window[0], window[8]), f32::max(window[0], window[8]));
    let pair_two = (f32::min(window[1], window[7]), f32::max(window[1], window[7]));
    let pair_three = (f32::min(window[2], window[6]), f32::max(window[2], window[6]));
    let pair_four = (f32::min(window[3], window[5]), f32::max(window[3], window[5]));

    [pair_one, pair_two, pair_three, pair_four]
}

fn repair_seventeen(value: f32, window: &[f32; 9], centre: f32) -> f32 {
    let pairs = window_pairs(window);
    let lower_bound = pairs.iter().map(|(mi, _)| *mi).fold(f32::MIN, f32::max);
    let upper_bound = pairs.iter().map(|(_, ma)| *ma).fold(f32::MAX, f32::min);
    let bounds_low = f32::min(lower_bound, upper_bound);
    let low = f32::min(bounds_low, centre);
    let bounds_high = f32::max(lower_bound, upper_bound);
    let high = f32::max(bounds_high, centre);

    value.clamp(low, high)
}

/// One pair's reach outward from `value`, for [repair_twenty_four], mirroring the kernel's
/// `pair_reach`.
fn pair_reach(mi: f32, ma: f32, value: f32) -> f32 {
    let above = f32::max(ma - value, 0.0);
    let below = f32::max(value - mi, 0.0);

    f32::max(above, below)
}

fn repair_twenty_four(value: f32, window: &[f32; 9], centre: f32) -> f32 {
    let pairs = window_pairs(window);
    let reach = pairs
        .iter()
        .map(|(mi, ma)| pair_reach(*mi, *ma, value))
        .fold(f32::MAX, f32::min);

    centre.clamp(value - reach, value + reach)
}

/// The 3x3 window of `values` centred on `x`, `y`, in row order, reading past the edge through
/// `reflect`.
fn reflected_window(values: &[f32], width: u32, height: u32, x: u32, y: u32) -> [f32; 9] {
    let mut window = [0.0f32; 9];

    for row in 0..3u32 {
        for column in 0..3u32 {
            let sample_row = reflect(y as i32 + row as i32 - 1, height as i32);
            let sample_column = reflect(x as i32 + column as i32 - 1, width as i32);
            let index = (sample_row as u32 * width + sample_column as u32) as usize;

            window[(row * 3 + column) as usize] = values[index];
        }
    }

    window
}

/// Runs RemoveGrain's `mode` on `input` on the CPU, mirroring
/// [remove_grain](crate::rgvs::remove_grain).
pub fn reference_remove_grain(
    input: &[f32],
    width: u32,
    height: u32,
    mode: RemoveGrainMode,
    border: GrainBorder,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let at_border = x == 0 || y == 0 || x + 1 == width || y + 1 == height;

            if at_border && border == GrainBorder::Copy {
                output.push(input[index]);
                continue;
            }

            let window = reflected_window(input, width, height, x, y);
            let centre = window[4];

            let cleaned = match mode {
                RemoveGrainMode::Seven => remove_grain_seven(&window, centre),
                RemoveGrainMode::Eleven => remove_grain_eleven(&window),
                RemoveGrainMode::Seventeen => remove_grain_seventeen(&window, centre),
                RemoveGrainMode::TwentyTwo => remove_grain_twenty_two(&window, centre),
            };

            output.push(cleaned);
        }
    }

    output
}

fn remove_grain_eleven(window: &[f32; 9]) -> f32 {
    let sum_edges = window[1] + window[3] + window[5] + window[7];
    let sum_corners = window[0] + window[2] + window[6] + window[8];

    (4.0 * window[4] + 2.0 * sum_edges + sum_corners) / 16.0
}

fn remove_grain_twenty_two(window: &[f32; 9], centre: f32) -> f32 {
    let average_one = (window[0] + window[8]) / 2.0;
    let average_two = (window[1] + window[7]) / 2.0;
    let average_three = (window[2] + window[6]) / 2.0;
    let average_four = (window[3] + window[5]) / 2.0;

    let low_one_two = f32::min(average_one, average_two);
    let low_three_four = f32::min(average_three, average_four);
    let low = f32::min(low_one_two, low_three_four);

    let high_one_two = f32::max(average_one, average_two);
    let high_three_four = f32::max(average_three, average_four);
    let high = f32::max(high_one_two, high_three_four);

    centre.clamp(low, high)
}

fn remove_grain_seventeen(window: &[f32; 9], centre: f32) -> f32 {
    let pairs = window_pairs(window);
    let lower_bound = pairs.iter().map(|(mi, _)| *mi).fold(f32::MIN, f32::max);
    let upper_bound = pairs.iter().map(|(_, ma)| *ma).fold(f32::MAX, f32::min);
    let low = f32::min(lower_bound, upper_bound);
    let high = f32::max(lower_bound, upper_bound);

    centre.clamp(low, high)
}

/// One pair's clamp and its cost, for [remove_grain_seven].
fn pair_cost(pair: (f32, f32), centre: f32) -> (f32, f32) {
    let (mi, ma) = pair;
    let distance = ma - mi;
    let clipped = centre.clamp(mi, ma);
    let cost = (centre - clipped).abs() + distance;

    (clipped, cost)
}

/// Seeds with pair 4, then walks pairs 2, 3 and 1, keeping the first strictly lowest cost.
fn remove_grain_seven(window: &[f32; 9], centre: f32) -> f32 {
    let pairs = window_pairs(window);
    let (clipped_four, cost_four) = pair_cost(pairs[3], centre);

    let mut chosen = clipped_four;
    let mut lowest_cost = cost_four;

    for pair_index in [1, 2, 0] {
        let (clipped, cost) = pair_cost(pairs[pair_index], centre);

        if cost < lowest_cost {
            chosen = clipped;
            lowest_cost = cost;
        }
    }

    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_clamps_the_clip_into_the_reference_neighbourhood() {
        // The centre's reference neighbourhood spans 10 to 18, so 40 clamps to 18 and 2 to 10.
        let reference: Vec<f32> = (10..19).map(|value| value as f32).collect();
        let mut high = vec![0.0; 9];
        high[4] = 40.0;
        let mut low = vec![0.0; 9];
        low[4] = 2.0;

        let raised = reference_repair(&high, &reference, 3, 3, RepairMode::One);
        let lowered = reference_repair(&low, &reference, 3, 3, RepairMode::One);

        assert_eq!(raised[4], 18.0);
        assert_eq!(lowered[4], 10.0);
    }

    #[test]
    fn repair_copies_the_border_of_the_clip() {
        let clip: Vec<f32> = (0..16).map(|value| value as f32 * 100.0).collect();
        let reference = vec![5.0; 16];

        let repaired = reference_repair(&clip, &reference, 4, 4, RepairMode::One);

        for index in [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15] {
            assert_eq!(repaired[index], clip[index], "sample {index}");
        }

        assert_eq!((repaired[5], repaired[10]), (5.0, 5.0));
    }

    #[test]
    fn repair_two_clamps_inside_the_second_extremes() {
        // Sorted with the centre, s0..s8 are 1..9 already, so s1 = 2 and s7 = 8.
        let reference: Vec<f32> = (1..=9).map(|value| value as f32).collect();
        let mut high = vec![0.0; 9];
        high[4] = 100.0;
        let mut low = vec![0.0; 9];
        low[4] = 0.0;

        let raised = reference_repair(&high, &reference, 3, 3, RepairMode::Two);
        let lowered = reference_repair(&low, &reference, 3, 3, RepairMode::Two);

        assert_eq!(raised[4], 8.0);
        assert_eq!(lowered[4], 2.0);
    }

    #[test]
    fn repair_thirteen_ignores_the_centre_rank() {
        // Neighbours sorted are 1 2 3 4 6 7 8 9, so s2 = 3 and s5 = 7, with c = 50.
        let reference = vec![1.0, 2.0, 3.0, 4.0, 50.0, 6.0, 7.0, 8.0, 9.0];
        let mut high = vec![0.0; 9];
        high[4] = 100.0;
        let mut low = vec![0.0; 9];
        low[4] = 0.0;

        let raised = reference_repair(&high, &reference, 3, 3, RepairMode::Thirteen);
        let lowered = reference_repair(&low, &reference, 3, 3, RepairMode::Thirteen);

        // max(s5, c) = max(7, 50) = 50, so 100 clamps down to 50.
        assert_eq!(raised[4], 50.0);
        // min(s2, c) = min(3, 50) = 3, so 0 clamps up to 3.
        assert_eq!(lowered[4], 3.0);
    }

    #[test]
    fn repair_thirteen_ties_follow_the_sort() {
        // Neighbours 5 5 5 5 5 5 9 9 sorted give s2 = 5 and s5 = 5, with c = 0.
        let reference = vec![5.0, 5.0, 5.0, 5.0, 0.0, 5.0, 5.0, 9.0, 9.0];
        let mut clip = vec![0.0; 9];
        clip[4] = 7.0;

        let repaired = reference_repair(&clip, &reference, 3, 3, RepairMode::Thirteen);

        // Bounds are min(5, 0) = 0 and max(5, 0) = 5, so 7 clamps down to 5.
        assert_eq!(repaired[4], 5.0);
    }

    #[test]
    fn repair_seventeen_uses_the_pair_bounds() {
        // Pairs (10, 30), (15, 25), (5, 40), (12, 28) give mi = 10 15 5 12 and ma = 30 25 40 28,
        // so l = max(mi) = 15 and u = min(ma) = 25. With c = 20, the bounds are min(15, 25, 20) =
        // 15 and max(15, 25, 20) = 25.
        let reference = vec![10.0, 15.0, 5.0, 12.0, 20.0, 28.0, 40.0, 25.0, 30.0];
        let mut high = vec![0.0; 9];
        high[4] = 100.0;
        let mut low = vec![0.0; 9];
        low[4] = 0.0;

        let raised = reference_repair(&high, &reference, 3, 3, RepairMode::Seventeen);
        let lowered = reference_repair(&low, &reference, 3, 3, RepairMode::Seventeen);

        assert_eq!(raised[4], 25.0);
        assert_eq!(lowered[4], 15.0);
    }

    #[test]
    fn repair_seventeen_pulls_the_bounds_towards_a_centre_outside_them() {
        // The same pairs give l = 15 and u = 25 as above, but here c = 5 sits below l. The bounds
        // become min(l, u, c) = min(15, 25, 5) = 5 and max(l, u, c) = max(15, 25, 5) = 25. A clip
        // of 8 sits inside 5..25, so it must pass through unchanged; dropping c from the bounds
        // and using only l and u would wrongly clamp it up to 15.
        let reference = vec![10.0, 15.0, 5.0, 12.0, 5.0, 28.0, 40.0, 25.0, 30.0];
        let mut clip = vec![0.0; 9];
        clip[4] = 8.0;

        let repaired = reference_repair(&clip, &reference, 3, 3, RepairMode::Seventeen);

        assert_eq!(repaired[4], 8.0);
    }

    #[test]
    fn repair_twenty_four_returns_the_clamped_reference_centre() {
        // Pairs (18, 22), (19, 21), (15, 25), (10, 30) with x = 20 give reaches 2, 1, 5, 10, so
        // u = min of those is 1. Bounds are x - u = 19 and x + u = 21, and c = 5 clamps up to 19.
        let reference = vec![18.0, 19.0, 15.0, 10.0, 5.0, 30.0, 25.0, 21.0, 22.0];
        let mut clip = vec![0.0; 9];
        clip[4] = 20.0;

        let repaired = reference_repair(&clip, &reference, 3, 3, RepairMode::TwentyFour);

        assert_eq!(repaired[4], 19.0);
    }

    #[test]
    fn remove_grain_seventeen_clamps_the_centre() {
        // Pairs (5, 25), (8, 30), (10, 20), (2, 40) give mi = 5 8 10 2 and ma = 25 30 20 40, so
        // l = max(mi) = 10 and u = min(ma) = 20. The spike centre of 200 clamps down to 20.
        let input = vec![5.0, 8.0, 10.0, 2.0, 200.0, 40.0, 20.0, 30.0, 25.0];

        let cleaned = reference_remove_grain(&input, 3, 3, RemoveGrainMode::Seventeen, GrainBorder::Copy);

        assert_eq!(cleaned[4], 20.0);
    }

    #[test]
    fn remove_grain_11_is_the_binomial() {
        // (4 * 5 + 2 * (2 + 4 + 6 + 8) + 1 + 3 + 7 + 9) / 16 = (20 + 40 + 20) / 16 = 5.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let cleaned = reference_remove_grain(&input, 3, 3, RemoveGrainMode::Eleven, GrainBorder::Copy);

        assert_eq!(cleaned[4], 5.0);

        // Every edge and corner is 0, so (4 * 16 + 0 + 0) / 16 = 4.
        let spike = vec![0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0];

        let cleaned_spike = reference_remove_grain(&spike, 3, 3, RemoveGrainMode::Eleven, GrainBorder::Copy);

        assert_eq!(cleaned_spike[4], 4.0);
    }

    #[test]
    fn remove_grain_22_clamps_between_the_pair_averages() {
        // Pair (a00, a22) averages (0 + 8) / 2 = 4, and the other three pairs average 0. The
        // centre 9 clamps down to 4.
        let input = vec![0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 8.0];

        let cleaned = reference_remove_grain(&input, 3, 3, RemoveGrainMode::TwentyTwo, GrainBorder::Copy);

        assert_eq!(cleaned[4], 4.0);

        // Every pair averages 2, so the centre becomes 2.
        let ring = vec![2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0];

        let cleaned_ring = reference_remove_grain(&ring, 3, 3, RemoveGrainMode::TwentyTwo, GrainBorder::Copy);

        assert_eq!(cleaned_ring[4], 2.0);
    }

    #[test]
    fn the_reflecting_border_filters_the_edge() {
        // Reflect folds column 0's neighbours onto column 1, and height 1 folds every row onto
        // row 0, so the window at x = 0 is 9 0 9 / 9 0 9 / 9 0 9 with centre 0. Mode 11 gives
        // (4 * 0 + 2 * (0 + 9 + 9 + 0) + 9 + 9 + 9 + 9) / 16 = (36 + 36) / 16 = 4.5.
        let input = vec![0.0, 9.0, 0.0];

        let reflected = reference_remove_grain(&input, 3, 1, RemoveGrainMode::Eleven, GrainBorder::Reflect);
        let copied = reference_remove_grain(&input, 3, 1, RemoveGrainMode::Eleven, GrainBorder::Copy);

        assert_eq!(reflected[0], 4.5);
        assert_eq!(copied[0], 0.0);
    }

    #[test]
    fn from_number_covers_the_supported_modes() {
        assert_eq!(RemoveGrainMode::from_number(7), Some(RemoveGrainMode::Seven));
        assert_eq!(RemoveGrainMode::from_number(11), Some(RemoveGrainMode::Eleven));
        assert_eq!(RemoveGrainMode::from_number(17), Some(RemoveGrainMode::Seventeen));
        assert_eq!(RemoveGrainMode::from_number(22), Some(RemoveGrainMode::TwentyTwo));

        assert_eq!(RemoveGrainMode::from_number(1), None);
        assert_eq!(RemoveGrainMode::from_number(12), None);
        assert_eq!(RemoveGrainMode::from_number(24), None);
    }

    #[test]
    fn remove_grain_seven_prefers_pair_four_on_a_tie() {
        // Pair 4 (40, 45) clips c = 50 to 45, cost |50 - 45| + 5 = 10.
        // Pair 2 (45, 55) clips c = 50 to 50, cost |50 - 50| + 10 = 10, tied with pair 4.
        // Pair 1 (0, 5) clips to 5, cost 45 + 5 = 50. Pair 3 (100, 110) clips to 100, cost 60.
        // The tie goes to pair 4, so the result is 45, not pair 2's 50.
        let input = vec![0.0, 45.0, 100.0, 40.0, 50.0, 45.0, 110.0, 55.0, 5.0];

        let cleaned = reference_remove_grain(&input, 3, 3, RemoveGrainMode::Seven, GrainBorder::Copy);

        assert_eq!(cleaned[4], 45.0);
    }

    #[test]
    fn remove_grain_copies_the_border() {
        let input: Vec<f32> = (0..16).map(|value| value as f32 * 3.0).collect();

        let seven = reference_remove_grain(&input, 4, 4, RemoveGrainMode::Seven, GrainBorder::Copy);
        let seventeen = reference_remove_grain(&input, 4, 4, RemoveGrainMode::Seventeen, GrainBorder::Copy);

        for index in [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15] {
            assert_eq!(seven[index], input[index], "seven sample {index}");
            assert_eq!(seventeen[index], input[index], "seventeen sample {index}");
        }
    }

    #[test]
    fn from_number_accepts_only_the_supported_modes() {
        assert_eq!(RepairMode::from_number(1), Some(RepairMode::One));
        assert_eq!(RepairMode::from_number(2), Some(RepairMode::Two));
        assert_eq!(RepairMode::from_number(13), Some(RepairMode::Thirteen));
        assert_eq!(RepairMode::from_number(17), Some(RepairMode::Seventeen));
        assert_eq!(RepairMode::from_number(24), Some(RepairMode::TwentyFour));

        assert_eq!(RepairMode::from_number(0), None);
        assert_eq!(RepairMode::from_number(3), None);
        assert_eq!(RepairMode::from_number(18), None);
        assert_eq!(RepairMode::from_number(25), None);
    }
}
