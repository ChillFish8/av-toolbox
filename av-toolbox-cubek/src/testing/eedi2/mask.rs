use super::{Grid, Scaled};

/// Whether the three field rows around `row` stay within `ten` of each other at `column`.
fn is_flat(field: &Grid, row: i64, column: i64, ten: i64) -> bool {
    let above = field.at(row - 1, column);
    let centre = field.at(row, column);
    let below = field.at(row + 1, column);

    (above - centre).abs() < ten && (centre - below).abs() < ten && (above - below).abs() < ten
}

/// Whether the 3 by 3 block around a pixel varies too little to hold an edge.
///
/// The sum shifts after summing while each square shifts first, so above 8 bits the variance can be
/// negative. The plugin's unsigned arithmetic wraps that to a huge value, so a negative variance is not low.
fn has_low_variance(field: &Grid, row: i64, column: i64, scaled: &Scaled) -> bool {
    let mut sum = 0;
    let mut sum_of_squares = 0;
    for row_offset in -1..=1 {
        for column_offset in -1..=1 {
            let sample = field.at(row + row_offset, column + column_offset);
            let reduced = sample >> scaled.shift;
            sum += sample;
            sum_of_squares += reduced * reduced;
        }
    }

    let reduced_sum = sum >> scaled.shift;
    let variance = 9 * sum_of_squares - reduced_sum * reduced_sum;
    (0..scaled.vthresh_81).contains(&variance)
}

/// Pass 1's test for one interior pixel.
fn is_edge(field: &Grid, row: i64, column: i64, scaled: &Scaled) -> bool {
    let flat_here = is_flat(field, row, column, scaled.ten);
    let flat_before = is_flat(field, row, column - 1, scaled.ten);
    let flat_after = is_flat(field, row, column + 1, scaled.ten);
    if flat_here || (flat_before && flat_after) {
        return false;
    }

    if has_low_variance(field, row, column, scaled) {
        return false;
    }

    let above = field.at(row - 1, column);
    let centre = field.at(row, column);
    let below = field.at(row + 1, column);
    let before = field.at(row, column - 1);
    let after = field.at(row, column + 1);

    let horizontal = (after - before).abs() >> scaled.shift;
    let vertical_step = (above - below)
        .abs()
        .max((above - centre).abs())
        .max((centre - below).abs());
    let vertical = vertical_step >> scaled.shift;
    if horizontal * horizontal + vertical * vertical >= scaled.mthresh_squared {
        return true;
    }

    let horizontal_curve = (before - 2 * centre + after).abs() >> scaled.shift;
    let vertical_curve = (above - 2 * centre + below).abs() >> scaled.shift;
    horizontal_curve + vertical_curve >= scaled.lthresh
}

/// Pass 1, the raw edge mask `E0`. The border stays clear.
pub(super) fn edge_mask(field: &Grid, scaled: &Scaled) -> Grid {
    let rows = field.rows();
    let width = field.width;

    let mut mask = Grid::filled(rows, width, 0);
    for row in 1..rows - 1 {
        for column in 1..width - 1 {
            let edge = is_edge(field, row, column, scaled);
            let value = i64::from(edge);
            mask.set(row, column, value);
        }
    }

    mask
}

/// The set pixels among the 8 neighbours of an interior pixel.
fn neighbours_set(mask: &Grid, row: i64, column: i64) -> i64 {
    let mut count = 0;
    for row_offset in -1..=1 {
        for column_offset in -1..=1 {
            let is_centre = row_offset == 0 && column_offset == 0;
            if !is_centre {
                count += mask.at(row + row_offset, column + column_offset);
            }
        }
    }

    count
}

/// Passes 2 and 4, clearing interior pixels with fewer than `strength` set neighbours.
pub(super) fn erode(mask: &Grid, strength: i64) -> Grid {
    let mut eroded = mask.clone();
    for row in 1..mask.rows() - 1 {
        for column in 1..mask.width - 1 {
            let is_set = mask.at(row, column) == 1;
            if is_set && neighbours_set(mask, row, column) < strength {
                eroded.set(row, column, 0);
            }
        }
    }

    eroded
}

/// Pass 3, setting interior pixels with at least `strength` set neighbours.
pub(super) fn dilate(mask: &Grid, strength: i64) -> Grid {
    let mut dilated = mask.clone();
    for row in 1..mask.rows() - 1 {
        for column in 1..mask.width - 1 {
            let is_clear = mask.at(row, column) == 0;
            if is_clear && neighbours_set(mask, row, column) >= strength {
                dilated.set(row, column, 1);
            }
        }
    }

    dilated
}

/// Pass 5, the final edge mask `E`, closing gaps of up to 3 pixels in a row and dropping lone pixels.
pub(super) fn remove_small_gaps(mask: &Grid) -> Grid {
    let mut closed = mask.clone();
    for row in 1..mask.rows() - 1 {
        for column in 3..mask.width - 3 {
            let set = |offset: i64| mask.at(row, column + offset) == 1;

            if set(0) {
                let alone = !(set(-3) || set(-2) || set(-1) || set(1) || set(2) || set(3));
                if alone {
                    closed.set(row, column, 0);
                }
            } else {
                let bridged = (set(1) && (set(-1) || set(-2) || set(-3)))
                    || (set(2) && (set(-1) || set(-2)))
                    || (set(3) && set(-1));
                if bridged {
                    closed.set(row, column, 1);
                }
            }
        }
    }

    closed
}
