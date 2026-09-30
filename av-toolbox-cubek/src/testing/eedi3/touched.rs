use std::fmt::Display;

use super::path::PathTables;

/// How close, relative to its size, a value must come to a decision's boundary for fused rounding to
/// be able to cross it.
const CLOSE_MARGIN: f64 = 1.0 / (1 << 20) as f64;

/// The largest share of compared samples that may be touched, as a fraction `numerator / denominator`.
const TOUCHED_BOUND: (usize, usize) = (2, 100);

/// The largest share of compared samples that may differ, as a fraction `numerator / denominator`.
const DIFFERING_BOUND: (usize, usize) = (5, 10_000);

/// Marks the columns whose direction a close path step can change.
///
/// The backtrack is followed with every state it could be in, adding each candidate that came within
/// the margin of the one that won. A column is touched when it could be in more than one state. An
/// inactive column decides nothing and only follows its backpointers.
pub(super) fn close_directions(tables: &PathTables, active: &[bool], width: i32, mdis: i32) -> Vec<bool> {
    let span = (2 * mdis + 1) as usize;

    let mut states = vec![false; span];
    states[mdis as usize] = true;
    let mut touched = vec![false; width as usize];
    for column in (0..width - 1).rev() {
        let mut next = vec![false; span];
        for direction in -mdis..=mdis {
            if !states[(direction + mdis) as usize] {
                continue;
            }

            let chosen = tables.back.get(column, direction);
            next[(chosen + mdis) as usize] = true;
            if !active[column as usize + 1] {
                continue;
            }

            let candidates = tables.candidates.get(column, direction);
            let winner_slot = chosen - direction + 1;
            let Some(best) = candidates.get(winner_slot as usize) else {
                continue;
            };
            let best = f64::from(*best);

            for (slot, candidate) in candidates.iter().enumerate() {
                let candidate = f64::from(*candidate);
                let close = candidate != f64::INFINITY && candidate - best <= CLOSE_MARGIN * best;
                if close {
                    let previous = direction + slot as i32 - 1;
                    next[(previous + mdis) as usize] = true;
                }
            }
        }

        let reachable = next.iter().filter(|state| **state).count();
        touched[column as usize] = reachable > 1;
        states = next;
    }

    touched
}

/// Whether `value` is not exactly an f32.
fn inexact(value: f64) -> bool {
    let rounded = value as f32;

    f64::from(rounded) != value
}

/// Whether fusing the blend `keep * sample + weight * fallback` can move its truncation to another
/// integer.
///
/// When both products are exact in f32, fusing cannot change the sum.
pub(super) fn blend_is_close(keep: f32, sample: i32, weight: f32, fallback: i32) -> bool {
    let kept_part = f64::from(keep) * f64::from(sample);
    let fallback_part = f64::from(weight) * f64::from(fallback);
    if !inexact(kept_part) && !inexact(fallback_part) {
        return false;
    }

    let value = kept_part + fallback_part;
    let distance = (value - value.round()).abs();
    distance <= CLOSE_MARGIN * value.max(1.0)
}

/// Counts how the planes a port computed compare with the reference's, over any number of planes.
///
/// Every sample that differs must be one the reference marks as touched. Over everything compared, at
/// most 2 % may be touched and at most 0.05 % may differ. Direction samples count toward the same totals
/// as output samples.
#[derive(Debug, Default)]
pub struct CloseDecisionTally {
    compared: usize,
    touched: usize,
    differing: usize,
}

impl CloseDecisionTally {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a plane `width` samples wide, panicking at the first difference at an untouched sample.
    pub fn compare(&mut self, label: &str, found: &[f32], expected: &[f32], touched: &[bool], width: u32) {
        self.compare_values(label, found, expected, touched, width);
    }

    /// Adds a direction map `width` samples wide, panicking at the first difference at an untouched
    /// direction.
    pub fn compare_directions(
        &mut self,
        label: &str,
        found: &[i32],
        expected: &[i32],
        touched: &[bool],
        width: u32,
    ) {
        self.compare_values(label, found, expected, touched, width);
    }

    fn compare_values<T: PartialEq + Display>(
        &mut self,
        label: &str,
        found: &[T],
        expected: &[T],
        touched: &[bool],
        width: u32,
    ) {
        assert_eq!(found.len(), expected.len(), "{label}: the planes differ in size");
        assert_eq!(
            touched.len(),
            expected.len(),
            "{label}: the touched set differs in size"
        );

        let columns = width as usize;
        for (index, (found_value, expected_value)) in found.iter().zip(expected).enumerate() {
            self.compared += 1;
            if touched[index] {
                self.touched += 1;
            }

            if found_value == expected_value {
                continue;
            }

            let row = index / columns;
            let column = index % columns;
            assert!(
                touched[index],
                "{label}: row {row}, column {column} differs at an untouched sample, found {found_value} and \
                 expected {expected_value}"
            );
            self.differing += 1;
        }
    }

    /// Panics when more than 2 % of the compared samples are touched or more than 0.05 % differ.
    pub fn assert_bounds(&self, label: &str) {
        let too_many_touched = self.touched * TOUCHED_BOUND.1 > self.compared * TOUCHED_BOUND.0;
        let too_many_differing = self.differing * DIFFERING_BOUND.1 > self.compared * DIFFERING_BOUND.0;

        assert!(
            !too_many_touched && !too_many_differing,
            "{label}: {} of {} compared samples are touched and {} differ, past the 2 % touched or 0.05 % \
             differing allowed",
            self.touched,
            self.compared,
            self.differing
        );
    }
}
