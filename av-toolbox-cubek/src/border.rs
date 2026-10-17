// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

/// Folds `index` back into a row of `length` samples without repeating the edge sample.
///
/// So `-1` reads `1` and `length` reads `length - 2`. An index any distance out folds back and
/// forth until it lands inside.
#[cube]
pub fn reflect(index: i32, length: i32) -> i32 {
    let mut reflected = 0i32;

    if length > 1 {
        let period = 2 * length - 2;
        let mut folded = index;
        if folded < 0 {
            folded = -folded;
        }

        folded %= period;
        if folded >= length {
            folded = period - folded;
        }

        reflected = folded;
    }

    reflected
}

/// Folds `index` back into a row of `length` samples, repeating the edge sample.
///
/// So `-1` reads `0` and `length` reads `length - 1`. An index any distance out folds back and
/// forth until it lands inside.
#[cube]
pub fn mirror(index: i32, length: i32) -> i32 {
    let period = 2 * length;
    let mut folded = index;
    if folded < 0 {
        folded = -folded - 1;
    }

    folded %= period;
    if folded >= length {
        folded = period - 1 - folded;
    }

    folded
}

/// Folds `index` into `0..length`, reflecting or clamping it when it falls outside.
#[cube]
pub(crate) fn fold(index: i32, length: u32, #[comptime] clamp: bool) -> u32 {
    let last = length as i32 - 1;
    let mut folded = index;

    if index < 0 || index > last {
        if clamp {
            folded = i32::clamp(index, 0, last);
        } else {
            folded = reflect(index, length as i32);
        }
    }

    folded as u32
}

/// What a pass reads past the edge of a plane.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Edge {
    /// Folds back without repeating the edge sample.
    Reflect,
    /// Repeats the edge sample.
    Clamp,
}
