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
