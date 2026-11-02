// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;

/// Adds the 3x3 taps of output `channel` around `column` and `row` to `sum`, in f32.
///
/// `input` holds `features` values per sample of a `region_width` by `region_height` region, and a tap past
/// its edge reads zero. `weights` are laid out by output channel, tap and feature.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the sum"
)]
pub(crate) fn add_taps<F: Float>(
    sum: f32,
    input: &[F],
    weights: &[F],
    column: i32,
    row: i32,
    region_width: u32,
    region_height: u32,
    channel: u32,
    features: u32,
) -> f32 {
    let mut total = sum;

    for tap in 0..9u32 {
        let source_column = column + (tap % 3) as i32 - 1;
        let source_row = row + (tap / 3) as i32 - 1;
        let inside = source_column >= 0
            && source_row >= 0
            && source_column < region_width as i32
            && source_row < region_height as i32;

        if inside {
            let source = (source_row as u32 * region_width + source_column as u32) * features;
            let tap_start = (channel * 9 + tap) * features;

            for feature in 0..features {
                let weight = f32::cast_from(weights[(tap_start + feature) as usize]);
                let value = f32::cast_from(input[(source + feature) as usize]);
                total += weight * value;
            }
        }
    }

    total
}
