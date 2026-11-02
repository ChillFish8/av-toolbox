// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// Stretches the TV range of one sample to between 0 and 1, scaled by `256 / 255`.
#[cube(launch_unchecked)]
fn normalise_kernel(
    input: &[f32],
    output: &mut [f32],
    black: f32,
    inverse_range: f32,
    stretch: f32,
    #[comptime] samples: u32,
) {
    if ABSOLUTE_POS >= samples as usize {
        terminate!();
    }

    let stretched = (input[ABSOLUTE_POS] - black) * inverse_range;
    let clamped = f32::clamp(stretched, 0.0f32, 1.0f32);

    output[ABSOLUTE_POS] = clamped * stretch;
}

/// Stretches TV-range luma code values to between 0 and 1, as `DitherLumaRebuild(s0 = 1)` and a division by
/// the peak do.
///
/// `input` holds code values of `bits` bits as f32. Values under black or over white clamp to the ends.
/// The ends are scaled by `256 / 255`, so white is a little over 1. It multiplies by the reciprocal of the
/// range, as devices divide with different rounding, so white can sit an ulp off `256 / 255`.
pub fn normalise(
    client: &Client,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    bits: u32,
) {
    assert!((8..=16).contains(&bits), "bits is {bits}, expected 8..=16");

    let width = input.width();
    let height = input.height();
    crate::assert_plane(&input, width, height, "input");
    crate::assert_plane(&output, width, height, "output");

    let scale = (1u32 << (bits - 8)) as f32;
    let black = 16.0 * scale;
    let inverse_range = 1.0 / (219.0 * scale);
    let stretch = 256.0f32 / 255.0;

    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let input_array = input.array();
    let output_array = output.array();

    // Safety: both planes hold `width` by `height` f32 samples, and each unit writes only its own sample.
    unsafe {
        normalise_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            black,
            inverse_range,
            stretch,
            input.samples() as u32,
        );
    }
}
