// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

#[cube(launch_unchecked)]
fn transpose_kernel(input: &Array<f32>, output: &mut Array<f32>, input_width: u32, input_height: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % input_height;
    let row = position / input_height;
    output[ABSOLUTE_POS] = input[(column * input_width + row) as usize];
}

/// Swaps the axes of `input`, so `output[x][y] = input[y][x]`.
///
/// The output is `input.height()` wide and `input.width()` tall.
pub fn transpose<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let input_width = input.width();
    let input_height = input.height();

    assert!(
        output.width() == input_height,
        "the output width {} must equal the input height {input_height}",
        output.width(),
    );
    assert!(
        output.height() == input_width,
        "the output height {} must equal the input width {input_width}",
        output.height(),
    );

    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the output holds its own samples, and every read lands inside the input because the
    // asserted sizes make each output position map to an input position.
    unsafe {
        transpose_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            input_width,
            input_height,
        );
    }
}
