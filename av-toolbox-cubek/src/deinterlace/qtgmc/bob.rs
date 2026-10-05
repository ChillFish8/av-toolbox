// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::resize::{Kernel, Taps, Weights, taps};
use crate::{DevicePlane, Dispatch, assert_plane};

/// Which rows of a frame a field holds, 0 for the top field and 1 for the bottom.
pub type Parity = u32;

/// The vertical taps that bob a field of `field_height` rows of `parity` to `2 * field_height` rows.
///
/// Output row `y` sits at field position `(y - parity) / 2`, as fmtconv places a field it resizes to a
/// frame. Taps past either end of the field are for the caller to clamp.
pub fn bob_taps(field_height: u32, parity: Parity) -> Taps {
    // fmtconv's `ResampleUtil.cpp` sets `cp_s_v` to 0.25 for a top field and 0.75 for a bottom field,
    // and `cp_d_v` to 0.5 for the frame it makes, for luma and for 4:2:0 `mpeg2` chroma alike.
    // `Scaler.cpp` then starts at `_src_step * center_pos_dst - 1 * center_pos_src`, which with a step
    // of 0.5 is 0 for the top field and -0.5 for the bottom.
    let offset = -(parity as f32) / 2.0;
    let kernel = Kernel::Bicubic { b: 0.0, c: 0.5 };

    taps(2 * field_height, 0.5, offset, &kernel)
}

/// The planes one bob reads and writes.
pub struct BobPlanes<'a> {
    /// The whole frame, both fields.
    pub frame: DevicePlane<'a>,
    /// The bobbed field, as high as the frame.
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn bob_kernel(
    frame: &Array<f32>,
    starts: &Array<i32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    field_height: u32,
    count: u32,
    parity: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;
    let start = starts[row as usize];
    let last = field_height as i32 - 1;

    let mut total = 0.0f32;
    for step in 0..count {
        let field_row = i32::clamp(start + step as i32, 0, last);
        let frame_row = field_row as u32 * 2 + parity;
        let sample = frame[(frame_row * width + column) as usize];
        let weight = weights[(row * count + step) as usize];
        total += sample * weight;
    }

    output[ABSOLUTE_POS] = total;
}

/// Bobs the field of `parity` in `frame` to a full-height plane.
///
/// `down` is [bob_taps] of the same parity. Rows past the field's ends read its first and last rows.
pub fn bob<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: BobPlanes<'_>,
    down: &Weights,
    parity: Parity,
) {
    let width = planes.frame.width();
    let height = planes.frame.height();
    assert!(
        height.is_multiple_of(2) && height >= 2,
        "the frame is {height} rows, expected an even count"
    );
    assert!(parity < 2, "the parity is {parity}, expected 0 or 1");
    assert_plane(&planes.frame, width, height, "frame");
    assert_plane(&planes.output, width, height, "output");

    let frame = planes.frame.array::<R>();
    let starts = down.starts::<R>();
    let weights = down.weights::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(planes.output.samples());
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `width` by `height` f32 samples, the tables `height` starts and
    // `height * count` weights, and every tap is clamped to a row of the field before it is read.
    unsafe {
        bob_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            frame,
            starts,
            weights,
            output,
            width,
            height / 2,
            down.count,
            parity,
        );
    }
}
