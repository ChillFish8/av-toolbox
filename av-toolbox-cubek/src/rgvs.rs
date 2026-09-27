// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// Which RemoveGrain Repair mode runs.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RepairMode {
    /// Clamps the clip between the least and greatest of the reference's 3x3 neighbourhood.
    One,
}

/// The planes one repair reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct RepairPlanes<'a> {
    pub clip: DevicePlane<'a>,
    pub reference: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn repair_one_kernel(
    clip: &Array<f32>,
    reference: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % width;
    let y = position / width;
    let value = clip[ABSOLUTE_POS];

    if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
        output[ABSOLUTE_POS] = value;
    } else {
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
    }
}

/// Repairs `clip` from `reference` with RemoveGrain's `mode`.
///
/// The first and last row and column of the clip are copied unchanged, as rgvs does.
pub fn repair<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: RepairPlanes<'_>,
    mode: RepairMode,
) {
    let width = planes.clip.width();
    let height = planes.clip.height();
    let samples = planes.output.samples();
    let clip = planes.clip.array::<R>();
    let reference = planes.reference.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    match mode {
        // Safety: every plane holds `samples` f32 values, and only inner samples read neighbours.
        RepairMode::One => unsafe {
            repair_one_kernel::launch_unchecked::<R>(
                client, cubes, unit, clip, reference, output, width, height,
            );
        },
    }
}
