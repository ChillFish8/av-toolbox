// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{MotionField, assert_field};
use crate::DevicePlane;

/// The units that count one field's blocks between them.
const REDUCE_UNITS: u32 = 64;

/// How many halvings reduce `REDUCE_UNITS` partial counts to one.
const REDUCE_STEPS: u32 = REDUCE_UNITS.trailing_zeros();

/// MVTools' scene-change rule. A field with more than `thscd2` of every 255 blocks above `thscd1` is a
/// scene change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneChange {
    /// Per 8x8 block, on the 8-bit scale.
    pub thscd1: f32,
    pub thscd2: u32,
}

impl SceneChange {
    /// `thscd1` as a normalised SAD of a block `blksize` across.
    pub fn threshold(&self, blksize: u32) -> f32 {
        normalised_threshold(self.thscd1, blksize)
    }
}

/// An 8-bit, 8x8 block threshold as a normalised SAD of a block `blksize` across.
pub(crate) fn normalised_threshold(value: f32, blksize: u32) -> f32 {
    let samples = (blksize * blksize) as f32;

    value * samples / 64.0 / 255.0
}

/// One field's flag in a plane of `u32` flags, 1 where the field is a scene change.
#[derive(Debug, Clone, Copy)]
pub struct SceneFlag<'a> {
    pub flags: DevicePlane<'a>,
    pub index: u32,
}

/// Asserts that `flag` points inside its plane.
pub(crate) fn assert_flag(flag: &SceneFlag<'_>) {
    let held = flag.flags.samples();

    assert!(
        (flag.index as usize) < held,
        "the flag index is {}, but its plane holds {held}",
        flag.index
    );
}

/// Counts the blocks of one field above `threshold` and sets its flag if there are too many.
///
/// Every cube takes one field, and its units count a share of the blocks each before halving their
/// counts down to one in shared memory.
#[cube(launch_unchecked)]
fn scene_flag_kernel(
    sads: &Sequence<Array<f32>>,
    flags: &mut Array<u32>,
    threshold: f32,
    #[comptime] count: u32,
    #[comptime] blocks: u32,
    #[comptime] thscd2: u32,
) {
    let mut partial = SharedMemory::<u32>::new(comptime!(REDUCE_UNITS as usize));

    #[unroll]
    for field in 0..count {
        if CUBE_POS == field as usize {
            let mut changed = 0u32;
            let mut block = UNIT_POS;
            while block < blocks {
                if sads.index(field as usize)[block as usize] > threshold {
                    changed += 1u32;
                }

                block += REDUCE_UNITS;
            }

            partial[UNIT_POS as usize] = changed;
            sync_cube();

            #[unroll]
            for step in 1..=REDUCE_STEPS {
                let stride = comptime!(REDUCE_UNITS >> step);
                if UNIT_POS < stride {
                    let other = partial[(UNIT_POS + stride) as usize];
                    partial[UNIT_POS as usize] += other;
                }

                sync_cube();
            }

            if UNIT_POS == 0 {
                let total = partial[0usize];
                let mut flag = 0u32;
                if total * 255u32 > thscd2 * blocks {
                    flag = 1u32;
                }

                flags[field as usize] = flag;
            }
        }
    }
}

/// Writes one flag per field into `flags`, 1 where the field is a scene change under `rule`.
///
/// `flags` is a plane of `u32`, `fields.len()` wide and 1 high. A field is a scene change when more than
/// `rule.thscd2` of every 255 blocks have a SAD above `rule.thscd1`. There must be at least one field,
/// all on one grid.
pub fn scene_change_flags<R: Runtime>(
    client: &ComputeClient<R>,
    fields: &[MotionField<'_>],
    flags: DevicePlane<'_>,
    rule: SceneChange,
) {
    let first = fields
        .first()
        .expect("scene_change_flags needs at least one field");
    let grid = first.grid;
    assert!(
        rule.thscd2 <= 255,
        "thscd2 is {}, expected at most 255",
        rule.thscd2
    );
    assert!(
        grid.blocks() <= (u32::MAX / 255) as usize,
        "{} blocks per field are too many to count",
        grid.blocks()
    );
    crate::assert_plane(&flags, fields.len() as u32, 1, "flags");

    let mut sads = SequenceArg::<R, Array<f32>>::new();
    for field in fields {
        assert_field(field);
        assert_eq!(field.grid, grid, "the fields' grids differ");

        sads.push(field.planes.sads.array::<R>());
    }

    let threshold = rule.threshold(grid.blksize);
    let cubes = CubeCount::Static(fields.len() as u32, 1, 1);
    let unit = CubeDim::new_1d(REDUCE_UNITS);
    let flag_array = flags.array::<R>();

    // Safety: every SADs plane holds a SAD for each block of `grid`, `flags` holds a flag for each field,
    // and each cube writes only its own field's flag.
    unsafe {
        scene_flag_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            sads,
            flag_array,
            threshold,
            fields.len() as u32,
            grid.blocks() as u32,
            rule.thscd2,
        );
    }
}
