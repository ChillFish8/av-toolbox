use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

/// The size of pyramid `level` for a frame `width` by `height`, each level half the one before and
/// never under 1.
pub fn level_dims(width: u32, height: u32, level: u32) -> (u32, u32) {
    let mut level_width = width;
    let mut level_height = height;
    for _ in 0..level {
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }

    (level_width, level_height)
}

/// One frame's luma at every level, each `f32` between 0 and 1.
///
/// Level 0 is the normalised luma itself, and each level after it is sized by [level_dims].
pub struct Pyramid<'a> {
    pub levels: Vec<DevicePlane<'a>>,
}

impl Pyramid<'_> {
    pub fn width(&self) -> u32 {
        self.levels[0].width()
    }

    pub fn height(&self) -> u32 {
        self.levels[0].height()
    }
}

#[cube(launch_unchecked)]
fn downscale_kernel(
    source: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] source_width: u32,
    #[comptime] source_height: u32,
    #[comptime] width: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % width;
    let y = position / width;

    let sx = x * 2;
    let sy = y * 2;
    let sx1 = if sx + 1 < source_width { sx + 1 } else { sx };
    let sy1 = if sy + 1 < source_height { sy + 1 } else { sy };

    let s00 = source[(sy * source_width + sx) as usize];
    let s10 = source[(sy * source_width + sx1) as usize];
    let s01 = source[(sy1 * source_width + sx) as usize];
    let s11 = source[(sy1 * source_width + sx1) as usize];

    let average = (s00 + s10 + s01 + s11) * 0.25f32;
    output[ABSOLUTE_POS] = average;
}

/// Fills every level of `pyramid` after the first from the level before it.
///
/// Each sample is the mean of a 2x2 group, the last row or column repeated where the group runs past the
/// edge.
pub fn build_pyramid<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, pyramid: &Pyramid<'_>) {
    let width = pyramid.width();
    let height = pyramid.height();
    for (level, plane) in pyramid.levels.iter().enumerate() {
        let (level_width, level_height) = level_dims(width, height, level as u32);
        let name = format!("pyramid level {level}");
        assert_plane(plane, level_width, level_height, &name);
    }

    for pair in pyramid.levels.windows(2) {
        let source = &pair[0];
        let output = &pair[1];

        let cubes = dispatch.cube_count(output.samples());
        let unit = dispatch.cube_dim();
        let source_array = source.array::<R>();
        let output_array = output.array::<R>();

        // Safety: each level holds its own size of f32 values, and each unit reads a 2x2 group of the
        // level before, clamped to it, and writes its own sample.
        unsafe {
            downscale_kernel::launch_unchecked::<R>(
                client,
                cubes,
                unit,
                source_array,
                output_array,
                source.width(),
                source.height(),
                output.width(),
            );
        }
    }
}
