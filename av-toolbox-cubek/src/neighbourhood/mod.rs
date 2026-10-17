mod chain;
mod median;
mod min_blur;
mod prewitt;
#[cfg(test)]
mod tests;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

pub use self::chain::{ChainPlanes, Step, chain, chain_with_edge};
pub use self::median::median;
pub use self::min_blur::{MinBlurPlanes, median_of_three, median_of_three_sample, min_blur};
pub use self::prewitt::prewitt;
pub use crate::Edge;
pub use crate::blur::Smooth;
use crate::blur::{self, FilterPlanes, Storage};
use crate::{DevicePlane, Dispatch, assert_plane, reflect};

/// The plane one neighbourhood operation reads and the plane it writes, both f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct NeighbourPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Extreme {
    Maximum,
    Minimum,
}

/// Which of the 8 neighbours an extreme looks at. The centre always counts.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Shape {
    /// All 8.
    Square,
    /// Top, left, right and bottom.
    Plus,
    /// Left and right.
    Horizontal,
    /// Top and bottom.
    Vertical,
    /// Top left, top right, bottom left and bottom right.
    Corners,
}

impl Shape {
    fn code(self) -> u32 {
        match self {
            Self::Square => 0,
            Self::Plus => 1,
            Self::Horizontal => 2,
            Self::Vertical => 3,
            Self::Corners => 4,
        }
    }
}

/// Moves a sample toward the mean of its 8 neighbours, as havsfunc's `std.Inflate`/`std.Deflate` do.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Spread {
    /// Replaces the sample with the mean when the mean is larger.
    Inflate,
    /// Replaces the sample with the mean when the mean is smaller.
    Deflate,
}

/// How a run of maximum or minimum passes grows a region, as havsfunc's `mt_expand_multi` does.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Expansion {
    Rectangle,
    Ellipse,
}

/// The shape of each pass that grows a region by `horizontal` and `vertical` steps.
///
/// While both axes have steps left a pass is square, or for an ellipse plus-shaped unless the
/// horizontal steps left leave 1 over a multiple of 3. Once only one axis has steps left each pass
/// moves along that axis.
pub fn expansion_passes(horizontal: u32, vertical: u32, expansion: Expansion) -> Vec<Shape> {
    let mut passes = Vec::new();
    let mut across = horizontal;
    let mut down = vertical;

    while across > 0 || down > 0 {
        let shape = if across > 0 && down > 0 {
            let plus = expansion == Expansion::Ellipse && across % 3 != 1;
            if plus { Shape::Plus } else { Shape::Square }
        } else if across > 0 {
            Shape::Horizontal
        } else {
            Shape::Vertical
        };

        passes.push(shape);
        across = across.saturating_sub(1);
        down = down.saturating_sub(1);
    }

    passes
}

#[cube]
fn read(input: &Array<f32>, width: u32, height: u32, column: i32, row: i32) -> f32 {
    let x = reflect(column, width as i32);
    let y = reflect(row, height as i32);
    let index = y as u32 * width + x as u32;

    input[index as usize]
}

#[cube]
fn read_edge(input: &Array<f32>, width: u32, height: u32, column: i32, row: i32, clamp: u32) -> f32 {
    let x = if clamp == 1u32 {
        i32::clamp(column, 0, width as i32 - 1)
    } else {
        reflect(column, width as i32)
    };
    let y = if clamp == 1u32 {
        i32::clamp(row, 0, height as i32 - 1)
    } else {
        reflect(row, height as i32)
    };
    let index = y as u32 * width + x as u32;

    input[index as usize]
}

#[cube]
fn in_shape(shape: u32, offset_x: i32, offset_y: i32) -> bool {
    let square = shape == 0u32;
    let plus = shape == 1u32 && (offset_x == 0 || offset_y == 0);
    let horizontal = shape == 2u32 && offset_y == 0;
    let vertical = shape == 3u32 && offset_x == 0;
    let corners = shape == 4u32 && offset_x != 0 && offset_y != 0;

    square || plus || horizontal || vertical || corners
}

#[cube(launch_unchecked)]
fn extreme_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    shape: u32,
    maximum: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;
    let mut best = input[ABSOLUTE_POS];

    for step_y in 0..3u32 {
        for step_x in 0..3u32 {
            let offset_x = step_x as i32 - 1;
            let offset_y = step_y as i32 - 1;

            if in_shape(shape, offset_x, offset_y) {
                let value = read(input, width, height, x + offset_x, y + offset_y);

                if maximum == 1u32 {
                    best = f32::max(best, value);
                } else {
                    best = f32::min(best, value);
                }
            }
        }
    }

    output[ABSOLUTE_POS] = best;
}

#[cube(launch_unchecked)]
fn spread_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    inflate: u32,
    clamp: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;
    let centre = input[ABSOLUTE_POS];
    let mut total = 0.0f32;

    for step_y in 0..3u32 {
        for step_x in 0..3u32 {
            let offset_x = step_x as i32 - 1;
            let offset_y = step_y as i32 - 1;

            if offset_x != 0 || offset_y != 0 {
                total += read_edge(input, width, height, x + offset_x, y + offset_y, clamp);
            }
        }
    }

    let mean = total / 8.0f32;
    let mut picked = f32::max(mean, centre);
    if inflate == 0u32 {
        picked = f32::min(mean, centre);
    }

    output[ABSOLUTE_POS] = picked;
}

/// Writes the largest or smallest value among each sample and its neighbours in `shape`.
pub fn extreme<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    extreme: Extreme,
    shape: Shape,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    assert_plane(&planes.input, width, height, "input");
    assert_plane(&planes.output, width, height, "output");
    let samples = planes.output.samples();
    let shape_code = shape.code();
    let maximum = u32::from(extreme == Extreme::Maximum);
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `width` by `height` f32 samples, and every read is reflected inside the
    // plane.
    unsafe {
        extreme_kernel::launch_unchecked::<R>(
            client, cubes, unit, input, output, width, height, shape_code, maximum,
        );
    }
}

/// Blurs a plane with a 3x3 kernel.
pub fn smooth<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    smooth: Smooth,
) {
    smooth_with_edge(client, dispatch, planes, smooth, Edge::Reflect);
}

/// The same as [smooth], reading past the plane's edge as `edge` says.
pub fn smooth_with_edge<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    smooth: Smooth,
    edge: Edge,
) {
    let filter_planes = FilterPlanes {
        input: planes.input,
        output: planes.output,
    };

    blur::smooth(client, dispatch, filter_planes, smooth, edge, Storage::F32);
}

/// Moves each sample to the mean of its 8 neighbours when the mean is larger for inflate, or
/// smaller for deflate.
pub fn spread<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    spread: Spread,
) {
    spread_with_edge(client, dispatch, planes, spread, Edge::Reflect);
}

/// The same as [spread], reading past the plane's edge as `edge` says.
pub fn spread_with_edge<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    spread: Spread,
    edge: Edge,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    assert_plane(&planes.input, width, height, "input");
    assert_plane(&planes.output, width, height, "output");
    let samples = planes.output.samples();
    let inflate = u32::from(spread == Spread::Inflate);
    let clamp = u32::from(edge == Edge::Clamp);
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `width` by `height` f32 samples, and every read is folded inside the
    // plane.
    unsafe {
        spread_kernel::launch_unchecked::<R>(
            client, cubes, unit, input, output, width, height, inflate, clamp,
        );
    }
}
