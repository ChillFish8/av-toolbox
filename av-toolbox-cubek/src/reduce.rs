// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// How many values one unit folds together in each pass.
const CHUNK: u32 = 256;

/// How many f32 values each of a reduction's two scratch planes has to hold.
pub const fn scratch_len(samples: usize) -> usize {
    2 * samples.div_ceil(CHUNK as usize)
}

/// A plane to reduce, two scratch planes to pass through, and where the result lands.
#[derive(Debug, Clone, Copy)]
pub struct ReducePlanes<'a> {
    pub input: DevicePlane<'a>,
    pub first: DevicePlane<'a>,
    pub second: DevicePlane<'a>,
    pub result: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn extremes_of_plane(input: &Array<f32>, output: &mut Array<f32>, length: u32, chunk: u32) {
    let group = ABSOLUTE_POS as u32;
    let start = group * chunk;
    if start >= length {
        terminate!();
    }

    let end = u32::min(start + chunk, length);
    let mut low = input[start as usize];
    let mut high = low;
    for index in start + 1..end {
        let value = input[index as usize];
        low = f32::min(low, value);
        high = f32::max(high, value);
    }

    output[(2 * group) as usize] = low;
    output[(2 * group + 1) as usize] = high;
}

#[cube(launch_unchecked)]
fn extremes_of_pairs(input: &Array<f32>, output: &mut Array<f32>, pairs: u32, chunk: u32) {
    let group = ABSOLUTE_POS as u32;
    let start = group * chunk;
    if start >= pairs {
        terminate!();
    }

    let end = u32::min(start + chunk, pairs);
    let mut low = input[(2 * start) as usize];
    let mut high = input[(2 * start + 1) as usize];
    for index in start + 1..end {
        low = f32::min(low, input[(2 * index) as usize]);
        high = f32::max(high, input[(2 * index + 1) as usize]);
    }

    output[(2 * group) as usize] = low;
    output[(2 * group + 1) as usize] = high;
}

#[cube(launch_unchecked)]
fn sums_of_plane(input: &Array<f32>, output: &mut Array<f32>, length: u32, chunk: u32) {
    let group = ABSOLUTE_POS as u32;
    let start = group * chunk;
    if start >= length {
        terminate!();
    }

    let end = u32::min(start + chunk, length);
    let mut total = 0.0f32;
    let mut squares = 0.0f32;
    for index in start..end {
        let value = input[index as usize];
        total += value;
        squares += value * value;
    }

    output[(2 * group) as usize] = total;
    output[(2 * group + 1) as usize] = squares;
}

#[cube(launch_unchecked)]
fn sums_of_pairs(input: &Array<f32>, output: &mut Array<f32>, pairs: u32, chunk: u32) {
    let group = ABSOLUTE_POS as u32;
    let start = group * chunk;
    if start >= pairs {
        terminate!();
    }

    let end = u32::min(start + chunk, pairs);
    let mut total = 0.0f32;
    let mut squares = 0.0f32;
    for index in start..end {
        total += input[(2 * index) as usize];
        squares += input[(2 * index + 1) as usize];
    }

    output[(2 * group) as usize] = total;
    output[(2 * group + 1) as usize] = squares;
}

#[derive(Debug, Clone, Copy)]
enum Statistic {
    Extremes,
    Sums,
}

/// Folds a plane into pairs, then pairs into fewer pairs, until one pair is left in the result.
fn reduce<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ReducePlanes<'_>,
    statistic: Statistic,
) {
    let length = planes.input.samples() as u32;
    let mut groups = length.div_ceil(CHUNK);

    let first_target = if groups == 1 { planes.result } else { planes.first };
    let input = planes.input.array::<R>();
    let output = first_target.array::<R>();
    let cubes = dispatch.cube_count(groups as usize);
    let shape = dispatch.cube_dim();

    // Safety: the input holds `length` f32 values and the target at least `2 * groups`, the most
    // the kernel writes.
    unsafe {
        match statistic {
            Statistic::Extremes => {
                extremes_of_plane::launch_unchecked::<R>(client, cubes, shape, input, output, length, CHUNK);
            },
            Statistic::Sums => {
                sums_of_plane::launch_unchecked::<R>(client, cubes, shape, input, output, length, CHUNK);
            },
        }
    }

    let mut current = planes.first;
    let mut spare = planes.second;

    while groups > 1 {
        let next = groups.div_ceil(CHUNK);
        let target = if next == 1 { planes.result } else { spare };

        let input = current.array::<R>();
        let output = target.array::<R>();
        let cubes = dispatch.cube_count(next as usize);
        let shape = dispatch.cube_dim();

        // Safety: `current` holds `2 * groups` f32 values and the target at least `2 * next`.
        unsafe {
            match statistic {
                Statistic::Extremes => {
                    extremes_of_pairs::launch_unchecked::<R>(
                        client, cubes, shape, input, output, groups, CHUNK,
                    );
                },
                Statistic::Sums => {
                    sums_of_pairs::launch_unchecked::<R>(client, cubes, shape, input, output, groups, CHUNK);
                },
            }
        }

        std::mem::swap(&mut current, &mut spare);
        groups = next;
    }
}

/// Writes the smallest and largest value of `input` to the result, in that order.
pub fn extremes<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, planes: ReducePlanes<'_>) {
    reduce(client, dispatch, planes, Statistic::Extremes);
}

/// Writes the sum and the sum of squares of `input` to the result, in that order.
pub fn sums<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, planes: ReducePlanes<'_>) {
    reduce(client, dispatch, planes, Statistic::Sums);
}
