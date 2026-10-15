// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::labels::{CountPlanes, KeepPlanes, LabelPlanes, count, keep, label};
use crate::{DevicePlane, Dispatch};

/// Every plane [hysteresis] touches, all one size.
#[derive(Debug, Clone, Copy)]
pub struct HysteresisPlanes<'a> {
    pub seeds_from: DevicePlane<'a>,
    pub region: DevicePlane<'a>,
    pub marks: DevicePlane<'a>,
    pub seeds: DevicePlane<'a>,
    pub labels: DevicePlane<'a>,
    pub counts: DevicePlane<'a>,
    pub kept: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn mark_region(region: &Array<f32>, marks: &mut Array<u8>) {
    if ABSOLUTE_POS >= marks.len() {
        terminate!();
    }

    if region[ABSOLUTE_POS] > 0.0f32 {
        marks[ABSOLUTE_POS] = u8::cast_from(255u32);
    } else {
        marks[ABSOLUTE_POS] = u8::cast_from(0u32);
    }
}

#[cube(launch_unchecked)]
fn mark_seeds(seeds_from: &Array<f32>, region: &Array<f32>, seeds: &mut Array<u8>) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }

    if seeds_from[ABSOLUTE_POS] > 0.0f32 && region[ABSOLUTE_POS] > 0.0f32 {
        seeds[ABSOLUTE_POS] = u8::cast_from(255u32);
    } else {
        seeds[ABSOLUTE_POS] = u8::cast_from(0u32);
    }
}

#[cube(launch_unchecked)]
fn spread_kept(kept: &Array<u8>, output: &mut Array<f32>, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    if u32::cast_from(kept[ABSOLUTE_POS]) != 0u32 {
        output[ABSOLUTE_POS] = peak;
    } else {
        output[ABSOLUTE_POS] = 0.0f32;
    }
}

/// Keeps the 8-connected components of `region` that hold at least one seed.
///
/// A seed only counts where it is set in both `seeds_from` and `region`. A kept component becomes
/// `peak` in `output`, everything else 0.
pub fn hysteresis<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: HysteresisPlanes<'_>,
    peak: f32,
) {
    let samples = planes.region.samples();

    let region = planes.region.array::<R>();
    let marks = planes.marks.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: `region` and `marks` each hold `samples` elements of their declared type.
    unsafe {
        mark_region::launch_unchecked::<R>(client, cubes, shape, region, marks);
    }

    let seeds_from = planes.seeds_from.array::<R>();
    let region = planes.region.array::<R>();
    let seeds = planes.seeds.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: `seeds_from`, `region` and `seeds` each hold `samples` elements of their declared type.
    unsafe {
        mark_seeds::launch_unchecked::<R>(client, cubes, shape, seeds_from, region, seeds);
    }

    let label_planes = LabelPlanes {
        marks: planes.marks,
        labels: planes.labels,
    };
    label(client, dispatch, label_planes);

    let count_planes = CountPlanes {
        labels: planes.labels,
        marks: planes.seeds,
        counts: planes.counts,
    };
    count(client, dispatch, count_planes);

    let keep_planes = KeepPlanes {
        labels: planes.labels,
        counts: planes.counts,
        output: planes.kept,
    };
    keep(client, dispatch, keep_planes, 1);

    let kept = planes.kept.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: `kept` and `output` each hold `samples` elements of their declared type.
    unsafe {
        spread_kept::launch_unchecked::<R>(client, cubes, shape, kept, output, peak);
    }
}
