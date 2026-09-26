// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// How many bins the range between the smallest and largest value is split into.
pub const BINS: u32 = 4096;

/// A plane, its range, the bins to count it into, and where the percentile lands.
#[derive(Debug, Clone, Copy)]
pub struct PercentilePlanes<'a> {
    pub input: DevicePlane<'a>,
    pub range: DevicePlane<'a>,
    pub bins: DevicePlane<'a>,
    pub result: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn clear_bins(bins: &mut Array<u32>) {
    if ABSOLUTE_POS >= bins.len() {
        terminate!();
    }

    bins[ABSOLUTE_POS] = 0u32;
}

#[cube(launch_unchecked)]
fn count_bins(input: &Array<f32>, range: &Array<f32>, bins: &mut Array<Atomic<u32>>, bin_count: u32) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let low = range[0];
    let span = range[1] - low;

    let mut bin = 0u32;
    if span > 0.0 {
        let scaled = (input[ABSOLUTE_POS] - low) / span * f32::cast_from(bin_count);
        let clamped = f32::max(scaled, 0.0);
        let floored = u32::cast_from(clamped);
        bin = u32::min(floored, bin_count - 1);
    }

    bins[bin as usize].fetch_add(1u32);
}

/// Finds the first bin whose running count reaches `target`, and writes its centre.
#[cube(launch_unchecked)]
fn find_percentile(
    bins: &Array<u32>,
    range: &Array<f32>,
    result: &mut Array<f32>,
    target: f32,
    bin_count: u32,
) {
    let low = range[0];
    let span = range[1] - low;
    let bin_width = span / f32::cast_from(bin_count);

    let mut running = 0u32;
    let mut found = false;
    for bin in 0..bin_count {
        running += bins[bin as usize];
        if !found && f32::cast_from(running) >= target {
            result[0] = low + (f32::cast_from(bin) + 0.5) * bin_width;
            found = true;
        }
    }

    if !found {
        result[0] = range[1];
    }
}

/// Writes the value `percentile` percent of the way through `input` to the result.
///
/// The value is the centre of the bin it falls in, so it is within half a bin of the exact
/// percentile. A plane whose values are all equal gives that value.
pub fn percentile<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: PercentilePlanes<'_>,
    percentile: f32,
) {
    let bins = planes.bins.array::<R>();
    let cubes = dispatch.cube_count(BINS as usize);
    let shape = dispatch.cube_dim();

    // Safety: every plane holds at least as many elements as the kernel reads or writes.
    unsafe {
        clear_bins::launch_unchecked::<R>(client, cubes, shape, bins);
    }

    let samples = planes.input.samples();
    let input = planes.input.array::<R>();
    let range = planes.range.array::<R>();
    let bins = planes.bins.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: the input holds `samples` f32 values, the range two and the bins `BINS` u32 values,
    // and the kernel reads and writes nothing past those lengths.
    unsafe {
        count_bins::launch_unchecked::<R>(client, cubes, shape, input, range, bins, BINS);
    }

    let target = percentile / 100.0 * samples as f32;
    let bins = planes.bins.array::<R>();
    let range = planes.range.array::<R>();
    let result = planes.result.array::<R>();
    let single = CubeCount::Static(1, 1, 1);
    let lone = CubeDim::new_1d(1);

    // Safety: the bins hold `BINS` u32 values, the range two and the result one, and the kernel
    // reads and writes nothing past those lengths.
    unsafe {
        find_percentile::launch_unchecked::<R>(client, single, lone, bins, range, result, target, BINS);
    }
}
