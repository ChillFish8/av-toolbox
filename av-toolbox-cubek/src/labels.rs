// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// The label of a pixel that is not set.
pub const UNSET: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub struct LabelPlanes<'a> {
    pub marks: DevicePlane<'a>,
    pub labels: DevicePlane<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct CountPlanes<'a> {
    pub labels: DevicePlane<'a>,
    pub marks: DevicePlane<'a>,
    pub counts: DevicePlane<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct KeepPlanes<'a> {
    pub labels: DevicePlane<'a>,
    pub counts: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn seed(marks: &Array<u8>, labels: &mut Array<u32>) {
    if ABSOLUTE_POS >= labels.len() {
        terminate!();
    }

    if u32::cast_from(marks[ABSOLUTE_POS]) == 0u32 {
        labels[ABSOLUTE_POS] = u32::MAX;
    } else {
        labels[ABSOLUTE_POS] = ABSOLUTE_POS as u32;
    }
}

#[cube(launch_unchecked)]
fn clear_u32(values: &mut Array<u32>) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    values[ABSOLUTE_POS] = 0u32;
}

/// Follows the links from `start` until one points at itself.
#[cube]
fn root(labels: &Array<Atomic<u32>>, start: u32) -> u32 {
    let mut current = start;
    let mut parent = labels[current as usize].load();
    while parent != current {
        current = parent;
        parent = labels[current as usize].load();
    }

    current
}

/// Joins the trees holding `first` and `second`, linking the larger root to the smaller.
///
/// When another pixel already relinked the larger root, the join carries on from the link it found, so no link is
/// ever lost.
#[cube]
fn join(labels: &Array<Atomic<u32>>, first: u32, second: u32) {
    let mut left = first;
    let mut right = second;
    let mut joined = false;

    while !joined {
        let left_root = root(labels, left);
        let right_root = root(labels, right);
        let larger = u32::max(left_root, right_root);
        let smaller = u32::min(left_root, right_root);

        if larger == smaller {
            joined = true;
        } else {
            let held = labels[larger as usize].fetch_min(smaller);
            if held == larger {
                joined = true;
            } else {
                left = held;
                right = smaller;
            }
        }
    }
}

#[cube]
fn join_if_set(marks: &Array<u8>, labels: &Array<Atomic<u32>>, position: u32, neighbour: u32) {
    if u32::cast_from(marks[neighbour as usize]) != 0u32 {
        join(labels, position, neighbour);
    }
}

/// Joins each set pixel with its set neighbours to the left, above left, above and above right.
#[cube(launch_unchecked)]
fn merge(marks: &Array<u8>, labels: &mut Array<Atomic<u32>>, width: u32) {
    if ABSOLUTE_POS >= labels.len() {
        terminate!();
    }

    if u32::cast_from(marks[ABSOLUTE_POS]) != 0u32 {
        let position = ABSOLUTE_POS as u32;
        let column = position % width;
        let row = position / width;

        if column > 0 {
            join_if_set(marks, labels, position, position - 1);
        }

        if row > 0 {
            let above = position - width;

            if column > 0 {
                join_if_set(marks, labels, position, above - 1);
            }

            join_if_set(marks, labels, position, above);

            if column + 1 < width {
                join_if_set(marks, labels, position, above + 1);
            }
        }
    }
}

/// Points each set pixel straight at its root.
#[cube(launch_unchecked)]
fn compress(labels: &mut Array<Atomic<u32>>) {
    if ABSOLUTE_POS >= labels.len() {
        terminate!();
    }

    let current = labels[ABSOLUTE_POS].load();
    if current != u32::MAX {
        let position = ABSOLUTE_POS as u32;
        let found = root(labels, position);
        labels[ABSOLUTE_POS].store(found);
    }
}

#[cube(launch_unchecked)]
fn count_marked(labels: &Array<u32>, marks: &Array<u8>, counts: &mut Array<Atomic<u32>>) {
    if ABSOLUTE_POS >= labels.len() {
        terminate!();
    }

    let current = labels[ABSOLUTE_POS];
    let marked = u32::cast_from(marks[ABSOLUTE_POS]) != 0u32;
    if current != u32::MAX && marked {
        counts[current as usize].fetch_add(1u32);
    }
}

#[cube(launch_unchecked)]
fn keep_counted(labels: &Array<u32>, counts: &Array<u32>, output: &mut Array<u8>, minimum: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let current = labels[ABSOLUTE_POS];
    let mut kept = false;
    if current != u32::MAX {
        kept = counts[current as usize] >= minimum;
    }

    if kept {
        output[ABSOLUTE_POS] = u8::cast_from(255u32);
    } else {
        output[ABSOLUTE_POS] = u8::cast_from(0u32);
    }
}

fn clear<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, plane: DevicePlane<'_>) {
    let values = plane.array::<R>();
    let samples = plane.samples();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: the plane holds `samples` u32 values and the kernel writes nothing past them.
    unsafe {
        clear_u32::launch_unchecked::<R>(client, cubes, shape, values);
    }
}

/// Labels every 8-connected component of the set marks with the smallest index inside it.
///
/// Unset pixels are labelled [UNSET]. It runs a fixed three launches whatever the marks hold.
pub fn label<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, planes: LabelPlanes<'_>) {
    let samples = planes.labels.samples();
    let width = planes.labels.width();

    let marks = planes.marks.array::<R>();
    let labels = planes.labels.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: the marks hold `samples` u8 values and the labels `samples` u32 values.
    unsafe {
        seed::launch_unchecked::<R>(client, cubes, shape, marks, labels);
    }

    let marks = planes.marks.array::<R>();
    let labels = planes.labels.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: the marks hold `samples` u8 values and the labels `samples` u32 values. Every link points at an
    // index below `samples`, since it starts as a set pixel's own index and only ever falls to another set pixel's.
    unsafe {
        merge::launch_unchecked::<R>(client, cubes, shape, marks, labels, width);
    }

    let labels = planes.labels.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: the labels hold `samples` u32 values, each set one linking to an index below `samples`.
    unsafe {
        compress::launch_unchecked::<R>(client, cubes, shape, labels);
    }
}

/// Counts, per label, the pixels whose mark is set.
pub fn count<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, planes: CountPlanes<'_>) {
    clear(client, dispatch, planes.counts);

    let samples = planes.labels.samples();
    let labels = planes.labels.array::<R>();
    let marks = planes.marks.array::<R>();
    let counts = planes.counts.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: every plane holds `samples` elements, and a label is always an index below `samples`.
    unsafe {
        count_marked::launch_unchecked::<R>(client, cubes, shape, labels, marks, counts);
    }
}

/// Selects the pixels whose label counted at least `minimum`.
pub fn keep<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: KeepPlanes<'_>,
    minimum: u32,
) {
    let samples = planes.output.samples();
    let labels = planes.labels.array::<R>();
    let counts = planes.counts.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: every plane holds `samples` elements, and a label is always an index below `samples`.
    unsafe {
        keep_counted::launch_unchecked::<R>(client, cubes, shape, labels, counts, output, minimum);
    }
}
