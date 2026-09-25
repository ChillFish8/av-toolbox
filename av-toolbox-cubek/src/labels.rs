// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// The label of a pixel that is not set.
pub const UNSET: u32 = u32::MAX;

/// How many propagate and jump passes run between reads of the changed flag.
const BATCH: u32 = 4;

/// How labelling finished.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Settled {
    /// A batch of passes changed nothing.
    Converged { passes: u32 },
    /// The pass cap was reached first, so a component may still carry more than one label.
    Capped { passes: u32 },
}

#[derive(Debug, Clone, Copy)]
pub struct LabelPlanes<'a> {
    pub marks: DevicePlane<'a>,
    pub labels: DevicePlane<'a>,
    pub changed: DevicePlane<'a>,
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

/// Takes the smallest label among a set pixel and its 8 neighbours.
#[cube(launch_unchecked)]
fn propagate(labels: &mut Array<u32>, changed: &mut Array<u32>, width: u32, height: u32) {
    if ABSOLUTE_POS >= labels.len() {
        terminate!();
    }

    let current = labels[ABSOLUTE_POS];
    if current != u32::MAX {
        let position = ABSOLUTE_POS as u32;
        let column = (position % width) as i32;
        let row = (position / width) as i32;

        let mut best = current;
        for dy in 0..3u32 {
            for dx in 0..3u32 {
                let x = column + dx as i32 - 1;
                let y = row + dy as i32 - 1;
                if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                    let index = y as u32 * width + x as u32;
                    best = u32::min(best, labels[index as usize]);
                }
            }
        }

        if best < current {
            labels[ABSOLUTE_POS] = best;
            changed[0] = 1u32;
        }
    }
}

/// Replaces each label with the label its root pixel holds, so long chains settle quickly.
#[cube(launch_unchecked)]
fn jump(labels: &mut Array<u32>, changed: &mut Array<u32>) {
    if ABSOLUTE_POS >= labels.len() {
        terminate!();
    }

    let current = labels[ABSOLUTE_POS];
    if current != u32::MAX {
        let root = labels[current as usize];
        if root < current {
            labels[ABSOLUTE_POS] = root;
            changed[0] = 1u32;
        }
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

fn changed_flag<R: Runtime>(client: &ComputeClient<R>, changed: DevicePlane<'_>) -> u32 {
    let handle = changed.handle().clone();
    let bytes = client
        .read_one(handle)
        .expect("the device should hand back the changed flag");

    u32::from_bytes(&bytes)[0]
}

/// Labels every 8-connected component of the set marks with the smallest index inside it.
///
/// Unset pixels are labelled [UNSET]. The passes stop once a batch changes nothing, or at the
/// plane's width plus height passes, whichever comes first.
pub fn label<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: LabelPlanes<'_>,
) -> Settled {
    let samples = planes.labels.samples();
    let width = planes.labels.width();
    let height = planes.labels.height();
    let cap = width + height;

    let marks = planes.marks.array::<R>();
    let labels = planes.labels.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: the marks hold `samples` u8 values and the labels `samples` u32 values.
    unsafe {
        seed::launch_unchecked::<R>(client, cubes, shape, marks, labels);
    }

    let mut passes = 0;
    loop {
        clear(client, dispatch, planes.changed);

        for _ in 0..BATCH {
            let labels = planes.labels.array::<R>();
            let changed = planes.changed.array::<R>();
            let cubes = dispatch.cube_count(samples);
            let shape = dispatch.cube_dim();

            // Safety: the labels hold `samples` u32 values and the flag one.
            unsafe {
                propagate::launch_unchecked::<R>(client, cubes, shape, labels, changed, width, height);
            }

            let labels = planes.labels.array::<R>();
            let changed = planes.changed.array::<R>();
            let cubes = dispatch.cube_count(samples);
            let shape = dispatch.cube_dim();

            // Safety: as above.
            unsafe {
                jump::launch_unchecked::<R>(client, cubes, shape, labels, changed);
            }
        }

        passes += BATCH;

        let flag = changed_flag(client, planes.changed);
        if flag == 0 {
            return Settled::Converged { passes };
        }

        if passes >= cap {
            return Settled::Capped { passes };
        }
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

    // Safety: as above.
    unsafe {
        keep_counted::launch_unchecked::<R>(client, cubes, shape, labels, counts, output, minimum);
    }
}
