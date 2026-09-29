use cubecl::prelude::{ComputeClient, Runtime};

use super::{Extreme, NeighbourPlanes, Shape, Spread, extreme, spread};
use crate::{DevicePlane, Dispatch};

/// One 3x3 pass of a [chain](crate::neighbourhood::chain).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Step {
    /// The largest or smallest value in the shape, as [extreme](crate::neighbourhood::extreme) writes.
    Extreme(Extreme, Shape),
    /// A move toward the neighbours' mean, as [spread](crate::neighbourhood::spread) writes.
    Spread(Spread),
}

/// The plane a [chain](crate::neighbourhood::chain) reads, the plane it bounces through and the plane
/// it writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct ChainPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub scratch: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// Runs each of `steps` in turn, feeding each step's result into the next.
///
/// The steps alternate between `scratch` and `output` so no step reads the plane it writes, and the
/// last step always lands in `output`. `input` is left untouched. `scratch` is overwritten when
/// there are two or more steps. The three planes must differ, and `steps` must not be empty.
pub fn chain<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ChainPlanes<'_>,
    steps: &[Step],
) {
    debug_assert!(!steps.is_empty(), "a chain needs at least one step");

    let mut source = planes.input;

    for (index, step) in steps.iter().enumerate() {
        let steps_after = steps.len() - 1 - index;
        let destination = if steps_after.is_multiple_of(2) {
            planes.output
        } else {
            planes.scratch
        };

        let pass_planes = NeighbourPlanes {
            input: source,
            output: destination,
        };
        match *step {
            Step::Extreme(which, shape) => extreme(client, dispatch, pass_planes, which, shape),
            Step::Spread(which) => spread(client, dispatch, pass_planes, which),
        }

        source = destination;
    }
}
