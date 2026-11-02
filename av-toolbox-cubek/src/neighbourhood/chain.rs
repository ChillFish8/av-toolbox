use cubecl::prelude::Client;

use super::{Edge, Extreme, NeighbourPlanes, Shape, Spread, extreme, spread_with_edge};
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
    /// The plane the first step reads.
    pub input: DevicePlane<'a>,
    /// The plane the steps bounce through.
    pub scratch: DevicePlane<'a>,
    /// The plane the last step writes.
    pub output: DevicePlane<'a>,
}

impl ChainPlanes<'_> {
    /// Whether the three planes are held by three separate handles.
    fn are_distinct(&self) -> bool {
        let input = self.input.handle();
        let scratch = self.scratch.handle();
        let output = self.output.handle();

        !std::ptr::eq(input, scratch) && !std::ptr::eq(input, output) && !std::ptr::eq(scratch, output)
    }
}

/// Runs each of `steps` in turn, feeding each step's result into the next.
///
/// The steps alternate between `scratch` and `output` so no step reads the plane it writes, and the
/// last step always lands in `output`. `input` is left untouched. `scratch` is overwritten when
/// there are two or more steps. The three planes must differ, and `steps` must not be empty.
pub fn chain(client: &Client, dispatch: &impl Dispatch, planes: ChainPlanes<'_>, steps: &[Step]) {
    chain_with_edge(client, dispatch, planes, steps, Edge::Reflect);
}

/// The same as [chain], reading past the plane's edge as `edge` says in each spread step.
pub fn chain_with_edge(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: ChainPlanes<'_>,
    steps: &[Step],
    edge: Edge,
) {
    debug_assert!(!steps.is_empty(), "a chain needs at least one step");
    debug_assert!(planes.are_distinct(), "a chain needs three distinct planes");

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
            Step::Spread(which) => spread_with_edge(client, dispatch, pass_planes, which, edge),
        }

        source = destination;
    }
}
