use cubecl::prelude::{CubeCount, CubeDim};

/// Sizes a kernel launch.
pub trait Dispatch {
    /// How many cubes cover `samples` units of work, one unit per sample.
    fn cube_count(&self, samples: usize) -> CubeCount;

    fn cube_dim(&self) -> CubeDim;
}
