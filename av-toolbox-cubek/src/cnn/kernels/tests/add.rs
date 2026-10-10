use cubecl::prelude::CubeCount;

use crate::cnn::kernels::add::spread_cubes;

#[test]
fn cubes_past_the_first_axis_spread_into_the_second() {
    let count = spread_cubes(70_000, (65535, 65535, 65535));

    assert!(matches!(count, CubeCount::Static(65535, 2, 1)), "{count:?}");
}
