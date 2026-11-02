// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::direct::{Quad, Sums, WRITE_WIDTH};

/// The units one workgroup of the add holds, at most.
const ADD_UNITS: u32 = 256;

#[cube(launch_unchecked)]
fn add_kernel(values: &mut [Quad], addend: &[Quad]) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    let sum = Sums::cast_from(values[ABSOLUTE_POS]) + Sums::cast_from(addend[ABSOLUTE_POS]);

    values[ABSOLUTE_POS] = Quad::cast_from(sum);
}

/// Adds the first `quads` groups of four f16 values of `addend` to those of `values`, summing in f32.
pub(super) fn add_maps(client: &Client, values: &Handle, addend: &Handle, quads: usize) {
    let values_count = quads * WRITE_WIDTH as usize;
    let bytes = (values_count * size_of::<half::f16>()) as u64;
    assert!(
        values.size_in_used() >= bytes,
        "the values are smaller than the quads"
    );
    assert!(
        addend.size_in_used() >= bytes,
        "the addend is smaller than the quads"
    );

    let hardware = &client.properties().hardware;
    let units = ADD_UNITS.min(hardware.max_units_per_cube);
    let cubes = quads.div_ceil(units as usize);
    let count = spread_cubes(cubes, hardware.max_cube_count);
    let unit = CubeDim::new_1d(units);

    // Safety: the values and the addend each hold `quads` quads, as just asserted, and units past the end
    // return before reading. The launch takes each length in f16 values, and the kernel sees it in quads.
    unsafe {
        let values_array = BufferArg::from_raw_parts(values.clone(), values_count);
        let addend_array = BufferArg::from_raw_parts(addend.clone(), values_count);

        add_kernel::launch_unchecked(client, count, unit, values_array, addend_array);
    }
}

/// `cubes` spread over the cube count's axes, each capped at the device's `limits`.
pub(super) fn spread_cubes(cubes: usize, limits: (u32, u32, u32)) -> CubeCount {
    let (max_x, max_y, max_z) = limits;
    let x = cubes.min(max_x as usize).max(1);
    let rows = cubes.div_ceil(x);
    let y = rows.min(max_y as usize).max(1);
    let layers = cubes.div_ceil(x * y);
    let z = layers.min(max_z as usize).max(1);

    CubeCount::Static(x as u32, y as u32, z as u32)
}
