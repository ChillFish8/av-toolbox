mod based;
mod eedi2;
mod eedi3;
mod merge;
mod sangnom;
mod transpose;

use cubecl::prelude::{CubeCount, CubeDim};

pub use self::based::{binarize_prewitt, double_mclip, double_mclip_turned, prewitt_threshold, pscale_blend};
pub use self::eedi2::{
    EEDI2_MAX_WIDTH,
    Eedi2Arguments,
    Eedi2Scratch,
    Eedi2ScratchSizes,
    Eedi2Settings,
    eedi2_field,
    eedi2_scratch,
};
#[cfg(any(test, feature = "test-support"))]
pub use self::eedi2::{Eedi2Stage, eedi2_fill_round, eedi2_passes_through};
#[cfg(any(test, feature = "test-support"))]
pub use self::eedi3::eedi3_mul_exact;
pub use self::eedi3::{
    Eedi3Arguments,
    Eedi3DeviceClips,
    Eedi3Scratch,
    Eedi3ScratchSizes,
    Eedi3Settings,
    Eedi3Weights,
    eedi3_field,
    eedi3_scratch,
};
pub use self::merge::merge_fields;
pub use self::sangnom::{SangnomPlanes, sangnom_aaf, sangnom_field, sangnom_max, sangnom_scratch};
pub use self::transpose::transpose;

/// The cubes that give each of `rows` rows a cube of its own.
///
/// The cubes are laid out 65535 to a line, the least every device allows.
fn row_cubes(rows: u32) -> CubeCount {
    let across = u32::min(rows, 65535);
    let lines = rows.div_ceil(across);

    CubeCount::Static(across, lines, 1)
}

/// The cubes and cube shape that launch one unit per row, `rows` in all.
///
/// Each unit walks a whole row, branching on its content, so every row gets a cube of its own and never
/// waits on another row's branches.
fn row_launch(rows: u32) -> (CubeCount, CubeDim) {
    let count = row_cubes(rows);
    let shape = CubeDim::new_1d(1);

    (count, shape)
}
