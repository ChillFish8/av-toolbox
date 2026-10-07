mod layers;
mod network;
mod tiling;
mod weights;

#[cfg(test)]
mod tests;

pub use self::layers::{
    HeadLayer,
    Placement,
    add_in_place,
    gather,
    head,
    matrix_product,
    scatter,
    tail,
    write_out,
};
pub use self::network::{Dpir, DpirMaps, DpirPlanes, column_bytes, level_bytes, run_region};
pub use self::tiling::{
    DPIR_BUDGET,
    OVERLAP,
    Padded,
    ScratchSizes,
    deblock,
    scratch_sizes,
    tile_sizes,
    warm_up,
};
pub use self::weights::{LayerKind, PACK, Variant, WIDTHS, WeightLayer, Weights, WeightsError, layer_plan};
