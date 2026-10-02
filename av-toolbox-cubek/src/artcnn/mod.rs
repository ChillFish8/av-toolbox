mod body;
mod ends;
mod fused;
mod network;
mod tiling;
mod weights;

#[cfg(test)]
mod tests;

pub use self::body::{Activation, ArtcnnError, BodyPath, epilogue};
#[cfg(any(test, feature = "test-support"))]
pub use self::body::{
    Finish,
    body_layer_with_blueprint,
    convolve_with_blueprint,
    convolve_without_bias,
    offers_cmma_16,
};
#[cfg(any(test, feature = "test-support"))]
pub use self::ends::padded_last_layer_with_blueprint;
pub use self::ends::{Region, first_layer, last_layer, padded_last_layer, shuffle};
pub use self::network::{Maps, Network, Precision, run_region};
pub use self::tiling::{
    ScratchSizes,
    TILE_BUDGET,
    artcnn,
    fits_whole,
    scratch_sizes,
    tile_side,
    tiles,
    warm_up,
};
pub use self::weights::{Layer, Model, Weights, WeightsError, layer_shapes};
