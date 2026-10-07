mod body;
mod ends;
mod network;
mod tiling;
mod weights;

#[cfg(test)]
mod tests;

pub use self::body::offered_candidates;
#[cfg(test)]
pub(crate) use self::body::transform_weights;
#[cfg(any(test, feature = "test-support"))]
pub use self::body::{
    body_layer_with_blueprint,
    convolve_with_blueprint,
    convolve_without_bias,
    epilogue,
    takes_forced_blueprint,
};
#[cfg(any(test, feature = "test-support"))]
pub use self::ends::padded_last_layer_with_blueprint;
pub use self::ends::{first_layer, last_layer, padded_last_layer, shuffle};
#[cfg(any(test, feature = "test-support"))]
pub use self::network::run_region_with_candidate;
pub use self::network::{Maps, Network, run_region};
pub use self::tiling::{ScratchSizes, TILE_BUDGET, artcnn, fits_whole, scratch_sizes, tile_side, warm_up};
pub use self::weights::{Layer, Model, Weights, WeightsError, layer_shapes, winograd_allowed};
pub use crate::cnn::{
    Activation,
    BodyPath,
    Candidate,
    CnnError,
    DirectVariant,
    Kernels,
    LayerRun,
    Precision,
    Region,
    WeightStaging,
    WinogradVariant,
    tiles,
};
#[cfg(any(test, feature = "test-support"))]
pub use crate::cnn::{
    BodyConvolution,
    Epilogue,
    Finish,
    TunedRun,
    direct_variants,
    fixed_candidate,
    offered_plane_width,
    offers_cmma_16,
    run_candidate,
    run_tuned,
    transform_input,
    winograd_variants,
};
