mod body;
mod ends;
mod network;
mod tiling;
mod weights;

#[cfg(test)]
mod tests;

pub use self::body::{Activation, ArtcnnError, BodyPath, epilogue};
pub use self::ends::{Region, first_layer, last_layer};
pub use self::network::{Maps, Network, Precision, run_region};
pub use self::tiling::{ScratchSizes, TILE_BUDGET, artcnn, scratch_sizes, tile_side, tiles, warm_up};
pub use self::weights::{Layer, Model, Weights, WeightsError, layer_shapes};
