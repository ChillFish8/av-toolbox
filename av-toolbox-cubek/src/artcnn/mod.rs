mod body;
mod ends;
mod network;
mod weights;

#[cfg(test)]
mod tests;

pub use self::body::{Activation, epilogue};
pub use self::ends::{Region, first_layer, last_layer};
pub use self::network::{Network, Precision};
pub use self::weights::{Layer, Model, Weights, WeightsError, layer_shapes};
