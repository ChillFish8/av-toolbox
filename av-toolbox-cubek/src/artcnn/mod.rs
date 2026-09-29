mod weights;

#[cfg(test)]
mod tests;

pub use self::weights::{Layer, Model, Weights, WeightsError, layer_shapes};
