mod weights;

#[cfg(test)]
mod tests;

pub use self::weights::{LayerKind, PACK, Variant, WIDTHS, WeightLayer, Weights, WeightsError, layer_plan};
