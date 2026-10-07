use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::artcnn::{LCG_INCREMENT, LCG_MULTIPLIER, next_unit};
use crate::dpir::{LayerKind, Variant, WeightLayer, Weights, layer_plan};

type Cache = Mutex<HashMap<(u32, u32), Weights>>;

/// The largest magnitude of a 3x3 body weight.
const BODY_SCALE: f64 = 0.01;

/// Deterministic weights for the `variant` network.
///
/// They keep activations bounded and the output mostly between 0 and 1, so the clamp does not hide a
/// difference from a reference. The weights of each seed are made once and kept.
pub fn synthetic_weights(variant: Variant, seed: u32) -> Weights {
    static CACHE: OnceLock<Cache> = OnceLock::new();

    let cache = CACHE.get_or_init(Default::default);
    let key = (variant.image_channels(), seed);
    let mut cached = cache.lock().expect("the weights cache is not poisoned");
    let weights = cached.entry(key).or_insert_with(|| build_weights(variant, seed));

    weights.clone()
}

fn build_weights(variant: Variant, seed: u32) -> Weights {
    let plan = layer_plan(variant);
    let last = plan.len() - 1;
    let mut state = u64::from(seed)
        .wrapping_mul(LCG_MULTIPLIER)
        .wrapping_add(LCG_INCREMENT);
    let mut layers = Vec::with_capacity(plan.len());

    for (index, (kind, in_channels, out_channels)) in plan.into_iter().enumerate() {
        let is_end = index == 0 || index == last;
        let (low, high) = value_range(kind, is_end, in_channels);
        let count = kind.value_count(in_channels, out_channels);
        let values = (0..count)
            .map(|_| (low + next_unit(&mut state) * (high - low)) as f32)
            .collect();

        layers.push(WeightLayer {
            kind,
            in_channels,
            out_channels,
            values,
        });
    }

    Weights::new(variant, layers).expect("the synthetic layers follow the plan")
}

/// The range a layer's values are drawn from.
fn value_range(kind: LayerKind, is_end: bool, in_channels: u32) -> (f64, f64) {
    match kind {
        LayerKind::Conv if is_end => (0.0, 2.0 / f64::from(9 * in_channels)),
        LayerKind::Conv => (-BODY_SCALE, BODY_SCALE),
        LayerKind::Down => symmetric_range(4 * in_channels),
        LayerKind::Up => symmetric_range(in_channels),
    }
}

fn symmetric_range(rows: u32) -> (f64, f64) {
    let bound = 1.0 / f64::from(rows).sqrt();

    (-bound, bound)
}
