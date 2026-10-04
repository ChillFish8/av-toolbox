use cubecl::prelude::*;

/// The two scalars a block's confidence is scored with, both on the normalised SAD scale.
///
/// `noise_floor` is the SAD two noisy copies of the same content reach by chance, and is taken off the
/// best SAD first. `thsad` is how far past the floor a block goes before its confidence reaches 0. It
/// must be above 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceScores {
    pub noise_floor: f32,
    pub thsad: f32,
}

/// How well a block with `best_sad` matched, between 0 and 1.
///
/// It is `(thsad² - excess²) / (thsad² + excess²)`, where `excess` is the SAD above `noise_floor`.
/// The last bits of the score depend on how the device divides.
#[cube]
pub(crate) fn confidence_of(best_sad: f32, noise_floor: f32, thsad: f32) -> f32 {
    let mut excess = best_sad - noise_floor;
    if excess < 0.0f32 {
        excess = 0.0f32;
    }

    let thsad_sq = thsad * thsad;
    let excess_sq = excess * excess;
    let mut confidence = (thsad_sq - excess_sq) / (thsad_sq + excess_sq);
    if confidence < 0.0f32 {
        confidence = 0.0f32;
    }

    confidence
}
