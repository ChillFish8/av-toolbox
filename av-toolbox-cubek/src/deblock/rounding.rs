// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

/// The zero every guarded kernel product is hidden behind, passed to kernels as a launch argument.
pub(super) const OPAQUE_ZERO: u32 = 0;

/// `left * right` rounded to f32, behind an always-zero `opaque_zero` so no compiler fuses it into an add.
#[cube]
pub(super) fn rounded_product(left: f32, right: f32, opaque_zero: u32) -> f32 {
    let product = left * right;
    let bits = u32::reinterpret(product) ^ opaque_zero;

    f32::reinterpret(bits)
}
