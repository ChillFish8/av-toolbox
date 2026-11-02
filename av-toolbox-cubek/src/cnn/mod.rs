//! The 3x3 convolutions shared by the networks
//!
//! A network runs its `F -> F` layers through these paths and kernels. On f16 maps on the convolution path
//! autotune picks a kernel per device and problem shape.

pub(crate) mod body;
pub(crate) mod convolution;
mod fused;
pub(crate) mod kernels;
mod layer;
mod region;
mod taps;

#[cfg(test)]
mod tests;

pub use self::body::{Activation, BodyPath, CnnError, Finish, LayerMaps, LayerRun, run_layer};
#[cfg(any(test, feature = "test-support"))]
pub use self::body::{body_layer_with_blueprint, convolve_with_blueprint, convolve_without_bias, epilogue};
pub use self::convolution::run_padded_layer;
#[cfg(any(test, feature = "test-support"))]
pub use self::convolution::{
    convolve_forced_past_the_shared_memory_limit,
    forced_shared_bytes,
    offers_cmma_16,
    takes_forced_blueprint,
};
pub(crate) use self::kernels::launch_candidate;
#[cfg(test)]
pub(crate) use self::kernels::transform_values;
pub use self::kernels::{
    BodyConvolution,
    Candidate,
    DirectVariant,
    Epilogue,
    Kernels,
    Stride,
    StridedLayer,
    StridedVariant,
    WeightStaging,
    WinogradVariant,
    offered_candidates,
    run_strided,
    strided_variant,
};
#[cfg(any(test, feature = "test-support"))]
pub use self::kernels::{
    TunedRun,
    direct_variants,
    fixed_candidate,
    offered_plane_width,
    run_candidate,
    run_tuned,
    transform_input,
    winograd_variants,
};
pub(crate) use self::layer::encode;
pub use self::layer::{ConvLayer, PADDED_CHANNELS, PaddedLastLayer, Precision};
pub use self::region::{Region, tiles};
pub(crate) use self::taps::add_taps;
