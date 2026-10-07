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

#[cfg(test)]
mod tests;

pub use self::body::{Activation, BodyPath, CnnError, LayerRun};
#[cfg(any(test, feature = "test-support"))]
pub use self::body::{
    Finish,
    body_layer_with_blueprint,
    convolve_with_blueprint,
    convolve_without_bias,
    epilogue,
};
#[cfg(any(test, feature = "test-support"))]
pub use self::convolution::{offers_cmma_16, takes_forced_blueprint};
#[cfg(any(test, feature = "test-support"))]
pub use self::kernels::{
    BodyConvolution,
    Epilogue,
    TunedRun,
    direct_variants,
    fixed_candidate,
    offered_candidates,
    offered_plane_width,
    run_candidate,
    run_tuned,
    transform_input,
    winograd_variants,
};
pub use self::kernels::{
    Candidate,
    DirectVariant,
    Kernels,
    WeightStaging,
    WinogradVariant,
    winograd_allowed,
};
pub use self::layer::Precision;
pub use self::region::{Region, tiles};
