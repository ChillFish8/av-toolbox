//! Kernels for removing compression blocking
//!
//! - [h264] for H.264's in-loop deblocking filter, run as a post-process.
//! - [qed] for Deblock_QED, which spreads a stronger H.264 pass's change into each block's interior.

pub mod h264;
pub mod qed;
