//! Kernels for removing compression blocking
//!
//! - [h264] for H.264's in-loop deblocking filter, run as a post-process.
//! - [qed] for Deblock_QED, which spreads a stronger H.264 pass's change into each block's interior.
//! - [pp7] for pp7, which thresholds the folded H.264 core transform of each sample's 7 by 7 neighbourhood.

pub mod h264;
pub mod pp7;
pub mod qed;
mod rounding;
