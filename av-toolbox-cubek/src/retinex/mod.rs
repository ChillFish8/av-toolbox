mod msr;
mod recursive;
#[cfg(test)]
mod tests;

pub use self::msr::{
    GAUSSIAN_MAX_SIGMA,
    RetinexBlurs,
    RetinexPlanes,
    RetinexSettings,
    retinex,
    uses_gaussian,
};
#[cfg(test)]
pub(crate) use self::recursive::solve_q;
pub use self::recursive::{RecursiveGaussian, RecursivePlanes, recursive_gaussian};
#[cfg(any(test, feature = "test-support"))]
pub(crate) use self::recursive::{sections, settling_length};
