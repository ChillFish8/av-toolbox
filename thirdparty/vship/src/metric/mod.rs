mod butteraugli;
mod handler;
mod ssimulacra2;

pub use self::butteraugli::{Butteraugli, ButteraugliScore};
pub use self::ssimulacra2::Ssimulacra2;
use crate::colour::{ColorSpace, Planes};

/// The two frames of one call and the colour spaces their handler was made with.
struct FramePair<'a> {
    reference_space: &'a ColorSpace,
    distorted_space: &'a ColorSpace,
    reference: &'a Planes<'a>,
    distorted: &'a Planes<'a>,
}
