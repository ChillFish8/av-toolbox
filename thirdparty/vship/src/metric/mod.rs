mod butteraugli;
mod cvvdp;
mod handler;
mod ssimulacra2;

pub use self::butteraugli::{Butteraugli, ButteraugliScore};
pub use self::cvvdp::{Cvvdp, CvvdpOptions};
pub use self::ssimulacra2::Ssimulacra2;
use crate::colour::{ColorSpace, Planes};
use crate::error::VshipError;

/// The two frames of one call and the colour spaces their handler was made with.
struct FramePair<'a> {
    reference_space: &'a ColorSpace,
    distorted_space: &'a ColorSpace,
    reference: &'a Planes<'a>,
    distorted: &'a Planes<'a>,
}

/// Refuses a reference and distorted pair whose frame sizes differ.
fn check_matching_sizes(reference: &ColorSpace, distorted: &ColorSpace) -> Result<(), VshipError> {
    let same_size = reference.width == distorted.width && reference.height == distorted.height;
    if !same_size {
        return Err(mismatched_sizes_error(reference, distorted));
    }

    Ok(())
}

fn mismatched_sizes_error(reference: &ColorSpace, distorted: &ColorSpace) -> VshipError {
    VshipError::new(format!(
        "the reference is {} by {} but the distorted frame is {} by {}",
        reference.width, reference.height, distorted.width, distorted.height
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::{ChromaLocation, Range, SampleType};

    fn yuv420(width: i64, height: i64) -> ColorSpace {
        ColorSpace {
            sample: SampleType::Uint8,
            width,
            height,
            range: Range::Limited,
            sub_sampling_w: 1,
            sub_sampling_h: 1,
            chroma_location: ChromaLocation::Left,
            matrix: 1,
            transfer: 1,
            primaries: 1,
        }
    }

    #[test]
    fn different_reference_and_distorted_sizes_are_refused() {
        let reference = yuv420(64, 64);
        let distorted = yuv420(64, 32);

        let error = check_matching_sizes(&reference, &distorted).expect_err("mismatched sizes passed");

        assert!(error.message().contains("64 by 32"), "{error}");
    }

    #[test]
    fn equal_sizes_pass() {
        let reference = yuv420(64, 64);
        let distorted = yuv420(64, 64);

        check_matching_sizes(&reference, &distorted).expect("equal sizes refused");
    }
}
