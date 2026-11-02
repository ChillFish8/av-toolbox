use strum_macros::{Display, EnumIter, EnumString};

/// A hardware backend that kernels can run on.
///
/// Only the backends whose crate feature is enabled exist at compile time, so a build without
/// the `cuda` feature has no `Accelerator::Cuda` variant at all.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, EnumString, EnumIter, Display)]
#[strum(serialize_all = "snake_case")]
pub enum Accelerator {
    /// Runs kernels through the Nvidia CUDA backend.
    ///
    /// Nvidia GPUs only.
    #[cfg(any(feature = "cuda", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    Cuda,
    /// Runs kernels through the wgpu Vulkan backend.
    ///
    /// This works on any platform and GPU that supports basic compute shaders.
    #[cfg(any(feature = "vulkan", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "vulkan")))]
    Vulkan,
    /// Runs kernels through CubeCL's native Metal runtime.
    ///
    /// Apple GPUs only, and needs Metal 3.
    #[cfg(any(feature = "metal", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "metal")))]
    Metal,
    /// Runs kernels through the AMD ROCm backend.
    ///
    /// AMD GPUs only. Vulkan is usually faster on AMD hardware and hits fewer driver problems.
    #[cfg(any(feature = "rocm", docsrs))]
    #[cfg_attr(docsrs, doc(cfg(feature = "rocm")))]
    Rocm,
}

/// Every accelerator this build enables, in the order to try them.
pub fn default_accelerators() -> Vec<Accelerator> {
    use strum::IntoEnumIterator;

    Accelerator::iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    use crate::testing::TEST_ACCELERATOR;

    #[test]
    fn every_accelerator_renders_in_lower_case_and_parses_back() {
        for accelerator in default_accelerators() {
            let printed = accelerator.to_string();
            assert_eq!(printed, printed.to_lowercase(), "{printed}");

            let parsed = printed.parse::<Accelerator>().expect("should parse");
            assert_eq!(parsed, accelerator, "{printed}");
        }
    }

    #[cfg(any(feature = "vulkan", feature = "metal"))]
    #[test]
    fn the_default_list_holds_the_backend_this_build_enables() {
        assert!(default_accelerators().contains(&TEST_ACCELERATOR));
    }

    #[test]
    fn parsing_rejects_an_unknown_backend() {
        assert!("unicorn".parse::<Accelerator>().is_err());
    }
}
