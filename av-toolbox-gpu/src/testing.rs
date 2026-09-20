/// The wgpu-backed accelerator this build tests against.
///
/// Which one it is follows the enabled backend feature, so the same tests run against Vulkan on a
/// Linux machine and against Metal on a macOS one. The two features are mutually exclusive, so it
/// is never defined twice. A build with neither does not define it at all.
///
/// `crate::Accelerator` is spelled out here rather than imported. A build with only `cuda` or
/// only `rocm` enabled never defines this constant, and a `use` for it would then be an unused
/// import, which `-D warnings` turns into a build failure.
#[cfg(feature = "vulkan")]
pub(crate) const TEST_ACCELERATOR: crate::Accelerator = crate::Accelerator::Vulkan;

#[cfg(feature = "metal")]
pub(crate) const TEST_ACCELERATOR: crate::Accelerator = crate::Accelerator::Metal;
