#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod batch;
#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod stream_order;
