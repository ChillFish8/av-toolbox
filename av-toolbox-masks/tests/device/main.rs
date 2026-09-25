#![cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

mod adaptive;
mod canny;
mod combine;
mod reference;
mod support;
mod xdog;
