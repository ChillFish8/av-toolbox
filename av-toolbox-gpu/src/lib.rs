#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]

//! Accelerator and device management for av-toolbox.
//!
//! An [Accelerator] names a backend, such as CUDA or Vulkan. A [Device] names the hardware
//! inside that backend. The two are chosen separately because most backends expose more than
//! one device.

#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "vulkan", feature = "metal")))]
compile_error!("av-toolbox-gpu needs at least one accelerator backend enabled: cuda, rocm, vulkan or metal");

#[cfg(all(feature = "vulkan", feature = "metal"))]
compile_error!("the vulkan and metal backends cannot both be enabled, pick one");

mod accelerator;
mod device;
mod enumerate;
mod probe;
mod sniff;

#[cfg(test)]
mod testing;

pub use self::accelerator::{Accelerator, default_accelerators};
pub use self::device::{Device, DeviceError, ParseDeviceError};
pub use self::enumerate::{BackendDevices, enumerate_devices};
pub use self::sniff::{auto_select_accelerator, sniff_best_accelerator};
