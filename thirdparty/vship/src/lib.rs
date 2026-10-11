//! GPU SSIMULACRA2 and Butteraugli through a statically linked Vship
//!
//! [device_count] and [device_info] list the devices Vship's backend can use.
//! [Ssimulacra2] and [Butteraugli] score a distorted frame against a reference on one device, and [Butteraugli] can also return a [DistortionMap].

mod colour;
mod device;
mod distmap;
mod error;
mod metric;
mod sys;
#[cfg(all(test, feature = "gpu-tests"))]
mod tests;

pub use self::colour::{ChromaLocation, ColorSpace, Planes, Range, SampleType};
pub use self::device::{DeviceInfo, check_device, device_count, device_info};
pub use self::distmap::DistortionMap;
pub use self::error::VshipError;
pub use self::metric::{Butteraugli, ButteraugliScore, Ssimulacra2};
