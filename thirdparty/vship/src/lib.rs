//! GPU SSIMULACRA2, Butteraugli and CVVDP through a statically linked Vship
//!
//! [device_count] and [device_info] list the devices Vship's backend can use.
//! [Ssimulacra2], [Butteraugli] and [Cvvdp] score a distorted frame against a reference on one device.
//! [Butteraugli] and [Cvvdp] can also return a [DistortionMap].

mod colour;
mod device;
mod display;
mod distmap;
mod error;
mod metric;
mod sys;
#[cfg(all(test, feature = "gpu-tests"))]
mod tests;

pub use self::colour::{ChromaLocation, ColorSpace, Planes, Range, SampleType};
pub use self::device::{DeviceInfo, check_device, device_count, device_info};
pub use self::display::DisplayModel;
pub use self::distmap::DistortionMap;
pub use self::error::VshipError;
pub use self::metric::{Butteraugli, ButteraugliScore, Cvvdp, CvvdpOptions, Ssimulacra2};
