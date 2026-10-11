//! GPU SSIMULACRA2 and Butteraugli through a statically linked Vship
//!
//! [device_count] and [device_info] list the devices Vship's backend can use.

mod colour;
mod device;
mod error;
mod sys;

pub use self::colour::{ChromaLocation, ColorSpace, Planes, Range, SampleType};
pub use self::device::{DeviceInfo, check_device, device_count, device_info};
pub use self::error::VshipError;
