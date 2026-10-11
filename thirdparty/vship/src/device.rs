use std::ffi::CStr;
use std::mem::MaybeUninit;

use crate::error::{VshipError, check};
use crate::sys;

/// A device as Vship's backend lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    /// Device memory in bytes.
    pub vram: u64,
    pub integrated: bool,
}

/// The number of devices Vship's backend can run on.
pub fn device_count() -> Result<i32, VshipError> {
    let mut count = 0;
    // SAFETY: `count` is a valid out pointer.
    let status = unsafe { sys::Vship_GetDeviceCount(&raw mut count) };
    check(status)?;

    Ok(count)
}

/// Describes device `gpu_id`, an index into Vship's backend's own device list.
pub fn device_info(gpu_id: i32) -> Result<DeviceInfo, VshipError> {
    let mut raw_info = MaybeUninit::<sys::Vship_DeviceInfo>::zeroed();
    // SAFETY: `raw_info` is valid for writes of a whole `Vship_DeviceInfo`.
    let status = unsafe { sys::Vship_GetDeviceInfo(raw_info.as_mut_ptr(), gpu_id) };
    check(status)?;

    // SAFETY: The struct is plain data, zero filled before the call and filled in by it.
    let mut info = unsafe { raw_info.assume_init() };
    let last = info.name.len() - 1;
    info.name[last] = 0;
    // SAFETY: The last byte of `name` was just set to NUL, so the string ends inside the array.
    let name = unsafe { CStr::from_ptr(info.name.as_ptr()) };

    Ok(DeviceInfo {
        name: name.to_string_lossy().into_owned(),
        vram: info.VRAMSize,
        integrated: info.integrated != 0,
    })
}

/// Checks that Vship can run every metric on device `gpu_id`.
pub fn check_device(gpu_id: i32) -> Result<(), VshipError> {
    // SAFETY: Takes a plain integer and touches no caller memory.
    let status = unsafe { sys::Vship_GPUFullCheck(gpu_id) };

    check(status)
}
