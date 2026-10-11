use std::ffi::c_void;
use std::ptr;

use super::FramePair;
use crate::distmap::MapSize;
use crate::error::{VshipError, check, check_handler};
use crate::sys;

/// A Vship metric handler, freed when dropped.
#[derive(Debug)]
pub(super) struct Handler {
    handle: sys::Vship_Handle,
}

// SAFETY: A Vship handler is not tied to the thread that made it. It is not safe to share, so it is
// not `Sync`.
unsafe impl Send for Handler {}

impl Handler {
    /// Makes a handler from a metric's init struct.
    ///
    /// # Safety
    ///
    /// `argument` must point at a live Vship init struct whose `structType` names its type, and every
    /// pointer inside it must stay valid until this returns.
    pub(super) unsafe fn init(argument: *mut c_void) -> Result<Self, VshipError> {
        let mut handle = ptr::null_mut();
        // SAFETY: The caller guarantees `argument`, and `handle` is a valid out pointer.
        let status = unsafe { sys::Vship_InitHandler(&raw mut handle, argument) };
        check(status)?;

        Ok(Self { handle })
    }

    /// Scores one frame pair into `score`.
    ///
    /// # Safety
    ///
    /// `score` must point at a live score struct whose `structType` matches this handler's metric. Its
    /// map pointer must be null, or come from a [DistortionMap](crate::DistortionMap) checked
    /// against this handler's map size.
    pub(super) unsafe fn compute(
        &mut self,
        score: *mut c_void,
        frames: &FramePair,
    ) -> Result<(), VshipError> {
        let mut inputs = FrameInputs::checked(frames)?;

        // SAFETY: Both plane sets were checked against the colour spaces the handler was made with, so
        // Vship reads only inside them. The caller guarantees `score`, and every pointer outlives the
        // call.
        let status = unsafe {
            sys::Vship_ComputeHandler(
                self.handle,
                score,
                inputs.reference_pointers.as_mut_ptr(),
                inputs.distorted_pointers.as_mut_ptr(),
                inputs.reference_lines.as_ptr(),
                inputs.distorted_lines.as_ptr(),
            )
        };

        check_handler(self.handle, status)
    }

    /// Adds a frame pair to a temporal metric's history without scoring it.
    pub(super) fn load_temporal(&mut self, frames: &FramePair) -> Result<(), VshipError> {
        let mut inputs = FrameInputs::checked(frames)?;

        // SAFETY: Both plane sets were checked against the colour spaces the handler was made with, so
        // Vship reads only inside them, and every pointer outlives the call.
        let status = unsafe {
            sys::Vship_LoadTemporalCVVDP2(
                self.handle,
                inputs.reference_pointers.as_mut_ptr(),
                inputs.distorted_pointers.as_mut_ptr(),
                inputs.reference_lines.as_ptr(),
                inputs.distorted_lines.as_ptr(),
            )
        };

        check_handler(self.handle, status)
    }

    /// Clears a temporal metric's frame history.
    pub(super) fn reset(&mut self) -> Result<(), VshipError> {
        // SAFETY: `self.handle` is a live handler.
        let status = unsafe { sys::Vship_Reset(self.handle) };

        check_handler(self.handle, status)
    }

    /// Restarts a temporal metric's running score.
    pub(super) fn reset_score(&mut self) -> Result<(), VshipError> {
        // SAFETY: `self.handle` is a live handler.
        let status = unsafe { sys::Vship_ResetScore(self.handle) };

        check_handler(self.handle, status)
    }

    /// The size of the distortion maps this handler writes.
    pub(super) fn distmap_size(&self) -> Result<MapSize, VshipError> {
        let mut width = 0i64;
        let mut height = 0i64;
        // SAFETY: `self.handle` is a live handler and both out pointers are valid.
        let status = unsafe { sys::Vship_GetDistmapResolution(self.handle, &raw mut width, &raw mut height) };
        check_handler(self.handle, status)?;

        MapSize::from_raw(width, height)
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        // SAFETY: `self.handle` came from `Vship_InitHandler` and only this value frees it.
        let status = unsafe { sys::Vship_FreeHandler(self.handle) };

        if status != sys::Vship_NoError {
            tracing::warn!(status, "vship could not free a metric handler");
        }
    }
}

/// Plane pointers and line sizes for one frame pair, taken after both are checked.
struct FrameInputs {
    reference_pointers: [*const u8; 3],
    distorted_pointers: [*const u8; 3],
    reference_lines: [i64; 3],
    distorted_lines: [i64; 3],
}

impl FrameInputs {
    fn checked(frames: &FramePair) -> Result<Self, VshipError> {
        frames.reference.check(frames.reference_space, "the reference")?;
        frames
            .distorted
            .check(frames.distorted_space, "the distorted frame")?;

        let reference_lines = frames.reference.line_sizes()?;
        let distorted_lines = frames.distorted.line_sizes()?;

        Ok(Self {
            reference_pointers: frames.reference.pointers(),
            distorted_pointers: frames.distorted.pointers(),
            reference_lines,
            distorted_lines,
        })
    }
}
