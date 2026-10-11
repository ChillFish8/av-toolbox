use std::ffi::c_void;
use std::ptr;

use crate::colour::{ColorSpace, Planes};
use crate::error::{VshipError, check};
use crate::sys;

/// The three Butteraugli norms of one frame pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButteraugliScore {
    pub norm_q: f64,
    pub norm3: f64,
    pub norm_inf: f64,
}

/// A SSIMULACRA2 metric on one device, for frames of fixed colour spaces.
#[derive(Debug)]
pub struct Ssimulacra2 {
    handle: sys::Vship_Handle,
    reference: ColorSpace,
    distorted: ColorSpace,
}

// SAFETY: A Vship handler is not tied to the thread that made it. It is not safe to share, so it is
// not `Sync`.
unsafe impl Send for Ssimulacra2 {}

impl Ssimulacra2 {
    pub fn new(reference: ColorSpace, distorted: ColorSpace, gpu_id: i32) -> Result<Self, VshipError> {
        let mut init = sys::Vship_InitSSIMULACRA2_1 {
            structType: sys::Vship_StructInitSSIMULACRA2_1,
            src_colorspace: reference.to_raw(),
            dis_colorspace: distorted.to_raw(),
            gpu_id,
        };
        let argument = ptr::from_mut(&mut init).cast::<c_void>();
        let handle = init_handler(argument)?;

        Ok(Self {
            handle,
            reference,
            distorted,
        })
    }

    /// Scores `distorted` against `reference`, where 100 means identical.
    pub fn compute(&mut self, reference: &Planes, distorted: &Planes) -> Result<f64, VshipError> {
        let mut score = sys::Vship_ScoreSSIMULACRA2 {
            structType: sys::Vship_StructScoreSSIMULACRA2,
            score: 0.0,
        };
        let score_pointer = ptr::from_mut(&mut score).cast::<c_void>();
        let frames = FramePair {
            reference_space: &self.reference,
            distorted_space: &self.distorted,
            reference,
            distorted,
        };
        compute_handler(self.handle, score_pointer, &frames)?;

        Ok(score.score)
    }
}

impl Drop for Ssimulacra2 {
    fn drop(&mut self) {
        free_handler(self.handle);
    }
}

/// A Butteraugli metric on one device, for frames of fixed colour spaces.
#[derive(Debug)]
pub struct Butteraugli {
    handle: sys::Vship_Handle,
    reference: ColorSpace,
    distorted: ColorSpace,
}

// SAFETY: A Vship handler is not tied to the thread that made it. It is not safe to share, so it is
// not `Sync`.
unsafe impl Send for Butteraugli {}

impl Butteraugli {
    /// Makes the metric, where `qnorm` picks the norm reported as `norm_q` and
    /// `intensity_multiplier` is the display brightness in nits.
    pub fn new(
        reference: ColorSpace,
        distorted: ColorSpace,
        qnorm: i32,
        intensity_multiplier: f32,
        gpu_id: i32,
    ) -> Result<Self, VshipError> {
        let mut init = sys::Vship_InitButteraugli_1 {
            structType: sys::Vship_StructInitButteraugli_1,
            src_colorspace: reference.to_raw(),
            dis_colorspace: distorted.to_raw(),
            Qnorm: qnorm,
            intensity_multiplier,
            gpu_id,
        };
        let argument = ptr::from_mut(&mut init).cast::<c_void>();
        let handle = init_handler(argument)?;

        Ok(Self {
            handle,
            reference,
            distorted,
        })
    }

    /// Scores `distorted` against `reference`, where 0 means identical.
    pub fn compute(
        &mut self,
        reference: &Planes,
        distorted: &Planes,
    ) -> Result<ButteraugliScore, VshipError> {
        let mut score = sys::Vship_ScoreButteraugli {
            structType: sys::Vship_StructScoreButteraugli,
            normQ: 0.0,
            norm3: 0.0,
            norminf: 0.0,
            dstp: ptr::null(),
            dststride: 0,
        };
        let score_pointer = ptr::from_mut(&mut score).cast::<c_void>();
        let frames = FramePair {
            reference_space: &self.reference,
            distorted_space: &self.distorted,
            reference,
            distorted,
        };
        compute_handler(self.handle, score_pointer, &frames)?;

        Ok(ButteraugliScore {
            norm_q: score.normQ,
            norm3: score.norm3,
            norm_inf: score.norminf,
        })
    }
}

impl Drop for Butteraugli {
    fn drop(&mut self) {
        free_handler(self.handle);
    }
}

/// The two frames of one compute call and the colour spaces their handler was made with.
struct FramePair<'a> {
    reference_space: &'a ColorSpace,
    distorted_space: &'a ColorSpace,
    reference: &'a Planes<'a>,
    distorted: &'a Planes<'a>,
}

fn init_handler(argument: *mut c_void) -> Result<sys::Vship_Handle, VshipError> {
    let mut handle = ptr::null_mut();
    // SAFETY: `argument` points at a live init struct whose `structType` names its type, and
    // `handle` is a valid out pointer.
    let status = unsafe { sys::Vship_InitHandler(&mut handle, argument) };
    check(status)?;

    Ok(handle)
}

fn compute_handler(
    handle: sys::Vship_Handle,
    score: *mut c_void,
    frames: &FramePair,
) -> Result<(), VshipError> {
    frames.reference.check(frames.reference_space, "the reference")?;
    frames
        .distorted
        .check(frames.distorted_space, "the distorted frame")?;

    let mut reference_pointers = frames.reference.pointers();
    let mut distorted_pointers = frames.distorted.pointers();
    let reference_lines = frames.reference.line_sizes()?;
    let distorted_lines = frames.distorted.line_sizes()?;

    // SAFETY: Both plane sets were checked against the colour spaces the handler was made with, so
    // Vship reads only inside them. `score` points at a live score struct whose `structType` matches
    // the handler's metric, and every pointer outlives the call.
    let status = unsafe {
        sys::Vship_ComputeHandler(
            handle,
            score,
            reference_pointers.as_mut_ptr(),
            distorted_pointers.as_mut_ptr(),
            reference_lines.as_ptr(),
            distorted_lines.as_ptr(),
        )
    };

    check(status)
}

fn free_handler(handle: sys::Vship_Handle) {
    // SAFETY: `handle` came from `Vship_InitHandler` and its owner frees it exactly once, on drop.
    let status = unsafe { sys::Vship_FreeHandler(handle) };

    if status != sys::Vship_NoError {
        tracing::warn!(status, "vship could not free a metric handler");
    }
}
