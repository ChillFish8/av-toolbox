use std::ffi::c_void;

use super::FramePair;
use super::handler::Handler;
use crate::colour::{ColorSpace, Planes};
use crate::error::VshipError;
use crate::sys;

/// A SSIMULACRA2 metric on one device, for frames of fixed colour spaces.
#[derive(Debug)]
pub struct Ssimulacra2 {
    handler: Handler,
    reference: ColorSpace,
    distorted: ColorSpace,
}

impl Ssimulacra2 {
    pub fn new(reference: ColorSpace, distorted: ColorSpace, gpu_id: i32) -> Result<Self, VshipError> {
        let mut init = sys::Vship_InitSSIMULACRA2_1 {
            structType: sys::Vship_StructInitSSIMULACRA2_1,
            src_colorspace: reference.to_raw(),
            dis_colorspace: distorted.to_raw(),
            gpu_id,
        };
        let argument = (&raw mut init).cast::<c_void>();
        // SAFETY: `argument` points at a live `Vship_InitSSIMULACRA2_1` whose `structType` names it.
        let handler = unsafe { Handler::init(argument) }?;

        Ok(Self {
            handler,
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
        let score_pointer = (&raw mut score).cast::<c_void>();
        let frames = FramePair {
            reference_space: &self.reference,
            distorted_space: &self.distorted,
            reference,
            distorted,
        };
        // SAFETY: `score_pointer` points at a live `Vship_ScoreSSIMULACRA2`, which is this handler's
        // metric.
        unsafe { self.handler.compute(score_pointer, &frames) }?;

        Ok(score.score)
    }
}
