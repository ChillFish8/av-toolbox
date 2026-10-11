use std::ffi::c_void;
use std::ptr;

use super::handler::Handler;
use super::{FramePair, check_matching_sizes};
use crate::colour::{ColorSpace, Planes};
use crate::distmap::{DistortionMap, MapSize};
use crate::error::VshipError;
use crate::sys;

/// The three Butteraugli norms of one frame pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButteraugliScore {
    pub norm_q: f64,
    pub norm3: f64,
    pub norm_inf: f64,
}

/// A Butteraugli metric on one device, for frames of fixed colour spaces.
#[derive(Debug)]
pub struct Butteraugli {
    handler: Handler,
    reference: ColorSpace,
    distorted: ColorSpace,
    map_size: MapSize,
}

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
        check_matching_sizes(&reference, &distorted)?;

        let mut init = sys::Vship_InitButteraugli_1 {
            structType: sys::Vship_StructInitButteraugli_1,
            src_colorspace: reference.to_raw(),
            dis_colorspace: distorted.to_raw(),
            Qnorm: qnorm,
            intensity_multiplier,
            gpu_id,
        };
        let argument = (&raw mut init).cast::<c_void>();
        // SAFETY: `argument` points at a live `Vship_InitButteraugli_1` whose `structType` names it.
        let handler = unsafe { Handler::init(argument) }?;
        let map_size = handler.distmap_size()?;

        Ok(Self {
            handler,
            reference,
            distorted,
            map_size,
        })
    }

    /// Scores `distorted` against `reference`, where 0 means identical.
    pub fn compute(
        &mut self,
        reference: &Planes,
        distorted: &Planes,
    ) -> Result<ButteraugliScore, VshipError> {
        self.score(reference, distorted, ptr::null(), 0)
    }

    /// Scores like [compute](crate::Butteraugli::compute) and fills `map` with the per-pixel error.
    pub fn compute_with_map(
        &mut self,
        reference: &Planes,
        distorted: &Planes,
        map: &mut DistortionMap,
    ) -> Result<ButteraugliScore, VshipError> {
        let (map_pointer, map_stride) = map.target(self.map_size)?;

        self.score(reference, distorted, map_pointer, map_stride)
    }

    /// A zero filled map at the size this metric writes.
    pub fn distortion_map(&self) -> DistortionMap {
        DistortionMap::new(self.map_size)
    }

    fn score(
        &mut self,
        reference: &Planes,
        distorted: &Planes,
        map_pointer: *const u8,
        map_stride: i64,
    ) -> Result<ButteraugliScore, VshipError> {
        let mut score = sys::Vship_ScoreButteraugli {
            structType: sys::Vship_StructScoreButteraugli,
            normQ: 0.0,
            norm3: 0.0,
            norminf: 0.0,
            dstp: map_pointer,
            dststride: map_stride,
        };
        let score_pointer = (&raw mut score).cast::<c_void>();
        let frames = FramePair {
            reference_space: &self.reference,
            distorted_space: &self.distorted,
            reference,
            distorted,
        };
        // SAFETY: `score_pointer` points at a live `Vship_ScoreButteraugli`, which is this handler's
        // metric. Its map pointer is null or came from a map checked against this handler's map size.
        unsafe { self.handler.compute(score_pointer, &frames) }?;

        Ok(ButteraugliScore {
            norm_q: score.normQ,
            norm3: score.norm3,
            norm_inf: score.norminf,
        })
    }
}
