use std::ffi::c_void;
use std::ptr;

use super::handler::Handler;
use super::{FramePair, check_matching_sizes};
use crate::colour::{ColorSpace, Planes};
use crate::display::DisplayModel;
use crate::distmap::{DistortionMap, MapSize};
use crate::error::VshipError;
use crate::sys;

/// How a [Cvvdp](crate::Cvvdp) models viewing.
#[derive(Debug, Clone, PartialEq)]
pub struct CvvdpOptions {
    /// The frame rate of the video, above 0 and at most 1000.
    pub fps: f32,
    /// Scales frames to the display's resolution before scoring.
    pub resize_to_display: bool,
    pub display: DisplayModel,
}

/// The largest frame rate accepted, as Vship sizes its frame history from it.
const MAX_FPS: f32 = 1000.0;

impl CvvdpOptions {
    fn check(&self) -> Result<(), VshipError> {
        let usable_fps = self.fps.is_finite() && self.fps > 0.0 && self.fps <= MAX_FPS;
        if !usable_fps {
            return Err(unusable_fps_error(self.fps));
        }

        self.display.check()
    }
}

/// A CVVDP metric on one device, scoring a video one frame pair at a time.
///
/// Its score covers every frame computed since the last [reset_score](crate::Cvvdp::reset_score), in
/// JOD where 10 means no visible difference.
#[derive(Debug)]
pub struct Cvvdp {
    handler: Handler,
    reference: ColorSpace,
    distorted: ColorSpace,
    map_size: MapSize,
}

impl Cvvdp {
    pub fn new(
        reference: ColorSpace,
        distorted: ColorSpace,
        options: &CvvdpOptions,
        gpu_id: i32,
    ) -> Result<Self, VshipError> {
        check_matching_sizes(&reference, &distorted)?;
        options.check()?;

        let display_strings = options.display.c_strings()?;
        let mut init = sys::Vship_InitCVVDP_1 {
            structType: sys::Vship_StructInitCVVDP_1,
            src_colorspace: reference.to_raw(),
            dis_colorspace: distorted.to_raw(),
            fps: options.fps,
            resizeToDisplay: options.resize_to_display,
            model_key_cstr: display_strings.key_pointer(),
            model_config_json_cstr: display_strings.config_pointer(),
            gpu_id,
        };
        let argument = (&raw mut init).cast::<c_void>();
        // SAFETY: `argument` points at a live `Vship_InitCVVDP_1` whose `structType` names it. Its
        // string pointers come from `display_strings`, which outlives this call, and Vship copies both
        // during init.
        let handler = unsafe { Handler::init(argument) }?;
        let map_size = handler.distmap_size()?;

        Ok(Self {
            handler,
            reference,
            distorted,
            map_size,
        })
    }

    /// Scores one more frame pair, returning the score of every frame since the last score reset.
    pub fn compute(&mut self, reference: &Planes, distorted: &Planes) -> Result<f64, VshipError> {
        self.score(reference, distorted, ptr::null(), 0)
    }

    /// Scores like [compute](crate::Cvvdp::compute) and fills `map` with this frame's error.
    pub fn compute_with_map(
        &mut self,
        reference: &Planes,
        distorted: &Planes,
        map: &mut DistortionMap,
    ) -> Result<f64, VshipError> {
        let (map_pointer, map_stride) = map.target(self.map_size)?;

        self.score(reference, distorted, map_pointer, map_stride)
    }

    /// Adds a frame pair to the history without scoring it.
    pub fn load_temporal(&mut self, reference: &Planes, distorted: &Planes) -> Result<(), VshipError> {
        let frames = FramePair {
            reference_space: &self.reference,
            distorted_space: &self.distorted,
            reference,
            distorted,
        };

        self.handler.load_temporal(&frames)
    }

    /// Clears the frame history and restarts the running score.
    pub fn reset(&mut self) -> Result<(), VshipError> {
        self.handler.reset()
    }

    /// Restarts the running score, for example at a scene change.
    pub fn reset_score(&mut self) -> Result<(), VshipError> {
        self.handler.reset_score()
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
    ) -> Result<f64, VshipError> {
        let mut score = sys::Vship_ScoreCVVDP {
            structType: sys::Vship_StructScoreCVVDP,
            score: 0.0,
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
        // SAFETY: `score_pointer` points at a live `Vship_ScoreCVVDP`, which is this handler's metric.
        // Its map pointer is null or came from a map checked against this handler's map size.
        unsafe { self.handler.compute(score_pointer, &frames) }?;

        Ok(score.score)
    }
}

fn unusable_fps_error(fps: f32) -> VshipError {
    VshipError::new(format!(
        "cvvdp needs a frame rate above 0 and at most {MAX_FPS}, not {fps}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options_at(fps: f32) -> CvvdpOptions {
        CvvdpOptions {
            fps,
            resize_to_display: false,
            display: DisplayModel::StandardFhd,
        }
    }

    #[test]
    fn a_zero_frame_rate_is_refused() {
        let options = options_at(0.0);

        let error = options.check().expect_err("a zero fps passed");

        assert!(error.message().contains("frame rate"), "{error}");
    }

    #[test]
    fn a_frame_rate_above_1000_is_refused() {
        let options = options_at(1000.5);

        let error = options.check().expect_err("a huge fps passed");

        assert!(error.message().contains("frame rate"), "{error}");
    }

    #[test]
    fn a_nan_frame_rate_is_refused() {
        let options = options_at(f32::NAN);

        let error = options.check().expect_err("a nan fps passed");

        assert!(error.message().contains("frame rate"), "{error}");
    }

    #[test]
    fn a_path_display_config_is_refused_through_the_options() {
        let options = CvvdpOptions {
            display: DisplayModel::Custom {
                name: "my_display".to_string(),
                json: "my_display.json".to_string(),
            },
            ..options_at(25.0)
        };

        let error = options.check().expect_err("a path passed");

        assert!(error.message().contains("inline JSON"), "{error}");
    }
}
