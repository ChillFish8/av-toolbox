use av_toolbox_gpu::{FrameContext, Mask as GpuMask};

use crate::{Inputs, StageError, StageInstance};

/// A [Mask] takes some input and produces a selection mask used for selecting pixels
/// from two or more sources.
pub trait Mask: StageInstance {
    fn select(&mut self, context: &mut FrameContext, inputs: Inputs) -> Result<GpuMask, StageError>;
}
