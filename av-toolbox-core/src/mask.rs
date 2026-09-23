use av_toolbox_gpu::{DeviceSide, FrameContext, Side};

use crate::{Inputs, StageError, StageInstance};

/// A [Mask] takes some input and produces a selection mask used for selecting pixels
/// from two or more sources.
///
/// The side defaults to the device.
pub trait Mask<S: Side = DeviceSide>: StageInstance {
    fn select(&mut self, context: &mut FrameContext, inputs: Inputs<S>) -> Result<S::Mask, StageError>;
}
