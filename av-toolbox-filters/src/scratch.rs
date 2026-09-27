use av_toolbox_cubek::DevicePlane;
use av_toolbox_gpu::{FrameContext, PlaneBuffer};
use cubecl::prelude::CubePrimitive;

/// Working memory for one frame, returned to the pool when it drops.
///
/// It is sized for one element type and only ever read as that type.
pub(crate) struct Scratch {
    buffer: PlaneBuffer,
    width: u32,
    height: u32,
}

impl Scratch {
    pub(crate) fn new<E: CubePrimitive>(context: &FrameContext, width: u32, height: u32) -> Self {
        let samples = width as usize * height as usize;
        let bytes = samples * E::type_size();
        let buffer = context.scratch(bytes);

        Self {
            buffer,
            width,
            height,
        }
    }

    pub(crate) fn plane(&self) -> DevicePlane<'_> {
        // Safety: the buffer holds `width * height` elements of the type it was made for, and every
        // caller reads it as that type.
        unsafe { DevicePlane::new(self.buffer.handle(), self.width, self.height) }
    }
}
