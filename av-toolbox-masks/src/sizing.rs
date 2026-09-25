use av_toolbox_cubek::reduce::{ReducePlanes, scratch_len};
use av_toolbox_cubek::{DevicePlane, Dispatch};
use av_toolbox_gpu::{FrameContext, Mask as GpuMask, PlaneBuffer};
use cubecl::prelude::{CubeCount, CubeDim, CubePrimitive};

/// Sizes launches from the frame context a stage is handed.
pub(crate) struct Sizing<'a>(pub(crate) &'a FrameContext);

impl Dispatch for Sizing<'_> {
    fn cube_count(&self, samples: usize) -> CubeCount {
        self.0.cube_count(samples)
    }

    fn cube_dim(&self) -> CubeDim {
        self.0.cube_dim()
    }
}

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

    pub(crate) fn line<E: CubePrimitive>(context: &FrameContext, len: usize) -> Self {
        Self::new::<E>(context, len as u32, 1)
    }

    pub(crate) fn plane(&self) -> DevicePlane<'_> {
        // Safety: the buffer holds `width * height` elements of the type it was made for, and every
        // caller reads it as that type.
        unsafe { DevicePlane::new(self.buffer.handle(), self.width, self.height) }
    }
}

/// The two scratch planes a reduction passes through.
pub(crate) struct Reduction {
    first: Scratch,
    second: Scratch,
}

impl Reduction {
    pub(crate) fn new(context: &FrameContext, samples: usize) -> Self {
        let len = scratch_len(samples);
        let first = Scratch::line::<f32>(context, len);
        let second = Scratch::line::<f32>(context, len);

        Self { first, second }
    }

    pub(crate) fn planes<'a>(&'a self, input: DevicePlane<'a>, result: &'a Scratch) -> ReducePlanes<'a> {
        ReducePlanes {
            input,
            first: self.first.plane(),
            second: self.second.plane(),
            result: result.plane(),
        }
    }
}

pub(crate) fn mask_plane(mask: &GpuMask) -> DevicePlane<'_> {
    let format = mask.format();
    let handle = mask.plane().buffer().handle();

    // Safety: a U8 mask holds exactly `width * height` bytes, and callers read it as u8.
    unsafe { DevicePlane::new(handle, format.width, format.height) }
}
