use cubecl::prelude::{ArrayArg, Runtime};
use cubecl::server::Handle;

/// A plane on the device, `width` samples across and `height` down, in row order.
#[derive(Debug, Clone, Copy)]
pub struct DevicePlane<'a> {
    handle: &'a Handle,
    width: u32,
    height: u32,
}

impl<'a> DevicePlane<'a> {
    /// # Safety
    ///
    /// The buffer behind `handle` must hold at least `width * height` elements of the type every
    /// kernel given this plane declares.
    pub const unsafe fn new(handle: &'a Handle, width: u32, height: u32) -> Self {
        Self {
            handle,
            width,
            height,
        }
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub const fn handle(&self) -> &'a Handle {
        self.handle
    }

    /// This plane as a kernel argument.
    pub fn array<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.handle.clone();
        let samples = self.samples();

        // Safety: `new` requires the buffer to hold `samples` elements of the type the kernel declares.
        unsafe { ArrayArg::from_raw_parts(handle, samples) }
    }
}
