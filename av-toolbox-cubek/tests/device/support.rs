use av_toolbox_cubek::{DevicePlane, Dispatch};
use av_toolbox_gpu::{Backend, Device, FrameContext};
use cubecl::prelude::*;
use cubecl::server::Handle;

pub fn backend() -> Backend {
    Backend::open_default(Device::Default).expect("a backend should open")
}

/// Sizes launches the way a stage would, from a frame context.
pub struct Sizing(pub FrameContext);

impl Dispatch for Sizing {
    fn cube_count(&self, samples: usize) -> CubeCount {
        self.0.cube_count(samples)
    }

    fn cube_dim(&self) -> CubeDim {
        self.0.cube_dim()
    }
}

pub fn upload_f32<R: Runtime>(client: &ComputeClient<R>, values: &[f32]) -> Handle {
    let bytes = f32::as_bytes(values);

    client.create_from_slice(bytes)
}

pub fn upload_u8<R: Runtime>(client: &ComputeClient<R>, values: &[u8]) -> Handle {
    client.create_from_slice(values)
}

pub fn empty<R: Runtime>(client: &ComputeClient<R>, bytes: usize) -> Handle {
    client.empty(bytes)
}

pub fn read_f32<R: Runtime>(client: &ComputeClient<R>, handle: &Handle) -> Vec<f32> {
    let owned = handle.clone();
    let bytes = client
        .read_one(owned)
        .expect("the device should hand back the buffer");

    f32::from_bytes(&bytes).to_vec()
}

pub fn read_u32<R: Runtime>(client: &ComputeClient<R>, handle: &Handle) -> Vec<u32> {
    let owned = handle.clone();
    let bytes = client
        .read_one(owned)
        .expect("the device should hand back the buffer");

    u32::from_bytes(&bytes).to_vec()
}

pub fn read_u8<R: Runtime>(client: &ComputeClient<R>, handle: &Handle) -> Vec<u8> {
    let owned = handle.clone();
    let bytes = client
        .read_one(owned)
        .expect("the device should hand back the buffer");

    bytes.to_vec()
}

/// A plane over `handle`, which the caller made large enough.
pub fn plane(handle: &Handle, width: u32, height: u32) -> DevicePlane<'_> {
    // Safety: every caller in these tests sizes `handle` for `width * height` elements of the type
    // the kernel reads it as.
    unsafe { DevicePlane::new(handle, width, height) }
}

pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());

    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "sample {index} was {got}, expected {want} within {tolerance}",
        );
    }
}
