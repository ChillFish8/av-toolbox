use av_toolbox_cubek::resize::Taps;
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

/// Mirrors [av_toolbox_cubek::reflect] on the CPU.
pub fn reflect(index: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }

    let period = 2 * length - 2;
    let folded = index.abs() % period;
    if folded >= length { period - folded } else { folded }
}

/// A CPU blur following the spec, reflect-101 borders and all.
pub fn reference_blur(input: &[f32], width: u32, height: u32, weights: &[f32]) -> Vec<f32> {
    let radius = (weights.len() / 2) as i32;
    let width = width as i32;
    let height = height as i32;

    let mut rows = vec![0.0f32; input.len()];
    for row in 0..height {
        for column in 0..width {
            let mut total = 0.0f32;
            for (tap, weight) in weights.iter().enumerate() {
                let source = reflect(column + tap as i32 - radius, width);
                total += input[(row * width + source) as usize] * weight;
            }
            rows[(row * width + column) as usize] = total;
        }
    }

    let mut output = vec![0.0f32; input.len()];
    for row in 0..height {
        for column in 0..width {
            let mut total = 0.0f32;
            for (tap, weight) in weights.iter().enumerate() {
                let source = reflect(row + tap as i32 - radius, height);
                total += rows[(source * width + column) as usize] * weight;
            }
            output[(row * width + column) as usize] = total;
        }
    }

    output
}

/// Applies `across` then `down` on the CPU, rounding to U8 at the end.
pub fn reference_resize(input: &[u8], width: u32, height: u32, across: &Taps, down: &Taps) -> Vec<u8> {
    let out_width = across.starts.len() as u32;
    let out_height = down.starts.len() as u32;

    let mut rows = vec![0.0f32; (out_width * height) as usize];
    for row in 0..height {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..across.count {
                let tap = across.starts[column as usize] + step as i32;
                let source = reflect(tap, width as i32) as u32;
                let weight = across.weights[(column * across.count + step) as usize];
                total += f32::from(input[(row * width + source) as usize]) * weight;
            }
            rows[(row * out_width + column) as usize] = total;
        }
    }

    let mut output = Vec::with_capacity((out_width * out_height) as usize);
    for row in 0..out_height {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..down.count {
                let tap = down.starts[row as usize] + step as i32;
                let source = reflect(tap, height as i32) as u32;
                let weight = down.weights[(row * down.count + step) as usize];
                total += rows[(source * out_width + column) as usize] * weight;
            }
            output.push(total.round().clamp(0.0, 255.0) as u8);
        }
    }

    output
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
