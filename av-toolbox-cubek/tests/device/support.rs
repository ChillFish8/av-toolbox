use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::{DevicePlane, Dispatch};
use av_toolbox_gpu::{Backend, FrameContext, test_device};
use cubecl::prelude::*;
use cubecl::server::Handle;

pub fn backend() -> Backend {
    Backend::open_default(test_device()).expect("a backend should open")
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

pub fn upload_f32(client: &Client, values: &[f32]) -> Handle {
    let bytes = f32::as_bytes(values);

    client.create_from_slice(bytes)
}

pub fn upload_f16(client: &Client, values: &[f32]) -> Handle {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
        .collect();

    client.create_from_slice(&bytes)
}

pub fn upload_u32(client: &Client, values: &[u32]) -> Handle {
    let bytes = u32::as_bytes(values);

    client.create_from_slice(bytes)
}

pub fn upload_u8(client: &Client, values: &[u8]) -> Handle {
    client.create_from_slice(values)
}

pub fn empty(client: &Client, bytes: usize) -> Handle {
    client.empty(bytes)
}

pub fn read_f32(client: &Client, handle: &Handle) -> Vec<f32> {
    let owned = handle.clone();
    let bytes = client
        .read_one(owned)
        .expect("the device should hand back the buffer");

    f32::from_bytes(&bytes).to_vec()
}

pub fn read_f16(client: &Client, handle: &Handle) -> Vec<f32> {
    let owned = handle.clone();
    let bytes = client
        .read_one(owned)
        .expect("the device should hand back the buffer");
    let (pairs, _) = bytes.as_chunks::<2>();

    pairs
        .iter()
        .map(|pair| half::f16::from_le_bytes(*pair).to_f32())
        .collect()
}

pub fn read_u32(client: &Client, handle: &Handle) -> Vec<u32> {
    let owned = handle.clone();
    let bytes = client
        .read_one(owned)
        .expect("the device should hand back the buffer");

    u32::from_bytes(&bytes).to_vec()
}

pub fn read_u8(client: &Client, handle: &Handle) -> Vec<u8> {
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

pub fn to_bytes(values: &[u16], element: Element) -> Vec<u8> {
    match element {
        Element::U8 => values.iter().map(|value| *value as u8).collect(),
        Element::U16 => values.iter().flat_map(|value| value.to_le_bytes()).collect(),
    }
}

pub fn from_bytes(bytes: &[u8], element: Element) -> Vec<u16> {
    match element {
        Element::U8 => bytes.iter().map(|value| u16::from(*value)).collect(),
        Element::U16 => {
            let (pairs, _) = bytes.as_chunks::<2>();
            pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
        },
    }
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

/// A plane of seeded pseudo-random samples between 0 and 1.
pub fn seeded_plane(width: u32, height: u32) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;

    (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}
