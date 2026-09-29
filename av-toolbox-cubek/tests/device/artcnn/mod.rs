mod body;
mod ends;
mod tiling;

use av_toolbox_cubek::artcnn::Precision;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::support::{read_f16, read_f32, upload_f16, upload_f32};

/// Waits for the device and fails the test if a kernel did not compile.
///
/// Reads after a failed launch hand back stale memory.
pub fn synced<R: Runtime>(client: &ComputeClient<R>) {
    let finished = cubecl::future::block_on(client.sync());
    finished.expect("every kernel compiled and ran");
}

pub fn upload_maps<R: Runtime>(client: &ComputeClient<R>, values: &[f32], precision: Precision) -> Handle {
    match precision {
        Precision::F16 => upload_f16(client, values),
        Precision::F32 => upload_f32(client, values),
    }
}

pub fn read_maps<R: Runtime>(client: &ComputeClient<R>, handle: &Handle, precision: Precision) -> Vec<f32> {
    match precision {
        Precision::F16 => read_f16(client, handle),
        Precision::F32 => read_f32(client, handle),
    }
}

/// A plane of `width` by `height` with texture, edges and flat areas, in `0..=1`.
pub fn textured(width: u32, height: u32) -> Vec<f32> {
    let mut plane = Vec::new();

    for row in 0..height {
        for column in 0..width {
            let wave = ((column as f32 * 0.7).sin() * (row as f32 * 0.45).cos() + 1.0) * 0.4;
            let edge = if column + row / 2 > width / 2 { 0.15 } else { 0.0 };
            plane.push((wave + edge).clamp(0.0, 1.0));
        }
    }

    plane
}
