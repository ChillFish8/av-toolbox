mod buffers;
mod compensate;
mod mask;
mod normalise;
mod quality;
mod scene;
mod search;

use av_toolbox_cubek::motion::{FieldBuffers, PlaneScale, Pyramid, level_dims};
use av_toolbox_cubek::testing::{ReferenceField, assert_same_plane};
use av_toolbox_gpu::Subsampling;
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::support::{empty, plane, read_f32, upload_f32};

/// The scale of the chroma planes of a 4:2:0 frame.
pub(crate) const CHROMA: PlaneScale = PlaneScale::from_factors(Subsampling::Yuv420.factors());

/// One frame's pyramid on the device, level 0 uploaded and the rest left for the pyramid build.
pub(crate) struct DevicePyramid {
    levels: Vec<(Handle, u32, u32)>,
}

impl DevicePyramid {
    pub(crate) fn upload(client: &Client, luma: &[f32], width: u32, height: u32, levels: u32) -> Self {
        let mut buffers = vec![(upload_f32(client, luma), width, height)];
        for level in 1..levels {
            let (level_width, level_height) = level_dims(width, height, level);
            let bytes = (level_width * level_height) as usize * size_of::<f32>();
            buffers.push((empty(client, bytes), level_width, level_height));
        }

        Self { levels: buffers }
    }

    pub(crate) fn pyramid(&self) -> Pyramid<'_> {
        let levels = self
            .levels
            .iter()
            .map(|(handle, width, height)| plane(handle, *width, *height))
            .collect();

        Pyramid { levels }
    }

    pub(crate) fn read(&self, client: &Client) -> Vec<Vec<f32>> {
        self.levels
            .iter()
            .map(|(handle, _, _)| read_f32(client, handle))
            .collect()
    }
}

/// The field in `buffers` as the CPU reference lays it out, its confidence empty unless `scored`.
pub(crate) fn read_field(client: &Client, buffers: &FieldBuffers, scored: bool) -> ReferenceField {
    let planes = buffers.field(1, scored).planes;
    let vector_bytes = client
        .read_one(planes.vectors.handle().clone())
        .expect("the device should hand back the vectors");
    let confidence = match planes.confidence {
        Some(confidence) => read_f32(client, confidence.handle()),
        None => Vec::new(),
    };

    ReferenceField {
        vectors: i32::from_bytes(&vector_bytes).to_vec(),
        sads: read_f32(client, planes.sads.handle()),
        confidence,
    }
}

/// Panics where the vectors or SADs of `found` differ from `expected`, a field `across` blocks wide.
fn assert_same_field(found: &ReferenceField, expected: &ReferenceField, across: u32, label: &str) {
    let vectors_label = format!("{label} vectors");
    let sads_label = format!("{label} sads");

    assert_same_plane(&found.vectors, &expected.vectors, 2 * across, &vectors_label);
    assert_same_plane(&found.sads, &expected.sads, across, &sads_label);
}
