mod compensate;
mod quality;
mod search;

use av_toolbox_cubek::motion::{BlockGrid, FieldPlanes, MotionField, PlaneScale, Pyramid, level_dims};
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
    pub(crate) fn upload<R: Runtime>(
        client: &ComputeClient<R>,
        luma: &[f32],
        width: u32,
        height: u32,
        levels: u32,
    ) -> Self {
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

    pub(crate) fn read<R: Runtime>(&self, client: &ComputeClient<R>) -> Vec<Vec<f32>> {
        self.levels
            .iter()
            .map(|(handle, _, _)| read_f32(client, handle))
            .collect()
    }
}

/// One field's buffers on the device.
pub(crate) struct DeviceField {
    vectors: Handle,
    sads: Handle,
    confidence: Handle,
    grid: BlockGrid,
}

impl DeviceField {
    pub(crate) fn new<R: Runtime>(client: &ComputeClient<R>, grid: &BlockGrid) -> Self {
        let bytes = grid.blocks() * size_of::<f32>();

        Self {
            vectors: empty(client, 2 * bytes),
            sads: empty(client, bytes),
            confidence: empty(client, bytes),
            grid: *grid,
        }
    }

    /// The buffers as a field in units of `1 / pel` pixel, with its confidence plane only if `scored`.
    pub(crate) fn field(&self, scored: bool, pel: u32) -> MotionField<'_> {
        let (vector_width, vector_height) = self.grid.vector_dims();
        let (sad_width, sad_height) = self.grid.sad_dims();
        let confidence = scored.then(|| plane(&self.confidence, sad_width, sad_height));

        let planes = FieldPlanes {
            vectors: plane(&self.vectors, vector_width, vector_height),
            sads: plane(&self.sads, sad_width, sad_height),
            confidence,
        };

        MotionField {
            planes,
            grid: self.grid,
            pel,
        }
    }

    /// The field as the CPU reference lays it out, its confidence empty unless `scored`.
    pub(crate) fn read<R: Runtime>(&self, client: &ComputeClient<R>, scored: bool) -> ReferenceField {
        let vector_bytes = client
            .read_one(self.vectors.clone())
            .expect("the device should hand back the vectors");
        let confidence = if scored {
            read_f32(client, &self.confidence)
        } else {
            Vec::new()
        };

        ReferenceField {
            vectors: i32::from_bytes(&vector_bytes).to_vec(),
            sads: read_f32(client, &self.sads),
            confidence,
        }
    }
}

/// Panics where the vectors or SADs of `found` differ from `expected`, a field `across` blocks wide.
fn assert_same_field(found: &ReferenceField, expected: &ReferenceField, across: u32, label: &str) {
    let vectors_label = format!("{label} vectors");
    let sads_label = format!("{label} sads");

    assert_same_plane(&found.vectors, &expected.vectors, 2 * across, &vectors_label);
    assert_same_plane(&found.sads, &expected.sads, across, &sads_label);
}
