mod search;

use av_toolbox_cubek::motion::{BlockGrid, FieldPlanes, Pyramid, level_dims};
use av_toolbox_cubek::testing::{ReferenceField, within_ulps};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::support::{empty, plane, read_f32, upload_f32};

/// One frame's pyramid on the device, level 0 uploaded and the rest left for the pyramid build.
struct DevicePyramid {
    levels: Vec<(Handle, u32, u32)>,
}

impl DevicePyramid {
    fn upload<R: Runtime>(
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

    fn pyramid(&self) -> Pyramid<'_> {
        let levels = self
            .levels
            .iter()
            .map(|(handle, width, height)| plane(handle, *width, *height))
            .collect();

        Pyramid { levels }
    }

    fn read<R: Runtime>(&self, client: &ComputeClient<R>) -> Vec<Vec<f32>> {
        self.levels
            .iter()
            .map(|(handle, _, _)| read_f32(client, handle))
            .collect()
    }
}

/// One field's buffers on the device.
struct DeviceField {
    vectors: Handle,
    sads: Handle,
    confidence: Handle,
    grid: BlockGrid,
}

impl DeviceField {
    fn new<R: Runtime>(client: &ComputeClient<R>, grid: &BlockGrid) -> Self {
        let bytes = grid.blocks() * size_of::<f32>();

        Self {
            vectors: empty(client, 2 * bytes),
            sads: empty(client, bytes),
            confidence: empty(client, bytes),
            grid: *grid,
        }
    }

    fn planes(&self, scored: bool) -> FieldPlanes<'_> {
        let confidence = scored.then(|| plane(&self.confidence, self.grid.across, self.grid.down));

        FieldPlanes {
            vectors: plane(&self.vectors, 2 * self.grid.across, self.grid.down),
            sads: plane(&self.sads, self.grid.across, self.grid.down),
            confidence,
        }
    }

    /// The field as the CPU reference lays it out, its confidence empty unless `scored`.
    fn read<R: Runtime>(&self, client: &ComputeClient<R>, scored: bool) -> ReferenceField {
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

/// Panics at the first value where `found` differs from `expected`.
fn assert_same<T: PartialEq + std::fmt::Debug>(found: &[T], expected: &[T], label: &str) {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong length");

    let differing = found
        .iter()
        .zip(expected)
        .position(|(actual, wanted)| actual != wanted);
    if let Some(index) = differing {
        panic!(
            "{label} differs first at value {index}: found {:?}, expected {:?}",
            found[index], expected[index],
        );
    }
}

/// Panics where the vectors or SADs of `found` differ from `expected`.
fn assert_same_field(found: &ReferenceField, expected: &ReferenceField, label: &str) {
    let vectors_label = format!("{label} vectors");
    let sads_label = format!("{label} sads");

    assert_same(&found.vectors, &expected.vectors, &vectors_label);
    assert_same(&found.sads, &expected.sads, &sads_label);
}

/// How far a confidence may sit from another device's or the CPU's, as division rounds differently.
const CONFIDENCE_ULPS: u32 = 2;

/// Panics at the first confidence further than [CONFIDENCE_ULPS] from `expected`.
fn assert_close_confidences(found: &[f32], expected: &[f32], label: &str) {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong length");

    let differing = found
        .iter()
        .zip(expected)
        .position(|(actual, wanted)| !within_ulps(*actual, *wanted, CONFIDENCE_ULPS));
    if let Some(index) = differing {
        panic!(
            "{label} differs first at block {index}: found {}, expected {}",
            found[index], expected[index],
        );
    }
}
