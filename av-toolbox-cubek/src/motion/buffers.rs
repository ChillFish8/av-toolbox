use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BlockGrid, FieldPlanes, MotionField, SceneFlag};
use crate::DevicePlane;

/// One field's vector, SAD and confidence buffers on the device.
pub struct FieldBuffers {
    vectors: Handle,
    sads: Handle,
    confidence: Handle,
    grid: BlockGrid,
}

impl FieldBuffers {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, grid: &BlockGrid) -> Self {
        let bytes = grid.blocks() * size_of::<f32>();

        Self {
            vectors: client.empty(2 * bytes),
            sads: client.empty(bytes),
            confidence: client.empty(bytes),
            grid: *grid,
        }
    }

    /// The buffers as a field in units of `1 / pel` pixel, with its confidence plane only if `scored`.
    pub fn field(&self, pel: u32, scored: bool) -> MotionField<'_> {
        let (vector_width, vector_height) = self.grid.vector_dims();
        let (sad_width, sad_height) = self.grid.sad_dims();

        // Safety: `new` allocates two 4 byte samples per block for the vectors, and one for each of the
        // SADs and the confidence, and the dimensions come from the same grid.
        let planes = unsafe {
            FieldPlanes {
                vectors: DevicePlane::new(&self.vectors, vector_width, vector_height),
                sads: DevicePlane::new(&self.sads, sad_width, sad_height),
                confidence: scored.then(|| DevicePlane::new(&self.confidence, sad_width, sad_height)),
            }
        };

        MotionField {
            planes,
            grid: self.grid,
            pel,
        }
    }
}

/// A device plane of scene-change flags, one per field.
pub struct SceneFlags {
    handle: Handle,
    count: u32,
}

impl SceneFlags {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, count: u32) -> Self {
        let bytes = count as usize * size_of::<u32>();

        Self {
            handle: client.empty(bytes),
            count,
        }
    }

    /// The flags as a plane `count` wide and 1 high.
    pub fn plane(&self) -> DevicePlane<'_> {
        // Safety: `new` allocates a 4 byte flag for each of the `count` fields.
        unsafe { DevicePlane::new(&self.handle, self.count, 1) }
    }

    /// The flag of the field at `index`.
    pub fn flag(&self, index: u32) -> SceneFlag<'_> {
        assert!(
            index < self.count,
            "the flag index is {index}, but there are {} flags",
            self.count
        );

        SceneFlag {
            flags: self.plane(),
            index,
        }
    }
}
