mod body;
mod ends;
mod fused;
mod kernels;
mod tiling;
mod tune;
mod winograd;

use av_toolbox_cubek::artcnn::{Precision, offers_cmma_16};
use cubecl::prelude::*;
use cubecl::server::Handle;

pub(crate) use self::body::{f16_step, supported_runs};
pub(crate) use self::kernels::{edged_maps, region};
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

/// Whether the device takes the forced blueprint on a large f16 region.
pub fn forced_eligible<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let plane_size_max = client.properties().hardware.plane_size_max;
    let offers_matrix = offers_cmma_16(client);

    plane_size_max == 64 && offers_matrix
}

/// `values` map values between -2 and 2, varying from each value to the next.
pub fn wavy_maps(values: usize) -> Vec<f32> {
    (0..values)
        .map(|index| (index as f32 * 0.37).sin() * 2.0)
        .collect()
}

pub fn bits_differ((forced, inferred): (&f32, &f32)) -> bool {
    forced.to_bits() != inferred.to_bits()
}

/// One convolution of the same maps with the forced blueprint allowed, and with the inferred one.
pub struct BlueprintRuns {
    pub forced: Vec<f32>,
    pub inferred: Vec<f32>,
    pub forced_ran: bool,
    pub eligible: bool,
}

impl BlueprintRuns {
    /// Asserts the forced blueprint ran exactly where the device is eligible, and matches the inferred one
    /// bit for bit there.
    pub fn assert_forced_matches_inferred(&self, name: &str) {
        println!(
            "{name}: forced ran {}, eligible {}",
            self.forced_ran, self.eligible
        );
        if !self.eligible {
            assert!(
                !self.forced_ran,
                "{name}: the forced blueprint ran on an ineligible device"
            );
            return;
        }

        assert!(
            self.forced_ran,
            "{name}: a large region on an eligible device kept the inferred blueprint"
        );
        assert_eq!(self.forced.len(), self.inferred.len());

        let values = self.forced.len();
        let pairs = self.forced.iter().zip(&self.inferred);
        let differing = pairs.clone().filter(|pair| bits_differ(*pair)).count();
        let first = pairs.clone().position(bits_differ);

        if let Some(index) = first {
            let forced = self.forced[index];
            let inferred = self.inferred[index];
            panic!(
                "{name}: {differing} of {values} values differ, first at {index}: forced {forced}, inferred \
                 {inferred}"
            );
        }
    }
}

/// How many of `values` differ from the separate passes, and the largest difference in f16 steps.
///
/// A fused write-out may apply the epilogue before the convolution is rounded to f16, so the step is taken at
/// the largest of the value, the separate passes' value and the bare convolution's.
pub fn steps_from_separate(values: &[f32], separate: &[f32], convolved: &[f32]) -> (usize, f32) {
    let mut differing = 0;
    let mut largest_share = 0.0f32;

    for (index, value) in values.iter().copied().enumerate() {
        let separate_value = separate[index];
        let convolved_value = convolved[index];
        let magnitude = value.abs().max(separate_value.abs()).max(convolved_value.abs());
        let step = f16_step(magnitude);
        let share = (value - separate_value).abs() / step;

        largest_share = largest_share.max(share);
        if value != separate_value {
            differing += 1;
        }
    }

    (differing, largest_share)
}
