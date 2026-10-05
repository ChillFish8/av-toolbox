mod average;
mod bob;
mod pointwise;
mod shimmer;

use av_toolbox_cubek::testing::QtgmcFixture;

/// How far a rounded reference sample may sit from havsfunc's.
const FIXTURE_TOLERANCE: i32 = 1;

/// How far a device sample may sit from the CPU, for fused multiply-adds.
const DEVICE_TOLERANCE: f32 = 1e-3;

fn codes(samples: &[i32]) -> Vec<f32> {
    samples.iter().map(|sample| *sample as f32).collect()
}

fn peak(fixture: &QtgmcFixture) -> f32 {
    ((1u32 << fixture.bits) - 1) as f32
}
