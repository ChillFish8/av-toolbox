mod average;
mod bob;
mod difference;
mod limits;
mod pointwise;
mod shimmer;
mod steps;

use av_toolbox_cubek::testing::QtgmcFixture;

/// How far a rounded reference sample may sit from havsfunc's.
const FIXTURE_TOLERANCE: i32 = 1;

/// How far a device sample may sit from the CPU, for fused multiply-adds.
const DEVICE_TOLERANCE: f32 = 1e-3;

fn codes(samples: &[i32]) -> Vec<f32> {
    samples.iter().map(|sample| *sample as f32).collect()
}

/// How a rounded reference plane sat against havsfunc's.
struct FixtureDifference {
    /// The largest distance from havsfunc over the plane.
    largest: i32,
    /// How many samples were further than [FIXTURE_TOLERANCE].
    over_tolerance: usize,
}

/// Rounds and clamps `found`, then checks each sample is within `allowed` of `expected`.
///
/// `allowed` gives the room for the sample at a position. A sample over it panics under `label`.
fn assert_fixture_within(
    found: &[f32],
    expected: &[i32],
    ceiling: f32,
    allowed: impl Fn(usize) -> i32,
    label: &str,
) -> FixtureDifference {
    assert_eq!(
        found.len(),
        expected.len(),
        "{label}: {} samples found, havsfunc has {}",
        found.len(),
        expected.len()
    );

    let mut report = FixtureDifference {
        largest: 0,
        over_tolerance: 0,
    };
    for (position, (value, wanted)) in found.iter().zip(expected).enumerate() {
        let rounded = value.round().clamp(0.0, ceiling) as i32;
        let difference = (rounded - wanted).abs();
        let room = allowed(position);
        assert!(
            difference <= room,
            "{label}: sample {position} is {value} (rounded {rounded}), havsfunc has {wanted}, \
             allowed {room}",
        );

        report.largest = report.largest.max(difference);
        if difference > FIXTURE_TOLERANCE {
            report.over_tolerance += 1;
        }
    }

    report
}

fn peak(fixture: &QtgmcFixture) -> f32 {
    ((1u32 << fixture.bits) - 1) as f32
}
