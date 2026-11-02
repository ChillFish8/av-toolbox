use av_toolbox_cubek::reduce::{ReducePlanes, extremes, scratch_len, sums};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// Reduces `input` on the device, handing back its extremes and its sums.
struct Reduce {
    input: Vec<f32>,
}

impl BuildTask for Reduce {
    type Output = (Vec<f32>, Vec<f32>);

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let samples = self.input.len();
        let scratch_bytes = scratch_len(samples) * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let first = empty(client, scratch_bytes);
        let second = empty(client, scratch_bytes);
        let low_high = empty(client, 2 * size_of::<f32>());
        let totals = empty(client, 2 * size_of::<f32>());

        let width = samples as u32;
        let scratch_width = scratch_len(samples) as u32;
        let input_plane = plane(&input, width, 1);
        let first_plane = plane(&first, scratch_width, 1);
        let second_plane = plane(&second, scratch_width, 1);
        let low_high_plane = plane(&low_high, 2, 1);
        let extremes_planes = ReducePlanes {
            input: input_plane,
            first: first_plane,
            second: second_plane,
            result: low_high_plane,
        };
        let totals_plane = plane(&totals, 2, 1);
        let sums_planes = ReducePlanes {
            result: totals_plane,
            ..extremes_planes
        };

        extremes(client, &sizing, extremes_planes);
        sums(client, &sizing, sums_planes);

        let found_extremes = read_f32(client, &low_high);
        let found_sums = read_f32(client, &totals);

        (found_extremes, found_sums)
    }
}

fn values(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|index| ((index * 7919) % 1000) as f32 / 1000.0 - 0.3)
        .collect()
}

fn check(samples: usize) {
    let input = values(samples);
    let low = input.iter().copied().fold(f32::INFINITY, f32::min);
    let high = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = input.iter().map(|value| f64::from(*value)).sum();
    let squares: f64 = input
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum();

    let (found_extremes, found_sums) = backend().build(Reduce { input });

    assert_eq!(found_extremes, vec![low, high]);
    let total_error = (f64::from(found_sums[0]) - total).abs();
    let squares_error = (f64::from(found_sums[1]) - squares).abs();
    assert!(
        total_error <= total.abs() * 1e-4 + 1e-3,
        "sum was {}, expected {total}",
        found_sums[0]
    );
    assert!(
        squares_error <= squares * 1e-4 + 1e-3,
        "squares were {}, expected {squares}",
        found_sums[1]
    );
}

#[test]
fn a_tiny_plane_reduces_in_one_pass() {
    check(35);
}

#[test]
fn a_plane_needing_two_passes_reduces() {
    check(1_000);
}

#[test]
fn a_plane_needing_three_passes_reduces() {
    check(70_001);
}
