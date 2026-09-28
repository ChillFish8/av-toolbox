use av_toolbox_cubek::edges::{EdgeLimits, Operator, add_capped, binarize, edges};
use av_toolbox_cubek::neighbourhood::NeighbourPlanes;
use av_toolbox_cubek::testing::{reference_add_capped, reference_binarize, reference_edges};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

/// The largest value `busy` ever produces.
const BUSY_RANGE: f32 = 255.0;

enum Operation {
    Edges(Operator, EdgeLimits),
    Binarize(f32, f32),
}

struct Edges {
    input: Vec<f32>,
    width: u32,
    height: u32,
    operation: Operation,
}

impl BuildTask for Edges {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = NeighbourPlanes {
            input: input_plane,
            output: output_plane,
        };

        match self.operation {
            Operation::Edges(operator, limits) => {
                edges(client, &sizing, planes, operator, limits);
            },
            Operation::Binarize(threshold, peak) => {
                binarize(client, &sizing, planes, threshold, peak);
            },
        }

        read_f32(client, &output)
    }
}

fn on_device(input: &[f32], width: u32, height: u32, operation: Operation) -> Vec<f32> {
    let task = Edges {
        input: input.to_vec(),
        width,
        height,
        operation,
    };

    backend().build(task)
}

struct AddCapped {
    first: Vec<f32>,
    second: Vec<f32>,
    width: u32,
    height: u32,
    peak: f32,
}

impl BuildTask for AddCapped {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.first.len() * size_of::<f32>();

        let first = upload_f32(client, &self.first);
        let second = upload_f32(client, &self.second);
        let output = empty(client, bytes);
        let first_plane = plane(&first, self.width, self.height);
        let second_plane = plane(&second, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);

        add_capped(
            client,
            &sizing,
            first_plane,
            second_plane,
            output_plane,
            self.peak,
        );

        read_f32(client, &output)
    }
}

fn on_device_add_capped(first: &[f32], second: &[f32], width: u32, height: u32, peak: f32) -> Vec<f32> {
    let task = AddCapped {
        first: first.to_vec(),
        second: second.to_vec(),
        width,
        height,
        peak,
    };

    backend().build(task)
}

/// Limits wide enough that no magnitude off `busy` reaches them.
fn open_limits() -> EdgeLimits {
    EdgeLimits {
        multi: 1.0,
        low: 0.0,
        high: BUSY_RANGE * 40.0,
        peak: BUSY_RANGE * 40.0,
    }
}

/// The midpoint between the magnitude `percentile` percent of the way through `sorted` (nearest-rank)
/// and its nearest distinct neighbour, so no sample sits within float error of the limit.
///
/// Prefers the next larger distinct magnitude, falling back to the next smaller one when the chosen
/// magnitude ties every sample above it. When every magnitude in `sorted` is the same value `v`, no
/// neighbour exists in either direction, so it returns `v / 2.0` instead, or `0.5` when `v` is `0.0`,
/// clearing a gap away from every sample without landing exactly on one.
fn limit_at_percentile(sorted: &[f32], percentile: f32) -> f32 {
    let position = (percentile / 100.0 * sorted.len() as f32) as usize;
    let index = position.min(sorted.len() - 1);
    let chosen = sorted[index];
    let above = &sorted[index..];
    let below = &sorted[..=index];
    let next_larger = above.iter().copied().find(|value| *value > chosen);
    let next_smaller = below.iter().copied().rev().find(|value| *value < chosen);

    match (next_larger, next_smaller) {
        (Some(larger), _) => (chosen + larger) / 2.0,
        (None, Some(smaller)) => (chosen + smaller) / 2.0,
        (None, None) if chosen == 0.0 => 0.5,
        (None, None) => chosen / 2.0,
    }
}

#[test]
fn every_operator_matches_the_reference_with_open_limits() {
    let operators = [Operator::Prewitt, Operator::Sobel, Operator::Kirsch];

    for operator in operators {
        for (width, height) in [(37, 23), (1, 5), (5, 1), (2, 2)] {
            let input = busy(width, height);
            let limits = open_limits();
            let operation = Operation::Edges(operator, limits);

            let actual = on_device(&input, width, height, operation);
            let expected = reference_edges(&input, width, height, operator, limits);

            assert_close(&actual, &expected, 1e-3);
        }
    }
}

/// Checks the branch shares at 37x23 only, as smaller planes hold too few samples. The
/// device-versus-reference comparison still runs at every size.
#[test]
fn the_percentile_limits_cut_fill_and_pass_a_real_share_each() {
    let sizes = [(37, 23), (1, 5), (5, 1), (2, 2)];

    for operator in [Operator::Prewitt, Operator::Sobel, Operator::Kirsch] {
        for (width, height) in sizes {
            let input = busy(width, height);
            let open = open_limits();
            let magnitudes = reference_edges(&input, width, height, operator, open);
            let mut sorted = magnitudes.clone();
            sorted.sort_by(f32::total_cmp);

            let low = limit_at_percentile(&sorted, 30.0);
            let high = limit_at_percentile(&sorted, 70.0);

            let on_a_limit = magnitudes.iter().any(|value| *value == low || *value == high);
            assert!(
                !on_a_limit,
                "{operator:?} {width}x{height} had a sample on low or high"
            );

            let limits = EdgeLimits {
                multi: 1.0,
                low,
                high,
                peak: BUSY_RANGE * 40.0,
            };

            if (width, height) == (37, 23) {
                let below = magnitudes.iter().filter(|value| **value < low).count();
                let above = magnitudes.iter().filter(|value| **value > high).count();
                let between = magnitudes.len() - below - above;
                let samples = magnitudes.len() as f32;

                assert!(below as f32 / samples >= 0.1, "{operator:?} below was {below}");
                assert!(above as f32 / samples >= 0.1, "{operator:?} above was {above}");
                assert!(
                    between as f32 / samples >= 0.1,
                    "{operator:?} between was {between}"
                );
            }

            let operation = Operation::Edges(operator, limits);
            let actual = on_device(&input, width, height, operation);
            let expected = reference_edges(&input, width, height, operator, limits);

            assert_close(&actual, &expected, 1e-3);
        }
    }
}

#[test]
fn binarize_matches_the_reference_at_the_median() {
    let width = 37;
    let height = 23;
    let input = busy(width, height);

    let mut sorted = input.clone();
    sorted.sort_by(f32::total_cmp);
    let median = sorted[sorted.len() / 2];

    let operation = Operation::Binarize(median, 255.0);
    let actual = on_device(&input, width, height, operation);
    let expected = reference_binarize(&input, median, 255.0);

    assert_eq!(actual, expected);
}

#[test]
fn add_capped_matches_the_reference_and_clamps_to_the_peak() {
    let width = 37;
    let height = 23;
    let peak = BUSY_RANGE;
    let first = busy(width, height);
    let second = busy(width, height);

    let actual = on_device_add_capped(&first, &second, width, height, peak);
    let expected = reference_add_capped(&first, &second, peak);

    assert_eq!(actual, expected);
    assert!(actual.iter().all(|value| *value <= peak));
}
