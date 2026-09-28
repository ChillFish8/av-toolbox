use av_toolbox_cubek::neighbourhood::{
    Extreme,
    MinBlurPlanes,
    NeighbourPlanes,
    Shape,
    Smooth,
    Spread,
    extreme,
    median,
    min_blur,
    prewitt,
    smooth,
    spread,
};
use av_toolbox_cubek::testing::{
    reference_extreme,
    reference_median,
    reference_min_blur,
    reference_prewitt,
    reference_smooth,
    reference_spread,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

/// Which neighbourhood operation a device run performs.
#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Extreme(Extreme, Shape),
    Smooth(Smooth),
    Median(u32),
    Prewitt(f32),
    MinBlur(u32),
    Spread(Spread),
}

struct Neighbours {
    input: Vec<f32>,
    width: u32,
    height: u32,
    operation: Operation,
}

impl BuildTask for Neighbours {
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

        match self.operation {
            Operation::Extreme(which, shape) => {
                let planes = NeighbourPlanes {
                    input: input_plane,
                    output: output_plane,
                };
                extreme(client, &sizing, planes, which, shape);
            },
            Operation::Smooth(which) => {
                let planes = NeighbourPlanes {
                    input: input_plane,
                    output: output_plane,
                };
                smooth(client, &sizing, planes, which);
            },
            Operation::Median(radius) => {
                let planes = NeighbourPlanes {
                    input: input_plane,
                    output: output_plane,
                };
                median(client, &sizing, planes, radius);
            },
            Operation::Prewitt(peak) => {
                let planes = NeighbourPlanes {
                    input: input_plane,
                    output: output_plane,
                };
                prewitt(client, &sizing, planes, peak);
            },
            Operation::MinBlur(radius) => {
                let blurred = empty(client, bytes);
                let median_plane = empty(client, bytes);
                let scratch = empty(client, bytes);
                let planes = MinBlurPlanes {
                    input: input_plane,
                    blurred: plane(&blurred, self.width, self.height),
                    median: plane(&median_plane, self.width, self.height),
                    scratch: plane(&scratch, self.width, self.height),
                    output: output_plane,
                };
                min_blur(client, &sizing, planes, radius);
            },
            Operation::Spread(which) => {
                let planes = NeighbourPlanes {
                    input: input_plane,
                    output: output_plane,
                };
                spread(client, &sizing, planes, which);
            },
        }

        read_f32(client, &output)
    }
}

pub fn on_device(input: &[f32], width: u32, height: u32, operation: Operation) -> Vec<f32> {
    let task = Neighbours {
        input: input.to_vec(),
        width,
        height,
        operation,
    };

    backend().build(task)
}

/// A busy integer-valued plane.
pub fn busy(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            ((column * 37 + row * 11 + (column * row) % 17) % 256) as f32
        })
        .collect()
}

#[test]
fn every_extreme_and_shape_matches_the_reference() {
    let shapes = [Shape::Square, Shape::Plus, Shape::Horizontal, Shape::Vertical];

    for which in [Extreme::Maximum, Extreme::Minimum] {
        for shape in shapes {
            for (width, height) in [(37, 23), (1, 5), (4, 1)] {
                let input = busy(width, height);
                let operation = Operation::Extreme(which, shape);

                let actual = on_device(&input, width, height, operation);
                let expected = reference_extreme(&input, width, height, which, shape);

                assert_eq!(actual, expected, "{which:?} {shape:?} {width}x{height}");
            }
        }
    }
}

#[test]
fn both_blurs_match_the_reference() {
    for which in [Smooth::Binomial, Smooth::Box] {
        for (width, height) in [(37, 23), (1, 5), (2, 5), (5, 1), (5, 2)] {
            let input = busy(width, height);
            let operation = Operation::Smooth(which);

            let actual = on_device(&input, width, height, operation);
            let expected = reference_smooth(&input, width, height, which);

            assert_close(&actual, &expected, 1e-3);
        }
    }
}

#[test]
fn every_median_radius_matches_the_reference() {
    for radius in 1..=3 {
        for (width, height) in [(29, 17), (2, 9)] {
            let input = busy(width, height);
            let operation = Operation::Median(radius);

            let actual = on_device(&input, width, height, operation);
            let expected = reference_median(&input, width, height, radius);

            assert_eq!(actual, expected, "radius {radius} {width}x{height}");
        }
    }
}

#[test]
fn prewitt_matches_the_reference() {
    for (width, height) in [(37, 23), (1, 5), (2, 5), (5, 1), (5, 2)] {
        let input = busy(width, height);
        let operation = Operation::Prewitt(255.0);

        let actual = on_device(&input, width, height, operation);
        let expected = reference_prewitt(&input, width, height, 255.0);

        assert_eq!(actual, expected, "{width}x{height}");
    }
}

fn triangle(value: u32, period: u32) -> u32 {
    let position = value % period;
    let half = period / 2;

    if position <= half {
        position
    } else {
        period - position
    }
}

/// A plane whose ramps and turning points spread MinBlur's radius 2 pick across all three of its
/// branches.
fn branchy(width: u32, height: u32) -> Vec<f32> {
    let noise = busy(width, height);

    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            let horizontal = triangle(column, 13) as f32 * 15.0;
            let vertical = triangle(row, 6) as f32 * 10.0;
            let local_noise = noise[index as usize] % 11.0;

            horizontal + vertical + local_noise
        })
        .collect()
}

/// How many samples of `input` land on each of MinBlur's radius 2 branches, on the CPU.
fn radius_2_branch_counts(input: &[f32], width: u32, height: u32) -> (usize, usize, usize) {
    let blurred_once = reference_smooth(input, width, height, Smooth::Binomial);
    let blurred = reference_smooth(&blurred_once, width, height, Smooth::Box);
    let median_plane = reference_median(input, width, height, 2);

    let mut source_count = 0;
    let mut blurred_count = 0;
    let mut median_count = 0;

    for index in 0..input.len() {
        let toward_blur = input[index] - blurred[index];
        let toward_median = input[index] - median_plane[index];

        if toward_blur * toward_median < 0.0 {
            source_count += 1;
        } else if toward_blur.abs() < toward_median.abs() {
            blurred_count += 1;
        } else {
            median_count += 1;
        }
    }

    (source_count, blurred_count, median_count)
}

#[test]
fn every_min_blur_radius_matches_the_reference() {
    for radius in 1..=3 {
        for (width, height) in [(29, 17), (2, 9)] {
            let input = busy(width, height);
            let operation = Operation::MinBlur(radius);

            let actual = on_device(&input, width, height, operation);
            let expected = reference_min_blur(&input, width, height, radius);

            assert_close(&actual, &expected, 1e-3);
        }
    }

    let width = 29;
    let height = 17;
    let input = branchy(width, height);
    let operation = Operation::MinBlur(2);

    let actual = on_device(&input, width, height, operation);
    let expected = reference_min_blur(&input, width, height, 2);
    assert_close(&actual, &expected, 1e-3);

    let samples = (width * height) as f32;
    let (source_count, blurred_count, median_count) = radius_2_branch_counts(&input, width, height);
    let source_share = source_count as f32 / samples;
    let blurred_share = blurred_count as f32 / samples;
    let median_share = median_count as f32 / samples;

    assert!(source_share >= 0.05, "source branch was only {source_share}");
    assert!(blurred_share >= 0.05, "blurred branch was only {blurred_share}");
    assert!(median_share >= 0.05, "median branch was only {median_share}");
}

/// How many samples of `expected` differ from `input` by more than `tolerance`, out of the total.
fn changed_share(input: &[f32], expected: &[f32], tolerance: f32) -> f32 {
    let changed = input
        .iter()
        .zip(expected.iter())
        .filter(|(before, after)| (*before - *after).abs() > tolerance)
        .count();

    changed as f32 / input.len() as f32
}

#[test]
fn both_spreads_match_the_reference() {
    for which in [Spread::Inflate, Spread::Deflate] {
        for (width, height) in [(37, 23), (1, 5), (2, 2)] {
            let input = busy(width, height);
            let operation = Operation::Spread(which);

            let actual = on_device(&input, width, height, operation);
            let expected = reference_spread(&input, width, height, which);

            assert_close(&actual, &expected, 1e-3);
        }
    }

    let width = 37;
    let height = 23;
    let input = busy(width, height);

    for which in [Spread::Inflate, Spread::Deflate] {
        let expected = reference_spread(&input, width, height, which);
        let changed_fraction = changed_share(&input, &expected, 1e-6);
        let unchanged_fraction = 1.0 - changed_fraction;

        assert!(
            changed_fraction >= 0.1,
            "{which:?} changed only {changed_fraction}"
        );
        assert!(
            unchanged_fraction >= 0.1,
            "{which:?} left only {unchanged_fraction} alone"
        );
    }
}
