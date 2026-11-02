use av_toolbox_cubek::retinex::{RecursiveGaussian, RecursivePlanes, recursive_gaussian};
use av_toolbox_cubek::testing::{reference_recursive_gaussian, reference_recursive_gaussian_cascade};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

/// The largest value `busy` ever produces.
const BUSY_RANGE: f32 = 255.0;

struct Recursive {
    input: Vec<f32>,
    width: u32,
    height: u32,
    filter: RecursiveGaussian,
}

impl BuildTask for Recursive {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = RecursivePlanes {
            input: input_plane,
            output: output_plane,
        };

        recursive_gaussian(client, &sizing, planes, self.filter);

        read_f32(client, &output)
    }
}

fn on_device(input: &[f32], width: u32, height: u32, filter: RecursiveGaussian) -> Vec<f32> {
    let task = Recursive {
        input: input.to_vec(),
        width,
        height,
        filter,
    };

    backend().build(task)
}

/// `busy` scaled into 0..1.
fn scaled_busy(width: u32, height: u32) -> Vec<f32> {
    let raw = busy(width, height);

    raw.iter().map(|value| value / BUSY_RANGE).collect()
}

/// A plane that steps from 0 to 1 halfway along its width (`along_width`) or its height.
fn step_plane(width: u32, height: u32, along_width: bool) -> Vec<f32> {
    let mut plane = vec![0.0f32; (width * height) as usize];

    for row in 0..height {
        for column in 0..width {
            let index = (row * width + column) as usize;
            let past_midpoint = if along_width {
                column >= width / 2
            } else {
                row >= height / 2
            };

            if past_midpoint {
                plane[index] = 1.0;
            }
        }
    }

    plane
}

/// A plane that steps from 0 to 1 `sigma` samples before the far end of its width
/// (`along_width`) or its height.
fn edge_step_plane(width: u32, height: u32, along_width: bool, sigma: f32) -> Vec<f32> {
    let offset = sigma as u32;
    let mut plane = vec![0.0f32; (width * height) as usize];

    for row in 0..height {
        for column in 0..width {
            let index = (row * width + column) as usize;
            let past_step = if along_width {
                column + offset >= width
            } else {
                row + offset >= height
            };

            if past_step {
                plane[index] = 1.0;
            }
        }
    }

    plane
}

#[test]
fn the_recursive_gaussian_matches_the_reference() {
    let sizes = [(37, 23), (1, 5), (5, 1), (2, 2), (1, 1)];
    let sigmas = [0.5, 2.0, 5.0, 50.0];

    for (width, height) in sizes {
        for sigma in sigmas {
            let input = scaled_busy(width, height);
            let filter = RecursiveGaussian::new(sigma);

            let actual = on_device(&input, width, height, filter);
            let expected = reference_recursive_gaussian(&input, width, height, sigma);

            assert_close(&actual, &expected, 1e-3);
        }
    }
}

#[test]
fn a_sigma_wider_than_the_plane_matches_the_reference() {
    let width = 16;
    let height = 16;
    let input = scaled_busy(width, height);
    let sigma = 350.0;
    let filter = RecursiveGaussian::new(sigma);

    let actual = on_device(&input, width, height, filter);
    let expected = reference_recursive_gaussian(&input, width, height, sigma);

    for value in &actual {
        assert!(value.is_finite(), "{value}");
    }

    assert_close(&actual, &expected, 1e-3);
}

#[test]
fn a_step_along_either_axis_matches_the_reference() {
    let sigmas = [200.0, 350.0];

    for sigma in sigmas {
        let filter = RecursiveGaussian::new(sigma);

        let wide = step_plane(1920, 2, true);
        let actual_wide = on_device(&wide, 1920, 2, filter);
        let expected_wide = reference_recursive_gaussian(&wide, 1920, 2, sigma);

        assert_close(&actual_wide, &expected_wide, 2e-3);

        let tall = step_plane(2, 1080, false);
        let actual_tall = on_device(&tall, 2, 1080, filter);
        let expected_tall = reference_recursive_gaussian(&tall, 2, 1080, sigma);

        assert_close(&actual_tall, &expected_tall, 2e-3);
    }
}

#[test]
fn a_wide_flat_plane_matches_the_reference_closely() {
    let width = 1920;
    let height = 2;
    let input = vec![0.5f32; (width * height) as usize];
    let sigma = 350.0;
    let filter = RecursiveGaussian::new(sigma);

    let actual = on_device(&input, width, height, filter);
    let expected = reference_recursive_gaussian(&input, width, height, sigma);

    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn the_edges_act_as_replicated_on_the_device() {
    let sigmas = [50.0, 350.0];

    for sigma in sigmas {
        let filter = RecursiveGaussian::new(sigma);

        let wide = edge_step_plane(1920, 2, true, sigma);
        let actual_wide = on_device(&wide, 1920, 2, filter);
        let expected_wide = reference_recursive_gaussian(&wide, 1920, 2, sigma);

        assert_close(&actual_wide, &expected_wide, 2e-4);

        let tall = edge_step_plane(2, 1080, false, sigma);
        let actual_tall = on_device(&tall, 2, 1080, filter);
        let expected_tall = reference_recursive_gaussian(&tall, 2, 1080, sigma);

        assert_close(&actual_tall, &expected_tall, 2e-4);
    }
}

#[test]
fn a_step_at_the_widest_sigma_matches_the_reference() {
    let sigma = 1024.0;
    let filter = RecursiveGaussian::new(sigma);
    let input = edge_step_plane(1920, 2, true, sigma);

    let actual = on_device(&input, 1920, 2, filter);
    let mirror = reference_recursive_gaussian_cascade(&input, 1920, 2, filter);
    let expected = reference_recursive_gaussian(&input, 1920, 2, sigma);

    assert_close(&actual, &mirror, 1e-5);
    assert_close(&mirror, &expected, 4e-3);
}
