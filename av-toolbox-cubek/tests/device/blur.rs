use av_toolbox_cubek::blur::{
    BlurPlanes,
    Gaussian,
    MAX_RADIUS,
    MAX_SIGMA,
    STAGED_MAX_RADIUS,
    Storage,
    TILED_MAX_RADIUS,
    WIDE_MIN_RADIUS,
    gaussian_weights,
};
use av_toolbox_cubek::testing::reference_blur;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{
    Sizing,
    assert_close,
    backend,
    empty,
    plane,
    read_f16,
    read_f32,
    upload_f16,
    upload_f32,
};

/// Blurs `input` on the device with `sigma` and the given radius.
struct Blur {
    input: Vec<f32>,
    width: u32,
    height: u32,
    sigma: f32,
    radius: u32,
    storage: Storage,
}

impl BuildTask for Blur {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * self.storage.bytes();

        let input = match self.storage {
            Storage::F32 => upload_f32(client, &self.input),
            Storage::F16 => upload_f16(client, &self.input),
        };
        let scratch = empty(client, bytes);
        let output = empty(client, bytes);
        let input_plane = plane(&input, self.width, self.height);
        let scratch_plane = plane(&scratch, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = BlurPlanes {
            input: input_plane,
            scratch: scratch_plane,
            output: output_plane,
        };

        let gaussian = Gaussian::with_radius(client, self.sigma, self.radius);
        gaussian.apply_as(client, &sizing, planes, self.storage);

        match self.storage {
            Storage::F32 => read_f32(client, &output),
            Storage::F16 => read_f16(client, &output),
        }
    }
}

fn seeded_plane(width: u32, height: u32) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;

    (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn step_edge_plane(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|index| if index % width < width / 2 { 0.0 } else { 1.0 })
        .collect()
}

fn check(input: Vec<f32>, width: u32, height: u32, sigma: f32, radius: u32) {
    check_as(input, width, height, sigma, radius, Storage::F32, 1e-5);
}

fn check_as(
    input: Vec<f32>,
    width: u32,
    height: u32,
    sigma: f32,
    radius: u32,
    storage: Storage,
    tolerance: f32,
) {
    let weights = gaussian_weights(sigma, radius);
    let expected = reference_blur(&input, width, height, &weights);

    let blurred = backend().build(Blur {
        input,
        width,
        height,
        sigma,
        radius,
        storage,
    });

    assert_close(&blurred, &expected, tolerance);
}

#[test]
fn an_impulse_blurs_into_the_kernel() {
    let mut input = vec![0.0f32; 9 * 7];
    input[3 * 9 + 4] = 1.0;

    check(input, 9, 7, 1.0, 3);
}

#[test]
fn a_ramp_of_odd_size_blurs_like_the_reference() {
    let input: Vec<f32> = (0..35).map(|index| index as f32 / 35.0).collect();

    check(input, 7, 5, 1.0, 4);
}

#[test]
fn a_radius_wider_than_the_plane_folds_back_inside() {
    let input: Vec<f32> = (0..35).map(|index| (index % 4) as f32).collect();

    check(input, 7, 5, 30.0, 100);
}

/// The wide limit of 1e-3 is scaled by the data's range of 3.
#[test]
fn a_wide_radius_wider_than_the_plane_folds_back_inside() {
    let input: Vec<f32> = (0..35).map(|index| (index % 4) as f32).collect();

    check_as(input, 7, 5, 25.0, 100, Storage::F32, 3e-3);
}

#[test]
fn a_single_row_blurs_only_along_the_row() {
    let input: Vec<f32> = (0..9).map(|index| index as f32).collect();

    check(input, 9, 1, 1.0, 2);
}

#[test]
fn an_f16_ramp_blurs_like_the_reference() {
    let input: Vec<f32> = (0..33 * 17)
        .map(|index| index as f32 / (33 * 17 - 1) as f32)
        .collect();

    check_as(input, 33, 17, 1.0, 4, Storage::F16, 2e-3);
}

#[test]
fn an_f16_plane_near_one_stays_within_rounding() {
    let input = vec![0.999f32; 24 * 20];

    check_as(input, 24, 20, 1.6, 7, Storage::F16, 2e-3);
}

#[test]
fn an_f16_zero_plane_stays_zero() {
    let input = vec![0.0f32; 16 * 16];
    let blurred = backend().build(Blur {
        input,
        width: 16,
        height: 16,
        sigma: 1.0,
        radius: 3,
        storage: Storage::F16,
    });

    assert!(blurred.iter().all(|sample| *sample == 0.0));
}

#[test]
fn every_tiled_radius_matches_the_reference() {
    for radius in 1..=TILED_MAX_RADIUS {
        let input = seeded_plane(1921, 1081);
        let sigma = radius as f32 / 4.0;

        check(input, 1921, 1081, sigma, radius);
    }
}

#[test]
fn a_tiled_blur_of_a_tiny_plane_folds_back_inside() {
    let sigma = TILED_MAX_RADIUS as f32 / 4.0;

    for (width, height) in [(1, 7), (7, 1), (3, 3)] {
        let input = seeded_plane(width, height);

        check(input, width, height, sigma, TILED_MAX_RADIUS);
    }
}

#[test]
fn a_tiled_blur_matches_in_f16() {
    let input = seeded_plane(1921, 1081);

    check_as(input, 1921, 1081, 1.0, 4, Storage::F16, 2e-3);
}

/// A sigma whose kernel is cut short at `radius`, so it stays on the staged tier.
fn truncated_sigma(radius: u32) -> f32 {
    radius as f32 / 3.0
}

#[test]
fn every_staged_radius_matches_the_reference() {
    for radius in [TILED_MAX_RADIUS + 1, 24, 48] {
        let input = seeded_plane(1921, 1081);
        let sigma = radius as f32 / 4.0;

        check(input, 1921, 1081, sigma, radius);
    }
}

#[test]
fn a_truncated_kernel_past_the_wide_minimum_matches_the_reference() {
    for radius in [100, 193] {
        let input = seeded_plane(1921, 1081);
        let sigma = truncated_sigma(radius);

        check(input, 1921, 1081, sigma, radius);
    }
}

#[test]
fn the_widest_staged_radius_matches_the_reference() {
    let input = seeded_plane(64, 48);

    check(input, 64, 48, MAX_SIGMA, STAGED_MAX_RADIUS - 1);
}

#[test]
fn a_staged_blur_of_a_tiny_plane_folds_back_inside() {
    for radius in [100, 193] {
        let sigma = truncated_sigma(radius);

        for (width, height) in [(7, 5), (1, 7), (7, 1)] {
            let input = seeded_plane(width, height);

            check(input, width, height, sigma, radius);
        }
    }
}

#[test]
fn a_staged_blur_matches_in_f16() {
    let input = seeded_plane(1921, 1081);

    check_as(input, 1921, 1081, 12.0, 48, Storage::F16, 2e-3);
}

#[test]
fn a_wide_blur_stays_within_the_limit() {
    let random = seeded_plane(1921, 1081);
    let step = step_edge_plane(1921, 1081);

    for input in [random, step] {
        check_as(input, 1921, 1081, 25.0, 100, Storage::F32, 1e-3);
    }
}

#[test]
fn every_wide_radius_stays_within_the_limit() {
    for radius in [WIDE_MIN_RADIUS, 64, 193] {
        let input = seeded_plane(1921, 1081);
        let sigma = radius as f32 / 4.0;

        check_as(input, 1921, 1081, sigma, radius, Storage::F32, 1e-3);
    }
}

#[test]
fn the_widest_blur_stays_within_the_limit() {
    let input = seeded_plane(64, 48);

    check_as(input, 64, 48, MAX_SIGMA, MAX_RADIUS, Storage::F32, 1e-3);
}

#[test]
fn a_wide_blur_of_a_tiny_plane_folds_back_inside() {
    for (width, height) in [(7, 5), (1, 7), (7, 1)] {
        let input = seeded_plane(width, height);

        check_as(input, width, height, 25.0, 100, Storage::F32, 1e-3);
    }
}

#[test]
fn a_wide_blur_matches_in_f16() {
    let input = seeded_plane(1921, 1081);

    check_as(input, 1921, 1081, 25.0, 100, Storage::F16, 2e-3);
}
