use av_toolbox_cubek::blur::{BlurPlanes, Gaussian, Storage, gaussian_weights};
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

    check(input, 7, 5, 25.0, 100);
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
