use av_toolbox_cubek::bilateral::{BilateralPlanes, BilateralSettings, bilateral};
use av_toolbox_cubek::blur::{gaussian_weights, radius_for};
use av_toolbox_cubek::testing::{reference_bilateral, reference_blur};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

struct Bilateral {
    input: Vec<f32>,
    reference: Vec<f32>,
    width: u32,
    height: u32,
    settings: BilateralSettings,
}

impl BuildTask for Bilateral {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let reference = upload_f32(client, &self.reference);
        let output = empty(client, bytes);
        let input_plane = plane(&input, self.width, self.height);
        let reference_plane = plane(&reference, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = BilateralPlanes {
            input: input_plane,
            reference: reference_plane,
            output: output_plane,
        };

        bilateral(client, &sizing, planes, self.settings);

        read_f32(client, &output)
    }
}

fn on_device(
    input: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
    settings: BilateralSettings,
) -> Vec<f32> {
    let task = Bilateral {
        input: input.to_vec(),
        reference: reference.to_vec(),
        width,
        height,
        settings,
    };

    backend().build(task)
}

/// A Gaussian blur of `input` on the CPU, standing in for the reference plane vszip's bilateral
/// takes from a separate source.
fn blurred_reference(input: &[f32], width: u32, height: u32, sigma_s: f32) -> Vec<f32> {
    let radius = radius_for(sigma_s);
    let weights = gaussian_weights(sigma_s, radius);

    reference_blur(input, width, height, &weights)
}

/// At sigma 16 on the 5x5 plane, most of the sample positions clamp past the plane's edge.
#[test]
fn the_bilateral_matches_the_reference() {
    let sizes = [(37, 23), (5, 5), (1, 9)];
    let sigma_values = [1.0f32, 5.0, 16.0];
    let sigma_r = 0.02;
    let peak = 255.0;

    let mut greatest_error = 0.0f32;

    for (width, height) in sizes {
        for sigma_s in sigma_values {
            let input = busy(width, height);
            let reference = blurred_reference(&input, width, height, sigma_s);
            let settings = BilateralSettings {
                sigma_s,
                sigma_r,
                peak,
            };

            let actual = on_device(&input, &reference, width, height, settings);
            let expected = reference_bilateral(&input, &reference, width, height, settings);

            for (got, want) in actual.iter().zip(&expected) {
                let error = (got - want).abs();
                greatest_error = greatest_error.max(error);
            }

            // The measured error across every size and sigma stayed at or under 1.7e-4, so 1e-3
            // gives it close to 6x headroom.
            assert_close(&actual, &expected, 1e-3);
        }
    }

    assert!(
        greatest_error < 5e-4,
        "the greatest error was {greatest_error}, expected under 5e-4"
    );
}

#[test]
fn a_flat_plane_stays_flat_on_the_device() {
    let width = 9;
    let height = 9;
    let value = 42.0;
    let input = vec![value; (width * height) as usize];
    let reference = input.clone();
    let settings = BilateralSettings {
        sigma_s: 5.0,
        sigma_r: 0.02,
        peak: 255.0,
    };

    let output = on_device(&input, &reference, width, height, settings);

    assert!(output.iter().all(|sample| sample.is_finite()));

    for sample in output {
        assert!((sample - value).abs() < 1e-4, "{sample} was not {value}");
    }
}
