use av_toolbox_cubek::histogram::BINS;
use av_toolbox_cubek::reduce::scratch_len;
use av_toolbox_cubek::retinex::{RetinexPlanes, RetinexSettings, retinex};
use av_toolbox_cubek::testing::reference_retinex;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// What `on_device` reads back. `output` is the stretched result and `msr` is the log response
/// before stretching.
struct RetinexOutput {
    output: Vec<f32>,
    msr: Vec<f32>,
}

struct Retinex {
    input: Vec<f32>,
    width: u32,
    height: u32,
    settings: RetinexSettings,
}

impl BuildTask for Retinex {
    type Output = RetinexOutput;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let samples = self.input.len();
        let scratch_width = scratch_len(samples) as u32;
        let scratch_bytes = scratch_len(samples) * size_of::<f32>();
        let plane_bytes = samples * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let normalised = empty(client, plane_bytes);
        let blurred = empty(client, plane_bytes);
        let msr = empty(client, plane_bytes);
        let output = empty(client, plane_bytes);
        let first = empty(client, scratch_bytes);
        let second = empty(client, scratch_bytes);
        let range = empty(client, 2 * size_of::<f32>());
        let bins = empty(client, BINS as usize * size_of::<u32>());
        let low = empty(client, size_of::<f32>());
        let high = empty(client, size_of::<f32>());

        let input_plane = plane(&input, self.width, self.height);
        let normalised_plane = plane(&normalised, self.width, self.height);
        let blurred_plane = plane(&blurred, self.width, self.height);
        let msr_plane = plane(&msr, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let first_plane = plane(&first, scratch_width, 1);
        let second_plane = plane(&second, scratch_width, 1);
        let range_plane = plane(&range, 2, 1);
        let bins_plane = plane(&bins, BINS, 1);
        let low_plane = plane(&low, 1, 1);
        let high_plane = plane(&high, 1, 1);
        let planes = RetinexPlanes {
            input: input_plane,
            normalised: normalised_plane,
            blurred: blurred_plane,
            msr: msr_plane,
            output: output_plane,
            first: first_plane,
            second: second_plane,
            range: range_plane,
            bins: bins_plane,
            low: low_plane,
            high: high_plane,
        };

        retinex(client, &sizing, planes, &self.settings);

        let output_values = read_f32(client, &output);
        let msr_values = read_f32(client, &msr);

        RetinexOutput {
            output: output_values,
            msr: msr_values,
        }
    }
}

fn on_device(input: &[f32], width: u32, height: u32, settings: &RetinexSettings) -> RetinexOutput {
    let task = Retinex {
        input: input.to_vec(),
        width,
        height,
        settings: settings.clone(),
    };

    backend().build(task)
}

fn default_settings(scale: f32) -> RetinexSettings {
    RetinexSettings::new(vec![50.0, 200.0, 350.0], 0.001, 0.005, scale)
}

/// A scene with a smooth dark half and a smooth bright half.
fn smooth_scene(width: u32, height: u32) -> Vec<f32> {
    let half = width / 2;

    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            let phase = (column + row) as f32 * 0.3;
            let base = if column < half { 40.0 } else { 200.0 };

            base + 15.0 * phase.sin()
        })
        .collect()
}

#[test]
fn the_retinex_matches_the_reference() {
    let scale = 255.0;
    let tight_tolerance = 1e-3 * scale;
    let loose_tolerance = 1e-2 * scale;
    let settings = default_settings(scale);
    let sizes = [(37, 23), (5, 1)];

    for (width, height) in sizes {
        let input = busy(width, height);

        let found = on_device(&input, width, height, &settings);
        let expected = reference_retinex(&input, width, height, &settings);

        for (value, reference) in found.output.iter().zip(expected.iter()) {
            assert!(value.is_finite(), "{value}");
            let error = (value - reference).abs();
            assert!(
                error <= loose_tolerance,
                "error {error} exceeded {loose_tolerance}"
            );
        }
    }

    let width = 48;
    let height = 40;
    let input = smooth_scene(width, height);

    let found = on_device(&input, width, height, &settings);
    let expected = reference_retinex(&input, width, height, &settings);

    let mut within_tight = 0;
    for (value, reference) in found.output.iter().zip(expected.iter()) {
        assert!(value.is_finite(), "{value}");
        let error = (value - reference).abs();
        assert!(
            error <= loose_tolerance,
            "error {error} exceeded {loose_tolerance}"
        );

        if error <= tight_tolerance {
            within_tight += 1;
        }
    }

    let share = within_tight as f32 / found.output.len() as f32;
    assert!(
        share >= 0.99,
        "only {share} of samples were within {tight_tolerance}"
    );
}

#[test]
fn a_flat_plane_gives_zero_on_the_device() {
    let width = 6;
    let height = 4;
    let input = vec![100.0f32; (width * height) as usize];
    let settings = default_settings(255.0);
    let count = settings.sigmas().len() as f32;

    let found = on_device(&input, width, height, &settings);

    for value in found.output {
        assert!(value.is_finite(), "{value}");
        assert_eq!(value, 0.0);
    }

    // A flat input normalises to 0, whose blur is also 0, so `accumulate` takes its `g <= 0`
    // branch throughout and adds `ln(1)` each time, so the log response settles at 0.
    let expected_response = 1.0f32.ln() / count;
    for value in found.msr {
        assert!(value.is_finite(), "{value}");
        assert_eq!(value, expected_response);
    }
}

#[test]
fn crossed_percentiles_give_zero_on_the_device() {
    let width = 9;
    let height = 7;
    let samples = (width * height) as usize;
    let input: Vec<f32> = (0..samples).map(|index| index as f32).collect();
    let settings = RetinexSettings::new(vec![2.0], 0.6, 0.6, 255.0);

    let found = on_device(&input, width, height, &settings);

    for value in found.output {
        assert!(value.is_finite(), "{value}");
        assert_eq!(value, 0.0);
    }
}

/// An isolated bright sample against a dark field pushes `intensity / gaussian` into the tens of
/// thousands at a sigma this wide, which overflows f32 across 8 sigmas when the response
/// multiplies a factor per sigma instead of summing its log.
#[test]
fn wide_sigmas_on_an_isolated_sample_stay_finite_on_the_device() {
    let width = 512;
    let height = 512;
    let mut input = vec![0.0f32; (width * height) as usize];
    input[(width * height / 2) as usize] = 1.0;
    let settings = RetinexSettings::new(vec![100_000.0; 8], 0.001, 0.005, 255.0);

    let found = on_device(&input, width, height, &settings);

    for value in &found.msr {
        assert!(value.is_finite(), "{value}");
    }
    assert!(found.msr.iter().any(|value| *value != 0.0));
    for value in &found.output {
        assert!(value.is_finite(), "{value}");
    }
}

#[test]
fn a_large_smooth_scene_matches_the_reference_loosely() {
    let width = 1024;
    let height = 1024;
    let scale = 255.0;
    let loose_tolerance = 2e-2 * scale;
    let settings = default_settings(scale);
    let input = smooth_scene(width, height);

    let found = on_device(&input, width, height, &settings);
    let expected = reference_retinex(&input, width, height, &settings);

    for value in &found.msr {
        assert!(value.is_finite(), "{value}");
    }
    for (value, reference) in found.output.iter().zip(expected.iter()) {
        assert!(value.is_finite(), "{value}");
        let error = (value - reference).abs();
        assert!(
            error <= loose_tolerance,
            "error {error} exceeded {loose_tolerance}"
        );
    }
}
