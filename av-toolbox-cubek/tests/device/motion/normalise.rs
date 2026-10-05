use av_toolbox_cubek::motion::normalise;
use av_toolbox_cubek::testing::{assert_same_plane, qtgmc_fixture, reference_normalise};
use av_toolbox_gpu::{BuildContext, BuildTask};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// How far white may sit from `256 / 255`, a few ulps.
const WHITE_TOLERANCE: f32 = 1e-6;

/// A plane of code values and the bit depth they are in.
struct Luma {
    samples: Vec<f32>,
    width: u32,
    bits: u32,
}

/// Normalises each plane on the device.
struct NormaliseRun {
    planes: Vec<Luma>,
}

impl BuildTask for NormaliseRun {
    type Output = Vec<Vec<f32>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());

        let mut outputs = Vec::new();
        for luma in &self.planes {
            let height = luma.samples.len() as u32 / luma.width;
            let input = upload_f32(client, &luma.samples);
            let output = empty(client, luma.samples.len() * size_of::<f32>());

            normalise(
                client,
                &sizing,
                plane(&input, luma.width, height),
                plane(&output, luma.width, height),
                luma.bits,
            );

            outputs.push(read_f32(client, &output));
        }

        outputs
    }
}

fn codes(samples: &[i32]) -> Vec<f32> {
    samples.iter().map(|sample| *sample as f32).collect()
}

#[test]
fn normalise_matches_dither_luma_rebuild() {
    let inputs = qtgmc_fixture("bob", "out");
    let rebuilt = qtgmc_fixture("normalise", "out");

    let mut planes = Vec::new();
    let mut wanted = Vec::new();
    for (input, fixture) in inputs.iter().zip(&rebuilt) {
        let scale = (1u32 << (input.bits - 8)) as f32;
        for (frame, fixture_frame) in input.frames.iter().zip(&fixture.frames) {
            planes.push(Luma {
                samples: codes(&frame[0].samples),
                width: frame[0].width as u32,
                bits: input.bits,
            });
            wanted.push((fixture_frame[0].samples.clone(), scale, input.clip));
        }
    }

    let found = backend().build(NormaliseRun {
        planes: planes.iter().map(clone_luma).collect(),
    });

    for (index, ((codes_wanted, scale, clip), luma)) in wanted.iter().zip(&planes).enumerate() {
        let reference = reference_normalise(&luma.samples, luma.bits);
        let label = format!("{clip} frame {index}");
        assert_same_plane(&found[index], &reference, luma.width, &label);

        for (sample, (reference_value, code)) in reference.iter().zip(codes_wanted).enumerate() {
            let rounded = (reference_value * 255.0 * scale).round();
            assert!(
                (rounded - *code as f32).abs() <= 1.0,
                "{label} sample {sample} is {rounded}, the fixture has {code}"
            );
        }
    }
}

fn clone_luma(luma: &Luma) -> Luma {
    Luma {
        samples: luma.samples.clone(),
        width: luma.width,
        bits: luma.bits,
    }
}

#[test]
fn normalise_clamps_outside_the_tv_range() {
    let mut planes = Vec::new();
    for bits in [8u32, 16] {
        let scale = (1u32 << (bits - 8)) as f32;
        let peak = ((1u32 << bits) - 1) as f32;

        planes.push(Luma {
            samples: vec![0.0, 16.0 * scale, 235.0 * scale, peak],
            width: 4,
            bits,
        });
    }

    let found = backend().build(NormaliseRun { planes });

    let white = 256.0f32 / 255.0;
    for (found_plane, bits) in found.iter().zip([8, 16]) {
        let wanted = [0.0, 0.0, white, white];
        for (sample, (value, expected)) in found_plane.iter().zip(wanted).enumerate() {
            assert!(
                (value - expected).abs() <= WHITE_TOLERANCE,
                "{bits} bit sample {sample} is {value}, expected {expected}"
            );
        }
    }
}
