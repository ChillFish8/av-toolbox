use av_toolbox_cubek::deinterlace::qtgmc::{repair_blend, vertical_blur};
use av_toolbox_cubek::testing::{
    assert_plane_within,
    assert_same_plane,
    qtgmc_fixture,
    reference_repair_blend,
    reference_vertical_blur,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::{DEVICE_TOLERANCE, FIXTURE_TOLERANCE, assert_fixture_within, codes, peak};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// One plane and its size.
struct Plane {
    samples: Vec<f32>,
    width: u32,
    height: u32,
}

/// Blurs every plane vertically on the device.
struct BlurRun {
    planes: Vec<Plane>,
}

impl BuildTask for BlurRun {
    type Output = Vec<Vec<f32>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());

        let mut outputs = Vec::with_capacity(self.planes.len());
        for source in &self.planes {
            let (width, height) = (source.width, source.height);
            let input = upload_f32(client, &source.samples);
            let output = empty(client, source.samples.len() * size_of::<f32>());
            vertical_blur(
                client,
                &sizing,
                plane(&input, width, height),
                plane(&output, width, height),
            );

            outputs.push(read_f32(client, &output));
        }

        outputs
    }
}

#[test]
fn the_vertical_blur_matches_std_convolution() {
    let inputs = qtgmc_fixture("inputs", "frames");
    let outputs = qtgmc_fixture("vblur", "out");

    for (input, output) in inputs.iter().zip(&outputs) {
        let ceiling = peak(input);
        let planes: Vec<Plane> = input
            .frames
            .iter()
            .flatten()
            .map(|source| Plane {
                samples: codes(&source.samples),
                width: source.width as u32,
                height: source.height as u32,
            })
            .collect();
        let expected_planes: Vec<_> = output.frames.iter().flatten().collect();
        let references: Vec<Vec<f32>> = planes
            .iter()
            .map(|source| {
                let (width, height) = (source.width as usize, source.height as usize);
                reference_vertical_blur(&source.samples, width, height)
            })
            .collect();

        for (index, (reference, expected)) in references.iter().zip(&expected_planes).enumerate() {
            let label = format!("{} plane {index}", input.clip);

            assert_fixture_within(
                reference,
                &expected.samples,
                ceiling,
                |_| FIXTURE_TOLERANCE,
                &label,
            );
        }

        let widths: Vec<u32> = planes.iter().map(|source| source.width).collect();
        let found = backend().build(BlurRun { planes });
        for (index, (blurred, reference)) in found.iter().zip(&references).enumerate() {
            let label = format!("{} plane {index} on the device", input.clip);

            assert_same_plane(blurred, reference, widths[index], &label);
        }
    }
}

/// Blends `original` toward `rebuilt` by `mask` on the device.
struct BlendRun {
    original: Vec<f32>,
    rebuilt: Vec<f32>,
    mask: Vec<f32>,
}

impl BuildTask for BlendRun {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let width = self.original.len() as u32;

        let original = upload_f32(client, &self.original);
        let rebuilt = upload_f32(client, &self.rebuilt);
        let mask = upload_f32(client, &self.mask);
        let output = empty(client, self.original.len() * size_of::<f32>());
        repair_blend(
            client,
            &sizing,
            plane(&original, width, 1),
            plane(&rebuilt, width, 1),
            plane(&mask, width, 1),
            plane(&output, width, 1),
        );

        read_f32(client, &output)
    }
}

#[test]
fn the_repair_blend_is_the_original_at_zero_and_the_rebuild_at_one() {
    let original: Vec<f32> = (0..64).map(|index| (index * 13 % 256) as f32).collect();
    let rebuilt: Vec<f32> = (0..64).map(|index| (index * 29 % 1024) as f32).collect();
    let width = original.len() as u32;

    for weight in [0.0f32, 1.0, 0.25, 0.6] {
        let mask = vec![weight; original.len()];
        let expected = reference_repair_blend(&original, &rebuilt, &mask);
        let run = BlendRun {
            original: original.clone(),
            rebuilt: rebuilt.clone(),
            mask,
        };
        let found = backend().build(run);

        let label = format!("mask {weight}");
        assert_plane_within(&found, &expected, DEVICE_TOLERANCE, width, &label);
        if weight == 0.0 {
            assert_eq!(found, original, "a mask of 0 keeps the original");
        }
        if weight == 1.0 {
            assert_eq!(found, rebuilt, "a mask of 1 gives the rebuild");
        }
    }
}
