use av_toolbox_cubek::deinterlace::qtgmc::{ShimmerPlanes, shimmer};
use av_toolbox_cubek::testing::{assert_plane_within, qtgmc_fixture, reference_shimmer};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::{FIXTURE_TOLERANCE, codes};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// One plane repaired by `shimmer` on the device.
struct ShimmerRun {
    input: Vec<f32>,
    reference: Vec<f32>,
    width: u32,
    height: u32,
    bits: u32,
}

impl BuildTask for ShimmerRun {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let reference = upload_f32(client, &self.reference);
        let difference = empty(client, bytes);
        let first = empty(client, bytes);
        let second = empty(client, bytes);
        let choke = empty(client, bytes);
        let output = empty(client, bytes);

        let planes = ShimmerPlanes {
            input: plane(&input, self.width, self.height),
            reference: plane(&reference, self.width, self.height),
            difference: plane(&difference, self.width, self.height),
            first: plane(&first, self.width, self.height),
            second: plane(&second, self.width, self.height),
            choke: plane(&choke, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };
        shimmer(client, &sizing, planes, self.bits);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

/// The input, reference and havsfunc output of every plane of every shimmer fixture frame.
struct ShimmerCase {
    label: String,
    input: Vec<f32>,
    reference: Vec<f32>,
    expected: Vec<i32>,
    size: (usize, usize),
    bits: u32,
}

fn shimmer_cases() -> Vec<ShimmerCase> {
    let inputs = qtgmc_fixture("average", "tr2_yuv");
    let references = qtgmc_fixture("bob", "out");
    let outputs = qtgmc_fixture("shimmer", "out");

    let mut cases = Vec::new();
    for ((input, reference), output) in inputs.iter().zip(&references).zip(&outputs) {
        for (index, planes) in input.frames.iter().enumerate() {
            for (plane_index, source) in planes.iter().enumerate() {
                let wanted = &reference.frames[index][plane_index];
                let expected = &output.frames[index][plane_index];

                cases.push(ShimmerCase {
                    label: format!("{} frame {index} plane {plane_index}", input.clip),
                    input: codes(&source.samples),
                    reference: codes(&wanted.samples),
                    expected: expected.samples.clone(),
                    size: (source.width, source.height),
                    bits: input.bits,
                });
            }
        }
    }

    cases
}

#[test]
fn shimmer_matches_havsfunc() {
    for case in shimmer_cases() {
        let (width, height) = case.size;
        let ceiling = ((1u32 << case.bits) - 1) as f32;
        let found = reference_shimmer(&case.input, &case.reference, width, height, case.bits);

        let mut largest = 0;
        for (position, (value, wanted)) in found.iter().zip(&case.expected).enumerate() {
            let rounded = value.round().clamp(0.0, ceiling) as i32;
            largest = largest.max((rounded - wanted).abs());
            assert!(
                (rounded - wanted).abs() <= FIXTURE_TOLERANCE,
                "{}: sample {position} is {value} (rounded {rounded}), havsfunc has {wanted}",
                case.label,
            );
        }

        println!("{}: largest difference {largest}", case.label);
    }
}

#[test]
fn the_device_shimmer_matches_the_reference() {
    for case in shimmer_cases() {
        let (width, height) = case.size;
        let expected = reference_shimmer(&case.input, &case.reference, width, height, case.bits);
        let task = ShimmerRun {
            input: case.input,
            reference: case.reference,
            width: width as u32,
            height: height as u32,
            bits: case.bits,
        };

        let found = backend().build(task);

        assert_plane_within(&found, &expected, 0.0, width as u32, &case.label);
    }
}

/// A 16x24 plane of 100 and a reference that is 200 over `rows`.
fn band_planes(rows: std::ops::Range<usize>) -> (Vec<f32>, Vec<f32>) {
    let input = vec![100.0f32; 16 * 24];

    let mut reference = input.clone();
    for row in rows {
        reference[row * 16..(row + 1) * 16].fill(200.0);
    }

    (input, reference)
}

fn repaired_on_both(input: Vec<f32>, reference: Vec<f32>) -> Vec<f32> {
    let expected = reference_shimmer(&input, &reference, 16, 24, 8);
    let task = ShimmerRun {
        input,
        reference,
        width: 16,
        height: 24,
        bits: 8,
    };

    let found = backend().build(task);

    assert_plane_within(&found, &expected, 0.0, 16, "device and reference");

    expected
}

#[test]
fn a_thin_band_is_kept_as_the_input_and_a_wide_band_is_restored_inside() {
    let (input, thin_reference) = band_planes(12..13);
    let thin = repaired_on_both(input.clone(), thin_reference);
    assert_eq!(thin, input, "a thin band is kept as the input has it");

    let (input, wide_reference) = band_planes(6..17);
    let wide = repaired_on_both(input, wide_reference.clone());
    for row in 8..=14 {
        let range = row * 16..(row + 1) * 16;
        assert_eq!(wide[range.clone()], wide_reference[range], "wide band, row {row}");
    }
}
