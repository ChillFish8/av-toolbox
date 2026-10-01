use av_toolbox_cubek::aa::{binarize_prewitt, double_mclip, pscale_blend};
use av_toolbox_cubek::neighbourhood::{Extreme, Shape, Smooth, median_of_three};
use av_toolbox_cubek::resize::{Kernel, ResizePlanes, apply_float, centred_taps, upload};
use av_toolbox_cubek::testing::{
    Eedi3PlaneKind,
    SplitMix,
    random_eedi3_mask,
    random_eedi3_plane,
    reference_binarize_prewitt,
    reference_box_halve,
    reference_double_mclip,
    reference_extreme,
    reference_median_of_three,
    reference_pscale_blend,
    reference_smooth,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::{ComputeClient, Runtime};
use cubecl::server::Handle;

use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

const DEPTHS: [u32; 3] = [8, 10, 16];

const KINDS: [Eedi3PlaneKind; 4] = [
    Eedi3PlaneKind::Noise,
    Eedi3PlaneKind::Edge,
    Eedi3PlaneKind::NoisyEdge,
    Eedi3PlaneKind::Level,
];

/// One launch of a step kernel over planes `width` by `height`.
enum Step {
    Binarize { threshold: u32, peak: f32 },
    Mclip { mask_peak: f32 },
    BoxHalve,
    Blend { keep: f32 },
    Median,
}

struct Job {
    inputs: Vec<Vec<f32>>,
    width: u32,
    height: u32,
    step: Step,
}

struct StepJobs(Vec<Job>);

impl BuildTask for StepJobs {
    type Output = Vec<Vec<f32>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);

        self.0
            .into_iter()
            .map(|job| run_job(client, &sizing, job))
            .collect()
    }
}

fn run_job<R: Runtime>(client: &ComputeClient<R>, sizing: &Sizing, job: Job) -> Vec<f32> {
    let width = job.width;
    let height = job.height;
    let handles: Vec<Handle> = job.inputs.iter().map(|input| upload_f32(client, input)).collect();
    let samples = (width * height) as usize;
    let bytes = samples * size_of::<f32>();

    let output = match job.step {
        Step::Binarize { threshold, peak } => {
            let output = empty(client, bytes);
            let input_plane = plane(&handles[0], width, height);
            let output_plane = plane(&output, width, height);

            binarize_prewitt(client, sizing, input_plane, output_plane, threshold, peak);
            output
        },
        Step::Mclip { mask_peak } => {
            let output = empty(client, 4 * bytes);
            let mask_plane = plane(&handles[0], width, height);
            let output_plane = plane(&output, 2 * width, 2 * height);

            double_mclip(client, sizing, mask_plane, output_plane, mask_peak);
            output
        },
        Step::BoxHalve => {
            let half_width = width / 2;
            let half_height = height / 2;
            let across_taps = centred_taps(width, half_width, &Kernel::Box);
            let down_taps = centred_taps(height, half_height, &Kernel::Box);
            let across = upload(client, &across_taps);
            let down = upload(client, &down_taps);
            let scratch = empty(client, bytes / 2);
            let output = empty(client, bytes / 4);
            let input_plane = plane(&handles[0], width, height);
            let scratch_plane = plane(&scratch, half_width, height);
            let output_plane = plane(&output, half_width, half_height);
            let planes = ResizePlanes {
                input: input_plane,
                scratch: scratch_plane,
                output: output_plane,
            };

            apply_float(client, sizing, planes, &across, &down);
            output
        },
        Step::Blend { keep } => {
            let output = empty(client, bytes);
            let aa_plane = plane(&handles[0], width, height);
            let luma_plane = plane(&handles[1], width, height);
            let down_plane = plane(&handles[2], width, height);
            let output_plane = plane(&output, width, height);

            pscale_blend(
                client,
                sizing,
                aa_plane,
                luma_plane,
                down_plane,
                output_plane,
                keep,
            );
            output
        },
        Step::Median => {
            let output = empty(client, bytes);
            let first_plane = plane(&handles[0], width, height);
            let second_plane = plane(&handles[1], width, height);
            let third_plane = plane(&handles[2], width, height);
            let output_plane = plane(&output, width, height);

            median_of_three(
                client,
                sizing,
                first_plane,
                second_plane,
                third_plane,
                output_plane,
            );
            output
        },
    };

    cubecl::future::block_on(client.sync()).expect("the device should finish");

    read_f32(client, &output)
}

/// Asserts that `output` equals `reference`, naming job `job`, the first differing sample and both values.
fn assert_exact(output: &[f32], reference: &[f32], job: usize) {
    assert_eq!(
        output.len(),
        reference.len(),
        "job {job} wrote the wrong number of samples"
    );

    let differing = output
        .iter()
        .zip(reference)
        .position(|(found, expected)| found.to_bits() != expected.to_bits());
    if let Some(index) = differing {
        let found = output[index];
        let expected = reference[index];
        panic!("job {job}: sample {index} was {found}, expected {expected}");
    }
}

fn on_device(jobs: Vec<Job>) -> Vec<Vec<f32>> {
    let task = StepJobs(jobs);

    backend().build(task)
}

/// One random plane of each kind.
fn random_planes(random: &mut SplitMix, width: u32, height: u32, bits: u32) -> Vec<Vec<f32>> {
    KINDS
        .iter()
        .map(|kind| random_eedi3_plane(random, width, height, bits, &[*kind]))
        .collect()
}

/// A plane stepping by `20 * 2^(bits - 8)` every second row, so most windows hold a gradient of exactly
/// `60 * 2^(bits - 8)` down. `across` turns the steps to columns.
fn tie_plane(width: u32, height: u32, bits: u32, across: bool) -> Vec<f32> {
    let step = 20u32 << (bits - 8);

    let mut plane = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let position = if across { column } else { row };
            let value = (position / 2 % 6) * step;
            plane.push(value as f32);
        }
    }

    plane
}

/// A plane of samples drawn from 0 and `peak` alone, so gradients reach their largest.
fn extreme_plane(random: &mut SplitMix, width: u32, height: u32, bits: u32) -> Vec<f32> {
    let peak = (1u32 << bits) - 1;
    let samples = width * height;

    (0..samples)
        .map(|_| {
            let high = random.below(2) == 1;
            if high { peak as f32 } else { 0.0 }
        })
        .collect()
}

/// A 16-bit 3x3 window whose gradients' squares sum to 7 under the threshold at `mask_thr` 255, which an
/// f32 square root rounds onto it.
const NEAR_MISS: [f32; 9] = [0.0, 61828.0, 0.0, 20947.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// A plane tiling [NEAR_MISS], so every window centred on a tile's centre holds it.
fn near_miss_plane(width: u32, height: u32) -> Vec<f32> {
    let mut plane = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let index = (row % 3) * 3 + column % 3;
            plane.push(NEAR_MISS[index as usize]);
        }
    }

    plane
}

/// Random, edge, tie and extreme planes at `bits`, and the near miss plane at 16 bit.
fn binarize_planes(random: &mut SplitMix, width: u32, height: u32, bits: u32) -> Vec<Vec<f32>> {
    let mut planes = random_planes(random, width, height, bits);
    let down_ties = tie_plane(width, height, bits, false);
    let across_ties = tie_plane(width, height, bits, true);
    let extremes = extreme_plane(random, width, height, bits);

    planes.push(down_ties);
    planes.push(across_ties);
    planes.push(extremes);

    if bits == 16 {
        let near_miss = near_miss_plane(width, height);
        planes.push(near_miss);
    }

    planes
}

fn squared_threshold(mask_thr: u32, bits: u32) -> u32 {
    let scaled = mask_thr << (bits - 8);

    scaled * scaled
}

#[test]
fn binarize_prewitt_matches_the_reference_exactly() {
    let width = 67;
    let height = 45;
    let mut random = SplitMix(21);

    let mut jobs = Vec::new();
    let mut expected = Vec::new();
    for bits in DEPTHS {
        let peak = ((1u32 << bits) - 1) as f32;
        let planes = binarize_planes(&mut random, width, height, bits);
        let drawn = random.below(256);

        for input in planes {
            for mask_thr in [60, drawn, 255] {
                let threshold = squared_threshold(mask_thr, bits);
                let reference = reference_binarize_prewitt(&input, width, height, threshold, peak);

                expected.push(reference);
                jobs.push(Job {
                    inputs: vec![input.clone()],
                    width,
                    height,
                    step: Step::Binarize { threshold, peak },
                });
            }
        }
    }

    let outputs = on_device(jobs);

    for (index, (output, reference)) in outputs.iter().zip(&expected).enumerate() {
        assert_exact(output, reference, index);
    }
}

/// The mask built from `luma` at `bits`, as `aa/based` builds it.
fn built_mask(luma: &[f32], width: u32, height: u32, bits: u32) -> Vec<f32> {
    let peak = ((1u32 << bits) - 1) as f32;
    let threshold = squared_threshold(60, bits);
    let binarized = reference_binarize_prewitt(luma, width, height, threshold, peak);
    let widened = reference_extreme(&binarized, width, height, Extreme::Maximum, Shape::Square);

    reference_smooth(&widened, width, height, Smooth::Box)
}

/// A mask holding `value` at one sample and 0 elsewhere.
fn lone_sample(width: u32, height: u32, value: f32) -> Vec<f32> {
    let samples = (width * height) as usize;
    let mut mask = vec![0.0; samples];
    mask[samples / 2] = value;

    mask
}

#[test]
fn double_mclip_matches_the_reference_exactly() {
    let width = 67;
    let height = 45;
    let mut random = SplitMix(22);

    let mut masks = Vec::new();
    for bits in DEPTHS {
        let peak = ((1u32 << bits) - 1) as f32;
        let planes = binarize_planes(&mut random, width, height, bits);

        for luma in planes {
            let mask = built_mask(&luma, width, height, bits);
            masks.push((mask, peak));
        }
    }

    for _ in 0..6 {
        let bound = random_eedi3_mask(&mut random, width, height);
        masks.push((bound, 255.0));
    }

    let exact_half = lone_sample(width, height, 8.0);
    let past_half = lone_sample(width, height, 9.0);
    masks.push((exact_half, 255.0));
    masks.push((past_half, 255.0));

    let mut jobs = Vec::new();
    let mut expected = Vec::new();
    for (mask, mask_peak) in masks {
        let reference = reference_double_mclip(&mask, width, height, mask_peak);

        expected.push(reference);
        jobs.push(Job {
            inputs: vec![mask],
            width,
            height,
            step: Step::Mclip { mask_peak },
        });
    }

    let outputs = on_device(jobs);

    for (index, (output, reference)) in outputs.iter().zip(&expected).enumerate() {
        assert_exact(output, reference, index);
    }
}

#[test]
fn box_halving_matches_the_reference_exactly() {
    let mut random = SplitMix(23);

    let mut jobs = Vec::new();
    let mut expected = Vec::new();
    for bits in DEPTHS {
        for (width, height) in [(66, 46), (64, 48)] {
            let planes = random_planes(&mut random, width, height, bits);

            for input in planes {
                let reference = reference_box_halve(&input, width, height);

                expected.push(reference);
                jobs.push(Job {
                    inputs: vec![input],
                    width,
                    height,
                    step: Step::BoxHalve,
                });
            }
        }
    }

    let outputs = on_device(jobs);

    for (index, (output, reference)) in outputs.iter().zip(&expected).enumerate() {
        assert_exact(output, reference, index);
    }
}

/// The halved anti-aliased plane, the luma and the halved supersample of one random scene.
fn blend_planes(random: &mut SplitMix, width: u32, height: u32, bits: u32) -> Vec<Vec<f32>> {
    let doubled_width = 2 * width;
    let doubled_height = 2 * height;
    let kinds = KINDS.as_slice();
    let aa2 = random_eedi3_plane(random, doubled_width, doubled_height, bits, kinds);
    let ss = random_eedi3_plane(random, doubled_width, doubled_height, bits, kinds);
    let luma = random_eedi3_plane(random, width, height, bits, kinds);
    let aa = reference_box_halve(&aa2, doubled_width, doubled_height);
    let down = reference_box_halve(&ss, doubled_width, doubled_height);

    vec![aa, luma, down]
}

/// Runs the blend at `keep` over random scenes at every depth, handing back each output beside its
/// reference.
fn blends(keep: f32, seed: u64) -> Vec<(Vec<f32>, Vec<f32>)> {
    let width = 67;
    let height = 45;
    let mut random = SplitMix(seed);

    let mut jobs = Vec::new();
    let mut expected = Vec::new();
    for bits in DEPTHS {
        for _ in 0..4 {
            let planes = blend_planes(&mut random, width, height, bits);
            let reference = reference_pscale_blend(&planes[0], &planes[1], &planes[2], keep);

            expected.push(reference);
            jobs.push(Job {
                inputs: planes,
                width,
                height,
                step: Step::Blend { keep },
            });
        }
    }

    let outputs = on_device(jobs);

    outputs.into_iter().zip(expected).collect()
}

#[test]
fn pscale_blend_at_0_matches_the_reference_exactly() {
    let results = blends(1.0, 24);

    for (index, (output, reference)) in results.iter().enumerate() {
        assert_exact(output, reference, index);
    }
}

#[test]
fn pscale_blend_at_half_matches_the_reference() {
    let results = blends(0.5, 25);

    for (output, reference) in &results {
        assert_close(output, reference, 1e-4);
    }
}

#[test]
fn pscale_blend_at_1_keeps_aa_exactly() {
    let results = blends(0.0, 26);

    for (index, (output, reference)) in results.iter().enumerate() {
        assert_exact(output, reference, index);
    }
}

#[test]
fn median_of_three_matches_the_reference() {
    let width = 67;
    let height = 45;
    let mut random = SplitMix(27);

    let mut jobs = Vec::new();
    let mut expected = Vec::new();
    for bits in DEPTHS {
        for kind in KINDS {
            let first = random_eedi3_plane(&mut random, width, height, bits, &[kind]);
            let second = random_eedi3_plane(&mut random, width, height, bits, &[kind]);
            let third = random_eedi3_plane(&mut random, width, height, bits, &[kind]);
            let reference = reference_median_of_three(&first, &second, &third);

            expected.push(reference);
            jobs.push(Job {
                inputs: vec![first, second, third],
                width,
                height,
                step: Step::Median,
            });
        }
    }

    let outputs = on_device(jobs);

    for (output, reference) in outputs.iter().zip(&expected) {
        assert_close(output, reference, 1e-4);
    }
}
