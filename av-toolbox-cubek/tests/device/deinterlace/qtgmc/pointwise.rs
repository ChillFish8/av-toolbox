use av_toolbox_cubek::deinterlace::qtgmc::{
    clamp_between,
    fmtc_gauss_taps,
    havsfunc_scale,
    merge,
    sharpness_strength,
    subtract,
    vertical_midpoint,
};
use av_toolbox_cubek::neighbourhood::{Edge, Extreme, Shape};
use av_toolbox_cubek::resize::{ResizePlanes, apply_float_with_edge, upload};
use av_toolbox_cubek::testing::{
    QtgmcFixture,
    assert_plane_within,
    qtgmc_fixture,
    reference_back_blend,
    reference_clamp,
    reference_extreme,
    reference_resharpen,
    reference_resize_float_with_edge,
    reference_search_clip,
    reference_search_inputs,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::steps::{STRENGTH, Step, luma_frames, run_step};
use super::{DEVICE_TOLERANCE, FIXTURE_TOLERANCE, assert_fixture_within, codes, peak};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// The largest difference seen for each clip and plane of one step.
#[derive(Default)]
struct StepDifferences {
    largest: Vec<(String, i32)>,
}

impl StepDifferences {
    fn record(&mut self, label: String, difference: i32) {
        match self.largest.iter_mut().find(|(known, _)| *known == label) {
            Some((_, seen)) => *seen = (*seen).max(difference),
            None => self.largest.push((label, difference)),
        }
    }

    fn print_largest(&self) {
        for (label, difference) in &self.largest {
            println!("{label}: largest difference {difference}");
        }
    }
}

/// The planes of one frame of a fixture as code values.
fn frame_codes(fixture: &QtgmcFixture, frame: usize) -> Vec<Vec<f32>> {
    fixture.frames[frame]
        .iter()
        .map(|source| codes(&source.samples))
        .collect()
}

/// The size of the luma plane of a fixture.
fn luma_size(fixture: &QtgmcFixture) -> (usize, usize) {
    (fixture.frames[0][0].width, fixture.frames[0][0].height)
}

/// How close to the threshold, in code values, a sample is a tie that f32 rounding decides.
pub(super) const TIE_GAP: f32 = 1e-3;

/// How close to the `scale(7)` threshold, in codes, a search clip sample may sit and be allowed more room.
const THRESHOLD_BAND: f32 = 1.5;

/// The share of a frame's samples that may differ from havsfunc by more than [FIXTURE_TOLERANCE].
const FLIP_SHARE_LIMIT: f32 = 0.03;

// The search clip compares its blur with the bob at a hard threshold. fmtc's fixed-point gaussian differs
// from the f32 one by about a code, which flips the branch for samples near the threshold. Those may
// differ by up to `ceil(1.5 * scale(1)) + 1` and the rest by 1. No more than 3% of a frame may differ by
// more than 1.
#[test]
fn the_search_clip_matches_havsfunc() {
    let repaired = luma_frames("shimmer", "out");
    let bobs = luma_frames("bob", "out");
    let outputs = qtgmc_fixture("search", "out");

    for ((output, repaired_frames), bob_frames) in outputs.iter().zip(&repaired).zip(&bobs) {
        let ceiling = peak(output);
        let (width, height) = luma_size(output);
        let threshold = havsfunc_scale(7.0, ceiling);
        let band_tolerance =
            (THRESHOLD_BAND * havsfunc_scale(1.0, ceiling)).ceil() as i32 + FIXTURE_TOLERANCE;

        let inputs = repaired_frames
            .iter()
            .zip(bob_frames)
            .map(|(repair, bob)| vec![repair.clone(), bob.clone()])
            .collect();
        let device = run_step(Step::Search, inputs, (width, height), output.bits);

        let mut largest = 0;
        let mut largest_share = 0.0f32;
        let mut flipped_total = 0;
        for (frame, found) in device.iter().enumerate() {
            let (blur, tweaked) = reference_search_inputs(
                &repaired_frames[frame],
                &bob_frames[frame],
                width,
                height,
                output.bits,
            );

            let expected = &output.frames[frame][0].samples;
            let allowed = |position: usize| {
                let gap = ((tweaked[position] - blur[position]).abs() - threshold).abs();

                if gap <= THRESHOLD_BAND {
                    band_tolerance
                } else {
                    FIXTURE_TOLERANCE
                }
            };
            let reference = reference_search_clip(
                &repaired_frames[frame],
                &bob_frames[frame],
                width,
                height,
                output.bits,
            );

            for (source, plane_found) in [("device", found), ("reference", &reference)] {
                let label = format!("{} {source} frame {frame}", output.clip);
                let report = assert_fixture_within(plane_found, expected, ceiling, allowed, &label);

                largest = largest.max(report.largest);
                flipped_total += report.over_tolerance;
                let share = report.over_tolerance as f32 / plane_found.len() as f32;
                largest_share = largest_share.max(share);
            }
        }

        println!(
            "{} search: largest difference {largest}, largest flip share {largest_share}, \
             {flipped_total} flipped",
            output.clip
        );
        assert!(
            largest_share < FLIP_SHARE_LIMIT,
            "{}: {largest_share} of a frame's samples differ by more than {FIXTURE_TOLERANCE}",
            output.clip,
        );
    }
}

#[test]
fn resharpen_matches_havsfunc() {
    let inputs = qtgmc_fixture("average", "tr2_yuv");
    let outputs = qtgmc_fixture("resharpen", "out");
    let mut differences = StepDifferences::default();

    for (input, output) in inputs.iter().zip(&outputs) {
        let ceiling = peak(output);

        for plane_index in 0..output.frames[0].len() {
            let source = &output.frames[0][plane_index];
            let size = (source.width, source.height);
            let frames = (0..output.frames.len())
                .map(|frame| vec![frame_codes(input, frame).swap_remove(plane_index)])
                .collect();
            let device = run_step(Step::Resharpen, frames, size, output.bits);

            for (frame, found) in device.iter().enumerate() {
                let expected = &output.frames[frame][plane_index].samples;

                let reference_input = &frame_codes(input, frame)[plane_index];
                let reference = reference_resharpen(reference_input, size.0, size.1, STRENGTH, output.bits);

                for (source, plane_found) in [("device", found), ("reference", &reference)] {
                    let label = format!("{} {source} resharpen plane {plane_index}", output.clip);
                    let report =
                        assert_fixture_within(plane_found, expected, ceiling, |_| FIXTURE_TOLERANCE, &label);
                    differences.record(label, report.largest);
                }
            }
        }
    }

    differences.print_largest();
}

#[test]
fn back_blend_matches_havsfunc() {
    let sharpened = luma_frames("resharpen", "out");
    let unsharpened = luma_frames("average", "tr2_yuv");
    let outputs = qtgmc_fixture("backblend", "out");
    let mut differences = StepDifferences::default();

    for ((output, thin_frames), lossed_frames) in outputs.iter().zip(&sharpened).zip(&unsharpened) {
        let ceiling = peak(output);
        let size = luma_size(output);
        let inputs = thin_frames
            .iter()
            .zip(lossed_frames)
            .map(|(thin, lossed)| vec![thin.clone(), lossed.clone()])
            .collect();
        let device = run_step(Step::BackBlend, inputs, size, output.bits);

        for (frame, found) in device.iter().enumerate() {
            let expected = &output.frames[frame][0].samples;
            let reference = reference_back_blend(
                &thin_frames[frame],
                &lossed_frames[frame],
                size.0,
                size.1,
                output.bits,
            );

            for (source, plane_found) in [("device", found), ("reference", &reference)] {
                let label = format!("{} {source} back blend luma", output.clip);
                let report =
                    assert_fixture_within(plane_found, expected, ceiling, |_| FIXTURE_TOLERANCE, &label);
                differences.record(label, report.largest);
            }
        }
    }

    differences.print_largest();
}

#[test]
fn clamp_matches_havsfunc() {
    let inputs = qtgmc_fixture("resharpen", "out");
    let uppers = qtgmc_fixture("clamp", "max");
    let lowers = qtgmc_fixture("clamp", "min");
    let outputs = qtgmc_fixture("clamp", "out");
    let mut differences = StepDifferences::default();

    for (((input, upper), lower), output) in inputs.iter().zip(&uppers).zip(&lowers).zip(&outputs) {
        let ceiling = peak(output);

        for frame in 0..output.frames.len() {
            let planes = frame_codes(input, frame);
            let highs = frame_codes(upper, frame);
            let lows = frame_codes(lower, frame);

            for (plane_index, source) in output.frames[frame].iter().enumerate() {
                let found = reference_clamp(&planes[plane_index], &highs[plane_index], &lows[plane_index]);

                let label = format!("{} clamp plane {plane_index}", output.clip);
                let report =
                    assert_fixture_within(&found, &source.samples, ceiling, |_| FIXTURE_TOLERANCE, &label);
                differences.record(label, report.largest);
            }
        }
    }

    differences.print_largest();
}

/// The kernels a [PointwiseRun] runs.
#[derive(Clone, Copy)]
enum Pointwise {
    Midpoint,
    Subtract,
    Clamp,
    Merge,
}

/// Planes put through one pointwise kernel on the device.
struct PointwiseRun {
    operation: Pointwise,
    planes: Vec<Vec<f32>>,
    width: u32,
    height: u32,
    bits: u32,
}

impl PointwiseRun {
    fn new(operation: Pointwise, planes: Vec<Vec<f32>>, size: (usize, usize), bits: u32) -> Self {
        Self {
            operation,
            planes,
            width: size.0 as u32,
            height: size.1 as u32,
            bits,
        }
    }
}

impl BuildTask for PointwiseRun {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let bytes = self.planes[0].len() * size_of::<f32>();
        let (width, height) = (self.width, self.height);

        let inputs: Vec<_> = self
            .planes
            .iter()
            .map(|values| upload_f32(client, values))
            .collect();
        let first = plane(&inputs[0], width, height);
        let output = empty(client, bytes);

        match self.operation {
            Pointwise::Midpoint => {
                vertical_midpoint(client, &sizing, first, plane(&output, width, height));
            },
            Pointwise::Subtract => {
                let second = plane(&inputs[1], width, height);
                let output_plane = plane(&output, width, height);
                subtract(client, &sizing, first, second, output_plane, self.bits);
            },
            Pointwise::Clamp => {
                let upper = plane(&inputs[1], width, height);
                let lower = plane(&inputs[2], width, height);
                clamp_between(
                    client,
                    &sizing,
                    first,
                    upper,
                    lower,
                    plane(&output, width, height),
                );
            },
            Pointwise::Merge => {
                let second = plane(&inputs[1], width, height);
                merge(client, &sizing, first, second, plane(&output, width, height), 0.3);
            },
        }

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

/// The frame 3 planes of the `pan_8` clip of a fixture step, as the device tests' input.
fn pan_frame(step: &str, name: &str) -> Vec<Vec<f32>> {
    let fixture = qtgmc_fixture(step, name).remove(0);

    frame_codes(&fixture, 3)
}

const PAN_SIZE: (usize, usize) = (128, 64);

#[test]
fn the_device_vertical_midpoint_matches_the_reference() {
    let input = pan_frame("average", "tr2_yuv").remove(0);
    let (width, height) = (PAN_SIZE.0 as u32, PAN_SIZE.1 as u32);
    let largest = reference_extreme(&input, width, height, Extreme::Maximum, Shape::Vertical);
    let smallest = reference_extreme(&input, width, height, Extreme::Minimum, Shape::Vertical);
    let expected: Vec<f32> = largest
        .iter()
        .zip(&smallest)
        .map(|(high, low)| ((high + low) / 2.0 + 0.5).floor())
        .collect();

    let found = backend().build(PointwiseRun::new(Pointwise::Midpoint, vec![input], PAN_SIZE, 8));

    assert_plane_within(&found, &expected, 0.0, width, "vertical midpoint");
}

#[test]
fn the_device_subtract_matches_the_reference() {
    let first = vec![10.0f32, 200.0, 255.0, 0.0, 128.0, 90.0, 5.0, 250.0];
    let second = vec![20.0f32, 100.0, 0.0, 10.0, 128.0, 91.0, 5.0, 300.0];
    let expected = vec![0.0f32, 100.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let task = PointwiseRun::new(Pointwise::Subtract, vec![first, second], (4, 2), 8);
    let found = backend().build(task);

    assert_plane_within(&found, &expected, 0.0, 4, "subtract");
}

#[test]
fn the_device_clamp_matches_the_reference_and_the_lower_bound_wins() {
    let input = vec![10.0f32, 50.0, 90.0, 50.0, 50.0, 50.0, 200.0, 0.0];
    let upper = vec![20.0f32, 60.0, 80.0, 40.0, 30.0, 100.0, 255.0, 0.0];
    let lower = vec![5.0f32, 55.0, 70.0, 45.0, 35.0, 10.0, 210.0, 1.0];
    let expected = reference_clamp(&input, &upper, &lower);

    assert_eq!(expected, vec![10.0, 55.0, 80.0, 45.0, 35.0, 50.0, 210.0, 1.0]);

    let task = PointwiseRun::new(Pointwise::Clamp, vec![input, upper, lower], (4, 2), 8);
    let found = backend().build(task);

    assert_plane_within(&found, &expected, 0.0, 4, "clamp");
}

#[test]
fn the_device_merge_weighs_the_second_plane_by_the_weight() {
    let first = vec![0.0f32, 100.0, 255.0, 40.0];
    let second = vec![100.0f32, 100.0, 0.0, 140.0];
    let expected = vec![30.0f32, 100.0, 178.5, 70.0];

    let task = PointwiseRun::new(Pointwise::Merge, vec![first, second], (2, 2), 8);
    let found = backend().build(task);

    assert_plane_within(&found, &expected, DEVICE_TOLERANCE, 2, "merge");
}

/// One axis of a plane resized on the device with the gaussian taps and a clamped edge.
struct GaussianRun {
    input: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for GaussianRun {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let scratch = empty(client, bytes);
        let output = empty(client, bytes);
        let across = upload(client, &fmtc_gauss_taps(self.width, 2.0));
        let down = upload(client, &fmtc_gauss_taps(self.height, 2.0));

        let planes = ResizePlanes {
            input: plane(&input, self.width, self.height),
            scratch: plane(&scratch, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };
        apply_float_with_edge(client, &sizing, planes, &across, &down, Edge::Clamp);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

#[test]
fn the_device_gaussian_matches_the_reference_with_a_clamped_edge() {
    let input = pan_frame("average", "tr2_yuv").remove(0);
    let (width, height) = PAN_SIZE;
    let across = fmtc_gauss_taps(width as u32, 2.0);
    let down = fmtc_gauss_taps(height as u32, 2.0);
    let expected =
        reference_resize_float_with_edge(&input, width as u32, height as u32, &across, &down, Edge::Clamp);

    let task = GaussianRun {
        input,
        width: width as u32,
        height: height as u32,
    };
    let found = backend().build(task);

    assert_plane_within(&found, &expected, DEVICE_TOLERANCE, width as u32, "gaussian");
}

#[test]
fn fmtc_gauss_taps_sum_to_one_and_peak_on_their_own_sample() {
    for sharpness in [2.0, 5.0] {
        let table = fmtc_gauss_taps(32, sharpness);
        let count = table.count as usize;

        for (index, row) in table.weights.chunks(count).enumerate() {
            let total: f32 = row.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-6,
                "a1 {sharpness} row {index} sums to {total}"
            );

            let peak_tap = row
                .iter()
                .enumerate()
                .max_by(|first, second| first.1.total_cmp(second.1))
                .map(|(tap, _)| tap)
                .expect("a row has taps");
            assert_eq!(
                table.starts[index] + peak_tap as i32,
                index as i32,
                "a1 {sharpness} row {index}"
            );
        }
    }
}

// fmtc reads one sample further on the far side of a sample than the near side, once the first few samples
// are past. The first 4 samples read 4 either way.
#[test]
fn fmtc_gauss_taps_read_one_sample_further_ahead_after_the_start() {
    let table = fmtc_gauss_taps(32, 2.0);
    let count = table.count as usize;
    let reach = |index: usize| {
        let row = &table.weights[index * count..(index + 1) * count];
        let first = row.iter().position(|weight| *weight > 0.0).expect("a weight") as i32;
        let last = row.iter().rposition(|weight| *weight > 0.0).expect("a weight") as i32;

        (
            index as i32 - (table.starts[index] + first),
            table.starts[index] + last - index as i32,
        )
    };

    assert_eq!(reach(0), (4, 4));
    assert_eq!(reach(3), (4, 4));
    assert_eq!(reach(4), (3, 4));
    assert_eq!(reach(31), (3, 4));
}

#[test]
fn the_default_sharpness_is_one_and_a_half() {
    assert!((sharpness_strength(1.0, 2, 1) - 1.5).abs() < 1e-6);
}

#[test]
fn sixteen_bit_thresholds_scale_with_the_depth() {
    let repaired = qtgmc_fixture("shimmer", "out").remove(0);
    let bobs = qtgmc_fixture("bob", "out").remove(0);
    let (width, height) = PAN_SIZE;

    // 257 is exactly 65535 / 255, so the thresholds scale exactly. A sample exactly on the threshold is
    // left out, as f32 rounding decides which side it falls on at each depth.
    let mut largest = 0.0f32;
    for frame in 0..repaired.frames.len() {
        let repaired_luma = &frame_codes(&repaired, frame)[0];
        let bob_luma = &frame_codes(&bobs, frame)[0];
        let eight_bit = reference_search_clip(repaired_luma, bob_luma, width, height, 8);

        let wide_repaired: Vec<f32> = repaired_luma.iter().map(|sample| sample * 257.0).collect();
        let wide_bob: Vec<f32> = bob_luma.iter().map(|sample| sample * 257.0).collect();
        let sixteen_bit = reference_search_clip(&wide_repaired, &wide_bob, width, height, 16);

        let (blur, tweaked) = reference_search_inputs(repaired_luma, bob_luma, width, height, 8);
        for (position, (wide, narrow)) in sixteen_bit.iter().zip(&eight_bit).enumerate() {
            let gap = ((tweaked[position] - blur[position]).abs() - havsfunc_scale(7.0, 255.0)).abs();
            if gap < TIE_GAP {
                continue;
            }

            largest = largest.max((wide - narrow * 257.0).abs());
        }
    }

    println!("sixteen bit search clip: largest difference {largest}");
    assert!(
        largest <= 1.0,
        "the largest difference is {largest}, expected at most 1"
    );
}
