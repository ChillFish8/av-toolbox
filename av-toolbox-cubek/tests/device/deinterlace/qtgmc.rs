use av_toolbox_cubek::deinterlace::qtgmc::{
    BobPlanes,
    MAX_AVERAGED,
    Parity,
    bob,
    bob_taps,
    tr0_weights,
    weighted_sum,
};
use av_toolbox_cubek::resize::upload;
use av_toolbox_cubek::testing::{
    QtgmcFixture,
    assert_plane_within,
    qtgmc_fixture,
    reference_bob,
    reference_tr0,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// How far a rounded reference sample may sit from havsfunc's.
const FIXTURE_TOLERANCE: i32 = 1;

/// How far a device sample may sit from the CPU, for fused multiply-adds.
const DEVICE_TOLERANCE: f32 = 1e-3;

/// One frame bobbed on the device.
struct BobRun {
    frame: Vec<f32>,
    width: u32,
    height: u32,
    parity: Parity,
}

impl BuildTask for BobRun {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());

        let frame = upload_f32(client, &self.frame);
        let output = empty(client, self.frame.len() * size_of::<f32>());
        let table = bob_taps(self.height / 2, self.parity);
        let down = upload(client, &table);

        let planes = BobPlanes {
            frame: plane(&frame, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };
        bob(client, &sizing, planes, &down, self.parity);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

fn codes(samples: &[i32]) -> Vec<f32> {
    samples.iter().map(|sample| *sample as f32).collect()
}

/// One plane of an input frame and the field of it to bob.
struct BobCase {
    label: String,
    frame: Vec<f32>,
    size: (usize, usize),
    parity: Parity,
}

fn bob_cases() -> Vec<BobCase> {
    let inputs = qtgmc_fixture("inputs", "frames");

    let mut cases = Vec::new();
    for input in &inputs {
        for (index, planes) in input.frames.iter().enumerate() {
            for (plane_index, source) in planes.iter().enumerate() {
                for parity in 0..2 {
                    let label = format!("{} frame {index} plane {plane_index} parity {parity}", input.clip);

                    cases.push(BobCase {
                        label,
                        frame: codes(&source.samples),
                        size: (source.width, source.height),
                        parity,
                    });
                }
            }
        }
    }

    cases
}

fn peak(fixture: &QtgmcFixture) -> f32 {
    ((1u32 << fixture.bits) - 1) as f32
}

#[test]
fn the_bob_reference_matches_havsfunc() {
    let outputs = qtgmc_fixture("bob", "out");
    let inputs = qtgmc_fixture("inputs", "frames");

    for (input, output) in inputs.iter().zip(&outputs) {
        let ceiling = peak(output);

        for (index, planes) in input.frames.iter().enumerate() {
            for (plane_index, source) in planes.iter().enumerate() {
                let frame = codes(&source.samples);
                let size = (source.width, source.height);

                for parity in 0..2 {
                    let found = reference_bob(&frame, size, parity);
                    let expected = &output.frames[2 * index + parity as usize][plane_index];

                    for (position, (value, wanted)) in found.iter().zip(&expected.samples).enumerate() {
                        let rounded = value.round().clamp(0.0, ceiling) as i32;
                        assert!(
                            (rounded - wanted).abs() <= FIXTURE_TOLERANCE,
                            "{} frame {index} plane {plane_index} parity {parity}: sample {position} is \
                             {value} (rounded {rounded}), havsfunc has {wanted}",
                            input.clip,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn the_device_bob_matches_the_reference() {
    for case in bob_cases() {
        let (width, height) = case.size;
        let expected = reference_bob(&case.frame, case.size, case.parity);
        let task = BobRun {
            frame: case.frame,
            width: width as u32,
            height: height as u32,
            parity: case.parity,
        };

        let found = backend().build(task);

        assert_plane_within(&found, &expected, DEVICE_TOLERANCE, width as u32, &case.label);
    }
}

/// A 16x8 frame whose field of `parity` ramps as `10 * (row + 1)` and whose other field is 1000.
fn ramp_frame(parity: Parity) -> Vec<f32> {
    let mut frame = Vec::with_capacity(16 * 8);
    for row in 0..8usize {
        let value = if row % 2 == parity as usize {
            10.0 * (row / 2 + 1) as f32
        } else {
            1000.0
        };
        frame.extend(std::iter::repeat_n(value, 16));
    }

    frame
}

/// The sample at the start of `row` in a plane 16 wide, after checking the whole row agrees.
fn row_value(plane: &[f32], row: usize) -> f32 {
    let samples = &plane[row * 16..(row + 1) * 16];
    assert!(
        samples.iter().all(|value| *value == samples[0]),
        "row {row} is {samples:?}"
    );

    samples[0]
}

/// Checks `rows` of a bob against hand-computed values on the CPU and the device.
fn assert_ramp_bob(parity: Parity, rows: &[(usize, f32)]) {
    let frame = ramp_frame(parity);
    let reference = reference_bob(&frame, (16, 8), parity);
    let task = BobRun {
        frame,
        width: 16,
        height: 8,
        parity,
    };
    let device = backend().build(task);

    for (name, plane) in [("reference", &reference), ("device", &device)] {
        assert!(
            plane.iter().all(|value| *value < 100.0),
            "{name} read the other field: {plane:?}"
        );

        for (row, wanted) in rows {
            let found = row_value(plane, *row);
            assert!(
                (found - wanted).abs() < DEVICE_TOLERANCE,
                "{name} row {row} is {found}, expected {wanted}"
            );
        }
    }
}

// Catmull-Rom at a half position weighs its four taps -1/16, 9/16, 9/16 and -1/16. The field rows are
// f0..f3 = 10, 20, 30, 40, and a tap past either end reads the border row.
#[test]
fn a_bottom_field_bob_clamps_at_both_ends() {
    // Row 0 is at -0.5. Taps at rows -2, -1, 0 and 1 clamp to 0, 0, 0 and 1, so the weights are
    // 17/16 on f0 and -1/16 on f1. 1.0625 * 10 - 0.0625 * 20 = 9.375.
    // Row 6 is at 2.5, over rows 1 to 4, and row 4 clamps to row 3 so f3 weighs 8/16.
    // -0.0625 * 20 + 0.5625 * 30 + 0.5 * 40 = -1.25 + 16.875 + 20 = 35.625.
    // Row 7 is at 3.0, exactly on f3.
    assert_ramp_bob(1, &[(0, 9.375), (6, 35.625), (7, 40.0)]);
}

#[test]
fn a_top_field_bob_clamps_at_the_last_row() {
    // Row 7 is at 3.5, over rows 2 to 5, and rows 4 and 5 clamp to row 3 so f3 weighs 17/16.
    // -0.0625 * 30 + 1.0625 * 40 = -1.875 + 42.5 = 40.625.
    // Row 0 is at 0.0, exactly on f0.
    assert_ramp_bob(0, &[(0, 10.0), (7, 40.625)]);
}

/// The luma plane of every bob frame of a clip.
fn luma_bobs(fixture: &QtgmcFixture) -> Vec<Vec<f32>> {
    fixture
        .frames
        .iter()
        .map(|planes| codes(&planes[0].samples))
        .collect()
}

fn bobs_by_clip() -> Vec<(QtgmcFixture, Vec<Vec<f32>>)> {
    qtgmc_fixture("bob", "out")
        .into_iter()
        .map(|fixture| {
            let bobs = luma_bobs(&fixture);

            (fixture, bobs)
        })
        .collect()
}

/// The bobs -2..=2 around `centre`, the nearest repeated at either end of the clip.
fn window_around(bobs: &[Vec<f32>], centre: usize) -> Vec<Vec<f32>> {
    let last = bobs.len() as isize - 1;

    (-2..=2isize)
        .map(|offset| bobs[(centre as isize + offset).clamp(0, last) as usize].clone())
        .collect()
}

/// Bobs weighed into one plane on the device.
struct AverageRun {
    window: Vec<Vec<f32>>,
    weights: [f32; MAX_AVERAGED],
    width: u32,
    height: u32,
}

impl BuildTask for AverageRun {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());

        let handles: Vec<_> = self.window.iter().map(|bob| upload_f32(client, bob)).collect();
        let planes: Vec<_> = handles
            .iter()
            .map(|handle| plane(handle, self.width, self.height))
            .collect();
        let output = empty(client, self.window[0].len() * size_of::<f32>());

        let output_plane = plane(&output, self.width, self.height);
        weighted_sum(client, &sizing, &planes, &self.weights, output_plane);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

#[test]
fn tr0_matches_havsfunc() {
    for (tr0, name) in [(1, "tr1"), (2, "tr2")] {
        let expected = qtgmc_fixture("average", name);

        for ((fixture, bobs), wanted) in bobs_by_clip().iter().zip(&expected) {
            let ceiling = peak(fixture);
            let (width, height) = (fixture.frames[0][0].width, fixture.frames[0][0].height);

            for centre in 0..bobs.len() {
                let found = reference_tr0(bobs, centre, tr0, width, height);
                let target = &wanted.frames[centre][0].samples;

                for (position, (value, code)) in found.iter().zip(target).enumerate() {
                    let rounded = value.round().clamp(0.0, ceiling) as i32;
                    assert!(
                        (rounded - code).abs() <= FIXTURE_TOLERANCE,
                        "{} tr0 {tr0} centre {centre}: sample {position} is {value} (rounded {rounded}), \
                         havsfunc has {code}",
                        fixture.clip,
                    );
                }
            }
        }
    }
}

#[test]
fn the_device_weighted_sum_matches_the_reference() {
    for (fixture, bobs) in bobs_by_clip() {
        let (width, height) = (fixture.frames[0][0].width, fixture.frames[0][0].height);

        for tr0 in 0..=2 {
            for centre in 0..bobs.len() {
                let expected = reference_tr0(&bobs, centre, tr0, width, height);
                let task = AverageRun {
                    window: window_around(&bobs, centre),
                    weights: tr0_weights(tr0),
                    width: width as u32,
                    height: height as u32,
                };

                let found = backend().build(task);

                let label = format!("{} tr0 {tr0} centre {centre}", fixture.clip);
                assert_plane_within(&found, &expected, DEVICE_TOLERANCE, width as u32, &label);
            }
        }
    }
}

#[test]
fn edge_bobs_repeat_the_nearest() {
    let (fixture, bobs) = bobs_by_clip().remove(0);
    let (width, height) = (fixture.frames[0][0].width, fixture.frames[0][0].height);

    let found = reference_tr0(&bobs, 0, 2, width, height);

    let mut padded = vec![bobs[0].clone(), bobs[0].clone()];
    padded.extend(bobs.iter().cloned());
    let expected = reference_tr0(&padded, 2, 2, width, height);

    assert_plane_within(&found, &expected, 0.0, width as u32, "centre 0 at tr0 2");
}

/// The bob before the cut of `cut_8`. The clip cuts at its frame 4 and the fixture starts at frame 2, so
/// the cut sits between input frames 1 and 2, which is between bobs 3 and 4.
const BEFORE_CUT: usize = 3;

/// The sum of absolute differences between two bobs.
fn bob_difference(first: &[f32], second: &[f32]) -> f32 {
    first
        .iter()
        .zip(second)
        .map(|(first_sample, second_sample)| (first_sample - second_sample).abs())
        .sum()
}

/// The index of the bob whose successor differs from it the most.
fn largest_jump_after(bobs: &[Vec<f32>]) -> usize {
    let jump = |index: usize| bob_difference(&bobs[index], &bobs[index + 1]);

    (0..bobs.len() - 1)
        .max_by(|first, second| jump(*first).total_cmp(&jump(*second)))
        .expect("the clip has bobs")
}

#[test]
fn tr0_averages_across_a_cut_as_qtgmc_does() {
    let (fixture, bobs) = bobs_by_clip()
        .into_iter()
        .find(|(fixture, _)| fixture.clip == "cut_8")
        .expect("cut_8 is a fixture clip");
    let (width, height) = (fixture.frames[0][0].width, fixture.frames[0][0].height);

    assert_eq!(
        largest_jump_after(&bobs),
        BEFORE_CUT,
        "the cut is not where the fixture puts it"
    );

    let found = reference_tr0(&bobs, BEFORE_CUT, 1, width, height);

    let blended: Vec<f32> = (0..width * height)
        .map(|position| {
            let earlier = bobs[BEFORE_CUT - 1][position];
            let centre = bobs[BEFORE_CUT][position];
            let later = bobs[BEFORE_CUT + 1][position];

            0.25 * earlier + 0.5 * centre + 0.25 * later
        })
        .collect();

    assert_plane_within(
        &found,
        &blended,
        DEVICE_TOLERANCE,
        width as u32,
        "the blend across the cut",
    );
}

#[test]
fn tr0_weights_sum_to_one() {
    for tr0 in 0..=2 {
        let total: f32 = tr0_weights(tr0).iter().sum();

        assert!((total - 1.0).abs() < 1e-6, "tr0 {tr0} weights sum to {total}");
    }
}
