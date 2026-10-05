use av_toolbox_cubek::deinterlace::qtgmc::{BobPlanes, Parity, bob, bob_taps};
use av_toolbox_cubek::resize::upload;
use av_toolbox_cubek::testing::{assert_plane_within, qtgmc_fixture, reference_bob};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::{DEVICE_TOLERANCE, FIXTURE_TOLERANCE, codes, peak};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

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
