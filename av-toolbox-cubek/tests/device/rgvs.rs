use av_toolbox_cubek::neighbourhood::Smooth;
use av_toolbox_cubek::rgvs::{
    GrainBorder,
    RemoveGrainMode,
    RemoveGrainPlanes,
    RepairMode,
    RepairPlanes,
    remove_grain,
    repair,
};
use av_toolbox_cubek::testing::{reference_remove_grain, reference_repair, reference_smooth};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, backend, empty, plane, read_f32, seeded_plane, upload_f32};

struct Repair {
    clip: Vec<f32>,
    reference: Vec<f32>,
    width: u32,
    height: u32,
    mode: RepairMode,
}

impl BuildTask for Repair {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let reference = upload_f32(client, &self.reference);
        let output = empty(client, bytes);
        let clip_plane = plane(&clip, self.width, self.height);
        let reference_plane = plane(&reference, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = RepairPlanes {
            clip: clip_plane,
            reference: reference_plane,
            output: output_plane,
        };
        repair(client, &sizing, planes, self.mode);

        read_f32(client, &output)
    }
}

struct RemoveGrain {
    input: Vec<f32>,
    width: u32,
    height: u32,
    mode: RemoveGrainMode,
    border: GrainBorder,
}

impl BuildTask for RemoveGrain {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = RemoveGrainPlanes {
            input: input_plane,
            output: output_plane,
        };
        remove_grain(client, &sizing, planes, self.mode, self.border);

        read_f32(client, &output)
    }
}

/// A plane split into a flat left half and a busy right half drawn from a small repeated set of
/// values.
///
/// The flat half ties the centre for RemoveGrain's averaging modes, and the busy half exercises
/// every mode's clamps, and mode 7's tied window costs, rather than always breaking a tie on
/// distinct neighbours.
fn repeated(width: u32, height: u32) -> Vec<f32> {
    let flat_width = width / 2;

    busy(width, height)
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let x = index as u32 % width;

            if x < flat_width {
                100.0
            } else {
                (*value as u32 % 6) as f32 * 50.0
            }
        })
        .collect()
}

/// The share of inner samples where `actual` differs from `clip`, so a test can check a mode's
/// clamps are really reached rather than passing on an accidental no-op.
fn changed_share(actual: &[f32], clip: &[f32], width: u32, height: u32) -> f64 {
    let mut changed = 0;
    let mut inner = 0;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = (y * width + x) as usize;
            inner += 1;

            if actual[index] != clip[index] {
                changed += 1;
            }
        }
    }

    changed as f64 / inner as f64
}

#[test]
fn repair_matches_the_reference() {
    let modes = [
        RepairMode::One,
        RepairMode::Two,
        RepairMode::Thirteen,
        RepairMode::Seventeen,
        RepairMode::TwentyFour,
    ];

    for mode in modes {
        for (width, height) in [(37, 23), (2, 2), (1, 6)] {
            let clip = busy(width, height);
            let reference: Vec<f32> = clip.iter().map(|value| (value * 7.0) % 256.0).collect();
            let task = Repair {
                clip: clip.clone(),
                reference: reference.clone(),
                width,
                height,
                mode,
            };

            let actual = backend().build(task);
            let expected = reference_repair(&clip, &reference, width, height, mode);

            assert_eq!(actual, expected, "{mode:?} {width}x{height}");

            if (width, height) == (37, 23) {
                let share = changed_share(&expected, &clip, width, height);

                assert!(
                    share >= 0.05,
                    "{mode:?} only changed {share} of the inner samples"
                );
            }
        }
    }
}

#[test]
fn every_remove_grain_mode_and_border_matches_the_reference() {
    let modes = [
        RemoveGrainMode::Seven,
        RemoveGrainMode::Eleven,
        RemoveGrainMode::Seventeen,
        RemoveGrainMode::TwentyTwo,
    ];
    let borders = [GrainBorder::Copy, GrainBorder::Reflect];

    for mode in modes {
        for border in borders {
            for (width, height) in [(37, 23), (1, 5), (5, 1), (2, 2)] {
                let input = repeated(width, height);
                let task = RemoveGrain {
                    input: input.clone(),
                    width,
                    height,
                    mode,
                    border,
                };

                let actual = backend().build(task);
                let expected = reference_remove_grain(&input, width, height, mode, border);

                assert_eq!(actual, expected, "{mode:?} {border:?} {width}x{height}");

                if (width, height) == (37, 23) {
                    let share = changed_share(&expected, &input, width, height);

                    assert!(
                        (0.10..=0.90).contains(&share),
                        "{mode:?} {border:?} changed {share} of the inner samples"
                    );
                }
            }
        }
    }
}

#[test]
fn mode_eleven_keeps_the_border_when_copying() {
    for (width, height) in [(33u32, 17u32), (2, 5), (5, 2)] {
        check_mode_eleven_border(width, height);
    }
}

fn check_mode_eleven_border(width: u32, height: u32) {
    let input = seeded_plane(width, height);
    let smoothed = reference_smooth(&input, width, height, Smooth::Binomial);

    let output = backend().build(RemoveGrain {
        input: input.clone(),
        width,
        height,
        mode: RemoveGrainMode::Eleven,
        border: GrainBorder::Copy,
    });

    for (index, sample) in output.iter().enumerate() {
        let x = index as u32 % width;
        let y = index as u32 / width;
        let at_border = x == 0 || y == 0 || x + 1 == width || y + 1 == height;

        if at_border {
            assert_eq!(
                *sample, input[index],
                "border sample {x}, {y} of {width}x{height}"
            );
        } else {
            let error = (sample - smoothed[index]).abs();
            assert!(
                error <= 1e-6,
                "sample {x}, {y} was {sample}, expected {}",
                smoothed[index]
            );
        }
    }
}
