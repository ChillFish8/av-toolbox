use av_toolbox_cubek::rgvs::{
    RemoveGrainMode,
    RemoveGrainPlanes,
    RepairMode,
    RepairPlanes,
    remove_grain,
    repair,
};
use av_toolbox_cubek::testing::{reference_remove_grain, reference_repair};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

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
        remove_grain(client, &sizing, planes, self.mode);

        read_f32(client, &output)
    }
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
fn remove_grain_matches_the_reference() {
    let modes = [RemoveGrainMode::Seven, RemoveGrainMode::Seventeen];

    for mode in modes {
        for (width, height) in [(37, 23), (2, 2), (1, 6)] {
            let input = busy(width, height);
            let task = RemoveGrain {
                input: input.clone(),
                width,
                height,
                mode,
            };

            let actual = backend().build(task);
            let expected = reference_remove_grain(&input, width, height, mode);

            assert_eq!(actual, expected, "{mode:?} {width}x{height}");
        }
    }
}
