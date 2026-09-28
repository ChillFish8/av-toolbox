use av_toolbox_cubek::dehalo::fine::{
    ContrastPlanes,
    OutsidePlanes,
    RampPlanes,
    contrasharpen,
    contrast_sample,
    outside_mask,
    outside_sample,
    ramp,
    scale_values,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::ranged_plane;
use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

struct Contrasharpen {
    dehaloed: Vec<f32>,
    blurred: Vec<f32>,
    repaired: Vec<f32>,
    clip: Vec<f32>,
    width: u32,
    height: u32,
    level: f32,
}

impl BuildTask for Contrasharpen {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.dehaloed.len() * size_of::<f32>();

        let dehaloed = upload_f32(client, &self.dehaloed);
        let blurred = upload_f32(client, &self.blurred);
        let repaired = upload_f32(client, &self.repaired);
        let clip = upload_f32(client, &self.clip);
        let output = empty(client, bytes);
        let planes = ContrastPlanes {
            dehaloed: plane(&dehaloed, self.width, self.height),
            blurred: plane(&blurred, self.width, self.height),
            repaired: plane(&repaired, self.width, self.height),
            clip: plane(&clip, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        contrasharpen(client, &sizing, planes, self.level);

        read_f32(client, &output)
    }
}

#[test]
fn contrasharpening_matches_the_reference() {
    let width = 37;
    let height = 23;
    let level = 1.5;

    // Shifted and moduli differently on purpose, so `detail` and `removed` each take both signs
    // and a spread of magnitudes relative to each other, and every branch of the restore gets hit.
    let blurred = ranged_plane(width, height, false, 5.0, 50.0, 1.0);
    let repaired = ranged_plane(width, height, true, 30.0, 50.0, 1.0);
    let dehaloed = ranged_plane(width, height, false, 10.0, 180.0, 1.0);
    let clip = ranged_plane(width, height, true, 40.0, 180.0, 1.0);
    let task = Contrasharpen {
        dehaloed: dehaloed.clone(),
        blurred: blurred.clone(),
        repaired: repaired.clone(),
        clip: clip.clone(),
        width,
        height,
        level,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| {
            let smoothed = dehaloed[index];
            let blurred_value = blurred[index];
            let repaired_value = repaired[index];
            let clip_value = clip[index];

            contrast_sample(smoothed, blurred_value, repaired_value, clip_value, level)
        })
        .collect();

    assert_close(&actual, &expected, 1e-3);

    // The kernel and its mirror branch three ways on `detail` and `removed`, and each branch needs
    // a real share of samples or the untested one could silently break.
    let mut opposite_signs = 0;
    let mut detail_wins = 0;
    let mut removed_wins = 0;
    for index in 0..dehaloed.len() {
        let detail = (blurred[index] - repaired[index]) * 2.49 * level;
        let removed = clip[index] - dehaloed[index];

        if detail * removed < 0.0 {
            opposite_signs += 1;
        } else if detail.abs() < removed.abs() {
            detail_wins += 1;
        } else {
            removed_wins += 1;
        }
    }

    let total = dehaloed.len() as f32;
    let opposite_share = opposite_signs as f32 / total;
    let detail_share = detail_wins as f32 / total;
    let removed_share = removed_wins as f32 / total;
    assert!(
        opposite_share > 0.05,
        "only {opposite_share} took the zero branch"
    );
    assert!(detail_share > 0.05, "only {detail_share} took the detail branch");
    assert!(
        removed_share > 0.05,
        "only {removed_share} took the removed branch"
    );
}

struct Ramp {
    input: Vec<f32>,
    width: u32,
    height: u32,
    low: f32,
    range: f32,
    peak: f32,
}

impl BuildTask for Ramp {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let planes = RampPlanes {
            input: plane(&input, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        ramp(client, &sizing, planes, self.low, self.range, self.peak);

        read_f32(client, &output)
    }
}

#[test]
fn the_ramp_matches_the_reference() {
    let width = 37;
    let height = 23;
    // `busy` spans 0..256, so this crosses below `low`, between `low` and `low + range`, and above it.
    let input = busy(width, height);
    let low = 80.0;
    let range = 48.0;
    let peak = 255.0;
    let task = Ramp {
        input: input.clone(),
        width,
        height,
        low,
        range,
        peak,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = input
        .iter()
        .map(|value| ((value - low) / range * 255.0).clamp(0.0, peak))
        .collect();

    assert_close(&actual, &expected, 1e-3);
}

struct Scale {
    input: Vec<f32>,
    width: u32,
    height: u32,
    factor: f32,
    peak: f32,
}

impl BuildTask for Scale {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let planes = RampPlanes {
            input: plane(&input, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        scale_values(client, &sizing, planes, self.factor, self.peak);

        read_f32(client, &output)
    }
}

#[test]
fn scaling_matches_the_reference() {
    let width = 37;
    let height = 23;
    let input = busy(width, height);
    let factor = 4.0;
    let peak = 255.0;
    let task = Scale {
        input: input.clone(),
        width,
        height,
        factor,
        peak,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = input
        .iter()
        .map(|value| (value * factor).clamp(0.0, peak))
        .collect();

    assert_close(&actual, &expected, 1e-3);
}

struct Outside {
    large: Vec<f32>,
    strong: Vec<f32>,
    shrink: Vec<f32>,
    width: u32,
    height: u32,
    excl: bool,
    edgeproc: f32,
    peak: f32,
}

impl BuildTask for Outside {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.large.len() * size_of::<f32>();

        let large = upload_f32(client, &self.large);
        let strong = upload_f32(client, &self.strong);
        let shrink = upload_f32(client, &self.shrink);
        let output = empty(client, bytes);
        let planes = OutsidePlanes {
            large: plane(&large, self.width, self.height),
            strong: plane(&strong, self.width, self.height),
            shrink: plane(&shrink, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        outside_mask(client, &sizing, planes, self.excl, self.edgeproc, self.peak);

        read_f32(client, &output)
    }
}

#[test]
fn the_outside_mask_matches_the_reference() {
    let width = 37;
    let height = 23;
    let peak = 255.0;
    // `large`'s range and scale run well past `peak` once doubled against `inside`, so this also
    // exercises the upper clamp rather than only the lower one.
    let large = ranged_plane(width, height, false, 0.0, 150.0, 2.5);
    let strong = ranged_plane(width, height, true, 30.0, 60.0, 1.5);
    let shrink = ranged_plane(width, height, false, 50.0, 40.0, 1.0);

    for excl in [true, false] {
        for edgeproc in [0.0, 0.5] {
            let task = Outside {
                large: large.clone(),
                strong: strong.clone(),
                shrink: shrink.clone(),
                width,
                height,
                excl,
                edgeproc,
                peak,
            };

            let actual = backend().build(task);
            let expected: Vec<f32> = (0..(width * height) as usize)
                .map(|index| {
                    let large_value = large[index];
                    let strong_value = strong[index];
                    let shrink_value = shrink[index];

                    outside_sample(large_value, strong_value, shrink_value, excl, edgeproc, peak)
                })
                .collect();

            assert_close(&actual, &expected, 1e-3);

            let unsaturated = expected
                .iter()
                .filter(|value| **value > 0.0 && **value < peak)
                .count();
            let share = unsaturated as f32 / expected.len() as f32;
            assert!(
                share > 0.1,
                "excl {excl} edgeproc {edgeproc}: only {share} of the band sat strictly between 0 and peak"
            );
        }
    }
}
