use av_toolbox_cubek::dehalo::alpha::{
    AlphaParams,
    ClampPlanes,
    HaloPlanes,
    StrengthPlanes,
    apply_strength,
    blend_halo_mask,
    clamp_between,
    halo_mask_sample,
    strength_sample,
};
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
use av_toolbox_cubek::dehalo::yahr::{YahrPlanes, yahr_combine};
use av_toolbox_cubek::dehalo::{DifferencePlanes, MergePlanes, difference, merge_masked};
use av_toolbox_cubek::testing::reference_merge_masked;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

fn defaults(bits: u32) -> AlphaParams {
    AlphaParams {
        rx: 2.0,
        ry: 2.0,
        darkstr: 1.0,
        brightstr: 1.0,
        lowsens: 50.0,
        highsens: 50.0,
        ss: 1.5,
        bits,
    }
}

/// Six busy planes of `width` by `height`, each shifted so they differ.
fn six_planes(width: u32, height: u32) -> Vec<Vec<f32>> {
    let base = busy(width, height);

    (0..6u32)
        .map(|plane| {
            base.iter()
                .map(|value| (value + plane as f32 * 13.0) % 256.0)
                .collect()
        })
        .collect()
}

/// A busy plane, optionally transposed and shifted, folded down to `0..modulus` then scaled.
///
/// Transposing swaps which axis `busy` reads as the row and which as the column, so it gives a
/// plane shaped differently from the untransposed one at the same size.
fn ranged_plane(width: u32, height: u32, transpose: bool, shift: f32, modulus: f32, scale: f32) -> Vec<f32> {
    let source = if transpose {
        busy(height, width)
    } else {
        busy(width, height)
    };

    source
        .iter()
        .map(|value| {
            let folded = (value + shift) % modulus;
            folded * scale
        })
        .collect()
}

/// Two planes whose difference, index by index, spans negative, zero and positive values.
fn signed_planes(width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let samples = (width * height) as i32;
    let first: Vec<f32> = (0..samples).map(|index| (index % 5) as f32 * 10.0).collect();
    let second: Vec<f32> = (0..samples)
        .map(|index| ((index * 3 + 2) % 5) as f32 * 10.0)
        .collect();

    (first, second)
}

fn pairwise_max(first: &[f32], second: &[f32]) -> Vec<f32> {
    first.iter().zip(second).map(|(a, b)| a.max(*b)).collect()
}

fn pairwise_min(first: &[f32], second: &[f32]) -> Vec<f32> {
    first.iter().zip(second).map(|(a, b)| a.min(*b)).collect()
}

/// The share of `values` that sit strictly between 0 and `peak`, so a mask cannot be checked while
/// quietly clamped at both ends of every sample.
fn share_unsaturated(values: &[f32], peak: f32) -> f32 {
    let unsaturated = values
        .iter()
        .filter(|value| **value > 0.0 && **value < peak)
        .count();

    unsaturated as f32 / values.len() as f32
}

struct HaloMask {
    clip: Vec<f32>,
    halos: Vec<f32>,
    clip_max: Vec<f32>,
    clip_min: Vec<f32>,
    halo_max: Vec<f32>,
    halo_min: Vec<f32>,
    width: u32,
    height: u32,
    params: AlphaParams,
}

impl BuildTask for HaloMask {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let halos = upload_f32(client, &self.halos);
        let clip_max = upload_f32(client, &self.clip_max);
        let clip_min = upload_f32(client, &self.clip_min);
        let halo_max = upload_f32(client, &self.halo_max);
        let halo_min = upload_f32(client, &self.halo_min);
        let output = empty(client, bytes);
        let planes = HaloPlanes {
            clip: plane(&clip, self.width, self.height),
            halos: plane(&halos, self.width, self.height),
            clip_max: plane(&clip_max, self.width, self.height),
            clip_min: plane(&clip_min, self.width, self.height),
            halo_max: plane(&halo_max, self.width, self.height),
            halo_min: plane(&halo_min, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        blend_halo_mask(client, &sizing, planes, &self.params);

        read_f32(client, &output)
    }
}

#[test]
fn the_halo_mask_matches_the_reference() {
    let width = 37;
    let height = 23;

    for bits in [8, 10] {
        let params = defaults(bits);
        let scale = params.scale(1.0);
        let peak = params.peak();

        // Each range comes from the max and min of two differently shaped planes, so it holds a
        // real spread of small values rather than one constant gap.
        let clip_a = ranged_plane(width, height, false, 0.0, 16.0, scale);
        let clip_b = ranged_plane(width, height, true, 3.0, 16.0, scale);
        let halo_a = ranged_plane(width, height, false, 5.0, 16.0, scale);
        let halo_b = ranged_plane(width, height, true, 9.0, 16.0, scale);

        let clip_max = pairwise_max(&clip_a, &clip_b);
        let clip_min = pairwise_min(&clip_a, &clip_b);
        let halo_max = pairwise_max(&halo_a, &halo_b);
        let halo_min = pairwise_min(&halo_a, &halo_b);

        let clip: Vec<f32> = busy(width, height).iter().map(|value| value * scale).collect();
        let halos: Vec<f32> = clip.iter().map(|value| (value + 40.0 * scale) % peak).collect();

        let task = HaloMask {
            clip: clip.clone(),
            halos: halos.clone(),
            clip_max: clip_max.clone(),
            clip_min: clip_min.clone(),
            halo_max: halo_max.clone(),
            halo_min: halo_min.clone(),
            width,
            height,
            params,
        };

        let actual = backend().build(task);
        let expected: Vec<f32> = (0..(width * height) as usize)
            .map(|index| {
                let clip_range = clip_max[index] - clip_min[index];
                let halo_range = halo_max[index] - halo_min[index];
                halo_mask_sample(clip[index], halos[index], clip_range, halo_range, &params)
            })
            .collect();

        assert_close(&actual, &expected, 1e-3);

        let share = share_unsaturated(&expected, peak);
        assert!(
            share > 0.1,
            "bits {bits}: only {share} of the mask sat strictly between 0 and peak"
        );
    }
}

struct Clamp {
    clip: Vec<f32>,
    highest: Vec<f32>,
    lowest: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Clamp {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let highest = upload_f32(client, &self.highest);
        let lowest = upload_f32(client, &self.lowest);
        let output = empty(client, bytes);
        let planes = ClampPlanes {
            clip: plane(&clip, self.width, self.height),
            highest: plane(&highest, self.width, self.height),
            lowest: plane(&lowest, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        clamp_between(client, &sizing, planes);

        read_f32(client, &output)
    }
}

#[test]
fn the_clamp_matches_the_reference() {
    let width = 37;
    let height = 23;
    let planes = six_planes(width, height);
    let task = Clamp {
        clip: planes[0].clone(),
        highest: planes[1].clone(),
        lowest: planes[2].clone(),
        width,
        height,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| planes[0][index].min(planes[1][index]).max(planes[2][index]))
        .collect();

    assert_eq!(actual, expected);
}

struct Strength {
    clip: Vec<f32>,
    remove: Vec<f32>,
    width: u32,
    height: u32,
    params: AlphaParams,
}

impl BuildTask for Strength {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let remove = upload_f32(client, &self.remove);
        let output = empty(client, bytes);
        let planes = StrengthPlanes {
            clip: plane(&clip, self.width, self.height),
            remove: plane(&remove, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        apply_strength(client, &sizing, planes, &self.params);

        read_f32(client, &output)
    }
}

fn run_strength_case(clip: &[f32], remove: &[f32], width: u32, height: u32, params: &AlphaParams) {
    let task = Strength {
        clip: clip.to_vec(),
        remove: remove.to_vec(),
        width,
        height,
        params: *params,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| strength_sample(clip[index], remove[index], params))
        .collect();

    assert_close(&actual, &expected, 1e-3);
}

#[test]
fn the_strength_matches_the_reference() {
    let width = 37;
    let height = 23;
    let planes = six_planes(width, height);
    let moderate = AlphaParams {
        darkstr: 0.5,
        brightstr: 0.8,
        ..defaults(8)
    };

    run_strength_case(&planes[0], &planes[1], width, height, &moderate);

    // Strengths above 1 push the pulled value past 0 or the peak, so this exercises the clamp.
    let strong = AlphaParams {
        darkstr: 1.5,
        brightstr: 1.5,
        ..defaults(8)
    };
    let near_zero_clip = vec![5.0; 16];
    let near_zero_remove = vec![200.0; 16];
    let near_peak_clip = vec![250.0; 16];
    let near_peak_remove = vec![50.0; 16];

    run_strength_case(&near_zero_clip, &near_zero_remove, 4, 4, &strong);
    run_strength_case(&near_peak_clip, &near_peak_remove, 4, 4, &strong);
}

struct Merge {
    clip: Vec<f32>,
    filtered: Vec<f32>,
    mask: Vec<f32>,
    width: u32,
    height: u32,
    peak: f32,
}

impl BuildTask for Merge {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let filtered = upload_f32(client, &self.filtered);
        let mask = upload_f32(client, &self.mask);
        let output = empty(client, bytes);
        let planes = MergePlanes {
            clip: plane(&clip, self.width, self.height),
            filtered: plane(&filtered, self.width, self.height),
            mask: plane(&mask, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        merge_masked(client, &sizing, planes, self.peak);

        read_f32(client, &output)
    }
}

#[test]
fn the_masked_merge_matches_the_reference() {
    let width = 37;
    let height = 23;
    let planes = six_planes(width, height);
    let task = Merge {
        clip: planes[0].clone(),
        filtered: planes[1].clone(),
        mask: planes[2].clone(),
        width,
        height,
        peak: 255.0,
    };

    let actual = backend().build(task);
    let expected = reference_merge_masked(&planes[0], &planes[1], &planes[2], 255.0);

    assert_close(&actual, &expected, 1e-3);
}

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

struct Difference {
    first: Vec<f32>,
    second: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Difference {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.first.len() * size_of::<f32>();

        let first = upload_f32(client, &self.first);
        let second = upload_f32(client, &self.second);
        let output = empty(client, bytes);
        let planes = DifferencePlanes {
            first: plane(&first, self.width, self.height),
            second: plane(&second, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        difference(client, &sizing, planes);

        read_f32(client, &output)
    }
}

#[test]
fn the_difference_matches_the_reference() {
    let width = 37;
    let height = 23;
    let (first, second) = signed_planes(width, height);
    let task = Difference {
        first: first.clone(),
        second: second.clone(),
        width,
        height,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = first
        .iter()
        .zip(&second)
        .map(|(value, other)| value - other)
        .collect();

    assert_close(&actual, &expected, 1e-3);

    let negative = expected.iter().filter(|value| **value < 0.0).count();
    let zero = expected.iter().filter(|value| **value == 0.0).count();
    let positive = expected.iter().filter(|value| **value > 0.0).count();
    assert!(negative > 0, "no negative differences");
    assert!(zero > 0, "no zero differences");
    assert!(positive > 0, "no positive differences");
}

struct YahrCombine {
    clip: Vec<f32>,
    detail: Vec<f32>,
    limited: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for YahrCombine {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let detail = upload_f32(client, &self.detail);
        let limited = upload_f32(client, &self.limited);
        let output = empty(client, bytes);
        let planes = YahrPlanes {
            clip: plane(&clip, self.width, self.height),
            detail: plane(&detail, self.width, self.height),
            limited: plane(&limited, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        yahr_combine(client, &sizing, planes);

        read_f32(client, &output)
    }
}

#[test]
fn the_yahr_combine_matches_the_reference() {
    let width = 37;
    let height = 23;
    let (detail, limited) = signed_planes(width, height);
    let clip = busy(width, height);
    let task = YahrCombine {
        clip: clip.clone(),
        detail: detail.clone(),
        limited: limited.clone(),
        width,
        height,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| clip[index] - (detail[index] - limited[index]))
        .collect();

    assert_close(&actual, &expected, 1e-3);

    let change = detail.iter().zip(&limited).map(|(value, other)| value - other);
    let negative = change.clone().filter(|value| *value < 0.0).count();
    let zero = change.clone().filter(|value| *value == 0.0).count();
    let positive = change.filter(|value| *value > 0.0).count();
    assert!(negative > 0, "no negative differences");
    assert!(zero > 0, "no zero differences");
    assert!(positive > 0, "no positive differences");
}
