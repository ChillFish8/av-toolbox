mod alpha;
mod edge_clean;
mod fine;
mod yahr;

use av_toolbox_cubek::dehalo::{DifferencePlanes, MergePlanes, difference, merge_masked};
use av_toolbox_cubek::testing::reference_merge_masked;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

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
