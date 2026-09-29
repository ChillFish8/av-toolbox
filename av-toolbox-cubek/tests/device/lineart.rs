use av_toolbox_cubek::lineart::{
    MergePlanes,
    VinverseParams,
    VinversePlanes,
    clamped_merge,
    merge,
    vertical_121,
    vinverse,
};
use av_toolbox_cubek::neighbourhood::NeighbourPlanes;
use av_toolbox_cubek::testing::{
    reference_clamped_merge,
    reference_merge,
    reference_vertical_121,
    reference_vinverse,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

const SIZES: [(u32, u32); 4] = [(37, 23), (1, 5), (5, 1), (2, 2)];

/// The largest difference allowed between the device merge and the reference. It measured 3.05e-5,
/// one f32 step at these values.
const MERGE_TOLERANCE: f32 = 1e-4;

/// The largest difference allowed between the device vinverse and the reference. It measured
/// 1.53e-5, the same order as the merge.
const VINVERSE_TOLERANCE: f32 = 1e-4;

struct Merge {
    input: Vec<f32>,
    filtered: Vec<f32>,
    weight: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Merge {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);

        let input = upload_f32(client, &self.input);
        let filtered = upload_f32(client, &self.filtered);
        let weight = upload_f32(client, &self.weight);
        let planes = MergePlanes {
            input: plane(&input, self.width, self.height),
            filtered: plane(&filtered, self.width, self.height),
            weight: plane(&weight, self.width, self.height),
        };
        merge(client, &sizing, planes, 255.0);

        read_f32(client, &filtered)
    }
}

/// Runs [clamped_merge](av_toolbox_cubek::lineart::clamped_merge) with a weight peak of 255.
struct ClampedMerge {
    merge: Merge,
    peak: f32,
}

impl BuildTask for ClampedMerge {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let width = self.merge.width;
        let height = self.merge.height;

        let input = upload_f32(client, &self.merge.input);
        let filtered = upload_f32(client, &self.merge.filtered);
        let weight = upload_f32(client, &self.merge.weight);
        let planes = MergePlanes {
            input: plane(&input, width, height),
            filtered: plane(&filtered, width, height),
            weight: plane(&weight, width, height),
        };
        clamped_merge(client, &sizing, planes, 255.0, self.peak);

        read_f32(client, &filtered)
    }
}

struct Vertical121 {
    input: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Vertical121 {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let planes = NeighbourPlanes {
            input: plane(&input, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };
        vertical_121(client, &sizing, planes);

        read_f32(client, &output)
    }
}

/// A weight plane stepping through 0, 255 and the values between.
fn weights(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|index| {
            let step = index % 7;
            match step {
                0 => 0.0,
                1 => 255.0,
                _ => (step * 41 % 256) as f32,
            }
        })
        .collect()
}

/// A plane filtered away from `busy` by a changing amount, in both directions.
fn shifted(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let shift = (index % 23) as f32 * 7.0 - 80.0;
            value + shift
        })
        .collect()
}

fn largest_difference(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0, f32::max)
}

#[test]
fn merge_matches_the_reference() {
    let mut worst = 0.0f32;

    for (width, height) in SIZES {
        let input = busy(width, height);
        let filtered = shifted(&input);
        let weight = weights(width, height);
        let task = Merge {
            input: input.clone(),
            filtered: filtered.clone(),
            weight: weight.clone(),
            width,
            height,
        };

        let actual = backend().build(task);
        let expected = reference_merge(&input, &filtered, &weight, 255.0);

        assert_eq!(actual.len(), expected.len());

        let difference = largest_difference(&actual, &expected);
        println!("merge {width}x{height}: largest difference {difference}");
        assert!(
            difference <= MERGE_TOLERANCE,
            "merge {width}x{height} differs by {difference}",
        );

        worst = worst.max(difference);
    }

    println!("merge: largest difference over every size {worst}");
}

#[test]
fn merge_is_exact_at_no_weight_and_full_weight() {
    let (width, height) = (37, 23);
    let input = busy(width, height);
    let filtered = shifted(&input);
    let weight = weights(width, height);
    let task = Merge {
        input: input.clone(),
        filtered: filtered.clone(),
        weight: weight.clone(),
        width,
        height,
    };

    let actual = backend().build(task);

    for (index, sample_weight) in weight.iter().enumerate() {
        if *sample_weight == 0.0 {
            assert_eq!(actual[index], input[index], "sample {index} at weight 0");
        }

        if *sample_weight == 255.0 {
            assert_eq!(actual[index], filtered[index], "sample {index} at weight 255");
        }
    }
}

#[test]
fn a_full_weight_lands_on_a_fractional_filtered_value_far_from_the_input() {
    let task = Merge {
        input: vec![1000.0, 0.3],
        filtered: vec![0.3, 1000.0],
        weight: vec![255.0, 255.0],
        width: 2,
        height: 1,
    };

    let actual = backend().build(task);

    assert_eq!(actual, vec![0.3, 1000.0]);
}

/// `shifted` carries `busy` up to 80 below 0 and 74 above 255, so the clamp holds both ends.
#[test]
fn clamped_merge_matches_the_reference() {
    let mut worst = 0.0f32;

    for (width, height) in SIZES {
        let input = busy(width, height);
        let filtered = shifted(&input);
        let weight = weights(width, height);
        let outside = filtered
            .iter()
            .filter(|value| !(0.0..=255.0).contains(*value))
            .count();
        let merge = Merge {
            input: input.clone(),
            filtered: filtered.clone(),
            weight: weight.clone(),
            width,
            height,
        };
        let task = ClampedMerge { merge, peak: 255.0 };

        let actual = backend().build(task);
        let expected = reference_clamped_merge(&input, &filtered, &weight, 255.0, 255.0);

        assert_eq!(actual.len(), expected.len());

        let difference = largest_difference(&actual, &expected);
        println!(
            "clamped merge {width}x{height}: largest difference {difference}, {outside} outside the range"
        );
        assert!(
            difference <= MERGE_TOLERANCE,
            "clamped merge {width}x{height} differs by {difference}",
        );

        worst = worst.max(difference);
    }

    println!("clamped merge: largest difference over every size {worst}");
}

/// A full weight hands back a filtered value below 0 as 0 and one above the peak as the peak.
#[test]
fn a_full_weight_hands_back_the_clamped_filtered_value() {
    let merge = Merge {
        input: vec![60.0, 200.0, 100.0],
        filtered: vec![-67.0, 300.0, 42.5],
        weight: vec![255.0, 255.0, 255.0],
        width: 3,
        height: 1,
    };
    let task = ClampedMerge { merge, peak: 255.0 };

    let actual = backend().build(task);

    assert_eq!(actual, vec![0.0, 255.0, 42.5]);
}

#[test]
fn vertical_121_matches_the_reference() {
    for (width, height) in SIZES {
        let input = busy(width, height);
        let task = Vertical121 {
            input: input.clone(),
            width,
            height,
        };

        let actual = backend().build(task);
        let expected = reference_vertical_121(&input, width, height);

        assert_eq!(actual, expected, "{width}x{height}");
    }
}

struct Vinverse {
    input: Vec<f32>,
    width: u32,
    height: u32,
    params: VinverseParams,
}

impl BuildTask for Vinverse {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let blurred = empty(client, bytes);
        let diff = empty(client, bytes);
        let diff_blurred = empty(client, bytes);
        let sbr = empty(client, bytes);
        let planes = VinversePlanes {
            input: plane(&input, self.width, self.height),
            output: plane(&output, self.width, self.height),
            blurred: plane(&blurred, self.width, self.height),
            diff: plane(&diff, self.width, self.height),
            diff_blurred: plane(&diff_blurred, self.width, self.height),
            sbr: plane(&sbr, self.width, self.height),
        };
        vinverse(client, &sizing, planes, &self.params);

        read_f32(client, &output)
    }
}

#[test]
fn vinverse_matches_the_reference() {
    let limits = [None, Some(9.0)];
    let mut worst = 0.0f32;

    for (width, height) in SIZES {
        for amnt in limits {
            let input: Vec<f32> = busy(width, height)
                .iter()
                .map(|value| value * 0.37 + 0.1)
                .collect();
            let params = VinverseParams {
                contra_str: 2.7,
                scl: 0.25,
                amnt,
            };
            let task = Vinverse {
                input: input.clone(),
                width,
                height,
                params,
            };

            let actual = backend().build(task);
            let expected = reference_vinverse(&input, width, height, 2.7, 0.25, amnt);
            let difference = largest_difference(&actual, &expected);

            println!("vinverse {width}x{height} amnt {amnt:?}: largest difference {difference}");
            assert!(
                difference <= VINVERSE_TOLERANCE,
                "vinverse {width}x{height} amnt {amnt:?} differs by {difference}",
            );

            worst = worst.max(difference);
        }
    }

    println!("vinverse: largest difference over every size {worst}");
}
