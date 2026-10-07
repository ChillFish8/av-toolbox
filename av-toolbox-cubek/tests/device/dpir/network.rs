use av_toolbox_cubek::cnn::{BodyPath, CnnError, LayerRun, Precision, Region};
use av_toolbox_cubek::dpir::{
    Dpir,
    DpirMaps,
    DpirPlanes,
    Placement,
    Variant,
    Weights,
    column_bytes,
    level_bytes,
    run_region,
};
use av_toolbox_cubek::testing::dpir::{reference_dpir, synthetic_weights};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::artcnn::{supports, synced, textured};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// The largest absolute error allowed on f32 maps.
const F32_BOUND: f64 = 1e-4;

/// The largest absolute error allowed on f16 maps, rounded up from twice the 2.03e-5 measured on Im2col.
/// The convolution path measured 1.94e-5.
const F16_BOUND: f64 = 5e-5;

/// The strength most tests run at.
const STRENGTH: f32 = 10.0;

/// The value the output planes are filled with before the pass.
const UNTOUCHED: f32 = -1.0;

/// How many ResBlock convolutions one region's pass runs.
const RESBLOCK_CONVOLUTIONS: usize = 56;

/// The share of a reference's outputs that must lie inside the clamp's margins.
const UNCLAMPED_SHARE: f64 = 0.9;

/// The path the device runs `precision` on, as a stage would choose it.
pub(super) fn body_path(precision: Precision) -> BodyPath {
    let offered = supports(BodyPath::Convolution, precision);

    if offered {
        BodyPath::Convolution
    } else {
        BodyPath::Im2col
    }
}

/// Runs the network once over the whole of `planes`, each `width` by `height` and a multiple of 8 on a side,
/// at a peak of 1.0. Gives the output planes and how each ResBlock convolution ran.
struct Forward<'a> {
    weights: &'a Weights,
    planes: &'a [Vec<f32>],
    width: u32,
    height: u32,
    strength: f32,
    path: BodyPath,
    precision: Precision,
}

impl BuildTask for Forward<'_> {
    type Output = Result<(Vec<Vec<f32>>, Vec<LayerRun>), CnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let dpir = Dpir::upload(client, self.weights, self.precision);
        let region = Region {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        };
        let placement = Placement {
            region,
            left: 0,
            top: 0,
        };
        let sizes = level_bytes(self.precision, self.width, self.height);
        let untouched = vec![UNTOUCHED; region.samples() as usize];

        let input_handles: Vec<_> = self
            .planes
            .iter()
            .map(|values| upload_f32(client, values))
            .collect();
        let output_handles: Vec<_> = self
            .planes
            .iter()
            .map(|_| upload_f32(client, &untouched))
            .collect();
        let map_handles: Vec<Vec<_>> = sizes
            .iter()
            .map(|bytes| (0..4).map(|_| empty(client, *bytes)).collect())
            .collect();
        let columns = match self.path {
            BodyPath::Convolution | BodyPath::UnfusedConvolution => None,
            BodyPath::Im2col => {
                let bytes = column_bytes(self.precision, self.width, self.height);
                Some(empty(client, bytes))
            },
        };

        let inputs: Vec<_> = input_handles
            .iter()
            .map(|handle| plane(handle, self.width, self.height))
            .collect();
        let outputs: Vec<_> = output_handles
            .iter()
            .map(|handle| plane(handle, self.width, self.height))
            .collect();
        let planes = DpirPlanes {
            inputs: &inputs,
            outputs: &outputs,
        };
        let levels = std::array::from_fn(|level| {
            let handles = &map_handles[level];
            [&handles[0], &handles[1], &handles[2], &handles[3]]
        });
        let maps = DpirMaps {
            levels,
            columns: columns.as_ref(),
        };

        let ran = run_region(
            client,
            &sizing,
            &dpir,
            self.path,
            &planes,
            self.strength,
            1.0,
            placement,
            region,
            maps,
        );
        synced(client);
        let runs = ran?;

        let values = output_handles
            .iter()
            .map(|handle| read_f32(client, handle))
            .collect();

        Ok((values, runs))
    }
}

/// `planes` widened to f64.
pub(super) fn widened(planes: &[Vec<f32>]) -> Vec<Vec<f64>> {
    planes
        .iter()
        .map(|values| values.iter().map(|value| f64::from(*value)).collect())
        .collect()
}

/// Three planes of `width` by `height` in `0..=1`, each different from the others.
fn distinct_planes(width: u32, height: u32) -> Vec<Vec<f32>> {
    let red = textured(width, height);
    let green: Vec<f32> = red.iter().rev().copied().collect();
    let blue: Vec<f32> = red.iter().map(|value| 0.2 + value * 0.6).collect();

    vec![red, green, blue]
}

/// Runs a [Forward] of `planes` and compares it with the reference within `allowed` absolute error.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the comparison"
)]
fn check_against_reference(
    name: &str,
    weights: &Weights,
    planes: &[Vec<f32>],
    width: u32,
    height: u32,
    strength: f32,
    path: BodyPath,
    precision: Precision,
    allowed: f64,
) {
    let inputs = widened(planes);
    let expected = reference_dpir(weights, &inputs, width, height, 1.0, f64::from(strength));

    let task = Forward {
        weights,
        planes,
        width,
        height,
        strength,
        path,
        precision,
    };
    let (actual, _) = backend().build(task).expect("the pass runs");

    let mut largest_error = 0.0f64;
    for (actual_plane, expected_plane) in actual.iter().zip(&expected) {
        assert_eq!(
            actual_plane.len(),
            expected_plane.len(),
            "{name}: the sizes differ"
        );

        for (got, want) in actual_plane.iter().zip(expected_plane) {
            let error = (f64::from(*got) - want).abs();
            largest_error = largest_error.max(error);
        }
    }

    println!("{name} {path:?} {precision:?}: largest error {largest_error:e}, bound {allowed:e}");
    assert_eq!(actual.len(), expected.len(), "{name}: the plane counts differ");
    assert!(
        largest_error <= allowed,
        "{name} {path:?} {precision:?}: error {largest_error:e} passes {allowed:e}"
    );
}

#[test]
fn gray_matches_the_reference_in_f32() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(24, 16)];
    let path = body_path(Precision::F32);

    check_against_reference(
        "gray",
        &weights,
        &planes,
        24,
        16,
        STRENGTH,
        path,
        Precision::F32,
        F32_BOUND,
    );
}

#[test]
fn colour_matches_the_reference_with_distinct_planes() {
    let weights = synthetic_weights(Variant::Colour, 1);
    let planes = distinct_planes(16, 16);
    let path = body_path(Precision::F32);

    check_against_reference(
        "colour",
        &weights,
        &planes,
        16,
        16,
        STRENGTH,
        path,
        Precision::F32,
        F32_BOUND,
    );
}

#[test]
fn strength_extremes_match_the_reference() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(24, 16)];
    let path = body_path(Precision::F32);

    for strength in [0.0, 255.0] {
        check_against_reference(
            "strength extreme",
            &weights,
            &planes,
            24,
            16,
            strength,
            path,
            Precision::F32,
            F32_BOUND,
        );
    }
}

#[test]
fn gray_matches_the_reference_in_f16() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(24, 16)];
    let path = body_path(Precision::F16);

    check_against_reference(
        "gray",
        &weights,
        &planes,
        24,
        16,
        STRENGTH,
        path,
        Precision::F16,
        F16_BOUND,
    );
}

#[test]
fn gray_matches_the_reference_on_im2col_in_f16() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(24, 16)];

    check_against_reference(
        "gray",
        &weights,
        &planes,
        24,
        16,
        STRENGTH,
        BodyPath::Im2col,
        Precision::F16,
        F16_BOUND,
    );
}

#[test]
fn the_reference_is_not_hidden_by_the_clamp() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(24, 16)];
    let inputs = widened(&planes);

    let expected = reference_dpir(&weights, &inputs, 24, 16, 1.0, f64::from(STRENGTH));

    let values = &expected[0];
    let inside = values
        .iter()
        .filter(|value| **value > 0.05 && **value < 0.95)
        .count();
    let share = inside as f64 / values.len() as f64;

    println!("reference outputs inside the clamp's margins: {share}");
    assert!(
        share >= UNCLAMPED_SHARE,
        "only {share} of the outputs avoid the clamp"
    );
}

#[test]
fn run_region_reports_every_resblock_in_forward_order() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(16, 16)];
    let precision = Precision::F16;
    let path = body_path(precision);

    let task = Forward {
        weights: &weights,
        planes: &planes,
        width: 16,
        height: 16,
        strength: STRENGTH,
        path,
        precision,
    };
    let (_, runs) = backend().build(task).expect("the pass runs");

    // `LayerRun` holds no feature count, so the reference comparisons pin the order and this the count.
    assert_eq!(runs.len(), RESBLOCK_CONVOLUTIONS);
}
