use av_toolbox_cubek::cnn::{BodyPath, Precision};
use av_toolbox_cubek::dpir::{PACK, Variant, Weights};
use av_toolbox_cubek::testing::dpir::{GOLDEN_RUNS, GoldenRun, planes_of};
use av_toolbox_models::Store;

use super::network::{body_path, widened};
use super::tiling::{Deblocked, largest_plane_error};
use crate::support::backend;

/// The largest absolute error allowed against ONNX Runtime in f32, rounded up from the 1.43e-6 measured.
const F32_BOUND: f64 = 1e-5;

/// The largest absolute error allowed against ONNX Runtime in f16, rounded up from twice the 6.64e-4 measured
/// on Im2col. The convolution path measured 4.47e-4.
const F16_BOUND: f64 = 2e-3;

/// The installed weights of both variants, or `None` with the reason printed when either is not installed.
fn installed() -> Option<(Weights, Weights)> {
    let gray = installed_variant(Variant::Gray)?;
    let colour = installed_variant(Variant::Colour)?;

    Some((gray, colour))
}

fn installed_variant(variant: Variant) -> Option<Weights> {
    let store = Store::from_environment();

    let located = match store.find(PACK, variant.file_name()) {
        Ok(located) => located,
        Err(missing) => {
            println!(
                "the {variant:?} DPIR model is not installed, so the goldens are not checked: {missing}"
            );
            return None;
        },
    };

    let bytes = located.read().expect("the installed weights read");
    let parsed = Weights::parse(&bytes, variant);

    Some(parsed.expect("the installed weights parse"))
}

/// Runs one golden in `precision` on `path`, giving its largest absolute error against ONNX Runtime.
fn golden_error(run: &GoldenRun, weights: &Weights, path: BodyPath, precision: Precision) -> f64 {
    let channels = run.variant.image_channels() as usize;
    let planes = planes_of(run.input, run.width, run.height, channels);
    let expected_planes = planes_of(run.expected, run.width, run.height, channels);
    let expected = widened(&expected_planes);

    let task = Deblocked {
        weights,
        planes: &planes,
        width: run.width,
        height: run.height,
        output_width: run.width,
        strength: run.strength,
        peak: 1.0,
        path,
        precision,
        tile: None,
    };
    let (actual, _) = backend().build(task).expect("the golden deblocks");

    largest_plane_error(&actual, &expected)
}

/// Checks every golden in `precision` on `path` within `allowed`, printing each run's error.
fn check_goldens(path: BodyPath, precision: Precision, allowed: f64) {
    let Some((gray, colour)) = installed() else {
        return;
    };

    let mut largest = 0.0f64;
    for run in &GOLDEN_RUNS {
        let weights = match run.variant {
            Variant::Gray => &gray,
            Variant::Colour => &colour,
        };

        let error = golden_error(run, weights, path, precision);
        println!("{} {path:?} {precision:?}: largest error {error:e}", run.name);
        largest = largest.max(error);
    }

    println!("every golden {path:?} {precision:?}: largest error {largest:e}, bound {allowed:e}");
    assert!(
        largest <= allowed,
        "{path:?} {precision:?}: error {largest:e} passes {allowed:e}"
    );
}

#[test]
fn every_golden_matches_in_f32() {
    let precision = Precision::F32;
    let path = body_path(precision);

    check_goldens(path, precision, F32_BOUND);
}

#[test]
fn every_golden_matches_in_f16() {
    let precision = Precision::F16;
    let path = body_path(precision);

    check_goldens(path, precision, F16_BOUND);
}

#[test]
fn every_golden_matches_on_im2col_in_f16() {
    check_goldens(BodyPath::Im2col, Precision::F16, F16_BOUND);
}
