use av_toolbox_cubek::artcnn::{
    ArtcnnError,
    BodyPath,
    Maps,
    Model,
    Network,
    Precision,
    Region,
    Weights,
    run_region,
};
use av_toolbox_cubek::testing::{reference_artcnn, synthetic_weights};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::{ComputeClient, CubePrimitive, Runtime};

use super::{synced, textured};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

const PATHS: [BodyPath; 2] = [BodyPath::Convolution, BodyPath::Im2col];

const PRECISIONS: [Precision; 2] = [Precision::F32, Precision::F16];

pub(super) const PEAK: f32 = 65535.0;

/// The code the output is filled with before the pass, which is -1 once divided by the peak.
pub(super) const UNTOUCHED: f32 = -PEAK;

/// Whether the device has a cooperative matrix taking `precision`'s type and accumulating in f32.
fn has_cooperative_matrix<R: Runtime>(client: &ComputeClient<R>, precision: Precision) -> bool {
    let operand_type = match precision {
        Precision::F16 => half::f16::as_type_native_unchecked(),
        Precision::F32 => f32::as_type_native_unchecked(),
    };
    let accumulator_type = f32::as_type_native_unchecked();
    let operand = operand_type.storage_type();
    let accumulator = accumulator_type.storage_type();
    let properties = client.properties();

    properties
        .features
        .matmul
        .cmma
        .iter()
        .any(|config| config.a_type == operand && config.b_type == operand && config.cd_type == accumulator)
}

struct Supported {
    path: BodyPath,
    precision: Precision,
}

impl BuildTask for Supported {
    type Output = bool;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();

        match self.path {
            BodyPath::Convolution => has_cooperative_matrix(client, self.precision),
            BodyPath::Im2col => true,
        }
    }
}

fn supports(path: BodyPath, precision: Precision) -> bool {
    let task = Supported { path, precision };

    backend().build(task)
}

pub(super) fn supported_runs() -> Vec<(BodyPath, Precision)> {
    let mut runs = Vec::new();

    for path in PATHS {
        for precision in PRECISIONS {
            let supported = supports(path, precision);
            println!("{path:?} at {precision:?}: supported {supported}");

            if supported {
                runs.push((path, precision));
            }
        }
    }

    runs
}

/// Runs the network over the whole of a plane in `0..=1`, and gives the doubled output divided by the peak.
struct Forward {
    weights: Weights,
    plane: Vec<f32>,
    width: u32,
    height: u32,
    path: BodyPath,
    precision: Precision,
}

impl BuildTask for Forward {
    type Output = Result<Vec<f32>, ArtcnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let network = Network::upload(client, &self.weights, self.precision);
        let region = Region {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        };
        let map_count = (region.samples() * network.features()) as usize;
        let map_bytes = map_count * self.precision.bytes();
        let output_samples = (region.samples() * 4) as usize;
        let untouched = vec![UNTOUCHED; output_samples];
        let scaled: Vec<f32> = self.plane.iter().map(|value| value * PEAK).collect();

        let input = upload_f32(client, &scaled);
        let output = upload_f32(client, &untouched);
        let conv0 = empty(client, map_bytes);
        let first = empty(client, map_bytes);
        let second = empty(client, map_bytes);
        let third = empty(client, map_bytes);
        let columns = match self.path {
            BodyPath::Convolution => None,
            BodyPath::Im2col => Some(empty(client, map_bytes * 9)),
        };

        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width * 2, self.height * 2);
        let maps = Maps {
            conv0: &conv0,
            first: &first,
            second: &second,
            third: &third,
            columns: columns.as_ref(),
        };

        let ran = run_region(
            client,
            &sizing,
            &network,
            self.path,
            input_plane,
            PEAK,
            region,
            region,
            maps,
            output_plane,
        );
        synced(client);
        ran?;

        let codes = read_f32(client, &output);
        let values = codes.iter().map(|code| code / PEAK).collect();

        Ok(values)
    }
}

pub(super) fn largest_error(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());

    actual
        .iter()
        .zip(expected)
        .map(|(got, want)| (f64::from(*got) - want).abs())
        .fold(0.0, f64::max)
}

/// Runs `path` at `precision` and gives the output and its largest difference from `expected`.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the run"
)]
fn measure(
    name: &str,
    weights: &Weights,
    plane: &[f32],
    width: u32,
    height: u32,
    path: BodyPath,
    precision: Precision,
    expected: &[f64],
) -> (Vec<f32>, f64) {
    let task = Forward {
        weights: weights.clone(),
        plane: plane.to_vec(),
        width,
        height,
        path,
        precision,
    };

    let ran = backend().build(task);
    let actual = ran.expect("a supported path runs");
    let error = largest_error(&actual, expected);
    let codes = error * 255.0;
    println!("{name} {path:?} at {precision:?}: largest error {error:e}, {codes:.3} 8-bit codes");

    (actual, error)
}

/// The largest difference from the reference on every supported path and precision.
fn largest_errors(
    name: &str,
    weights: &Weights,
    plane: &[f32],
    width: u32,
    height: u32,
) -> Vec<(BodyPath, Precision, f64)> {
    let expected = reference_artcnn(plane, width, height, weights);
    let runs = supported_runs();
    let mut errors = Vec::new();

    for (path, precision) in runs {
        let (_, error) = measure(name, weights, plane, width, height, path, precision, &expected);
        errors.push((path, precision, error));
    }

    errors
}

pub(super) fn r8f64() -> Weights {
    let model = Model::R8F64;
    let builtin = av_toolbox_models::builtin("artcnn", model.file_name());
    let bytes = builtin.expect("R8F64 is built in");
    let parsed = Weights::parse(bytes, model.features(), model.blocks());

    parsed.expect("the builtin weights parse")
}

#[test]
fn a_one_by_one_image_matches_the_reference() {
    let weights = synthetic_weights(8, 2, 1);
    let plane = [0.6];

    let errors = largest_errors("1x1", &weights, &plane, 1, 1);

    for (path, precision, error) in errors {
        let bound = match precision {
            // Measured 6.40e-6 on im2col, so 3.1x headroom. Rounding to 16-bit codes alone gives up to
            // 7.63e-6.
            Precision::F32 => 2e-5,
            // Measured 2.17e-5 on both paths, so 4.6x headroom.
            Precision::F16 => 1e-4,
        };
        assert!(error <= bound, "{path:?} at {precision:?}: {error:e}");
    }
}

#[test]
fn odd_sizes_match_the_reference() {
    let weights = synthetic_weights(8, 2, 2);
    let plane = textured(13, 11);

    let errors = largest_errors("13x11", &weights, &plane, 13, 11);

    for (path, precision, error) in errors {
        let bound = match precision {
            // Measured 7.66e-6 on im2col, of which up to 7.63e-6 is rounding to 16-bit codes, so 2.6x
            // headroom.
            Precision::F32 => 2e-5,
            // Measured 3.43e-4 on both paths, so 2.9x headroom.
            Precision::F16 => 1e-3,
        };
        assert!(error <= bound, "{path:?} at {precision:?}: {error:e}");
    }
}

#[test]
fn sixteen_blocks_keep_every_buffer_live() {
    let weights = synthetic_weights(16, 16, 3);
    let plane = textured(10, 9);

    let errors = largest_errors("16 blocks", &weights, &plane, 10, 9);

    for (path, precision, error) in errors {
        let bound = match precision {
            // Measured 7.55e-6 on im2col, so 2.6x headroom. Rounding to 16-bit codes alone gives up to
            // 7.63e-6.
            Precision::F32 => 2e-5,
            // Measured 2.95e-4 on the convolution path and 3.38e-4 on im2col, so 3x headroom.
            Precision::F16 => 1e-3,
        };
        assert!(error <= bound, "{path:?} at {precision:?}: {error:e}");
    }
}

#[test]
fn a_width_of_96_matches_the_reference() {
    let weights = synthetic_weights(96, 1, 4);
    let plane = textured(12, 10);

    let errors = largest_errors("width 96", &weights, &plane, 12, 10);

    for (path, precision, error) in errors {
        let bound = match precision {
            // Measured 7.89e-6 on im2col, of which up to 7.63e-6 is rounding to 16-bit codes, so 2.5x
            // headroom.
            Precision::F32 => 2e-5,
            // Measured 1.67e-4 on the convolution path and 1.97e-4 on im2col, so 2.5x headroom.
            Precision::F16 => 5e-4,
        };
        assert!(error <= bound, "{path:?} at {precision:?}: {error:e}");
    }
}

#[test]
fn r8f64_matches_the_reference() {
    let weights = r8f64();
    let plane = textured(24, 20);

    let errors = largest_errors("R8F64", &weights, &plane, 24, 20);

    for (path, precision, error) in errors {
        let bound = match precision {
            // Measured 8.19e-6 on im2col, of which up to 7.63e-6 is rounding to 16-bit codes, so 2.4x
            // headroom.
            Precision::F32 => 2e-5,
            // Measured 1.45e-4 on the convolution path and 1.85e-4 on im2col, which is 0.047 of an 8-bit
            // code, so 2.7x headroom.
            Precision::F16 => 5e-4,
        };
        assert!(error <= bound, "{path:?} at {precision:?}: {error:e}");
    }
}

#[test]
fn r8f64_stays_finite_on_flat_extremes() {
    let weights = r8f64();
    let precision = Precision::F16;
    let path = match supports(BodyPath::Convolution, precision) {
        true => BodyPath::Convolution,
        false => BodyPath::Im2col,
    };

    for level in [0.0f32, 1.0] {
        let plane = vec![level; 144];
        let expected = reference_artcnn(&plane, 12, 12, &weights);
        let name = format!("flat {level}");

        let (actual, error) = measure(&name, &weights, &plane, 12, 12, path, precision, &expected);

        for value in &actual {
            assert!(value.is_finite(), "{value} is not finite");
            assert!((0.0..=1.0).contains(value), "{value} lies outside 0..=1");
        }

        // Measured 2.21e-5 on the flat zeros and 2.14e-4 on the flat ones, so 2.3x headroom.
        assert!(error <= 5e-4, "{error:e}");
    }
}

#[test]
fn an_unsupported_convolution_is_refused() {
    let weights = synthetic_weights(8, 1, 6);
    let plane = textured(7, 5);
    let path = BodyPath::Convolution;
    let precision = Precision::F32;
    let supported = supports(path, precision);

    if supported {
        let expected = reference_artcnn(&plane, 7, 5, &weights);

        let (_, error) = measure("refusal", &weights, &plane, 7, 5, path, precision, &expected);

        // Unmeasured, since it was written on a device without an f32 cooperative matrix. The bound is the
        // f32 im2col bound, whose measurements peak at 8.19e-6.
        // Re-measure this bound on a device that offers an f32 cooperative matrix.
        assert!(error <= 2e-5, "{error:e}");
        return;
    }

    let task = Forward {
        weights,
        plane,
        width: 7,
        height: 5,
        path,
        precision,
    };

    let ran = backend().build(task);

    println!("refusal: {ran:?}");
    assert!(matches!(ran, Err(ArtcnnError::Convolution(_))), "{ran:?}");
}
