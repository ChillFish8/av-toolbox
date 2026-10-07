use av_toolbox_cubek::artcnn::{
    Activation,
    BodyPath,
    Candidate,
    CnnError,
    Finish,
    LayerRun,
    Maps,
    Model,
    Network,
    Precision,
    Region,
    Weights,
    body_layer_with_blueprint,
    convolve_with_blueprint,
    convolve_without_bias,
    epilogue,
    run_region,
    run_region_with_candidate,
    scratch_sizes,
};
use av_toolbox_cubek::testing::{reference_artcnn, synthetic_weights};
use av_toolbox_gpu::{BuildContext, BuildTask, ElementType};
use cubecl::prelude::Runtime;
use half::f16;

use super::{BlueprintRuns, forced_eligible, read_maps, synced, textured, upload_maps, wavy_maps};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

const PATHS: [BodyPath; 3] = [
    BodyPath::Convolution,
    BodyPath::UnfusedConvolution,
    BodyPath::Im2col,
];

const PRECISIONS: [Precision; 2] = [Precision::F32, Precision::F16];

pub(super) const PEAK: f32 = 65535.0;

/// The code the output is filled with before the pass, which is -1 once divided by the peak.
pub(super) const UNTOUCHED: f32 = -PEAK;

struct Supported {
    path: BodyPath,
    precision: Precision,
}

impl BuildTask for Supported {
    type Output = bool;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let capabilities = build.capabilities();
        let operands = match self.precision {
            Precision::F16 => ElementType::F16,
            Precision::F32 => ElementType::F32,
        };

        match self.path {
            BodyPath::Convolution | BodyPath::UnfusedConvolution => {
                capabilities.has_cooperative_matrix(operands, ElementType::F32)
            },
            BodyPath::Im2col => true,
        }
    }
}

pub(crate) fn supports(path: BodyPath, precision: Precision) -> bool {
    let task = Supported { path, precision };

    backend().build(task)
}

pub(crate) fn supported_runs() -> Vec<(BodyPath, Precision)> {
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
pub(super) struct Forward {
    pub(super) weights: Weights,
    pub(super) plane: Vec<f32>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) path: BodyPath,
    pub(super) precision: Precision,
}

impl BuildTask for Forward {
    type Output = Result<Vec<f32>, CnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        run_forward(self, build, None, None)
    }
}

/// The same as [Forward] on the convolution path, with every body convolution run by `candidate`.
pub(super) struct CandidateForward {
    pub(super) forward: Forward,
    pub(super) candidate: Candidate,
}

impl BuildTask for CandidateForward {
    type Output = Result<Vec<f32>, CnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        run_forward(self.forward, build, None, Some(self.candidate))
    }
}

/// The same as [Forward] with the weights uploaded as `model`, and every body convolution run by `candidate`
/// if given. Gives the output and how each body layer ran.
pub(super) struct ModelForward {
    pub(super) forward: Forward,
    pub(super) model: Model,
    pub(super) candidate: Option<Candidate>,
}

impl BuildTask for ModelForward {
    type Output = Result<(Vec<f32>, Vec<LayerRun>), CnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let model = Some(self.model);

        run_forward_as(self.forward, build, None, self.candidate, model)
    }
}

/// The same as [Forward], with `conv0` holding `conv0_bytes` rather than what the scratch sizes ask for.
struct ForwardWithConv0 {
    forward: Forward,
    conv0_bytes: usize,
}

impl BuildTask for ForwardWithConv0 {
    type Output = Result<Vec<f32>, CnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        run_forward(self.forward, build, Some(self.conv0_bytes), None)
    }
}

/// Runs `task`, with `conv0` holding `conv0_bytes` if given and otherwise what the scratch sizes ask for, and
/// every body convolution run by `candidate` if given.
fn run_forward<R: Runtime>(
    task: Forward,
    build: &BuildContext<R>,
    conv0_bytes: Option<usize>,
    candidate: Option<Candidate>,
) -> Result<Vec<f32>, CnnError> {
    let (values, _) = run_forward_as(task, build, conv0_bytes, candidate, None)?;

    Ok(values)
}

/// The same as [run_forward] with the weights uploaded as `model` if given, giving how each body layer ran
/// beside the output.
fn run_forward_as<R: Runtime>(
    task: Forward,
    build: &BuildContext<R>,
    conv0_bytes: Option<usize>,
    candidate: Option<Candidate>,
    model: Option<Model>,
) -> Result<(Vec<f32>, Vec<LayerRun>), CnnError> {
    let client = build.client();
    let frame_context = build.frame_context();
    let sizing = Sizing(frame_context);
    let uploaded = match model {
        Some(model) => Network::upload_model(client, model, &task.weights, task.precision),
        None => Network::upload(client, &task.weights, task.precision),
    };
    let network = uploaded.expect("the weights are valid");
    let region = Region {
        x: 0,
        y: 0,
        width: task.width,
        height: task.height,
    };
    let map_count = (region.samples() * network.features()) as usize;
    let map_bytes = map_count * task.precision.bytes();
    let side = task.width.max(task.height);
    let sizes = scratch_sizes(&network, task.path, side, task.width, task.height);
    let conv0_size = conv0_bytes.unwrap_or(sizes.conv0_bytes);
    let output_samples = (region.samples() * 4) as usize;
    let untouched = vec![UNTOUCHED; output_samples];
    let scaled: Vec<f32> = task.plane.iter().map(|value| value * PEAK).collect();

    let input = upload_f32(client, &scaled);
    let output = upload_f32(client, &untouched);
    let conv0 = empty(client, conv0_size);
    let first = empty(client, map_bytes);
    let second = empty(client, map_bytes);
    let third = empty(client, map_bytes);
    let columns = match task.path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => None,
        BodyPath::Im2col => Some(empty(client, map_bytes * 9)),
    };

    let input_plane = plane(&input, task.width, task.height);
    let output_plane = plane(&output, task.width * 2, task.height * 2);
    let maps = Maps {
        conv0: &conv0,
        first: &first,
        second: &second,
        third: &third,
        columns: columns.as_ref(),
    };

    let ran = match candidate {
        Some(candidate) => run_region_with_candidate(
            client,
            &sizing,
            &network,
            candidate,
            input_plane,
            PEAK,
            region,
            region,
            maps,
            output_plane,
        ),
        None => run_region(
            client,
            &sizing,
            &network,
            task.path,
            input_plane,
            PEAK,
            region,
            region,
            maps,
            output_plane,
        ),
    };
    synced(client);
    let runs = ran?;

    let codes = read_f32(client, &output);
    let values = codes.iter().map(|code| code / PEAK).collect();

    Ok((values, runs))
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
            // Measured 3.14e-4 on the convolution path and 3.43e-4 on im2col, so 2.9x headroom.
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
            // Measured 4.03e-4 on the convolution path and 3.38e-4 on im2col, so 2.5x headroom.
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
            // Measured 2.10e-4 on the convolution path and 1.97e-4 on im2col, so 2.4x headroom.
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
            // Measured 2.55e-4 on the convolution path, 2.10e-4 on the unfused convolution path and 1.85e-4
            // on im2col. The largest is 0.065 of an 8-bit code, so 1.96x headroom.
            Precision::F16 => 5e-4,
        };
        assert!(error <= bound, "{path:?} at {precision:?}: {error:e}");
    }
}

/// Runs R8F64's first body layer over a small region on `path`, as a forward pass does.
struct OneLayer {
    path: BodyPath,
}

impl BuildTask for OneLayer {
    type Output = LayerRun;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let precision = Precision::F16;
        let network = Network::upload(client, &r8f64(), precision).expect("the weights are valid");
        let region = Region {
            x: 0,
            y: 0,
            width: 24,
            height: 20,
        };
        let values = (region.samples() * network.features()) as usize;
        let maps = wavy_maps(values);
        let finish = Finish {
            activation: Activation::Silu,
            residual: None,
        };

        let input = upload_maps(client, &maps, precision);
        let output = empty(client, values * precision.bytes());

        let ran = body_layer_with_blueprint(
            client, &sizing, &network, self.path, 1, &input, &output, region, finish, true,
        );
        synced(client);

        ran.expect("a supported path runs")
    }
}

#[test]
fn only_the_convolution_path_fuses_its_epilogues() {
    let precision = Precision::F16;
    if !supports(BodyPath::Convolution, precision) {
        println!("fused epilogues: no f16 cooperative matrix, so the convolution paths cannot run");
        return;
    }

    let weights = r8f64();
    let plane = textured(24, 20);
    let expected = reference_artcnn(&plane, 24, 20, &weights);
    let unfused_path = BodyPath::UnfusedConvolution;

    let (_, error) = measure(
        "R8F64",
        &weights,
        &plane,
        24,
        20,
        unfused_path,
        precision,
        &expected,
    );
    let fused_run = backend().build(OneLayer {
        path: BodyPath::Convolution,
    });
    let unfused_run = backend().build(OneLayer { path: unfused_path });

    println!("fused epilogues: convolution {fused_run:?}, unfused convolution {unfused_run:?}");
    // Measured 2.10e-4, so 2.4x headroom.
    assert!(error <= 5e-4, "{error:e}");
    assert!(
        fused_run.fused,
        "the convolution path ran its epilogues in a separate pass"
    );
    assert!(
        !unfused_run.fused,
        "the unfused convolution path fused its epilogues"
    );
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

        // Measured 2.36e-5 on the flat zeros and 1.94e-4 on the flat ones, so 2.6x headroom.
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
    assert!(matches!(ran, Err(CnnError::Convolution(_))), "{ran:?}");
}

/// One body convolution of the same maps with the forced blueprint allowed, and with the inferred one.
struct Blueprints {
    weights: Weights,
    maps: Vec<f32>,
    region: Region,
}

impl BuildTask for Blueprints {
    type Output = BlueprintRuns;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let eligible = forced_eligible(client);
        let precision = Precision::F16;
        let network = Network::upload(client, &self.weights, precision).expect("the weights are valid");
        let map_bytes = self.maps.len() * precision.bytes();

        let input = upload_maps(client, &self.maps, precision);
        let forced_output = empty(client, map_bytes);
        let inferred_output = empty(client, map_bytes);

        let forced_ran = convolve_with_blueprint(
            client,
            &sizing,
            &network,
            1,
            &input,
            &forced_output,
            self.region,
            true,
        );
        let inferred_ran = convolve_with_blueprint(
            client,
            &sizing,
            &network,
            1,
            &input,
            &inferred_output,
            self.region,
            false,
        );
        synced(client);

        let forced_ran = forced_ran.expect("the forced convolution runs");
        let inferred_forced = inferred_ran.expect("the inferred convolution runs");
        assert!(!inferred_forced, "the inferred run took the forced blueprint");

        let forced = read_maps(client, &forced_output, precision);
        let inferred = read_maps(client, &inferred_output, precision);

        BlueprintRuns {
            forced,
            inferred,
            forced_ran,
            eligible,
        }
    }
}

#[test]
fn the_forced_blueprint_matches_the_inferred_one_exactly() {
    let precision = Precision::F16;
    if !supports(BodyPath::Convolution, precision) {
        println!("forced blueprint: no f16 cooperative matrix, so there is nothing to compare");
        return;
    }

    let region = Region {
        x: 0,
        y: 0,
        width: 200,
        height: 120,
    };
    let values = (region.samples() * 64) as usize;
    let task = Blueprints {
        weights: synthetic_weights(64, 1, 5),
        maps: wavy_maps(values),
        region,
    };

    let outputs = backend().build(task);

    outputs.assert_forced_matches_inferred("forced blueprint");
}

/// One body convolution of the same maps with the bias inside it, and without it before a separate bias add.
struct BiasPlacement {
    weights: Weights,
    maps: Vec<f32>,
    region: Region,
}

struct BiasOutputs {
    fused: Vec<f32>,
    unbiased: Vec<f32>,
    separate: Vec<f32>,
}

impl BuildTask for BiasPlacement {
    type Output = BiasOutputs;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let precision = Precision::F16;
        let network = Network::upload(client, &self.weights, precision).expect("the weights are valid");
        let map_bytes = self.maps.len() * precision.bytes();
        let samples = self.region.samples();

        let input = upload_maps(client, &self.maps, precision);
        let fused_output = empty(client, map_bytes);
        let separate_output = empty(client, map_bytes);

        let fused_ran = convolve_with_blueprint(
            client,
            &sizing,
            &network,
            1,
            &input,
            &fused_output,
            self.region,
            true,
        );
        let unbiased_ran = convolve_without_bias(client, &network, 1, &input, &separate_output, self.region);
        synced(client);

        fused_ran.expect("the fused convolution runs");
        unbiased_ran.expect("the unbiased convolution runs");
        let unbiased = read_maps(client, &separate_output, precision);

        epilogue(
            client,
            &sizing,
            &network,
            &separate_output,
            1,
            samples,
            Activation::None,
            None,
        );
        synced(client);

        let fused = read_maps(client, &fused_output, precision);
        let separate = read_maps(client, &separate_output, precision);

        BiasOutputs {
            fused,
            unbiased,
            separate,
        }
    }
}

/// The spacing of f16 values at `magnitude`.
pub(crate) fn f16_step(magnitude: f32) -> f32 {
    let bits = f16::from_f32(magnitude.abs()).to_bits();
    let exponent = i32::from(bits >> 10);

    match exponent {
        0 => 2.0f32.powi(-24),
        _ => 2.0f32.powi(exponent - 25),
    }
}

#[test]
fn the_fused_bias_matches_a_separate_bias_add_within_one_f16_step() {
    let precision = Precision::F16;
    if !supports(BodyPath::Convolution, precision) {
        println!("fused bias: no f16 cooperative matrix, so there is nothing to compare");
        return;
    }

    let region = Region {
        x: 0,
        y: 0,
        width: 200,
        height: 120,
    };
    let values = (region.samples() * 64) as usize;
    let task = BiasPlacement {
        weights: synthetic_weights(64, 1, 5),
        maps: wavy_maps(values),
        region,
    };

    let outputs = backend().build(task);

    let mut differing = 0;
    let mut largest_share = 0.0f32;
    for index in 0..values {
        let fused = outputs.fused[index];
        let separate = outputs.separate[index];
        let unbiased = outputs.unbiased[index];
        // The separate add rounds the convolution to f16 before adding the bias, so the step is taken at the
        // largest of the fused, separate and unbiased values.
        let magnitude = fused.abs().max(separate.abs()).max(unbiased.abs());
        let step = f16_step(magnitude);
        let share = (fused - separate).abs() / step;

        largest_share = largest_share.max(share);
        if fused != separate {
            differing += 1;
        }
    }

    println!("fused bias: {differing} of {values} values differ, largest {largest_share} f16 steps");
    // Measured 398,720 values differing, by at most one step, from the separate add's extra rounding.
    assert!(largest_share <= 1.0, "{largest_share} f16 steps");
}

#[test]
fn a_network_narrower_than_32_runs_with_conv0_widened() {
    let weights = synthetic_weights(8, 1, 9);
    let plane = textured(11, 9);
    let expected = reference_artcnn(&plane, 11, 9, &weights);
    let precision = Precision::F16;
    let path = BodyPath::Convolution;
    if !supports(path, precision) {
        println!("narrow width: no f16 cooperative matrix, so the convolution path cannot run");
        return;
    }

    // Eight f16 features are 16 bytes a sample, so `conv0` alone holds the padded last layer's 64.
    let task = ForwardWithConv0 {
        forward: Forward {
            weights,
            plane,
            width: 11,
            height: 9,
            path,
            precision,
        },
        conv0_bytes: 11 * 9 * 64,
    };

    let ran = backend().build(task);
    let actual = ran.expect("a supported path runs");

    let error = largest_error(&actual, &expected);
    println!("narrow width: largest error {error:e}");
    // The same bound as the other 8 wide f16 runs, whose measurements peak at 3.43e-4.
    assert!(error <= 1e-3, "{error:e}");
}

#[test]
#[should_panic(expected = "conv0 is smaller than the region's last layer needs")]
fn a_conv0_of_one_map_is_refused_on_a_narrow_convolution_path() {
    let weights = synthetic_weights(8, 1, 9);
    let plane = textured(11, 9);
    let task = ForwardWithConv0 {
        forward: Forward {
            weights,
            plane,
            width: 11,
            height: 9,
            path: BodyPath::Convolution,
            precision: Precision::F16,
        },
        conv0_bytes: 11 * 9 * 8 * 2,
    };

    let ran = backend().build(task);

    println!("narrow conv0: {ran:?}");
}
