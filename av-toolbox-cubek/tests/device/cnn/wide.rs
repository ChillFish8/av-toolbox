use av_toolbox_cubek::cnn::{
    Activation,
    BodyConvolution,
    BodyPath,
    Candidate,
    ConvLayer,
    Epilogue,
    Finish,
    Kernels,
    LayerMaps,
    Precision,
    body_layer_with_blueprint,
    offered_candidates,
    offers_cmma_16,
    run_candidate,
    run_layer,
};
use av_toolbox_cubek::testing::{reference_layer, synthetic_layer};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;
use half::f16;

use crate::artcnn::{edged_maps, f16_step, read_maps, region, supported_runs, synced, upload_maps};
use crate::support::{Sizing, backend, empty};

/// The widths of feature maps the layers run at.
const FEATURES: [u32; 4] = [64, 128, 256, 512];

/// What a layer applies after its convolution and bias.
#[derive(Debug, Clone, Copy)]
enum Ending {
    /// The first convolution of a ResBlock.
    Relu,
    /// The second convolution of a ResBlock, adding the block's input.
    Residual,
    /// The last convolution before an up layer or the tail, adding the block's input and the level's skip.
    ResidualSkip,
}

const ENDINGS: [Ending; 3] = [Ending::Relu, Ending::Residual, Ending::ResidualSkip];

impl Ending {
    fn activation(self) -> Activation {
        match self {
            Self::Relu => Activation::Relu,
            Self::Residual | Self::ResidualSkip => Activation::None,
        }
    }

    fn epilogue(self) -> Epilogue {
        match self {
            Self::Relu => Epilogue::Relu,
            Self::Residual => Epilogue::Residual,
            Self::ResidualSkip => Epilogue::ResidualSkip,
        }
    }

    fn adds_residual(self) -> bool {
        matches!(self, Self::Residual | Self::ResidualSkip)
    }

    fn adds_skip(self) -> bool {
        matches!(self, Self::ResidualSkip)
    }
}

/// What runs the layer.
#[derive(Debug, Clone, Copy)]
enum Runner {
    Path(BodyPath),
    Candidate(Candidate),
    /// The fused convolution path with the blueprint `cubek-convolution` infers, bypassing the tuner.
    InferredFused,
}

impl Runner {
    /// Whether the runner rounds the value to the maps' type before it adds the skip.
    fn rounds_before_skip(self) -> bool {
        matches!(self, Self::Candidate(Candidate::FusedCubek) | Self::InferredFused)
    }
}

/// One synthetic layer run over edged maps of a `width` by `height` region.
struct WideLayer {
    features: u32,
    bias: bool,
    ending: Ending,
    precision: Precision,
    width: u32,
    height: u32,
    runner: Runner,
    winograd: bool,
}

struct WideOutputs {
    actual: Vec<f32>,
    expected: Vec<f64>,
    /// The convolution and bias alone, before the activation and the residual.
    convolved: Vec<f64>,
    /// The value before the skip is added, for a runner that rounds it before adding the skip.
    before_skip: Option<Vec<f64>>,
}

/// `values` as the maps' type holds them.
fn as_stored(values: &[f32], precision: Precision) -> Vec<f32> {
    match precision {
        Precision::F16 => values
            .iter()
            .map(|value| f16::from_f32(*value).to_f32())
            .collect(),
        Precision::F32 => values.to_vec(),
    }
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|value| f64::from(*value)).collect()
}

impl BuildTask for WideLayer {
    type Output = WideOutputs;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let frame = region(self.width, self.height);
        let synthetic = synthetic_layer(self.features, self.bias, u64::from(self.features));
        let bias = synthetic.bias.as_deref();
        let layer = ConvLayer::upload(
            client,
            &synthetic.weights,
            bias,
            self.features,
            self.precision,
            self.winograd,
        );

        let values = (frame.samples() * self.features) as usize;
        let edged = edged_maps(frame, self.features);
        let maps = as_stored(&edged, self.precision);
        let waves: Vec<f32> = (0..values).map(|index| (index as f32 * 0.11).cos()).collect();
        let residual_maps = as_stored(&waves, self.precision);
        let ripples: Vec<f32> = (0..values).map(|index| (index as f32 * 0.07).sin()).collect();
        let skip_maps = as_stored(&ripples, self.precision);

        let input = upload_maps(client, &maps, self.precision);
        let residual = upload_maps(client, &residual_maps, self.precision);
        let skip = upload_maps(client, &skip_maps, self.precision);
        let output = empty(client, values * self.precision.bytes());
        let columns = empty(client, 9 * values * self.precision.bytes());
        let finish = Finish {
            activation: self.ending.activation(),
            residual: self.ending.adds_residual().then_some(&residual),
            skip: self.ending.adds_skip().then_some(&skip),
        };

        match self.runner {
            Runner::Path(path) => {
                let layer_maps = LayerMaps {
                    input: &input,
                    output: &output,
                    columns: Some(&columns),
                };
                let ran = run_layer(
                    client,
                    &sizing,
                    &Kernels::Tuned,
                    path,
                    &layer,
                    layer_maps,
                    frame,
                    finish,
                    self.ending.epilogue(),
                );
                ran.expect("the layer runs");
            },
            Runner::Candidate(candidate) => {
                let convolution = BodyConvolution::of_conv(&layer, &input, &output, frame, finish);
                let ran = run_candidate(client, candidate, &convolution);
                ran.expect("the candidate runs");
            },
            Runner::InferredFused => {
                let path = BodyPath::Convolution;
                let ran = body_layer_with_blueprint(
                    client, &sizing, &layer, path, &input, &output, frame, finish, false,
                );
                ran.expect("the layer runs");
            },
        }

        synced(client);

        let actual = read_maps(client, &output, self.precision);
        let input_values = widen(&maps);
        let residual_values = widen(&residual_maps);
        let skip_values = widen(&skip_maps);
        let added_values: Vec<f64> = match self.ending.adds_skip() {
            true => residual_values
                .iter()
                .zip(&skip_values)
                .map(|(residual, skip)| residual + skip)
                .collect(),
            false => residual_values,
        };
        let added = self.ending.adds_residual().then_some(added_values.as_slice());
        let (width, height) = (self.width, self.height);
        let activation = self.ending.activation();
        let expected = reference_layer(&synthetic, &input_values, width, height, activation, added);
        let convolved = reference_layer(&synthetic, &input_values, width, height, Activation::None, None);
        let rounds_before_skip = self.ending.adds_skip() && self.runner.rounds_before_skip();
        let before_skip = rounds_before_skip.then(|| {
            expected
                .iter()
                .zip(&skip_values)
                .map(|(value, skip)| value - skip)
                .collect()
        });

        WideOutputs {
            actual,
            expected,
            convolved,
            before_skip,
        }
    }
}

/// Asserts every value lies within the precision's bound of the reference, and is exactly zero wherever ReLU
/// clips a clearly negative sum.
///
/// f32 maps allow 1e-4. f16 maps allow two f16 steps at the largest of the value, the bare convolution,
/// which the unfused paths round before the epilogue, and the value before the skip where the runner rounds
/// it before adding the skip. They also allow 1e-4 for the f32 accumulation near zero.
/// `layer_steps` adds that many f16 steps of the layer's largest bare convolution to every value's bound.
fn assert_matches_reference(
    name: &str,
    outputs: &WideOutputs,
    precision: Precision,
    ending: Ending,
    layer_steps: f32,
) {
    assert_eq!(outputs.actual.len(), outputs.expected.len(), "{name}");

    let layer_largest = outputs
        .convolved
        .iter()
        .fold(0.0f64, |largest, value| largest.max(value.abs()));
    let layer_step = layer_steps * f16_step(layer_largest as f32);
    let layer_bound = f64::from(layer_step);
    let mut largest = 0.0f64;

    for (index, actual) in outputs.actual.iter().copied().enumerate() {
        let expected = outputs.expected[index];
        let convolved = outputs.convolved[index];
        let before_skip = outputs
            .before_skip
            .as_ref()
            .map_or(0.0, |values| values[index].abs());
        let error = (f64::from(actual) - expected).abs();
        let magnitude = expected.abs().max(convolved.abs()).max(before_skip) as f32;
        let bound = match precision {
            Precision::F32 => 1e-4,
            Precision::F16 => 2.0 * f64::from(f16_step(magnitude)) + 1e-4 + layer_bound,
        };

        assert!(
            error <= bound,
            "{name}: value {index} is {actual}, the reference {expected}, error {error:e} over {bound:e}"
        );

        let clipped = matches!(ending, Ending::Relu) && convolved < -0.01;
        if clipped {
            assert_eq!(actual, 0.0, "{name}: value {index} of a clipped sum");
        }

        largest = largest.max(error);
    }

    println!("{name}: largest error {largest:e}");
}

/// The region sizes each width runs at: an odd size (an even one at 512), and one smaller than any direct block.
fn sizes_for(features: u32) -> [(u32, u32); 2] {
    match features {
        512 => [(12, 10), (5, 3)],
        _ => [(19, 13), (5, 3)],
    }
}

#[test]
fn every_path_matches_the_reference_from_64_to_512_features() {
    let runs = supported_runs();

    for features in FEATURES {
        for (width, height) in sizes_for(features) {
            for bias in [true, false] {
                for ending in ENDINGS {
                    for (path, precision) in runs.iter().copied() {
                        let name = format!(
                            "{path:?} {precision:?} {features} {width}x{height} bias {bias} {ending:?}"
                        );
                        let task = WideLayer {
                            features,
                            bias,
                            ending,
                            precision,
                            width,
                            height,
                            runner: Runner::Path(path),
                            winograd: false,
                        };

                        let outputs = backend().build(task);

                        assert_matches_reference(&name, &outputs, precision, ending, 0.0);
                    }
                }
            }
        }
    }
}

#[test]
fn the_fused_path_with_an_inferred_blueprint_matches_the_reference_from_64_to_512_features() {
    let runs = supported_runs();
    if !runs.contains(&(BodyPath::Convolution, Precision::F16)) {
        println!("no f16 cooperative matrix, so the fused path does not run");
        return;
    }

    for features in FEATURES {
        for (width, height) in sizes_for(features) {
            for bias in [true, false] {
                for ending in ENDINGS {
                    let name = format!("inferred fused {features} {width}x{height} bias {bias} {ending:?}");
                    let task = WideLayer {
                        features,
                        bias,
                        ending,
                        precision: Precision::F16,
                        width,
                        height,
                        runner: Runner::InferredFused,
                        winograd: false,
                    };

                    let outputs = backend().build(task);

                    assert_matches_reference(&name, &outputs, Precision::F16, ending, 0.0);
                }
            }
        }
    }
}

/// The candidates offered for maps `features` wide, or `None` without an f16 cooperative matrix.
struct Offered {
    features: u32,
    winograd: bool,
}

impl BuildTask for Offered {
    type Output = Option<Vec<Candidate>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        if !offers_cmma_16(client) {
            return None;
        }

        let properties = client.properties();
        let offered = offered_candidates(properties, self.features, self.winograd);

        Some(offered)
    }
}

/// How far a Winograd candidate may sit from the reference, in f16 steps of the layer's largest bare
/// convolution.
///
/// Rounding the transformed weights to f16 moves every value by about the same amount, however small the
/// value is, so the bound is about three times the error seen at 64 features.
const WINOGRAD_STEPS: f32 = 2.0;

/// Runs every candidate offered at `features` against the reference at each of its sizes, with and without a
/// bias.
fn assert_every_candidate_matches(features: u32, winograd: bool) {
    let Some(candidates) = backend().build(Offered { features, winograd }) else {
        println!("{features}: no f16 cooperative matrix, so no candidate runs");
        return;
    };

    assert!(
        candidates.contains(&Candidate::FusedCubek),
        "{features}: {candidates:?}"
    );

    for candidate in candidates {
        let layer_steps = match candidate {
            Candidate::Winograd(_) => WINOGRAD_STEPS,
            _ => 0.0,
        };

        for (width, height) in sizes_for(features) {
            for bias in [true, false] {
                for ending in ENDINGS {
                    let name = format!(
                        "{} at {features} {width}x{height} bias {bias} {ending:?}",
                        candidate.name()
                    );
                    let task = WideLayer {
                        features,
                        bias,
                        ending,
                        precision: Precision::F16,
                        width,
                        height,
                        runner: Runner::Candidate(candidate),
                        winograd,
                    };

                    let outputs = backend().build(task);

                    assert_matches_reference(&name, &outputs, Precision::F16, ending, layer_steps);
                }
            }
        }
    }
}

#[test]
fn every_offered_candidate_matches_the_reference_at_128_256_and_512_features() {
    for features in [128, 256, 512] {
        assert_every_candidate_matches(features, false);
    }
}

#[test]
fn every_candidate_of_a_winograd_layer_matches_the_reference_with_relu_and_without_a_bias() {
    assert_every_candidate_matches(64, true);
}

struct SkipIsOutput;

impl BuildTask for SkipIsOutput {
    type Output = ();

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame = region(4, 4);
        let synthetic = synthetic_layer(64, true, 64);
        let layer = ConvLayer::upload(
            client,
            &synthetic.weights,
            synthetic.bias.as_deref(),
            64,
            Precision::F16,
            false,
        );
        let bytes = (frame.samples() * 64) as usize * Precision::F16.bytes();
        let input = empty(client, bytes);
        let output = empty(client, bytes);
        let finish = Finish {
            activation: Activation::None,
            residual: Some(&input),
            skip: Some(&output),
        };
        let convolution = BodyConvolution::of_conv(&layer, &input, &output, frame, finish);
        let candidate = Candidate::FusedCubek;

        let _ = run_candidate(client, candidate, &convolution);
    }
}

#[test]
#[should_panic(expected = "the output is the skip")]
fn a_skip_that_is_the_output_is_refused() {
    backend().build(SkipIsOutput);
}
