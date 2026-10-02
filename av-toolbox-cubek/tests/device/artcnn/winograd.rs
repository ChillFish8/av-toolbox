use av_toolbox_cubek::artcnn::{
    ArtcnnError,
    BodyConvolution,
    BodyPath,
    Candidate,
    Kernels,
    LayerRun,
    Model,
    Network,
    Precision,
    TunedRun,
    Weights,
    WeightsError,
    WinogradVariant,
    offered_candidates,
    offers_cmma_16,
    run_tuned,
    transform_input,
    winograd_variants,
};
use av_toolbox_cubek::testing::{reference_artcnn, synthetic_weights};
use av_toolbox_gpu::{BuildContext, BuildTask};
use av_toolbox_models::Store;
use cubecl::prelude::Runtime;
use half::f16;

use super::body::{Forward, ModelForward, f16_step, largest_error, r8f64, supports};
use super::kernels::{
    CandidateLayer,
    EPILOGUES,
    Epilogue,
    Runner,
    UNWRITTEN,
    edged_maps,
    offers_variants,
    region,
};
use super::{read_maps, steps_from_separate, synced, textured, upload_maps};
use crate::support::{backend, empty};

/// The largest difference a Winograd variant's body layer may have from the separate passes, in f16 steps
/// of the layer's largest value.
///
/// Measured at most 2 steps, on SiLU at width 96, so the bound leaves 2x headroom. Rounding the transformed
/// input and weights to f16 moves each value by up to a step of its own magnitude, where the direct
/// convolution reads the maps and weights as they are.
const LAYER_STEPS: f32 = 4.0;

/// The model a Winograd layer runs as at `features`, and its weights: R8F64's at 64, and synthetic ones of
/// R16F96's shape at 96.
fn model_weights_of(features: u32) -> (Model, Weights) {
    match features {
        64 => (Model::R8F64, r8f64()),
        _ => (Model::R16F96, synthetic_weights(96, 16, 4)),
    }
}

/// The Winograd variants the test device offers for maps `features` wide.
struct Offered {
    features: u32,
}

impl BuildTask for Offered {
    type Output = Vec<WinogradVariant>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let properties = build.client().properties();

        winograd_variants(properties, self.features)
    }
}

/// The Winograd variants the test device offers for maps `features` wide, asserted to be some wherever the
/// device can run them.
fn offered_winograd(features: u32) -> Vec<WinogradVariant> {
    let task = Offered { features };

    let variants = backend().build(task);
    println!("Winograd offered at width {features}: {variants:?}");

    if offers_variants() {
        assert!(
            !variants.is_empty(),
            "no Winograd variant is offered at width {features}"
        );
    }

    variants
}

/// Runs `variant` over a `width` by `height` region of maps `features` wide with `epilogue`, and gives its
/// largest difference from the separate passes in f16 steps of the layer's largest value.
///
/// Asserts the variant ran on its own and left no value unwritten.
fn layer_steps(variant: WinogradVariant, features: u32, width: u32, height: u32, epilogue: Epilogue) -> f32 {
    let candidate = Candidate::Winograd(variant);
    let name = format!("{} {epilogue:?} {width}x{height} at {features}", candidate.name());
    let (model, weights) = model_weights_of(features);
    let task = CandidateLayer {
        runner: Runner::One(candidate),
        weights,
        model: Some(model),
        region: region(width, height),
        epilogue,
    };

    let outputs = backend()
        .build(task)
        .expect("a device offering the variant has the matrix");

    let values = &outputs.candidate;
    let separate = &outputs.separate;
    let (_, local_steps) = steps_from_separate(values, separate, &outputs.convolved);
    let largest = separate
        .iter()
        .fold(0.0f32, |largest, value| largest.max(value.abs()));
    let difference = values
        .iter()
        .zip(separate)
        .fold(0.0f32, |largest, (value, expected)| {
            largest.max((value - expected).abs())
        });
    let steps = difference / f16_step(largest);
    println!(
        "{name}: largest difference {difference:e}, {steps} steps of {largest}, {local_steps} local steps"
    );

    assert!(!values.contains(&UNWRITTEN), "{name}: values left unwritten");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{name}: a value is not finite"
    );

    steps
}

/// Runs every Winograd variant the device offers for maps `features` wide against the separate passes, on a
/// large region, odd sizes and a region smaller than every block, with each epilogue.
fn assert_every_offered_variant_matches_separate(features: u32) {
    let variants = offered_winograd(features);

    for variant in variants {
        for (width, height) in [(200, 120), (13, 11), (31, 7), (5, 3)] {
            for epilogue in EPILOGUES {
                let steps = layer_steps(variant, features, width, height, epilogue);

                assert!(
                    steps <= LAYER_STEPS,
                    "{variant:?} {width}x{height} {epilogue:?}: {steps}"
                );
            }
        }
    }
}

#[test]
fn every_offered_winograd_variant_matches_the_separate_passes_at_width_64() {
    assert_every_offered_variant_matches_separate(64);
}

#[test]
fn every_offered_winograd_variant_matches_the_separate_passes_at_width_96() {
    assert_every_offered_variant_matches_separate(96);
}

/// A plane and the bound its R8F64 and R16F96 outputs keep from the reference on the f16 path.
struct Case {
    name: &'static str,
    plane: Vec<f32>,
    width: u32,
    height: u32,
    bound: f64,
}

/// The planes the network tests run, with their bounds.
fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "textured 24x20",
            plane: textured(24, 20),
            width: 24,
            height: 20,
            bound: 5e-4,
        },
        Case {
            name: "textured 60x40",
            plane: textured(60, 40),
            width: 60,
            height: 40,
            bound: 1e-3,
        },
        Case {
            name: "flat 0",
            plane: vec![0.0; 144],
            width: 12,
            height: 12,
            bound: 5e-4,
        },
        Case {
            name: "flat 1",
            plane: vec![1.0; 144],
            width: 12,
            height: 12,
            bound: 5e-4,
        },
    ]
}

/// Runs `model` over `case` with every body convolution on `candidate`, or as tuned when it is `None`, and
/// gives the output, how each body layer ran and its largest difference from `expected`.
fn model_error(
    model: Model,
    weights: &Weights,
    candidate: Option<Candidate>,
    case: &Case,
    expected: &[f64],
) -> (Vec<f32>, Vec<LayerRun>, f64) {
    let task = ModelForward {
        forward: Forward {
            weights: weights.clone(),
            plane: case.plane.clone(),
            width: case.width,
            height: case.height,
            path: BodyPath::Convolution,
            precision: Precision::F16,
        },
        model,
        candidate,
    };

    let ran = backend().build(task);
    let (actual, runs) = ran.expect("the network runs");

    let error = largest_error(&actual, expected);
    let name = match candidate {
        Some(candidate) => candidate.name(),
        None => "tuned".to_string(),
    };
    println!("{model:?} {name} {}: largest error {error:e}", case.name);

    for value in &actual {
        assert!(value.is_finite(), "{name}: {value} is not finite");
        assert!((0.0..=1.0).contains(value), "{name}: {value} lies outside 0..=1");
    }

    (actual, runs, error)
}

/// Asserts `model` stays within every case's bound with each Winograd variant the device offers, and on the
/// tuned path, whose every body layer must report a candidate offered to the model.
fn assert_model_matches_reference(model: Model, weights: &Weights) {
    let precision = Precision::F16;
    if !supports(BodyPath::Convolution, precision) {
        println!("{model:?}: no f16 cooperative matrix, so no Winograd variant runs");
        return;
    }

    let features = model.features();
    let variants = offered_winograd(features);
    let offered = offered_for(model);

    for case in cases() {
        let expected = reference_artcnn(&case.plane, case.width, case.height, weights);

        for variant in &variants {
            let candidate = Candidate::Winograd(*variant);
            let (_, runs, error) = model_error(model, weights, Some(candidate), &case, &expected);

            assert!(
                runs.iter().all(|run| run.candidate == Some(candidate)),
                "{runs:?}"
            );
            assert!(
                error <= case.bound,
                "{} {}: {error:e}",
                candidate.name(),
                case.name
            );
        }

        let (_, runs, error) = model_error(model, weights, None, &case, &expected);

        for run in &runs {
            let candidate = run.candidate.expect("a body layer took the tuned path");
            assert!(offered.contains(&candidate), "{candidate:?} was not offered");
        }

        assert!(error <= case.bound, "tuned {}: {error:e}", case.name);
    }
}

/// The candidates the test device offers `model`.
struct OfferedForModel {
    model: Model,
}

impl BuildTask for OfferedForModel {
    type Output = Vec<Candidate>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let properties = build.client().properties();
        let features = self.model.features();

        offered_candidates(properties, features, Some(self.model))
    }
}

fn offered_for(model: Model) -> Vec<Candidate> {
    let task = OfferedForModel { model };

    backend().build(task)
}

#[test]
fn r8f64_on_every_winograd_variant_and_the_tuned_path_matches_the_reference() {
    let weights = r8f64();

    assert_model_matches_reference(Model::R8F64, &weights);
}

/// The installed R16F96 weights, or `None` with the reason printed when they are not installed.
fn installed_r16f96() -> Option<Weights> {
    let model = Model::R16F96;
    let store = Store::from_environment();

    let located = match store.find("artcnn", model.file_name()) {
        Ok(located) => located,
        Err(missing) => {
            println!("R16F96 is not installed, so its network is not checked: {missing}");
            return None;
        },
    };

    let bytes = located.read().expect("the installed weights read");
    let parsed = Weights::parse(&bytes, model.features(), model.blocks());

    Some(parsed.expect("the installed weights parse"))
}

#[test]
fn r16f96_on_every_winograd_variant_and_the_tuned_path_matches_the_reference() {
    let Some(weights) = installed_r16f96() else {
        return;
    };

    assert_model_matches_reference(Model::R16F96, &weights);
}

#[test]
fn the_tuner_may_choose_winograd_for_r8f64_and_reports_the_winner() {
    let precision = Precision::F16;
    if !supports(BodyPath::Convolution, precision) {
        println!("no f16 cooperative matrix, so there is nothing to tune");
        return;
    }

    let model = Model::R8F64;
    let weights = r8f64();
    let offered = offered_for(model);
    let winograd_offered = offered
        .iter()
        .any(|candidate| matches!(candidate, Candidate::Winograd(_)));
    let variants = offered_winograd(model.features());
    assert_eq!(winograd_offered, !variants.is_empty(), "{offered:?}");

    let task = ModelForward {
        forward: Forward {
            weights,
            plane: textured(200, 120),
            width: 200,
            height: 120,
            path: BodyPath::Convolution,
            precision,
        },
        model,
        candidate: None,
    };

    let ran = backend().build(task);
    let (_, runs) = ran.expect("the tuned network runs");

    assert_eq!(runs.len(), 25);
    for run in &runs {
        let candidate = run.candidate.expect("a body layer took the tuned path");
        println!("chose {}", candidate.name());

        assert!(offered.contains(&candidate), "{candidate:?} was not offered");
        assert!(run.fused, "{run:?}");
        assert_eq!(run.winograd_offered, winograd_offered, "{run:?}");
    }
}

/// Whether body layer 1 of a network holds Winograd weights, and the model it reports, uploaded as `model`
/// if given.
struct HoldsTransformed {
    weights: Weights,
    model: Option<Model>,
}

impl BuildTask for HoldsTransformed {
    type Output = Result<(bool, Option<Model>), WeightsError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let precision = Precision::F16;
        let network = match self.model {
            Some(model) => Network::upload_model(client, model, &self.weights, precision)?,
            None => Network::upload(client, &self.weights, precision)?,
        };
        let frame = region(4, 4);
        let maps = empty(client, 16 * network.features() as usize * precision.bytes());
        let output = empty(client, 16 * network.features() as usize * precision.bytes());
        let finish = Epilogue::Silu.finish(&maps);
        let convolution = BodyConvolution::of_layer(&network, 1, &maps, &output, frame, finish);

        Ok((convolution.transformed.is_some(), convolution.model))
    }
}

fn holds_transformed(weights: Weights, model: Option<Model>) -> Result<(bool, Option<Model>), WeightsError> {
    let task = HoldsTransformed { weights, model };

    backend().build(task)
}

#[test]
fn only_a_network_uploaded_as_an_allow_listed_model_holds_winograd_weights() {
    let as_model = holds_transformed(r8f64(), Some(Model::R8F64));
    let as_weights = holds_transformed(r8f64(), None);
    let synthetic = holds_transformed(synthetic_weights(96, 16, 4), Some(Model::R16F96));

    assert_eq!(as_model, Ok((true, Some(Model::R8F64))));
    assert_eq!(as_weights, Ok((false, None)));
    assert_eq!(synthetic, Ok((true, Some(Model::R16F96))));
}

#[test]
fn weights_of_another_shape_are_refused_as_a_model() {
    let refused = holds_transformed(synthetic_weights(64, 1, 4), Some(Model::R8F64));

    let expected = WeightsError::ModelShape {
        found: [64, 1],
        expected: [64, 8],
    };
    assert_eq!(refused, Err(expected));
}

/// Lavin and Gray's F(2x2, 3x3) input transform, `Bᵀ`.
const INPUT_TRANSFORM: [[f64; 4]; 4] = [
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0],
];

/// The f16 maps of a `width` by `height` region `features` wide, and their windows as the device transforms
/// them.
struct TransformedInput {
    width: u32,
    height: u32,
    features: u32,
}

impl BuildTask for TransformedInput {
    type Output = Option<(Vec<f32>, Vec<f32>)>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        if !offers_cmma_16(client) {
            return None;
        }

        let precision = Precision::F16;
        let frame = region(self.width, self.height);
        let maps = edged_maps(frame, self.features);

        let input = upload_maps(client, &maps, precision);
        let transformed = transform_input(client, &input, self.width, self.height, self.features);
        synced(client);

        let stored = read_maps(client, &input, precision);
        let windows = read_maps(client, &transformed, precision);

        Some((stored, windows))
    }
}

/// `Bᵀ d B` of every 4x4 window of `maps` around a 2x2 tile, in f64, by position, tile and channel.
fn reference_transform(maps: &[f32], width: u32, height: u32, features: u32) -> Vec<f64> {
    let columns = width.div_ceil(2) as usize;
    let rows = height.div_ceil(2) as usize;
    let channels = features as usize;
    let tiles = rows * columns;
    let mut transformed = vec![0.0; 16 * tiles * channels];

    for tile in 0..tiles {
        for channel in 0..channels {
            let window = window_of(maps, width, height, features, tile, channel);

            for position in 0..16 {
                let position_row = position / 4;
                let position_column = position % 4;
                let mut sum = 0.0;

                for (window_row, samples) in window.iter().enumerate() {
                    for (window_column, sample) in samples.iter().enumerate() {
                        let left = INPUT_TRANSFORM[position_row][window_row];
                        let right = INPUT_TRANSFORM[position_column][window_column];
                        sum += left * sample * right;
                    }
                }

                transformed[(position * tiles + tile) * channels + channel] = sum;
            }
        }
    }

    transformed
}

/// The 4x4 window of `channel` around tile `tile` of `maps`, zero outside the region.
fn window_of(
    maps: &[f32],
    width: u32,
    height: u32,
    features: u32,
    tile: usize,
    channel: usize,
) -> [[f64; 4]; 4] {
    let columns = width.div_ceil(2) as usize;
    let first_row = (tile / columns * 2) as i64 - 1;
    let first_column = (tile % columns * 2) as i64 - 1;
    let mut window = [[0.0; 4]; 4];

    for (window_row, samples) in window.iter_mut().enumerate() {
        for (window_column, sample) in samples.iter_mut().enumerate() {
            let row = first_row + window_row as i64;
            let column = first_column + window_column as i64;
            let inside = row >= 0 && column >= 0 && row < i64::from(height) && column < i64::from(width);

            if inside {
                let index = (row as usize * width as usize + column as usize) * features as usize + channel;
                *sample = f64::from(maps[index]);
            }
        }
    }

    window
}

#[test]
fn the_input_transform_rounds_each_position_to_f16_once() {
    for (width, height, features) in [(13, 11, 64), (6, 4, 96)] {
        let task = TransformedInput {
            width,
            height,
            features,
        };

        let Some((stored, windows)) = backend().build(task) else {
            println!("no f16 cooperative matrix, so no Winograd variant runs");
            return;
        };

        let expected = reference_transform(&stored, width, height, features);
        let mut largest_share = 0.0f64;

        for (value, exact) in windows.iter().zip(&expected) {
            let rounded = f16::from_f64(*exact).to_f64();
            let step = f64::from(f16_step(rounded as f32));
            let share = (f64::from(*value) - exact).abs() / step;

            largest_share = largest_share.max(share);
        }

        println!(
            "{width}x{height} at {features}: largest {largest_share} f16 steps from the exact transform"
        );
        // Rounding the exact value to f16 once is at most half a step off. The f32 sums add at most a
        // rounding of their own, far below a step.
        assert!(
            largest_share <= 0.5 + 1e-3,
            "{width}x{height} at {features}: {largest_share}"
        );
    }
}

/// One tuned body convolution of R8F64 uploaded as its model, then of the same weights uploaded without a
/// model, in one process so the second sees the plans the first made.
struct TunedTwice;

impl BuildTask for TunedTwice {
    type Output = Option<(TunedRun, Result<TunedRun, ArtcnnError>)>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        if !offers_cmma_16(client) {
            return None;
        }

        let precision = Precision::F16;
        let weights = r8f64();
        let as_model = Network::upload_model(client, Model::R8F64, &weights, precision);
        let as_model = as_model.expect("the weights are R8F64's");
        let without_model = Network::upload(client, &weights, precision).expect("the weights are valid");
        let frame = region(24, 20);
        let maps = edged_maps(frame, 64);
        let epilogue = Epilogue::Silu;
        let layer = epilogue.layer(&weights);

        let input = upload_maps(client, &maps, precision);
        let output = empty(client, maps.len() * precision.bytes());
        // SiLU adds no residual, so the input only fills its place.
        let finish = epilogue.finish(&input);
        let first = BodyConvolution::of_layer(&as_model, layer, &input, &output, frame, finish);
        let second = BodyConvolution::of_layer(&without_model, layer, &input, &output, frame, finish);

        let first_run = run_tuned(client, &Kernels::Tuned, &first, epilogue.kind());
        let first_run = first_run.expect("R8F64 tunes");
        let second_run = run_tuned(client, &Kernels::Tuned, &second, epilogue.kind());
        synced(client);

        Some((first_run, second_run))
    }
}

#[test]
fn a_network_of_no_model_is_not_offered_winograd_after_an_allow_listed_one() {
    let variants = offered_winograd(64);

    let Some((first, second)) = backend().build(TunedTwice) else {
        println!("no f16 cooperative matrix, so there is nothing to tune");
        return;
    };

    println!("as R8F64 {first:?}, without a model {second:?}");
    let second = second.expect("a network of no model tunes without Winograd weights");
    let second_candidate = second.run.candidate.expect("the tuner reports its candidate");

    assert_eq!(first.run.winograd_offered, !variants.is_empty());
    assert!(!second.run.winograd_offered, "{second:?}");
    assert!(
        !matches!(second_candidate, Candidate::Winograd(_)),
        "{second_candidate:?}"
    );
}
