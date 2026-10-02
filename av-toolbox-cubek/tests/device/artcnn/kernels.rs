use av_toolbox_cubek::artcnn::{
    self,
    Activation,
    BodyConvolution,
    BodyPath,
    Candidate,
    DirectVariant,
    Finish,
    Kernels,
    Model,
    Network,
    Precision,
    Region,
    TunedRun,
    WeightStaging,
    Weights,
    body_layer_with_blueprint,
    convolve_with_blueprint,
    direct_variants,
    offers_cmma_16,
    run_candidate,
    run_tuned,
};
use av_toolbox_cubek::testing::{reference_artcnn, synthetic_weights};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;
use cubecl::server::Handle;

use super::body::{CandidateForward, Forward, largest_error, r8f64};
use super::{read_maps, steps_from_separate, synced, textured, upload_maps, wavy_maps};
use crate::support::{Sizing, backend, empty};

/// The value the candidate's output holds before it runs, far outside what any layer writes.
pub(super) const UNWRITTEN: f32 = -1000.0;

/// What a body convolution applies as it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Epilogue {
    /// The first convolution of a block, through SiLU.
    Silu,
    /// The last convolution of a block, adding the block's input.
    Residual,
    /// The last body convolution, adding the first layer's maps.
    Join,
}

pub(super) const EPILOGUES: [Epilogue; 3] = [Epilogue::Silu, Epilogue::Residual, Epilogue::Join];

impl Epilogue {
    /// The body layer of `weights` that applies this epilogue in a forward pass.
    pub(super) fn layer(self, weights: &Weights) -> usize {
        match self {
            Self::Silu => 1,
            Self::Residual => 3,
            Self::Join => 3 * weights.blocks as usize + 1,
        }
    }

    /// The epilogue the tuner keys this one by.
    pub(super) fn kind(self) -> artcnn::Epilogue {
        match self {
            Self::Silu => artcnn::Epilogue::Silu,
            Self::Residual => artcnn::Epilogue::Residual,
            Self::Join => artcnn::Epilogue::Join,
        }
    }

    pub(super) fn finish(self, residual: &Handle) -> Finish<'_> {
        match self {
            Self::Silu => Finish {
                activation: Activation::Silu,
                residual: None,
            },
            Self::Residual | Self::Join => Finish {
                activation: Activation::None,
                residual: Some(residual),
            },
        }
    }
}

fn direct() -> Candidate {
    let variant = DirectVariant::default();

    Candidate::Direct(variant)
}

/// What runs a body layer: one candidate, or the candidate some kernels choose.
#[derive(Debug, Clone)]
pub(super) enum Runner {
    One(Candidate),
    Chosen(Kernels),
}

/// One body layer of the same maps run by a runner, through the separate passes, and as its bare
/// convolution and bias. The weights are uploaded as `model` when one is given.
pub(super) struct CandidateLayer {
    pub(super) runner: Runner,
    pub(super) weights: Weights,
    pub(super) model: Option<Model>,
    pub(super) region: Region,
    pub(super) epilogue: Epilogue,
}

pub(super) struct CandidateOutputs {
    pub(super) candidate: Vec<f32>,
    pub(super) separate: Vec<f32>,
    pub(super) convolved: Vec<f32>,
    /// How the chosen candidate ran, when kernels chose it.
    pub(super) chosen: Option<TunedRun>,
}

impl BuildTask for CandidateLayer {
    type Output = Option<CandidateOutputs>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        if !offers_cmma_16(client) {
            return None;
        }

        let sizing = Sizing(build.frame_context());
        let precision = Precision::F16;
        let uploaded = match self.model {
            Some(model) => Network::upload_model(client, model, &self.weights, precision),
            None => Network::upload(client, &self.weights, precision),
        };
        let network = uploaded.expect("the weights are valid");
        let features = network.features();
        let values = (self.region.samples() * features) as usize;
        let maps = edged_maps(self.region, features);
        let residual_maps: Vec<f32> = (0..values)
            .map(|index| (index as f32 * 0.11).cos() * 3.0)
            .collect();
        let unwritten = vec![UNWRITTEN; values];
        let layer = self.epilogue.layer(&self.weights);

        let input = upload_maps(client, &maps, precision);
        let residual = upload_maps(client, &residual_maps, precision);
        let candidate_output = upload_maps(client, &unwritten, precision);
        let separate_output = empty(client, values * precision.bytes());
        let convolved_output = empty(client, values * precision.bytes());

        let finish = self.epilogue.finish(&residual);
        let convolution =
            BodyConvolution::of_layer(&network, layer, &input, &candidate_output, self.region, finish);

        let candidate_ran = match &self.runner {
            Runner::One(candidate) => run_candidate(client, *candidate, &convolution).map(|()| None),
            Runner::Chosen(kernels) => {
                let kind = self.epilogue.kind();

                run_tuned(client, kernels, &convolution, kind).map(Some)
            },
        };
        let separate_ran = body_layer_with_blueprint(
            client,
            &sizing,
            &network,
            BodyPath::UnfusedConvolution,
            layer,
            &input,
            &separate_output,
            self.region,
            finish,
            false,
        );
        let convolved_ran = convolve_with_blueprint(
            client,
            &sizing,
            &network,
            layer,
            &input,
            &convolved_output,
            self.region,
            false,
        );
        synced(client);

        let chosen = candidate_ran.expect("the candidate runs");
        separate_ran.expect("the separate passes run");
        convolved_ran.expect("the bare convolution runs");

        let outputs = CandidateOutputs {
            candidate: read_maps(client, &candidate_output, precision),
            separate: read_maps(client, &separate_output, precision),
            convolved: read_maps(client, &convolved_output, precision),
            chosen,
        };

        Some(outputs)
    }
}

/// Maps of `region` between -2 and 2, varying from each value to the next, raised by 1.5 on the frame's edge.
pub(super) fn edged_maps(region: Region, features: u32) -> Vec<f32> {
    let values = (region.samples() * features) as usize;
    let mut maps = wavy_maps(values);

    for (index, value) in maps.iter_mut().enumerate() {
        let sample = index as u32 / features;
        let column = sample % region.width;
        let row = sample / region.width;
        let on_edge = column == 0 || row == 0 || column + 1 == region.width || row + 1 == region.height;

        if on_edge {
            *value += 1.5;
        }
    }

    maps
}

pub(super) fn region(width: u32, height: u32) -> Region {
    Region {
        x: 0,
        y: 0,
        width,
        height,
    }
}

/// Weights `features` wide. They are R8F64's at 64, and synthetic ones of one block otherwise.
pub(super) fn weights_of(features: u32) -> Weights {
    match features {
        64 => r8f64(),
        _ => synthetic_weights(features, 1, 4),
    }
}

/// Runs `candidate` over a `width` by `height` region of maps `features` wide with `epilogue`, and asserts
/// it lies within one f16 step of the separate passes. Skips without an f16 cooperative matrix.
fn assert_matches_separate(candidate: Candidate, features: u32, width: u32, height: u32, epilogue: Epilogue) {
    let runner = Runner::One(candidate);

    runner_matches_separate(runner, features, width, height, epilogue);
}

/// The same as [assert_matches_separate] with `runner`, giving the outputs, or `None` without an f16
/// cooperative matrix.
pub(super) fn runner_matches_separate(
    runner: Runner,
    features: u32,
    width: u32,
    height: u32,
    epilogue: Epilogue,
) -> Option<CandidateOutputs> {
    let name = format!("{runner:?} {epilogue:?} {width}x{height} at {features}");
    let task = CandidateLayer {
        runner,
        weights: weights_of(features),
        model: None,
        region: region(width, height),
        epilogue,
    };

    let Some(outputs) = backend().build(task) else {
        println!("{name}: no f16 cooperative matrix, so the candidate cannot run");
        return None;
    };

    let candidate_values = &outputs.candidate;
    let separate = &outputs.separate;
    let convolved = &outputs.convolved;
    let (differing, largest_share) = steps_from_separate(candidate_values, separate, convolved);
    let values = candidate_values.len();
    println!("{name}: {differing} of {values} values differ, largest {largest_share} f16 steps");

    assert!(
        !candidate_values.contains(&UNWRITTEN),
        "{name}: the candidate left values unwritten"
    );
    // The candidate rounds to f16 once, after the epilogue, where the separate passes round the convolution
    // first.
    assert!(largest_share <= 1.0, "{name}: {largest_share} f16 steps");

    Some(outputs)
}

#[test]
fn the_direct_candidate_matches_the_separate_passes_at_width_64() {
    for epilogue in EPILOGUES {
        assert_matches_separate(direct(), 64, 200, 120, epilogue);
    }
}

#[test]
fn the_direct_candidate_matches_the_separate_passes_at_width_96() {
    for epilogue in EPILOGUES {
        assert_matches_separate(direct(), 96, 200, 120, epilogue);
    }
}

#[test]
fn the_direct_candidate_matches_the_separate_passes_on_odd_sizes() {
    for features in [64, 96] {
        for (width, height) in [(13, 11), (31, 7)] {
            for epilogue in EPILOGUES {
                assert_matches_separate(direct(), features, width, height, epilogue);
            }
        }
    }
}

#[test]
fn the_direct_candidate_matches_the_separate_passes_on_a_region_smaller_than_its_block() {
    let variant = DirectVariant::default();
    let (width, height) = (5, 3);
    assert!(
        width < variant.columns && height < variant.rows,
        "the region fills the default block"
    );

    for features in [64, 96] {
        for epilogue in EPILOGUES {
            assert_matches_separate(direct(), features, width, height, epilogue);
        }
    }
}

#[test]
fn the_direct_candidate_with_staged_weights_matches_the_separate_passes() {
    for weights in [WeightStaging::Single, WeightStaging::Double] {
        let variant = DirectVariant {
            weights,
            ..DirectVariant::default()
        };
        let candidate = Candidate::Direct(variant);

        for features in [64, 96] {
            for (width, height) in [(200, 120), (31, 7)] {
                for epilogue in EPILOGUES {
                    assert_matches_separate(candidate, features, width, height, epilogue);
                }
            }
        }
    }
}

/// Runs R8F64 over `plane` with every body convolution on the direct candidate, and gives the output and its
/// largest difference from the reference, or `None` without an f16 cooperative matrix.
fn direct_network_error(name: &str, plane: &[f32], width: u32, height: u32) -> Option<(Vec<f32>, f64)> {
    candidate_network_error(name, direct(), plane, width, height)
}

/// The same as [direct_network_error] with every body convolution on `candidate`.
fn candidate_network_error(
    name: &str,
    candidate: Candidate,
    plane: &[f32],
    width: u32,
    height: u32,
) -> Option<(Vec<f32>, f64)> {
    let precision = Precision::F16;
    if !super::body::supports(BodyPath::Convolution, precision) {
        println!("{name}: no f16 cooperative matrix, so the direct candidate cannot run");
        return None;
    }

    let weights = r8f64();
    let expected = reference_artcnn(plane, width, height, &weights);
    let task = CandidateForward {
        forward: Forward {
            weights,
            plane: plane.to_vec(),
            width,
            height,
            path: BodyPath::Convolution,
            precision,
        },
        candidate,
    };

    let ran = backend().build(task);
    let actual = ran.expect("the direct candidate runs");

    let error = largest_error(&actual, &expected);
    let codes = error * 255.0;
    println!("{name}: largest error {error:e}, {codes:.3} 8-bit codes");

    Some((actual, error))
}

#[test]
fn r8f64_on_the_direct_candidate_matches_the_reference() {
    let plane = textured(24, 20);

    let Some((_, error)) = direct_network_error("direct R8F64 24x20", &plane, 24, 20) else {
        return;
    };

    // The bound of the R8F64 24x20 case on every path.
    assert!(error <= 5e-4, "{error:e}");
}

#[test]
fn r8f64_on_the_direct_candidate_matches_the_reference_at_60_by_40() {
    let plane = textured(60, 40);

    let Some((_, error)) = direct_network_error("direct R8F64 60x40", &plane, 60, 40) else {
        return;
    };

    // The bound of the tiled R8F64 60x40 case.
    assert!(error <= 1e-3, "{error:e}");
}

#[test]
fn r8f64_on_the_direct_candidate_stays_finite_on_flat_extremes() {
    for level in [0.0f32, 1.0] {
        let plane = vec![level; 144];
        let name = format!("direct R8F64 flat {level}");

        let Some((actual, error)) = direct_network_error(&name, &plane, 12, 12) else {
            return;
        };

        for value in &actual {
            assert!(value.is_finite(), "{value} is not finite");
            assert!((0.0..=1.0).contains(value), "{value} lies outside 0..=1");
        }

        // The bound of the R8F64 flat cases.
        assert!(error <= 5e-4, "{error:e}");
    }
}

/// The direct variants the test device offers for maps `features` wide.
struct Offered {
    features: u32,
}

impl BuildTask for Offered {
    type Output = Vec<DirectVariant>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let properties = build.client().properties();

        direct_variants(properties, self.features)
    }
}

pub(super) fn offered_variants(features: u32) -> Vec<DirectVariant> {
    let task = Offered { features };

    let variants = backend().build(task);
    println!("offered at width {features}: {variants:?}");

    variants
}

/// Runs every direct variant the device offers for maps `features` wide against the separate passes, on a
/// large region, odd sizes and a region smaller than every block, with each epilogue.
fn assert_every_offered_variant_matches_separate(features: u32) {
    let variants = offered_variants(features);

    for variant in variants {
        let candidate = Candidate::Direct(variant);

        for (width, height) in [(200, 120), (13, 11), (31, 7), (5, 3)] {
            for epilogue in EPILOGUES {
                assert_matches_separate(candidate, features, width, height, epilogue);
            }
        }
    }
}

#[test]
fn every_offered_variant_matches_the_separate_passes_at_width_64() {
    assert_every_offered_variant_matches_separate(64);
}

#[test]
fn every_offered_variant_matches_the_separate_passes_at_width_96() {
    assert_every_offered_variant_matches_separate(96);
}

#[test]
fn r8f64_on_every_offered_variant_matches_the_reference() {
    let plane = textured(24, 20);
    let variants = offered_variants(64);

    for variant in variants {
        let candidate = Candidate::Direct(variant);
        let name = format!("{} R8F64 24x20", candidate.name());

        let Some((actual, error)) = candidate_network_error(&name, candidate, &plane, 24, 20) else {
            return;
        };

        for value in &actual {
            assert!(value.is_finite(), "{name}: {value} is not finite");
        }

        // The bound of the R8F64 24x20 case on every path.
        assert!(error <= 5e-4, "{name}: {error:e}");
    }
}

#[test]
fn a_variant_splitting_the_output_channels_matches_the_separate_passes() {
    let variant = DirectVariant {
        channel_groups: 2,
        planes_n: 1,
        ..DirectVariant::default()
    };
    let candidate = Candidate::Direct(variant);

    for features in [64, 96] {
        for (width, height) in [(200, 120), (31, 7)] {
            for epilogue in EPILOGUES {
                assert_matches_separate(candidate, features, width, height, epilogue);
            }
        }
    }
}
