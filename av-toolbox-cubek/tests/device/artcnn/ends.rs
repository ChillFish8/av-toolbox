use av_toolbox_cubek::artcnn::{
    Activation,
    BodyPath,
    Network,
    Precision,
    Region,
    Weights,
    epilogue,
    first_layer,
    last_layer,
    padded_last_layer,
    padded_last_layer_with_blueprint,
    shuffle,
};
use av_toolbox_cubek::testing::{
    reference_conv3x3,
    reference_depth_to_space,
    reference_silu,
    synthetic_weights,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;
use half::f16;

use super::body::supports;
use super::{BlueprintRuns, forced_eligible, read_maps, synced, textured, upload_maps};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

const PRECISIONS: [Precision; 2] = [Precision::F32, Precision::F16];

/// The relative spacing of f16 values, `2^-10`.
const F16_EPSILON: f64 = 1.0 / 1024.0;

/// The value the output plane is filled with before the last layer runs.
const UNTOUCHED: f32 = -1.0;

/// `values` as the device holds them in `precision`, widened to f64.
pub(crate) fn as_stored(values: &[f32], precision: Precision) -> Vec<f64> {
    match precision {
        Precision::F16 => values
            .iter()
            .map(|value| f16::from_f32(*value).to_f64())
            .collect(),
        Precision::F32 => values.iter().map(|value| f64::from(*value)).collect(),
    }
}

/// `count` values between -`scale` and `scale` from a small LCG.
pub(crate) fn signed_values(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(count);

    for _ in 0..count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
        values.push((unit * 2.0 - 1.0) * scale);
    }

    values
}

/// The samples of `region` from a plane `width` across, divided by `peak`.
fn region_samples(plane: &[f32], width: u32, region: Region, peak: f32) -> Vec<f64> {
    let mut samples = Vec::with_capacity(region.samples() as usize);

    for row in region.y..region.y + region.height {
        for column in region.x..region.x + region.width {
            let value = plane[(row * width + column) as usize];
            samples.push(f64::from(value) / f64::from(peak));
        }
    }

    samples
}

struct FirstLayer {
    weights: Weights,
    plane: Vec<f32>,
    width: u32,
    height: u32,
    region: Region,
    peak: f32,
    precision: Precision,
}

impl BuildTask for FirstLayer {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let network = Network::upload(client, &self.weights, self.precision).expect("the weights are valid");
        let map_count = (self.region.samples() * network.features()) as usize;
        let map_bytes = map_count * self.precision.bytes();

        let input = upload_f32(client, &self.plane);
        let output = empty(client, map_bytes);
        let input_plane = plane(&input, self.width, self.height);

        first_layer(
            client,
            &sizing,
            &network,
            input_plane,
            self.peak,
            self.region,
            &output,
        );
        synced(client);

        read_maps(client, &output, self.precision)
    }
}

/// The second region ends at the plane's right and bottom edges.
#[test]
fn the_first_layer_matches_the_reference_over_offset_regions() {
    let width = 20;
    let height = 15;
    let peak = 255.0f32;
    let weights = synthetic_weights(8, 1, 3);
    let regions = [
        Region {
            x: 3,
            y: 2,
            width: 11,
            height: 9,
        },
        Region {
            x: 12,
            y: 8,
            width: 8,
            height: 7,
        },
    ];
    let plane: Vec<f32> = textured(width, height)
        .into_iter()
        .map(|value| value * peak)
        .collect();

    for region in regions {
        let scaled = region_samples(&plane, width, region, peak);
        let expected = reference_conv3x3(&scaled, region.width, region.height, &weights.layers[0]);

        for precision in PRECISIONS {
            let task = FirstLayer {
                weights: weights.clone(),
                plane: plane.clone(),
                width,
                height,
                region,
                peak,
                precision,
            };

            let actual = backend().build(task);
            assert_eq!(actual.len(), expected.len());

            let mut largest_error = 0.0f64;
            let mut largest_share_of_rounding = 0.0f64;
            for (got, want) in actual.iter().zip(&expected) {
                let error = (f64::from(*got) - want).abs();
                let rounding = want.abs() * F16_EPSILON + 1e-4;
                largest_error = largest_error.max(error);
                largest_share_of_rounding = largest_share_of_rounding.max(error / rounding);
            }

            println!(
                "first layer {precision:?} at ({}, {}): largest error {largest_error:e}, \
                 largest share of the f16 rounding bound {largest_share_of_rounding:.3}",
                region.x, region.y,
            );

            match precision {
                // Measured 1.59e-7 and 1.52e-7 over the two regions, so the bound leaves over 6x headroom.
                Precision::F32 => assert!(largest_error <= 1e-6, "{largest_error:e}"),
                // Measured 0.439 and 0.390 of the bound over the two regions, so the bound leaves over 2x
                // headroom.
                Precision::F16 => assert!(largest_share_of_rounding <= 1.0, "{largest_share_of_rounding}"),
            }
        }
    }
}

struct LastLayer {
    weights: Weights,
    maps: Vec<f32>,
    region: Region,
    core: Region,
    output_width: u32,
    output_height: u32,
    peak: f32,
    precision: Precision,
}

impl BuildTask for LastLayer {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let network = Network::upload(client, &self.weights, self.precision).expect("the weights are valid");
        let output_samples = (self.output_width * self.output_height) as usize;
        let untouched = vec![UNTOUCHED; output_samples];

        let input = upload_maps(client, &self.maps, self.precision);
        let output = upload_f32(client, &untouched);
        let output_plane = plane(&output, self.output_width, self.output_height);

        last_layer(
            client,
            &sizing,
            &network,
            &input,
            self.region,
            self.core,
            self.peak,
            output_plane,
        );
        synced(client);

        read_f32(client, &output)
    }
}

/// The region is offset in the plane, and the core touches its left, right and bottom edges.
#[test]
fn the_last_layer_shuffles_clamps_and_rounds_the_core() {
    let features = 8;
    let peak = 65535.0f32;
    let weights = synthetic_weights(features, 1, 4);
    let last = weights.layers.last().expect("a last layer");
    let plane_width = 12;
    let plane_height = 10;
    let region = Region {
        x: 1,
        y: 1,
        width: 9,
        height: 7,
    };
    let core = Region {
        x: 1,
        y: 3,
        width: 9,
        height: 5,
    };
    let output_width = plane_width * 2;
    let output_height = plane_height * 2;
    let doubled_region_width = region.width * 2;
    let map_count = (region.samples() * features) as usize;
    let maps = signed_values(map_count, 11, 1.0);

    for precision in PRECISIONS {
        let stored = as_stored(&maps, precision);
        let channels = reference_conv3x3(&stored, region.width, region.height, last);
        let doubled = reference_depth_to_space(&channels, region.width, region.height);
        let expected: Vec<f64> = doubled
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0) * f64::from(peak))
            .collect();

        let task = LastLayer {
            weights: weights.clone(),
            maps: maps.clone(),
            region,
            core,
            output_width,
            output_height,
            peak,
            precision,
        };

        let actual = backend().build(task);
        assert_eq!(actual.len(), (output_width * output_height) as usize);

        let mut largest_error = 0.0f64;
        let mut clamped_samples = 0;
        for row in 0..output_height {
            for column in 0..output_width {
                let index = (row * output_width + column) as usize;
                let got = actual[index];
                let inside_core = column >= core.x * 2
                    && row >= core.y * 2
                    && column < (core.x + core.width) * 2
                    && row < (core.y + core.height) * 2;

                if !inside_core {
                    assert_eq!(
                        got, UNTOUCHED,
                        "({column}, {row}) lies outside the core but was written"
                    );
                    continue;
                }

                let region_row = row - region.y * 2;
                let region_column = column - region.x * 2;
                let want = expected[(region_row * doubled_region_width + region_column) as usize];
                assert_eq!(
                    got,
                    got.round(),
                    "({column}, {row}) was {got}, which is not rounded"
                );

                let error = (f64::from(got) - want).abs();
                largest_error = largest_error.max(error);

                if want == 0.0 || want == f64::from(peak) {
                    clamped_samples += 1;
                }
            }
        }

        println!(
            "last layer {precision:?}: largest error {largest_error:e} codes, \
             {clamped_samples} clamped samples"
        );

        // Rounding alone gives up to 0.5 codes. Measured 0.493 for f32 and 0.495 for f16, with 87 of the 180
        // samples clamped, so 0.01 codes is left for the accumulation error.
        assert!(largest_error <= 0.51, "{largest_error:e}");
        assert!(clamped_samples > 0, "no sample of the core reached a clamp");
        assert!(
            clamped_samples < core.samples() * 4,
            "every sample of the core was clamped"
        );
    }
}

struct Epilogue {
    weights: Weights,
    values: Vec<f32>,
    residual: Option<Vec<f32>>,
    layer: usize,
    samples: u32,
    activation: Activation,
    precision: Precision,
}

impl BuildTask for Epilogue {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let precision = self.precision;
        let network = Network::upload(client, &self.weights, precision).expect("the weights are valid");

        let values = upload_maps(client, &self.values, precision);
        let residual = self
            .residual
            .map(|residual| upload_maps(client, &residual, precision));

        epilogue(
            client,
            &sizing,
            &network,
            &values,
            self.layer,
            self.samples,
            self.activation,
            residual.as_ref(),
        );
        synced(client);

        read_maps(client, &values, precision)
    }
}

fn reference_epilogue(
    values: &[f64],
    bias: &[f16],
    residual: Option<&[f64]>,
    activation: Activation,
) -> Vec<f64> {
    let mut output = Vec::with_capacity(values.len());

    for (index, value) in values.iter().enumerate() {
        let channel = index % bias.len();
        let mut result = value + bias[channel].to_f64();

        if activation == Activation::Silu {
            result = reference_silu(result);
        }

        if let Some(residual) = residual {
            result += residual[index];
        }

        output.push(result);
    }

    output
}

#[test]
fn the_epilogue_adds_the_bias_then_silu_then_the_residual() {
    let features = 8;
    let samples = 6;
    let layer = 1;
    let weights = synthetic_weights(features, 1, 5);
    let bias = &weights.layers[layer].bias;
    let count = (samples * features) as usize;
    let values = signed_values(count, 21, 3.0);
    let residual = signed_values(count, 22, 3.0);
    let activations = [Activation::None, Activation::Silu];
    let residuals = [None, Some(residual)];

    for precision in PRECISIONS {
        let stored_values = as_stored(&values, precision);
        let mut largest_error = 0.0f64;
        let mut largest_share_of_rounding = 0.0f64;

        for activation in activations {
            for residual in &residuals {
                let stored_residual = residual.as_ref().map(|residual| as_stored(residual, precision));
                let expected =
                    reference_epilogue(&stored_values, bias, stored_residual.as_deref(), activation);

                let task = Epilogue {
                    weights: weights.clone(),
                    values: values.clone(),
                    residual: residual.clone(),
                    layer,
                    samples,
                    activation,
                    precision,
                };

                let actual = backend().build(task);
                assert_eq!(actual.len(), expected.len());

                for (got, want) in actual.iter().zip(&expected) {
                    let error = (f64::from(*got) - want).abs();
                    let rounding = want.abs() * F16_EPSILON + 1e-6;
                    largest_error = largest_error.max(error);
                    largest_share_of_rounding = largest_share_of_rounding.max(error / rounding);
                }
            }
        }

        println!(
            "epilogue {precision:?}: largest error {largest_error:e}, \
             largest share of the f16 rounding bound {largest_share_of_rounding:.3}"
        );

        match precision {
            // Measured 2.38e-7, so the bound leaves over 4x headroom.
            Precision::F32 => assert!(largest_error <= 1e-6, "{largest_error:e}"),
            // Measured 0.486 of the bound, with an error of 1.30e-3, which is half an f16 step, so the bound
            // leaves 2x headroom.
            Precision::F16 => assert!(largest_share_of_rounding <= 1.0, "{largest_share_of_rounding}"),
        }
    }
}

#[test]
#[should_panic(expected = "does not map the features to themselves")]
fn the_epilogue_refuses_a_layer_that_does_not_map_the_features_to_themselves() {
    let features = 8;
    let samples = 6;
    let weights = synthetic_weights(features, 1, 5);
    let last = weights.layers.len() - 1;
    let values = signed_values((samples * features) as usize, 21, 1.0);
    let task = Epilogue {
        weights,
        values,
        residual: None,
        layer: last,
        samples,
        activation: Activation::None,
        precision: Precision::F32,
    };

    backend().build(task);
}

/// The last layer over the same maps both directly and as the padded convolution and shuffle.
struct PaddedAgainstDirect {
    weights: Weights,
    maps: Vec<f32>,
    region: Region,
    core: Region,
    output_width: u32,
    output_height: u32,
    peak: f32,
}

struct LastLayerOutputs {
    direct: Vec<f32>,
    padded: Vec<f32>,
}

impl BuildTask for PaddedAgainstDirect {
    type Output = LastLayerOutputs;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let precision = Precision::F16;
        let network = Network::upload(client, &self.weights, precision).expect("the weights are valid");
        let output_samples = (self.output_width * self.output_height) as usize;
        let untouched = vec![UNTOUCHED; output_samples];
        let scratch_bytes = self.region.samples() as usize * 16 * size_of::<f32>();

        let input = upload_maps(client, &self.maps, precision);
        let scratch = empty(client, scratch_bytes);
        let direct_output = upload_f32(client, &untouched);
        let padded_output = upload_f32(client, &untouched);
        let direct_plane = plane(&direct_output, self.output_width, self.output_height);
        let padded_plane = plane(&padded_output, self.output_width, self.output_height);

        last_layer(
            client,
            &sizing,
            &network,
            &input,
            self.region,
            self.core,
            self.peak,
            direct_plane,
        );
        let ran = padded_last_layer(client, &network, &input, self.region, &scratch);
        shuffle(
            client,
            &sizing,
            &scratch,
            self.region,
            self.core,
            self.peak,
            padded_plane,
        );
        synced(client);
        ran.expect("the padded last layer runs");

        let direct = read_f32(client, &direct_output);
        let padded = read_f32(client, &padded_output);

        LastLayerOutputs { direct, padded }
    }
}

/// A 200x120 region offset in a 210x130 plane, with a core that touches its left, right and bottom edges.
fn large_region_and_core() -> (Region, Region) {
    let region = Region {
        x: 4,
        y: 3,
        width: 200,
        height: 120,
    };
    let core = Region {
        x: 4,
        y: 10,
        width: 200,
        height: 113,
    };

    (region, core)
}

/// Runs the padded last layer against the direct one at `peak`, and gives how many output samples differ.
///
/// It fails if any sample differs by more than one code, or if fewer than a quarter of the core's samples lie
/// strictly between the clamps.
fn padded_samples_differing(peak: f32) -> usize {
    let (region, core) = large_region_and_core();
    let output_width = 210 * 2;
    let output_height = 130 * 2;
    let map_count = (region.samples() * 64) as usize;
    let task = PaddedAgainstDirect {
        weights: synthetic_weights(64, 1, 12),
        maps: signed_values(map_count, 13, 1.0),
        region,
        core,
        output_width,
        output_height,
        peak,
    };

    let outputs = backend().build(task);
    assert_eq!(outputs.direct.len(), outputs.padded.len());

    let pairs = outputs.direct.iter().zip(&outputs.padded);
    let differing = pairs.clone().filter(|(direct, padded)| direct != padded).count();
    let largest = pairs
        .clone()
        .map(|(direct, padded)| (direct - padded).abs())
        .fold(0.0f32, f32::max);
    let unclamped = outputs
        .direct
        .iter()
        .filter(|code| **code > 0.0 && **code < peak)
        .count();
    let core_samples = (core.samples() * 4) as usize;
    let share = differing as f64 / core_samples as f64 * 100.0;
    println!(
        "padded last layer at peak {peak}: {differing} of {core_samples} samples differ ({share:.4} %), \
         largest {largest} codes, {unclamped} unclamped"
    );

    assert!(largest <= 1.0, "a sample differs by {largest} codes");
    assert!(
        unclamped * 4 >= core_samples,
        "only {unclamped} of {core_samples} samples lie between the clamps"
    );

    differing
}

#[test]
fn the_padded_last_layer_matches_the_direct_one_at_8_bits() {
    if !supports(BodyPath::Convolution, Precision::F16) {
        println!("padded last layer: no f16 cooperative matrix, so there is nothing to compare");
        return;
    }

    let differing = padded_samples_differing(255.0);

    let (_, core) = large_region_and_core();
    let core_samples = (core.samples() * 4) as usize;
    // The two sum in another order, so a sample right on a rounding boundary may move by one code. Measured 2
    // of 90,400 samples, against the 9 the bound allows.
    assert!(
        differing * 10_000 <= core_samples,
        "{differing} of {core_samples} samples differ"
    );
}

#[test]
fn the_padded_last_layer_matches_the_direct_one_at_16_bits() {
    if !supports(BodyPath::Convolution, Precision::F16) {
        println!("padded last layer: no f16 cooperative matrix, so there is nothing to compare");
        return;
    }

    let differing = padded_samples_differing(65535.0);

    let (_, core) = large_region_and_core();
    let core_samples = (core.samples() * 4) as usize;
    // At 16 bits a code is 257 times finer, so more samples sit close enough to a rounding boundary to move.
    // Measured 434 of 90,400 samples, so 2x headroom.
    assert!(
        differing * 100 <= core_samples,
        "{differing} of {core_samples} samples differ"
    );
}

/// The padded last layer over the same maps with the forced blueprint allowed, and with the inferred one.
struct PaddedBlueprints {
    weights: Weights,
    maps: Vec<f32>,
    region: Region,
}

impl BuildTask for PaddedBlueprints {
    type Output = BlueprintRuns;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let eligible = forced_eligible(client);
        let precision = Precision::F16;
        let network = Network::upload(client, &self.weights, precision).expect("the weights are valid");
        let scratch_bytes = self.region.samples() as usize * 16 * size_of::<f32>();

        let input = upload_maps(client, &self.maps, precision);
        let forced_output = empty(client, scratch_bytes);
        let inferred_output = empty(client, scratch_bytes);

        let forced_ran =
            padded_last_layer_with_blueprint(client, &network, &input, self.region, &forced_output, true);
        let inferred_ran =
            padded_last_layer_with_blueprint(client, &network, &input, self.region, &inferred_output, false);
        synced(client);

        let forced_ran = forced_ran.expect("the forced padded last layer runs");
        let inferred_forced = inferred_ran.expect("the inferred padded last layer runs");
        assert!(!inferred_forced, "the inferred run took the forced blueprint");

        let forced = read_f32(client, &forced_output);
        let inferred = read_f32(client, &inferred_output);

        BlueprintRuns {
            forced,
            inferred,
            forced_ran,
            eligible,
        }
    }
}

#[test]
fn the_padded_last_layer_forced_blueprint_matches_the_inferred_one_exactly() {
    if !supports(BodyPath::Convolution, Precision::F16) {
        println!("padded blueprint: no f16 cooperative matrix, so there is nothing to compare");
        return;
    }

    let region = Region {
        x: 0,
        y: 0,
        width: 200,
        height: 120,
    };
    let map_count = (region.samples() * 64) as usize;
    let task = PaddedBlueprints {
        weights: synthetic_weights(64, 1, 12),
        maps: signed_values(map_count, 14, 1.0),
        region,
    };

    let outputs = backend().build(task);

    outputs.assert_forced_matches_inferred("padded blueprint");
}
