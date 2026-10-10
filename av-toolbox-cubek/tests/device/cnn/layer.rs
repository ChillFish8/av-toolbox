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
    LayerRun,
    PaddedLastLayer,
    Precision,
    Region,
    offers_cmma_16,
    run_layer,
    run_tuned,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;
use half::f16;

use crate::artcnn::{read_maps, synced, upload_maps, wavy_maps};
use crate::support::{Sizing, backend, empty};

/// Weights `features` wide whose values are exact in f16, by output channel, row, column and input channel.
fn exact_weights(features: u32) -> Vec<f32> {
    let count = (features * 9 * features) as usize;

    (0..count)
        .map(|index| f16::from_f32((index % 13) as f32 * 0.125 - 0.75).to_f32())
        .collect()
}

fn exact_bias(features: u32) -> Vec<f32> {
    (0..features).map(|channel| channel as f32 * 0.25).collect()
}

/// Uploads a layer of [exact_weights] and reads its weights and bias back.
struct Uploaded {
    features: u32,
    bias: bool,
    precision: Precision,
    winograd: bool,
}

struct ReadBack {
    weights: Vec<f32>,
    bias: Option<Vec<f32>>,
    transformed: bool,
    winograd: bool,
}

impl BuildTask for Uploaded {
    type Output = ReadBack;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let weights = exact_weights(self.features);
        let bias_values = exact_bias(self.features);
        let bias = self.bias.then_some(bias_values.as_slice());

        let layer = ConvLayer::upload(
            client,
            &weights,
            bias,
            self.features,
            self.precision,
            self.winograd,
        );
        synced(client);

        let read_weights = read_maps(client, &layer.weights, self.precision);
        let read_bias = layer
            .bias
            .as_ref()
            .map(|handle| read_maps(client, handle, self.precision));

        ReadBack {
            weights: read_weights,
            bias: read_bias,
            transformed: layer.transformed.is_some(),
            winograd: layer.winograd,
        }
    }
}

#[test]
fn a_conv_layer_uploads_its_weights_and_bias_in_the_maps_type() {
    for precision in [Precision::F32, Precision::F16] {
        let task = Uploaded {
            features: 16,
            bias: true,
            precision,
            winograd: false,
        };

        let read = backend().build(task);

        assert_eq!(read.weights, exact_weights(16), "{precision:?}");
        assert_eq!(read.bias, Some(exact_bias(16)), "{precision:?}");
        assert!(!read.transformed, "{precision:?}");
    }
}

#[test]
fn a_conv_layer_without_a_bias_holds_none() {
    let task = Uploaded {
        features: 16,
        bias: false,
        precision: Precision::F16,
        winograd: false,
    };

    let read = backend().build(task);

    assert_eq!(read.bias, None);
}

#[test]
fn only_an_f16_winograd_layer_holds_transformed_weights() {
    let cases = [
        (Precision::F16, true, true),
        (Precision::F16, false, false),
        (Precision::F32, true, false),
    ];

    for (precision, asked, expected) in cases {
        let task = Uploaded {
            features: 16,
            bias: true,
            precision,
            winograd: asked,
        };

        let read = backend().build(task);

        assert_eq!(read.transformed, expected, "{precision:?} asked {asked}");
        assert_eq!(read.winograd, expected, "{precision:?} asked {asked}");
    }
}

/// Uploads a layer whose weights, or else its bias, are one value short.
struct ShortUpload {
    short_weights: bool,
}

impl BuildTask for ShortUpload {
    type Output = ();

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let mut weights = exact_weights(16);
        let mut bias = exact_bias(16);

        if self.short_weights {
            weights.pop();
        } else {
            bias.pop();
        }

        ConvLayer::upload(client, &weights, Some(&bias), 16, Precision::F32, false);
    }
}

#[test]
#[should_panic(expected = "the weights do not map the features to themselves")]
fn a_conv_layer_with_too_few_weights_is_refused() {
    backend().build(ShortUpload { short_weights: true });
}

#[test]
#[should_panic(expected = "the bias does not hold one value a feature")]
fn a_conv_layer_with_too_short_a_bias_is_refused() {
    backend().build(ShortUpload { short_weights: false });
}

/// Uploads a padded layer of `out_channels` from 16 features, and reads its weights and bias back.
struct Padded {
    out_channels: u32,
}

impl BuildTask for Padded {
    type Output = (Vec<f32>, Vec<f32>);

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let count = (self.out_channels * 9 * 16) as usize;
        let weights = vec![0.5f32; count];
        let bias = vec![0.25f32; self.out_channels as usize];

        let out_channels = self.out_channels;
        let layer = PaddedLastLayer::upload(client, &weights, Some(&bias), 16, out_channels, Precision::F32);
        synced(client);

        assert_eq!(layer.out_channels, self.out_channels);

        let read_weights = read_maps(client, &layer.weights, Precision::F32);
        let read_bias = read_maps(client, &layer.bias, Precision::F32);

        (read_weights, read_bias)
    }
}

#[test]
fn a_padded_layer_is_zero_past_its_real_channels() {
    for out_channels in [1, 3, 4] {
        let (weights, bias) = backend().build(Padded { out_channels });

        let real_weights = (out_channels * 9 * 16) as usize;
        let real_bias = out_channels as usize;
        assert_eq!(weights.len(), 16 * 9 * 16, "{out_channels}");
        assert!(
            weights[..real_weights].iter().all(|value| *value == 0.5),
            "{out_channels}"
        );
        assert!(
            weights[real_weights..].iter().all(|value| *value == 0.0),
            "{out_channels}"
        );
        assert_eq!(bias.len(), 16, "{out_channels}");
        assert!(
            bias[..real_bias].iter().all(|value| *value == 0.25),
            "{out_channels}"
        );
        assert!(
            bias[real_bias..].iter().all(|value| *value == 0.0),
            "{out_channels}"
        );
    }
}

/// Tunes a 64-wide Winograd layer, then the same weights as a layer without Winograd, and gives how each ran.
/// Gives `None` without an f16 cooperative matrix.
struct TunedAfterWinograd;

impl BuildTask for TunedAfterWinograd {
    type Output = Option<(LayerRun, LayerRun)>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        if !offers_cmma_16(client) {
            return None;
        }

        let precision = Precision::F16;
        let weights = exact_weights(64);
        let bias = exact_bias(64);
        let frame = Region {
            x: 0,
            y: 0,
            width: 24,
            height: 20,
        };
        let maps = wavy_maps((frame.samples() * 64) as usize);

        let with_winograd = ConvLayer::upload(client, &weights, Some(&bias), 64, precision, true);
        let without_winograd = ConvLayer::upload(client, &weights, Some(&bias), 64, precision, false);
        let input = upload_maps(client, &maps, precision);
        let output = empty(client, maps.len() * precision.bytes());
        let finish = Finish {
            activation: Activation::Silu,
            residual: None,
            skip: None,
        };

        let first = BodyConvolution::of_conv(&with_winograd, &input, &output, frame, finish);
        let second = BodyConvolution::of_conv(&without_winograd, &input, &output, frame, finish);
        let first_run = run_tuned(client, &Kernels::Tuned, &first, Epilogue::Silu);
        let first_run = first_run.expect("the Winograd layer tunes");
        let second_run = run_tuned(client, &Kernels::Tuned, &second, Epilogue::Silu);
        let second_run = second_run.expect("the layer without Winograd tunes");
        synced(client);

        Some((first_run.run, second_run.run))
    }
}

#[test]
fn a_conv_layer_without_winograd_is_never_offered_winograd() {
    let Some((first, second)) = backend().build(TunedAfterWinograd) else {
        println!("no f16 cooperative matrix, so there is nothing to tune");
        return;
    };

    println!("with Winograd {first:?}, without {second:?}");
    let candidate = second.candidate.expect("the tuner reports its candidate");

    assert!(!second.winograd_offered, "{second:?}");
    assert!(!matches!(candidate, Candidate::Winograd(_)), "{candidate:?}");
}

/// Runs the Im2col path on a layer whose weights or bias handle is a single byte.
struct ShortHandle {
    short_weights: bool,
}

impl BuildTask for ShortHandle {
    type Output = ();

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let precision = Precision::F32;
        let features = 16;
        let frame = Region {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        };
        let weights = exact_weights(features);
        let bias = exact_bias(features);
        let whole = ConvLayer::upload(client, &weights, Some(&bias), features, precision, false);
        let short = client.empty(1);
        let layer = if self.short_weights {
            ConvLayer {
                weights: short,
                ..whole
            }
        } else {
            ConvLayer {
                bias: Some(short),
                ..whole
            }
        };

        let map_bytes = layer.map_bytes(frame) as usize;
        let input = empty(client, map_bytes);
        let output = empty(client, map_bytes);
        let columns = empty(client, map_bytes * 9);
        let maps = LayerMaps {
            input: &input,
            output: &output,
            columns: Some(&columns),
        };
        let finish = Finish {
            activation: Activation::Silu,
            residual: None,
            skip: None,
        };

        let dispatch = Sizing(build.frame_context());
        let _ = run_layer(
            client,
            &dispatch,
            &Kernels::Tuned,
            BodyPath::Im2col,
            &layer,
            maps,
            frame,
            finish,
            Epilogue::Silu,
        );
    }
}

#[test]
#[should_panic(expected = "the weights are smaller than the layer's")]
fn the_im2col_path_refuses_a_layer_with_short_weights() {
    backend().build(ShortHandle { short_weights: true });
}

#[test]
#[should_panic(expected = "the bias is smaller than the features")]
fn the_im2col_path_refuses_a_layer_with_a_short_bias() {
    backend().build(ShortHandle { short_weights: false });
}
