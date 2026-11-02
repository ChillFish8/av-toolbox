use cubecl::ir::DeviceProperties;
#[cfg(any(test, feature = "test-support"))]
use cubecl::prelude::Client;
use cubecl::server::Handle;
#[cfg(test)]
use half::f16;

#[cfg(any(test, feature = "test-support"))]
use crate::Dispatch;
#[cfg(test)]
use crate::artcnn::Layer;
use crate::artcnn::{Model, Network, winograd_allowed};
use crate::cnn::{self, Activation, BodyConvolution, Candidate, Epilogue, Finish, Region};
#[cfg(any(test, feature = "test-support"))]
use crate::cnn::{BodyPath, CnnError, LayerRun};

impl<'a> BodyConvolution<'a> {
    /// Body layer `layer` of `network` over the maps of `region`, applying `finish`.
    ///
    /// Panics if the network's maps are not f16 or the layer does not map the features to themselves.
    pub fn of_layer(
        network: &'a Network,
        layer: usize,
        input: &'a Handle,
        output: &'a Handle,
        region: Region,
        finish: Finish<'a>,
    ) -> Self {
        let conv = network.body_layer(layer);

        Self::of_conv(conv, input, output, region, finish)
    }
}

impl Epilogue {
    /// The epilogue body layer `layer` of `network` applies with `finish`.
    pub(crate) fn of_layer(network: &Network, layer: usize, finish: Finish<'_>) -> Self {
        let last_body_layer = 3 * network.blocks() as usize + 1;

        if finish.activation == Activation::Silu {
            return Self::Silu;
        }

        if layer == last_body_layer {
            return Self::Join;
        }

        Self::Residual
    }
}

/// The candidates a device with `properties` offers `model` for maps of `features` channels: the direct
/// variants first, then the Winograd variants for an allow-listed model, and the fused cubek convolution
/// last.
pub fn offered_candidates(
    properties: &DeviceProperties,
    features: u32,
    model: Option<Model>,
) -> Vec<Candidate> {
    let winograd = model.is_some_and(winograd_allowed);

    cnn::offered_candidates(properties, features, winograd)
}

/// The transformed weights of `layer`, by position, output channel and input channel, each computed in f64
/// and rounded to f16 once.
#[cfg(test)]
pub(crate) fn transform_weights(layer: &Layer) -> Vec<f16> {
    let widened: Vec<f32> = layer.weights.iter().map(|value| value.to_f32()).collect();

    cnn::transform_values(&widened, layer.out_channels, layer.in_channels)
}

/// Adds body layer `layer`'s bias to the NHWC maps in `values`, then applies `activation`, then adds
/// `residual` if given.
///
/// The maps hold `samples` samples of the network's features, and are updated in place. The layer must map
/// the features to themselves.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn epilogue(
    client: &Client,
    dispatch: &impl Dispatch,
    network: &Network,
    values: &Handle,
    layer: usize,
    samples: u32,
    activation: Activation,
    residual: Option<&Handle>,
) {
    let conv = network.body_layer(layer);

    cnn::epilogue(client, dispatch, conv, values, samples, activation, residual);
}

/// Convolves the NHWC maps of `region` in `input` with body layer `layer`'s `F -> F` weights and bias on
/// cooperative-matrix tiles, into `output`. Gives whether the forced blueprint ran.
///
/// With `allow_forced` the convolution picks its blueprint as a forward pass does, and otherwise it keeps
/// the inferred blueprint.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn convolve_with_blueprint(
    client: &Client,
    dispatch: &impl Dispatch,
    network: &Network,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
    allow_forced: bool,
) -> Result<bool, CnnError> {
    let conv = network.body_layer(layer);

    cnn::convolve_with_blueprint(client, dispatch, conv, input, output, region, allow_forced)
}

/// Convolves the NHWC maps of `region` in `input` with body layer `layer`'s `F -> F` weights on
/// cooperative-matrix tiles, into `output` without the bias. Gives whether the forced blueprint ran.
#[cfg(any(test, feature = "test-support"))]
pub fn convolve_without_bias(
    client: &Client,
    network: &Network,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
) -> Result<bool, CnnError> {
    let conv = network.body_layer(layer);

    cnn::convolve_without_bias(client, conv, input, output, region)
}

/// Runs body layer `layer` on `path` as a forward pass does, keeping the inferred blueprint unless
/// `allow_forced` is set.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the layer"
)]
pub fn body_layer_with_blueprint(
    client: &Client,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    allow_forced: bool,
) -> Result<LayerRun, CnnError> {
    let conv = network.body_layer(layer);

    cnn::body_layer_with_blueprint(
        client,
        dispatch,
        conv,
        path,
        input,
        output,
        region,
        finish,
        allow_forced,
    )
}

/// Whether a body convolution of `region` on the network's maps takes the forced blueprint on the client's
/// device.
#[cfg(any(test, feature = "test-support"))]
pub fn takes_forced_blueprint(client: &Client, network: &Network, region: Region) -> bool {
    let features = network.features();
    let precision = network.precision();

    cnn::takes_forced_blueprint(client, features, precision, region)
}
