use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use crate::artcnn::{
    Layer,
    Model,
    Weights,
    WeightsError,
    first_layer,
    last_layer,
    padded_last_layer,
    shuffle,
    winograd_allowed,
};
use crate::cnn::{
    Activation,
    BodyConvolution,
    BodyPath,
    Candidate,
    CnnError,
    ConvLayer,
    Epilogue,
    Finish,
    Kernels,
    LayerMaps,
    LayerRun,
    PADDED_CHANNELS,
    PaddedLastLayer,
    Precision,
    Region,
    encode,
    launch_candidate,
    run_layer,
};
use crate::{DevicePlane, Dispatch};

/// The bytes one sample of the padded last layer's f32 output takes.
const PADDED_SAMPLE_BYTES: usize = PADDED_CHANNELS * size_of::<f32>();

/// One layer's weights and bias on the device, in the feature maps' type.
pub(crate) struct LayerBuffers {
    pub(crate) weights: Handle,
    pub(crate) bias: Handle,
    pub(crate) out_channels: u32,
    pub(crate) in_channels: u32,
}

/// An ArtCNN network's weights on the device, in the feature maps' type, and how its body convolutions
/// choose their kernel.
pub struct Network {
    model: Option<Model>,
    features: u32,
    blocks: u32,
    precision: Precision,
    first: LayerBuffers,
    body: Vec<ConvLayer>,
    last: LayerBuffers,
    padded_last: PaddedLastLayer,
    kernels: Kernels,
}

impl Network {
    /// Uploads `weights` in `precision`'s type, refusing weights whose layers do not match their model.
    pub fn upload<R: Runtime>(
        client: &ComputeClient<R>,
        weights: &Weights,
        precision: Precision,
    ) -> Result<Self, WeightsError> {
        upload_network(client, None, weights, precision)
    }

    /// Uploads `model`'s `weights` in `precision`'s type, refusing weights of another shape.
    ///
    /// On f16 maps, a model on the Winograd allow-list also holds its body layers' transformed weights.
    pub fn upload_model<R: Runtime>(
        client: &ComputeClient<R>,
        model: Model,
        weights: &Weights,
        precision: Precision,
    ) -> Result<Self, WeightsError> {
        let found = [weights.features, weights.blocks];
        let expected = [model.features(), model.blocks()];

        if found != expected {
            return Err(WeightsError::ModelShape { found, expected });
        }

        upload_network(client, Some(model), weights, precision)
    }

    /// The same network, with its f16 body convolutions on the convolution path choosing their kernel by
    /// `kernels`.
    pub fn with_kernels(self, kernels: Kernels) -> Self {
        Self { kernels, ..self }
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    /// The model the network was uploaded as, if any.
    pub fn model(&self) -> Option<Model> {
        self.model
    }

    pub fn features(&self) -> u32 {
        self.features
    }

    pub fn blocks(&self) -> u32 {
        self.blocks
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// How many samples past a tile's edge the network reads, one per layer.
    pub fn halo(&self) -> u32 {
        self.body.len() as u32 + 2
    }

    pub(crate) fn first_layer(&self) -> &LayerBuffers {
        &self.first
    }

    /// Body layer `index` of the forward pass, counting the first layer as 0.
    ///
    /// Panics if `index` is the first or last layer, which do not map the features to themselves.
    pub(crate) fn body_layer(&self, index: usize) -> &ConvLayer {
        let position = index
            .checked_sub(1)
            .filter(|position| *position < self.body.len());

        match position {
            Some(position) => &self.body[position],
            None => panic!("layer {index} does not map the features to themselves"),
        }
    }

    pub(crate) fn last_layer(&self) -> &LayerBuffers {
        &self.last
    }

    pub(crate) fn padded_last_layer(&self) -> &PaddedLastLayer {
        &self.padded_last
    }
}

/// Uploads `weights` in `precision`'s type as `model`, refusing weights whose layers do not match their
/// model.
fn upload_network<R: Runtime>(
    client: &ComputeClient<R>,
    model: Option<Model>,
    weights: &Weights,
    precision: Precision,
) -> Result<Network, WeightsError> {
    weights.validate()?;

    let winograd = model.is_some_and(winograd_allowed);
    let last_index = weights.layers.len() - 1;
    let mut body = Vec::new();

    for layer in &weights.layers[1..last_index] {
        let widened_weights = widen(&layer.weights);
        let widened_bias = widen(&layer.bias);
        let uploaded = ConvLayer::upload(
            client,
            &widened_weights,
            Some(&widened_bias),
            weights.features,
            precision,
            winograd,
        );

        body.push(uploaded);
    }

    let first = upload_buffers(client, &weights.layers[0], precision);
    let last_layer = &weights.layers[last_index];
    let last = upload_buffers(client, last_layer, precision);
    let last_weights = widen(&last_layer.weights);
    let last_bias = widen(&last_layer.bias);
    let padded_last = PaddedLastLayer::upload(
        client,
        &last_weights,
        Some(&last_bias),
        weights.features,
        last_layer.out_channels,
        precision,
    );

    Ok(Network {
        model,
        features: weights.features,
        blocks: weights.blocks,
        precision,
        first,
        body,
        last,
        padded_last,
        kernels: Kernels::default(),
    })
}

fn upload_buffers<R: Runtime>(
    client: &ComputeClient<R>,
    layer: &Layer,
    precision: Precision,
) -> LayerBuffers {
    let widened_weights = widen(&layer.weights);
    let widened_bias = widen(&layer.bias);
    let weight_bytes = encode(&widened_weights, precision);
    let bias_bytes = encode(&widened_bias, precision);
    let weights_handle = client.create_from_slice(&weight_bytes);
    let bias_handle = client.create_from_slice(&bias_bytes);

    LayerBuffers {
        weights: weights_handle,
        bias: bias_handle,
        out_channels: layer.out_channels,
        in_channels: layer.in_channels,
    }
}

fn widen(values: &[f16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

/// The device buffers one region's forward pass works in.
///
/// The four maps must be distinct buffers, each holding at least the region's samples times `F` values of the
/// precision's type. On the convolution paths the last layer writes 16 f32 values a sample into `conv0`, so
/// it holds at least 64 bytes a sample. `columns` holds nine times a map, and is needed on
/// [BodyPath::Im2col](crate::artcnn::BodyPath::Im2col) only.
///
/// A layer reads one map while writing another, so two maps sharing a buffer give wrong results. Only a
/// handle passed twice is caught, and only in debug builds.
pub struct Maps<'a> {
    pub conv0: &'a Handle,
    pub first: &'a Handle,
    pub second: &'a Handle,
    pub third: &'a Handle,
    pub columns: Option<&'a Handle>,
}

/// Runs the network over `region` of `input`, divided by `peak`, and writes the doubled `core` into `output`.
/// Gives how each body layer ran, in order.
///
/// The region's edge is padded with zero. Each output sample is clamped to `0..=1`, scaled by `peak` and
/// rounded. The four maps must be distinct buffers, since a layer reads one while writing another.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn run_region<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
    input: DevicePlane<'_>,
    peak: f32,
    region: Region,
    core: Region,
    maps: Maps<'_>,
    output: DevicePlane<'_>,
) -> Result<Vec<LayerRun>, CnnError> {
    let body = Body {
        path,
        candidate: None,
    };

    run_region_on(
        client, dispatch, network, body, input, peak, region, core, maps, output,
    )
}

/// The same as [run_region](crate::artcnn::run_region) on the convolution path, with every body convolution
/// run by `candidate`.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn run_region_with_candidate<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    candidate: Candidate,
    input: DevicePlane<'_>,
    peak: f32,
    region: Region,
    core: Region,
    maps: Maps<'_>,
    output: DevicePlane<'_>,
) -> Result<Vec<LayerRun>, CnnError> {
    let body = Body {
        path: BodyPath::Convolution,
        candidate: Some(candidate),
    };

    run_region_on(
        client, dispatch, network, body, input, peak, region, core, maps, output,
    )
}

/// How a region's body layers run. They run on `path`, or by `candidate` when one is given.
#[derive(Clone, Copy)]
struct Body {
    path: BodyPath,
    candidate: Option<Candidate>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
fn run_region_on<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    body: Body,
    input: DevicePlane<'_>,
    peak: f32,
    region: Region,
    core: Region,
    maps: Maps<'_>,
    output: DevicePlane<'_>,
) -> Result<Vec<LayerRun>, CnnError> {
    let path = body.path;
    let every_map = [maps.conv0, maps.first, maps.second, maps.third];
    debug_assert!(are_distinct(&every_map), "two of the maps are the same buffer");

    let bytes = map_bytes(network, region);
    for map in every_map {
        assert!(
            map.size_in_used() >= bytes,
            "a map is smaller than the region's maps"
        );
    }

    let conv0_bytes = conv0_bytes(network, path, region);
    assert!(
        maps.conv0.size_in_used() >= conv0_bytes,
        "conv0 is smaller than the region's last layer needs"
    );

    if path == BodyPath::Im2col {
        let columns = maps.columns.ok_or(CnnError::MissingColumns)?;
        assert!(
            columns.size_in_used() >= bytes * 9,
            "the columns are smaller than nine of the region's maps"
        );
    }

    first_layer(client, dispatch, network, input, peak, region, maps.conv0);

    let spare = [maps.first, maps.second, maps.third];
    let mut runs = Vec::new();
    let mut state = maps.conv0;
    let mut layer = 1;
    let activated = Finish {
        activation: Activation::Silu,
        residual: None,
    };

    for _ in 0..network.blocks() {
        let [first, second] = two_others(&spare, state);
        let joined = Finish {
            activation: Activation::None,
            residual: Some(state),
        };

        let first_run = run_body_layer(
            client,
            dispatch,
            network,
            body,
            layer,
            state,
            first,
            maps.columns,
            region,
            activated,
        )?;
        let second_run = run_body_layer(
            client,
            dispatch,
            network,
            body,
            layer + 1,
            first,
            second,
            maps.columns,
            region,
            activated,
        )?;
        let third_run = run_body_layer(
            client,
            dispatch,
            network,
            body,
            layer + 2,
            second,
            first,
            maps.columns,
            region,
            joined,
        )?;
        runs.extend([first_run, second_run, third_run]);

        state = first;
        layer += 3;
    }

    let [joined, _] = two_others(&spare, state);
    let skipped = Finish {
        activation: Activation::None,
        residual: Some(maps.conv0),
    };
    let join_run = run_body_layer(
        client,
        dispatch,
        network,
        body,
        layer,
        state,
        joined,
        maps.columns,
        region,
        skipped,
    )?;
    runs.push(join_run);

    match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => {
            // `conv0` is dead once the join has read it, so the padded last layer writes into it.
            padded_last_layer(client, network, joined, region, maps.conv0)?;
            shuffle(client, dispatch, maps.conv0, region, core, peak, output);
        },
        BodyPath::Im2col => last_layer(client, dispatch, network, joined, region, core, peak, output),
    }

    Ok(runs)
}

/// Runs body layer `layer` as `body` says, over the maps of `region` in `input`, into `output`.
///
/// On the convolution path with f16 maps whose features are a multiple of 16, the layer runs with the
/// candidate the network's kernels choose.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the layer"
)]
fn run_body_layer<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    body: Body,
    layer: usize,
    input: &Handle,
    output: &Handle,
    columns: Option<&Handle>,
    region: Region,
    finish: Finish<'_>,
) -> Result<LayerRun, CnnError> {
    if let Some(candidate) = body.candidate {
        let convolution = BodyConvolution::of_layer(network, layer, input, output, region, finish);
        let forced = launch_candidate(client, candidate, &convolution)?;
        let run = LayerRun::of_candidate(candidate, forced);

        return Ok(run);
    }

    let conv = network.body_layer(layer);
    let epilogue = Epilogue::of_layer(network, layer, finish);
    let maps = LayerMaps {
        input,
        output,
        columns,
    };

    run_layer(
        client,
        dispatch,
        network.kernels(),
        body.path,
        conv,
        maps,
        region,
        finish,
        epilogue,
    )
}

/// The bytes one feature map of `region` takes in the network's precision.
fn map_bytes(network: &Network, region: Region) -> u64 {
    let map_count = u64::from(region.samples()) * u64::from(network.features());

    map_count * network.precision().bytes() as u64
}

/// The bytes `conv0` needs for `region` on `path`.
fn conv0_bytes(network: &Network, path: BodyPath, region: Region) -> u64 {
    let features = network.features();
    let precision = network.precision();
    let sample_bytes = conv0_sample_bytes(features, precision, path);

    u64::from(region.samples()) * sample_bytes as u64
}

/// The bytes one sample of `conv0` needs on `path`.
///
/// That is one map's sample, widened on the convolution paths to the 64 bytes of the padded last layer's 16
/// f32 values.
pub(crate) fn conv0_sample_bytes(features: u32, precision: Precision, path: BodyPath) -> usize {
    let map_sample_bytes = features as usize * precision.bytes();

    match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => map_sample_bytes.max(PADDED_SAMPLE_BYTES),
        BodyPath::Im2col => map_sample_bytes,
    }
}

/// The first two of `spare` that are not `state`.
fn two_others<'a>(spare: &[&'a Handle; 3], state: &Handle) -> [&'a Handle; 2] {
    let others: Vec<&'a Handle> = spare
        .iter()
        .copied()
        .filter(|candidate| !std::ptr::eq(*candidate, state))
        .collect();

    [others[0], others[1]]
}

fn are_distinct(handles: &[&Handle]) -> bool {
    for (index, handle) in handles.iter().enumerate() {
        let later = &handles[index + 1..];

        if later.iter().any(|other| std::ptr::eq(*other, *handle)) {
            return false;
        }
    }

    true
}
