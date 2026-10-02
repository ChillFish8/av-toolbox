use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use super::body::{Finish, body_layer};
use super::kernels::{BodyConvolution, Candidate, run_candidate};
use crate::artcnn::{
    Activation,
    ArtcnnError,
    BodyPath,
    Layer,
    Region,
    Weights,
    WeightsError,
    first_layer,
    last_layer,
    padded_last_layer,
    shuffle,
};
use crate::{DevicePlane, Dispatch};

/// How many output channels the last layer is padded to on the convolution path.
pub(crate) const PADDED_CHANNELS: usize = 16;

/// The bytes one sample of the padded last layer's f32 output takes.
const PADDED_SAMPLE_BYTES: usize = PADDED_CHANNELS * size_of::<f32>();

/// The type feature maps are held in. Accumulation is f32 either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F16,
    F32,
}

impl Precision {
    pub fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// One layer's weights and bias on the device, in the feature maps' type.
pub(crate) struct LayerBuffers {
    pub(crate) weights: Handle,
    pub(crate) bias: Handle,
    pub(crate) out_channels: u32,
    pub(crate) in_channels: u32,
}

/// The last layer padded to 16 output channels, for the convolution path.
///
/// The weights are in the feature maps' type with output channels 4 to 15 zero. The bias is 16 f32 values,
/// since the padded layer writes f32, with the last 12 zero.
pub(crate) struct PaddedLastLayer {
    pub(crate) weights: Handle,
    pub(crate) bias: Handle,
}

/// An ArtCNN network's weights on the device, in the feature maps' type.
pub struct Network {
    features: u32,
    blocks: u32,
    precision: Precision,
    layers: Vec<LayerBuffers>,
    padded_last: PaddedLastLayer,
}

impl Network {
    /// Uploads `weights` in `precision`'s type, refusing weights whose layers do not match their model.
    pub fn upload<R: Runtime>(
        client: &ComputeClient<R>,
        weights: &Weights,
        precision: Precision,
    ) -> Result<Self, WeightsError> {
        weights.validate()?;

        let mut layers = Vec::new();

        for layer in &weights.layers {
            let weight_bytes = encode(&layer.weights, precision);
            let bias_bytes = encode(&layer.bias, precision);
            let weights_handle = client.create_from_slice(&weight_bytes);
            let bias_handle = client.create_from_slice(&bias_bytes);

            layers.push(LayerBuffers {
                weights: weights_handle,
                bias: bias_handle,
                out_channels: layer.out_channels,
                in_channels: layer.in_channels,
            });
        }

        let last = weights
            .layers
            .last()
            .expect("validated weights have a last layer");
        let padded_weights = padded_weights(last);
        let padded_bias = padded_bias(last);
        let weight_bytes = encode(&padded_weights, precision);
        let bias_bytes = f32::as_bytes(&padded_bias);
        let padded_weights_handle = client.create_from_slice(&weight_bytes);
        let padded_bias_handle = client.create_from_slice(bias_bytes);
        let padded_last = PaddedLastLayer {
            weights: padded_weights_handle,
            bias: padded_bias_handle,
        };

        Ok(Self {
            features: weights.features,
            blocks: weights.blocks,
            precision,
            layers,
            padded_last,
        })
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
        self.layers.len() as u32
    }

    pub(crate) fn layer(&self, index: usize) -> &LayerBuffers {
        &self.layers[index]
    }

    pub(crate) fn last_layer(&self) -> &LayerBuffers {
        let last_index = self.layers.len() - 1;

        &self.layers[last_index]
    }

    pub(crate) fn padded_last_layer(&self) -> &PaddedLastLayer {
        &self.padded_last
    }
}

/// The weights of `layer`, followed by zeros up to 16 output channels.
fn padded_weights(layer: &Layer) -> Vec<f16> {
    let in_channels = layer.in_channels as usize;
    let mut padded = layer.weights.clone();
    padded.resize(PADDED_CHANNELS * 9 * in_channels, f16::ZERO);

    padded
}

/// The bias of `layer` in f32, followed by zeros up to 16 output channels.
fn padded_bias(layer: &Layer) -> Vec<f32> {
    let mut padded: Vec<f32> = layer.bias.iter().map(|value| value.to_f32()).collect();
    padded.resize(PADDED_CHANNELS, 0.0);

    padded
}

/// Little-endian bytes of `values` in `precision`'s type.
fn encode(values: &[f16], precision: Precision) -> Vec<u8> {
    match precision {
        Precision::F16 => values.iter().flat_map(|value| value.to_le_bytes()).collect(),
        Precision::F32 => values
            .iter()
            .flat_map(|value| value.to_f32().to_le_bytes())
            .collect(),
    }
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
) -> Result<(), ArtcnnError> {
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
) -> Result<(), ArtcnnError> {
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
) -> Result<(), ArtcnnError> {
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
        let columns = maps.columns.ok_or(ArtcnnError::MissingColumns)?;
        assert!(
            columns.size_in_used() >= bytes * 9,
            "the columns are smaller than nine of the region's maps"
        );
    }

    first_layer(client, dispatch, network, input, peak, region, maps.conv0);

    let spare = [maps.first, maps.second, maps.third];
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

        run_body_layer(
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
        run_body_layer(
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
        run_body_layer(
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

        state = first;
        layer += 3;
    }

    let [joined, _] = two_others(&spare, state);
    let skipped = Finish {
        activation: Activation::None,
        residual: Some(maps.conv0),
    };
    run_body_layer(
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

    match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => {
            // `conv0` is dead once the join has read it, so the padded last layer writes into it.
            padded_last_layer(client, network, joined, region, maps.conv0)?;
            shuffle(client, dispatch, maps.conv0, region, core, peak, output);
        },
        BodyPath::Im2col => last_layer(client, dispatch, network, joined, region, core, peak, output),
    }

    Ok(())
}

/// Runs body layer `layer` as `body` says, over the maps of `region` in `input`, into `output`.
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
) -> Result<(), ArtcnnError> {
    let Some(candidate) = body.candidate else {
        body_layer(
            client, dispatch, network, body.path, layer, input, output, columns, region, finish,
        )?;

        return Ok(());
    };

    let convolution = BodyConvolution::of_layer(network, layer, input, output, region, finish);

    run_candidate(client, candidate, &convolution)
}

/// The bytes one feature map of `region` takes in the network's precision.
pub(crate) fn map_bytes(network: &Network, region: Region) -> u64 {
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
