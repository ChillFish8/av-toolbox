use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use super::body::convolve;
use crate::artcnn::{
    Activation,
    ArtcnnError,
    BodyPath,
    Region,
    Weights,
    WeightsError,
    epilogue,
    first_layer,
    last_layer,
};
use crate::{DevicePlane, Dispatch};

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

/// An ArtCNN network's weights on the device, in the feature maps' type.
pub struct Network {
    features: u32,
    blocks: u32,
    precision: Precision,
    layers: Vec<LayerBuffers>,
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

        Ok(Self {
            features: weights.features,
            blocks: weights.blocks,
            precision,
            layers,
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
/// precision's type. `columns` holds nine times that, and is needed on
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
    let every_map = [maps.conv0, maps.first, maps.second, maps.third];
    debug_assert!(are_distinct(&every_map), "two of the maps are the same buffer");

    let samples = region.samples();
    let bytes = map_bytes(network, region);
    for map in every_map {
        assert!(
            map.size_in_used() >= bytes,
            "a map is smaller than the region's maps"
        );
    }

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

    for _ in 0..network.blocks() {
        let [first, second] = two_others(&spare, state);

        convolve(
            client,
            dispatch,
            network,
            path,
            layer,
            state,
            first,
            maps.columns,
            region,
        )?;
        epilogue(
            client,
            dispatch,
            network,
            first,
            layer,
            samples,
            Activation::Silu,
            None,
        );

        convolve(
            client,
            dispatch,
            network,
            path,
            layer + 1,
            first,
            second,
            maps.columns,
            region,
        )?;
        epilogue(
            client,
            dispatch,
            network,
            second,
            layer + 1,
            samples,
            Activation::Silu,
            None,
        );

        convolve(
            client,
            dispatch,
            network,
            path,
            layer + 2,
            second,
            first,
            maps.columns,
            region,
        )?;
        epilogue(
            client,
            dispatch,
            network,
            first,
            layer + 2,
            samples,
            Activation::None,
            Some(state),
        );

        state = first;
        layer += 3;
    }

    let [joined, _] = two_others(&spare, state);
    convolve(
        client,
        dispatch,
        network,
        path,
        layer,
        state,
        joined,
        maps.columns,
        region,
    )?;
    epilogue(
        client,
        dispatch,
        network,
        joined,
        layer,
        samples,
        Activation::None,
        Some(maps.conv0),
    );

    last_layer(client, dispatch, network, joined, region, core, peak, output);

    Ok(())
}

/// The bytes one feature map of `region` takes in the network's precision.
pub(crate) fn map_bytes(network: &Network, region: Region) -> u64 {
    let map_count = u64::from(region.samples()) * u64::from(network.features());

    map_count * network.precision().bytes() as u64
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
