use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use crate::artcnn::Weights;

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
    pub fn upload<R: Runtime>(client: &ComputeClient<R>, weights: &Weights, precision: Precision) -> Self {
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

        Self {
            features: weights.features,
            blocks: weights.blocks,
            precision,
            layers,
        }
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
