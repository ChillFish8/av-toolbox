use cubecl::prelude::Client;
use cubecl::server::Handle;
use half::f16;

use crate::cnn::Region;
use crate::cnn::kernels::transform_values;

/// How many output channels a padded layer writes, its real channels followed by zeros.
pub const PADDED_CHANNELS: usize = 16;

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

/// One `F -> F` 3x3 convolution on the device, in the feature maps' type, and what its kernel is chosen by.
///
/// The weights are laid out by output channel, row, column and input channel. A Winograd layer on f16 maps
/// also holds its transformed weights, by position, output channel and input channel.
pub struct ConvLayer {
    pub weights: Handle,
    pub bias: Option<Handle>,
    pub transformed: Option<Handle>,
    pub features: u32,
    pub precision: Precision,
    pub winograd: bool,
}

impl ConvLayer {
    /// Uploads `weights` and `bias` in `precision`'s type. A layer asked for Winograd on f16 maps also
    /// uploads its transformed weights, and is offered the Winograd candidates.
    ///
    /// Panics if `weights` does not hold `features * 9 * features` values, or `bias` does not hold
    /// `features`.
    pub fn upload(
        client: &Client,
        weights: &[f32],
        bias: Option<&[f32]>,
        features: u32,
        precision: Precision,
        winograd: bool,
    ) -> Self {
        let channels = features as usize;
        assert_eq!(
            weights.len(),
            channels * 9 * channels,
            "the weights do not map the features to themselves"
        );

        if let Some(values) = bias {
            assert_eq!(
                values.len(),
                channels,
                "the bias does not hold one value a feature"
            );
        }

        let weight_bytes = encode(weights, precision);
        let weights_handle = client.create_from_slice(&weight_bytes);
        let bias_handle = bias.map(|values| upload_encoded(client, values, precision));
        let transforms = winograd && precision == Precision::F16;
        let transformed = transforms.then(|| upload_transformed(client, weights, features));

        Self {
            weights: weights_handle,
            bias: bias_handle,
            transformed,
            features,
            precision,
            winograd: transforms,
        }
    }

    /// The bytes one feature map of `region` takes in the layer's precision.
    pub fn map_bytes(&self, region: Region) -> u64 {
        let values = u64::from(region.samples()) * u64::from(self.features);

        values * self.precision.bytes() as u64
    }
}

/// A 3x3 convolution from the features to at most 16 channels, padded to 16 output channels, writing f32
/// maps.
///
/// The weights are in the feature maps' type and the bias in f32, both zero past the real channels.
pub struct PaddedLastLayer {
    pub weights: Handle,
    pub bias: Handle,
    pub in_channels: u32,
    pub out_channels: u32,
    pub precision: Precision,
}

impl PaddedLastLayer {
    /// Uploads `weights`, laid out by output channel, row, column and input channel, and `bias`, padded with
    /// zeros to 16 output channels. A missing bias is zero.
    ///
    /// Panics if `out_channels` is not between 1 and 16, or the values do not match the channels.
    pub fn upload(
        client: &Client,
        weights: &[f32],
        bias: Option<&[f32]>,
        in_channels: u32,
        out_channels: u32,
        precision: Precision,
    ) -> Self {
        let real = out_channels as usize;
        let taps = 9 * in_channels as usize;
        assert!(
            (1..=PADDED_CHANNELS).contains(&real),
            "a padded layer writes between 1 and 16 channels"
        );
        assert_eq!(
            weights.len(),
            real * taps,
            "the weights do not match the channels"
        );

        let mut padded_weights = weights.to_vec();
        padded_weights.resize(PADDED_CHANNELS * taps, 0.0);

        let mut padded_bias = vec![0.0f32; PADDED_CHANNELS];
        if let Some(values) = bias {
            assert_eq!(
                values.len(),
                real,
                "the bias does not hold one value an output channel"
            );
            padded_bias[..real].copy_from_slice(values);
        }

        let weight_bytes = encode(&padded_weights, precision);
        let bias_bytes = encode(&padded_bias, Precision::F32);
        let weights_handle = client.create_from_slice(&weight_bytes);
        let bias_handle = client.create_from_slice(&bias_bytes);

        Self {
            weights: weights_handle,
            bias: bias_handle,
            in_channels,
            out_channels,
            precision,
        }
    }
}

fn upload_encoded(client: &Client, values: &[f32], precision: Precision) -> Handle {
    let bytes = encode(values, precision);

    client.create_from_slice(&bytes)
}

/// The f16 Winograd weights of a `features` wide layer on the device.
fn upload_transformed(client: &Client, weights: &[f32], features: u32) -> Handle {
    let transformed = transform_values(weights, features, features);
    let bytes: Vec<u8> = transformed.iter().flat_map(|value| value.to_le_bytes()).collect();

    client.create_from_slice(&bytes)
}

/// Little-endian bytes of `values` in `precision`'s type.
pub(crate) fn encode(values: &[f32], precision: Precision) -> Vec<u8> {
    match precision {
        Precision::F16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect(),
        Precision::F32 => values.iter().flat_map(|value| value.to_le_bytes()).collect(),
    }
}
