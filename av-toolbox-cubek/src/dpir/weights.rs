/// The bytes every weights file starts with.
const MAGIC: &[u8; 8] = b"AVTBDPR1";

/// How many convolutions make up one block at each level.
const BLOCK_CONVOLUTIONS: usize = 8;

/// How many layers a network has.
const LAYER_COUNT: u32 = 64;

/// The pack the DPIR weights install in.
pub const PACK: &str = "dpir";

/// The features of each of DRUNet's four levels.
pub const WIDTHS: [u32; 4] = [64, 128, 256, 512];

/// Which of DPIR's deblocking networks a set of weights is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Gray,
    Colour,
}

impl Variant {
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Gray => "drunet_deblock_gray.bin",
            Self::Colour => "drunet_deblock_color.bin",
        }
    }

    pub fn image_channels(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::Colour => 3,
        }
    }

    fn code(self) -> u32 {
        match self {
            Self::Gray => 0,
            Self::Colour => 1,
        }
    }

    fn from_code(code: u32) -> Result<Self, WeightsError> {
        match code {
            0 => Ok(Self::Gray),
            1 => Ok(Self::Colour),
            _ => Err(WeightsError::UnknownVariant(code)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// A 3x3 convolution, by output channel, row, column and input channel.
    Conv,
    /// A 2x2 stride 2 convolution, as a `[4 * in, out]` matrix.
    Down,
    /// A 2x2 stride 2 transposed convolution, as an `[in, 4 * out]` matrix.
    Up,
}

impl LayerKind {
    fn code(self) -> u32 {
        match self {
            Self::Conv => 0,
            Self::Down => 1,
            Self::Up => 2,
        }
    }

    fn from_code(code: u32, index: usize) -> Result<Self, WeightsError> {
        match code {
            0 => Ok(Self::Conv),
            1 => Ok(Self::Down),
            2 => Ok(Self::Up),
            _ => Err(WeightsError::UnknownKind { index, kind: code }),
        }
    }

    /// How many values a layer of this kind holds.
    pub(crate) fn value_count(self, in_channels: u32, out_channels: u32) -> usize {
        let channel_pairs = in_channels as usize * out_channels as usize;

        match self {
            Self::Conv => channel_pairs * 9,
            Self::Down | Self::Up => channel_pairs * 4,
        }
    }
}

/// One layer's kind, input channels and output channels.
type LayerShape = (LayerKind, u32, u32);

/// One layer of weights, in the layout its [LayerKind] describes.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightLayer {
    pub kind: LayerKind,
    pub in_channels: u32,
    pub out_channels: u32,
    pub values: Vec<f32>,
}

/// The weights of one DRUNet deblocking network.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    variant: Variant,
    layers: Vec<WeightLayer>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum WeightsError {
    #[error("the weights do not start with the AVTBDPR1 magic")]
    Magic,

    #[error("the weights are for the {found:?} network, not {expected:?}")]
    Variant { found: Variant, expected: Variant },

    #[error("the weights hold {found} layers, not 64")]
    LayerCount { found: u32 },

    #[error("layer {index} is {found:?}, not {expected:?}")]
    LayerShape {
        index: usize,
        found: LayerShape,
        expected: LayerShape,
    },

    #[error("layer {index} holds {found} values, not {expected}")]
    LayerValues {
        index: usize,
        found: usize,
        expected: usize,
    },

    #[error("the weights end in the middle of layer {index}")]
    Truncated { index: usize },

    #[error("the weights run {extra} bytes past their last layer")]
    Trailing { extra: usize },

    #[error("variant {0} is neither 0 (gray) nor 1 (colour)")]
    UnknownVariant(u32),

    #[error("layer {index} has an unknown kind {kind}")]
    UnknownKind { index: usize, kind: u32 },

    #[error("layer {index} holds a value that is not finite")]
    NotFinite { index: usize },
}

/// Each layer's kind, input and output channels in forward order, for `variant`.
pub fn layer_plan(variant: Variant) -> Vec<LayerShape> {
    let image_channels = variant.image_channels();
    let mut plan = vec![(LayerKind::Conv, image_channels + 1, WIDTHS[0])];

    for pair in WIDTHS.windows(2) {
        let (width, next_width) = (pair[0], pair[1]);
        plan.extend([(LayerKind::Conv, width, width); BLOCK_CONVOLUTIONS]);
        plan.push((LayerKind::Down, width, next_width));
    }

    let deepest = WIDTHS[3];
    plan.extend([(LayerKind::Conv, deepest, deepest); BLOCK_CONVOLUTIONS]);

    for pair in WIDTHS.windows(2).rev() {
        let (next_width, width) = (pair[0], pair[1]);
        plan.push((LayerKind::Up, width, next_width));
        plan.extend([(LayerKind::Conv, next_width, next_width); BLOCK_CONVOLUTIONS]);
    }

    plan.push((LayerKind::Conv, WIDTHS[0], image_channels));
    plan
}

fn truncated(index: usize) -> WeightsError {
    WeightsError::Truncated { index }
}

fn variant_mismatch(found: Variant, expected: Variant) -> WeightsError {
    WeightsError::Variant { found, expected }
}

fn shape_mismatch(index: usize, found: LayerShape, expected: LayerShape) -> WeightsError {
    WeightsError::LayerShape {
        index,
        found,
        expected,
    }
}

impl Weights {
    /// Reads the weights of the `variant` network, refusing a file that does not match its layer plan.
    pub fn parse(bytes: &[u8], variant: Variant) -> Result<Self, WeightsError> {
        let mut reader = Reader { bytes };
        let magic = reader.take(MAGIC.len()).ok_or(WeightsError::Magic)?;

        if magic != MAGIC {
            return Err(WeightsError::Magic);
        }

        let found_code = reader.u32().ok_or(truncated(0))?;
        let found_variant = Variant::from_code(found_code)?;

        if found_variant != variant {
            return Err(variant_mismatch(found_variant, variant));
        }

        let found_count = reader.u32().ok_or(truncated(0))?;

        if found_count != LAYER_COUNT {
            return Err(WeightsError::LayerCount { found: found_count });
        }

        let plan = layer_plan(variant);
        let mut layers = Vec::with_capacity(plan.len());

        for (index, expected) in plan.into_iter().enumerate() {
            let layer = reader.layer(index, expected)?;
            layers.push(layer);
        }

        if !reader.bytes.is_empty() {
            return Err(WeightsError::Trailing {
                extra: reader.bytes.len(),
            });
        }

        Ok(Self { variant, layers })
    }

    /// Builds weights from `layers`, checking they follow the layer plan of `variant`.
    pub fn new(variant: Variant, layers: Vec<WeightLayer>) -> Result<Self, WeightsError> {
        let found_count = layers.len() as u32;

        if found_count != LAYER_COUNT {
            return Err(WeightsError::LayerCount { found: found_count });
        }

        let plan = layer_plan(variant);

        for (index, (layer, expected)) in layers.iter().zip(plan).enumerate() {
            let found = (layer.kind, layer.in_channels, layer.out_channels);

            if found != expected {
                return Err(shape_mismatch(index, found, expected));
            }

            let expected_values = layer.kind.value_count(layer.in_channels, layer.out_channels);

            if layer.values.len() != expected_values {
                return Err(WeightsError::LayerValues {
                    index,
                    found: layer.values.len(),
                    expected: expected_values,
                });
            }
        }

        Ok(Self { variant, layers })
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    pub fn layers(&self) -> &[WeightLayer] {
        &self.layers
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&self.variant.code().to_le_bytes());
        bytes.extend_from_slice(&LAYER_COUNT.to_le_bytes());

        for layer in &self.layers {
            bytes.extend_from_slice(&layer.kind.code().to_le_bytes());
            bytes.extend_from_slice(&layer.in_channels.to_le_bytes());
            bytes.extend_from_slice(&layer.out_channels.to_le_bytes());

            for value in &layer.values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }

        bytes
    }
}

/// Reads little-endian values from the front of a byte slice.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        if count > self.bytes.len() {
            return None;
        }

        let (taken, rest) = self.bytes.split_at(count);
        self.bytes = rest;

        Some(taken)
    }

    fn u32(&mut self) -> Option<u32> {
        let taken = self.take(4)?;
        let (words, _) = taken.as_chunks::<4>();

        Some(u32::from_le_bytes(words[0]))
    }

    /// Reads the layer at `index`, which must be the `expected` one.
    fn layer(&mut self, index: usize, expected: LayerShape) -> Result<WeightLayer, WeightsError> {
        let kind_code = self.u32().ok_or(truncated(index))?;
        let in_channels = self.u32().ok_or(truncated(index))?;
        let out_channels = self.u32().ok_or(truncated(index))?;
        let kind = LayerKind::from_code(kind_code, index)?;
        let found = (kind, in_channels, out_channels);

        if found != expected {
            return Err(shape_mismatch(index, found, expected));
        }

        let value_count = kind.value_count(in_channels, out_channels);
        let raw = self.take(value_count * 4).ok_or(truncated(index))?;
        let (words, _) = raw.as_chunks::<4>();
        let values: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();

        if values.iter().any(|value| !value.is_finite()) {
            return Err(WeightsError::NotFinite { index });
        }

        Ok(WeightLayer {
            kind,
            in_channels,
            out_channels,
            values,
        })
    }
}
