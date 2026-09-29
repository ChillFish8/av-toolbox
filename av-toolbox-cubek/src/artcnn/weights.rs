use half::f16;

/// The bytes every weights file starts with.
const MAGIC: &[u8; 8] = b"AVTBCNN1";

/// The ArtCNN networks av-toolbox can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    R8F64,
    R16F96,
}

impl Model {
    pub fn features(self) -> u32 {
        match self {
            Self::R8F64 => 64,
            Self::R16F96 => 96,
        }
    }

    pub fn blocks(self) -> u32 {
        match self {
            Self::R8F64 => 8,
            Self::R16F96 => 16,
        }
    }

    pub fn file_name(self) -> &'static str {
        match self {
            Self::R8F64 => "ArtCNN_R8F64.bin",
            Self::R16F96 => "ArtCNN_R16F96.bin",
        }
    }
}

/// One 3x3 convolution, with its weights laid out by output channel, row, column and input channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub out_channels: u32,
    pub in_channels: u32,
    pub weights: Vec<f16>,
    pub bias: Vec<f16>,
}

/// The layers of an ArtCNN network `features` wide with `blocks` residual blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    pub features: u32,
    pub blocks: u32,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WeightsError {
    #[error("the weights do not start with AVTBCNN1")]
    BadMagic,

    #[error("the weights end after {available} bytes, but {needed} are needed")]
    Truncated { needed: usize, available: usize },

    #[error("the weights have {found} layers, but the model has {expected}")]
    LayerCount { found: u32, expected: u32 },

    #[error("layer {index} has the shape {found:?}, but the model needs {expected:?}")]
    LayerShape {
        index: usize,
        found: [u32; 4],
        expected: [u32; 4],
    },

    #[error("the weights have {count} bytes after their last layer")]
    TrailingBytes { count: usize },

    #[error("layer {index} holds {found:?} weight and bias values, but its shape needs {expected:?}")]
    LayerLength {
        index: usize,
        found: [usize; 2],
        expected: [usize; 2],
    },
}

/// The shape of each layer of a network `features` wide with `blocks` residual blocks.
///
/// Each shape is the output channels, input channels, rows and columns.
pub fn layer_shapes(features: u32, blocks: u32) -> Vec<[u32; 4]> {
    let body_count = 3 * blocks + 1;
    let mut shapes = vec![[features, 1, 3, 3]];

    for _ in 0..body_count {
        shapes.push([features, features, 3, 3]);
    }

    shapes.push([4, features, 3, 3]);
    shapes
}

impl Weights {
    /// Reads weights, refusing a file whose layers do not match a network `features` wide with `blocks`
    /// blocks.
    pub fn parse(bytes: &[u8], features: u32, blocks: u32) -> Result<Self, WeightsError> {
        let mut reader = Reader { bytes, offset: 0 };
        let magic = reader.take(MAGIC.len())?;

        if magic != MAGIC {
            return Err(WeightsError::BadMagic);
        }

        let expected_shapes = layer_shapes(features, blocks);
        let found_count = reader.u32()?;
        let expected_count = expected_shapes.len() as u32;

        if found_count != expected_count {
            return Err(WeightsError::LayerCount {
                found: found_count,
                expected: expected_count,
            });
        }

        let mut layers = Vec::new();
        for (index, expected) in expected_shapes.into_iter().enumerate() {
            let found = [reader.u32()?, reader.u32()?, reader.u32()?, reader.u32()?];

            if found != expected {
                return Err(WeightsError::LayerShape {
                    index,
                    found,
                    expected,
                });
            }

            let [out_channels, in_channels, rows, columns] = found;
            let weight_count = (out_channels * in_channels * rows * columns) as usize;
            let weights = reader.halves(weight_count)?;
            let bias = reader.halves(out_channels as usize)?;

            layers.push(Layer {
                out_channels,
                in_channels,
                weights,
                bias,
            });
        }

        let remaining = bytes.len() - reader.offset;
        if remaining != 0 {
            return Err(WeightsError::TrailingBytes { count: remaining });
        }

        Ok(Self {
            features,
            blocks,
            layers,
        })
    }

    /// Checks the layers are those of a network `features` wide with `blocks` blocks.
    ///
    /// Each layer must have the model's shape, and hold as many weights and bias values as that shape needs.
    pub fn validate(&self) -> Result<(), WeightsError> {
        let expected_shapes = layer_shapes(self.features, self.blocks);
        let found_count = self.layers.len() as u32;
        let expected_count = expected_shapes.len() as u32;

        if found_count != expected_count {
            return Err(WeightsError::LayerCount {
                found: found_count,
                expected: expected_count,
            });
        }

        for (index, (layer, expected)) in self.layers.iter().zip(expected_shapes).enumerate() {
            let found = [layer.out_channels, layer.in_channels, 3, 3];

            if found != expected {
                return Err(WeightsError::LayerShape {
                    index,
                    found,
                    expected,
                });
            }

            let found_lengths = [layer.weights.len(), layer.bias.len()];
            let weight_count = (layer.out_channels * layer.in_channels * 9) as usize;
            let expected_lengths = [weight_count, layer.out_channels as usize];

            if found_lengths != expected_lengths {
                return Err(WeightsError::LayerLength {
                    index,
                    found: found_lengths,
                    expected: expected_lengths,
                });
            }
        }

        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        let count = self.layers.len() as u32;
        bytes.extend_from_slice(&count.to_le_bytes());

        for layer in &self.layers {
            let header = [layer.out_channels, layer.in_channels, 3, 3];

            for value in header {
                bytes.extend_from_slice(&value.to_le_bytes());
            }

            for value in layer.weights.iter().chain(&layer.bias) {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }

        bytes
    }

    /// How many samples past a tile's edge the network reads, one per layer.
    pub fn halo(&self) -> u32 {
        self.layers.len() as u32
    }
}

/// Reads little-endian values from the front of a byte slice.
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], WeightsError> {
        let needed = self.offset + count;

        if needed > self.bytes.len() {
            return Err(WeightsError::Truncated {
                needed,
                available: self.bytes.len(),
            });
        }

        let taken = &self.bytes[self.offset..needed];
        self.offset = needed;

        Ok(taken)
    }

    fn u32(&mut self) -> Result<u32, WeightsError> {
        let taken = self.take(4)?;
        let array = [taken[0], taken[1], taken[2], taken[3]];

        Ok(u32::from_le_bytes(array))
    }

    fn halves(&mut self, count: usize) -> Result<Vec<f16>, WeightsError> {
        let taken = self.take(count * 2)?;
        let (pairs, _) = taken.as_chunks::<2>();

        Ok(pairs.iter().map(|pair| f16::from_le_bytes(*pair)).collect())
    }
}
