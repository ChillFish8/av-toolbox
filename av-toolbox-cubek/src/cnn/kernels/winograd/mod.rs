mod kernel;

use cubecl::ir::DeviceProperties;
use cubecl::prelude::*;
#[cfg(any(test, feature = "test-support"))]
use cubecl::server::Handle;
use half::f16;

#[cfg(any(test, feature = "test-support"))]
use self::kernel::input_transform_kernel;
use self::kernel::winograd_kernel;
use super::BodyConvolution;
use super::direct::{
    LOAD_WIDTH,
    SAMPLE_PADDING,
    TILE,
    TILE_VALUES,
    WRITE_WIDTH,
    check_limits,
    offered_plane_width,
};
use crate::cnn::{Activation, CnnError};

/// Lavin and Gray's F(2x2, 3x3) input transform, `Bᵀ`.
const INPUT_TRANSFORM: [[i32; 4]; 4] = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]];

/// Lavin and Gray's F(2x2, 3x3) kernel transform, `G`.
const KERNEL_TRANSFORM: [[f64; 3]; 4] = [
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.0, 0.0, 1.0],
];

/// Lavin and Gray's F(2x2, 3x3) output transform, `Aᵀ`.
const OUTPUT_TRANSFORM: [[i32; 4]; 2] = [[1, 1, 1, 0], [0, 1, -1, -1]];

/// The samples on each side of the input window a 2x2 output tile is computed from.
const WINDOW: u32 = 4;

/// The positions of a transformed 4x4 window, each its own matrix product.
pub(super) const POSITIONS: u32 = WINDOW * WINDOW;

/// The output samples of one 2x2 tile.
const TILE_OUTPUTS: u32 = 4;

/// The units a workgroup of the test-support input transform holds.
#[cfg(any(test, feature = "test-support"))]
const TRANSFORM_UNITS: u32 = 256;

/// `G g Gᵀ` of one 3x3 kernel, in f64.
pub(crate) fn transform_kernel(kernel: &[[f64; 3]; 3]) -> [[f64; 4]; 4] {
    let mut transformed = [[0.0; 4]; 4];

    for (row, transformed_row) in transformed.iter_mut().enumerate() {
        for (column, value) in transformed_row.iter_mut().enumerate() {
            let mut sum = 0.0;

            for (tap_row, kernel_row) in kernel.iter().enumerate() {
                for (tap_column, weight) in kernel_row.iter().enumerate() {
                    let left = KERNEL_TRANSFORM[row][tap_row];
                    let right = KERNEL_TRANSFORM[column][tap_column];
                    sum += left * weight * right;
                }
            }

            *value = sum;
        }
    }

    transformed
}

/// The transformed weights of a 3x3 layer from `in_channels` to `out_channels`, by position, output channel
/// and input channel, each computed in f64 and rounded to f16 once.
///
/// `weights` is laid out by output channel, row, column and input channel.
pub(crate) fn transform_values(weights: &[f32], out_channels: u32, in_channels: u32) -> Vec<f16> {
    let out_channels = out_channels as usize;
    let in_channels = in_channels as usize;
    let positions = POSITIONS as usize;
    let mut transformed = vec![f16::ZERO; positions * out_channels * in_channels];

    for out_channel in 0..out_channels {
        for in_channel in 0..in_channels {
            let mut kernel = [[0.0; 3]; 3];

            for (tap_row, kernel_row) in kernel.iter_mut().enumerate() {
                for (tap_column, weight) in kernel_row.iter_mut().enumerate() {
                    let tap = (out_channel * 3 + tap_row) * 3 + tap_column;
                    *weight = f64::from(weights[tap * in_channels + in_channel]);
                }
            }

            let positioned = transform_kernel(&kernel);

            for (position, value) in positioned.iter().flatten().enumerate() {
                let index = (position * out_channels + out_channel) * in_channels + in_channel;
                transformed[index] = f16::from_f64(*value);
            }
        }
    }

    transformed
}

/// How a Winograd F(2x2, 3x3) body convolution divides its work.
///
/// A workgroup computes a block of `tile_rows` by `tile_columns` 2x2 output tiles for every output channel,
/// one transformed position at a time, transforming its input windows into shared memory. Its planes form
/// a grid of `planes_m` by `planes_n`, splitting the block's groups of 16 tiles and its channels between
/// them. Each plane is a row of `plane_width` units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WinogradVariant {
    pub tile_rows: u32,
    pub tile_columns: u32,
    pub planes_m: u32,
    pub planes_n: u32,
    pub plane_width: u32,
}

/// The tile blocks and plane grids the Winograd convolution is offered in, before the plane width is chosen.
const LAYOUTS: [WinogradVariant; 6] = [
    layout(1, 16, 1, 4),
    layout(1, 16, 1, 6),
    layout(2, 8, 1, 4),
    layout(2, 8, 1, 6),
    layout(1, 32, 2, 4),
    layout(1, 16, 1, 2),
];

const fn layout(tile_rows: u32, tile_columns: u32, planes_m: u32, planes_n: u32) -> WinogradVariant {
    WinogradVariant {
        tile_rows,
        tile_columns,
        planes_m,
        planes_n,
        plane_width: 64,
    }
}

impl WinogradVariant {
    pub fn planes(&self) -> u32 {
        self.planes_m * self.planes_n
    }

    /// The units one workgroup holds.
    pub fn units(&self) -> u32 {
        self.plane_width * self.planes()
    }

    /// The shape the kernel compiles for this variant on maps of `features` channels.
    ///
    /// Gives an error naming the first constraint the variant breaks.
    pub(crate) fn shape(
        &self,
        features: u32,
        activation: Activation,
        residual: bool,
    ) -> Result<WinogradShape, String> {
        let tiles = self.tile_rows * self.tile_columns;
        let tile_groups = tiles / TILE;
        let channel_tiles = features / TILE;

        if tiles == 0 || !tiles.is_multiple_of(TILE) {
            return Err(format!(
                "a block of {} by {} tiles is not a positive multiple of 16 tiles",
                self.tile_rows, self.tile_columns
            ));
        }

        if features == 0 || !features.is_multiple_of(TILE) {
            return Err(format!("{features} features do not split into 16-channel tiles"));
        }

        if self.planes_m == 0 || !tile_groups.is_multiple_of(self.planes_m) {
            return Err(format!(
                "{} planes do not split the block's {tile_groups} groups of 16 tiles",
                self.planes_m
            ));
        }

        if self.planes_n == 0 || !channel_tiles.is_multiple_of(self.planes_n) {
            return Err(format!(
                "{} planes do not split the {channel_tiles} channel tiles",
                self.planes_n
            ));
        }

        if self.plane_width == 0 || !TILE_QUADS.is_multiple_of(self.plane_width) {
            return Err(format!(
                "a plane of {} units does not split a tile's 64 quads",
                self.plane_width
            ));
        }

        Ok(WinogradShape {
            features,
            tile_rows: self.tile_rows,
            tile_columns: self.tile_columns,
            planes_m: self.planes_m,
            planes_n: self.planes_n,
            plane_width: self.plane_width,
            activation,
            residual,
            skip: false,
            bias: true,
        })
    }
}

/// The quads of four values one 16x16 tile holds.
const TILE_QUADS: u32 = TILE_VALUES / WRITE_WIDTH;

/// Everything about a Winograd convolution that is fixed when its kernels compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct WinogradShape {
    features: u32,
    tile_rows: u32,
    tile_columns: u32,
    planes_m: u32,
    planes_n: u32,
    plane_width: u32,
    activation: Activation,
    residual: bool,
    skip: bool,
    bias: bool,
}

impl WinogradShape {
    /// The same shape for a layer without a bias, whose sums start at zero.
    pub(crate) fn without_bias(self) -> Self {
        Self { bias: false, ..self }
    }

    /// The same shape for a layer that adds a skip after its residual.
    pub(crate) fn with_skip(self) -> Self {
        Self { skip: true, ..self }
    }

    fn planes(&self) -> u32 {
        self.planes_m * self.planes_n
    }

    fn tiles(&self) -> u32 {
        self.tile_rows * self.tile_columns
    }

    /// The halves between one tile's transformed channels and the next in shared memory.
    fn tile_stride(&self) -> u32 {
        self.features + SAMPLE_PADDING
    }

    fn groups_per_plane(&self) -> u32 {
        self.tiles() / TILE / self.planes_m
    }

    fn channel_tiles_per_plane(&self) -> u32 {
        self.features / TILE / self.planes_n
    }

    /// The products each plane keeps, one per group of 16 tiles and channel tile.
    fn fragments_per_plane(&self) -> u32 {
        self.groups_per_plane() * self.channel_tiles_per_plane()
    }

    /// The quads of each fragment a unit folds into the output tiles.
    fn quads_per_unit(&self) -> u32 {
        TILE_QUADS / self.plane_width
    }

    /// The quads of transformed input shared memory holds for one position.
    fn transformed_quads(&self) -> u32 {
        self.tiles() * self.tile_stride() / WRITE_WIDTH
    }

    /// The quads of products one plane stores for a position.
    fn plane_scratch_quads(&self) -> u32 {
        self.fragments_per_plane() * TILE_QUADS
    }

    fn scratch_quads(&self) -> u32 {
        self.planes() * self.plane_scratch_quads()
    }

    /// The output quads a unit sums, four for each fragment quad it folds.
    fn sums_per_unit(&self) -> u32 {
        self.fragments_per_plane() * self.quads_per_unit() * TILE_OUTPUTS
    }

    /// The bytes of shared memory the kernel declares.
    pub(crate) fn shared_bytes(&self) -> usize {
        let transformed_bytes = self.transformed_quads() as usize * WRITE_WIDTH as usize * size_of::<f16>();
        let scratch_bytes = self.scratch_quads() as usize * WRITE_WIDTH as usize * size_of::<f32>();

        transformed_bytes + scratch_bytes
    }
}

/// Whether a device with `properties` can run `variant` compiled as `shape`.
fn check_device(
    properties: &DeviceProperties,
    variant: &WinogradVariant,
    shape: &WinogradShape,
) -> Result<(), String> {
    let shared_bytes = shape.shared_bytes();
    let units = variant.units();

    check_limits(properties, variant.plane_width, units, shared_bytes)
}

/// Whether a device with `properties` can run `variant` on maps of `features` channels.
pub(crate) fn check_variant(
    properties: &DeviceProperties,
    variant: &WinogradVariant,
    features: u32,
) -> Result<(), String> {
    let shape = variant.shape(features, Activation::Silu, false)?;

    check_device(properties, variant, &shape)
}

/// The Winograd variants offered on a device with `properties` for maps of `features` channels.
///
/// A variant is offered when the device can run it and its planes are exactly the device's widest subgroup.
pub fn winograd_variants(properties: &DeviceProperties, features: u32) -> Vec<WinogradVariant> {
    let mut offered = Vec::new();

    let Some(plane_width) = offered_plane_width(properties) else {
        return offered;
    };

    for layout in LAYOUTS {
        let variant = WinogradVariant {
            plane_width,
            ..layout
        };
        let runs = check_variant(properties, &variant, features);

        if runs.is_ok() {
            offered.push(variant);
        }
    }

    offered
}

fn no_transformed_weights() -> CnnError {
    let message = "the layer holds no Winograd weights".to_string();

    CnnError::Convolution(message)
}

/// Runs `convolution` as Winograd F(2x2, 3x3), as `variant` describes.
///
/// Gives an error if the layer holds no transformed weights, or the variant cannot run on the device.
pub(crate) fn run_winograd(
    client: &Client,
    variant: WinogradVariant,
    convolution: &BodyConvolution<'_>,
) -> Result<(), CnnError> {
    convolution.check();

    let transformed = convolution.transformed.ok_or_else(no_transformed_weights)?;
    let features = convolution.features;
    let residual = convolution.residual.is_some();
    let properties = client.properties();
    let has_bias = convolution.bias.is_some();
    let shape = variant.shape(features, convolution.activation, residual);
    let shape = shape.map_err(CnnError::Convolution)?;
    let shape = match has_bias {
        true => shape,
        false => shape.without_bias(),
    };
    let shape = match convolution.skip.is_some() {
        true => shape.with_skip(),
        false => shape,
    };
    let fits = check_device(properties, &variant, &shape);
    fits.map_err(CnnError::Convolution)?;

    let width = convolution.width;
    let height = convolution.height;
    let map_values = width as usize * height as usize * features as usize;
    let weight_values = (POSITIONS * features * features) as usize;
    assert!(
        u32::try_from(map_values).is_ok(),
        "the maps hold more values than the kernel can index"
    );

    let column_blocks = width.div_ceil(2).div_ceil(variant.tile_columns);
    let row_blocks = height.div_ceil(2).div_ceil(variant.tile_rows);
    let cubes = CubeCount::Static(column_blocks, row_blocks, 1);
    let unit = CubeDim::new_2d(variant.plane_width, variant.planes());
    // Without a residual the input stands in for it, and is never read through it.
    let residual = convolution.residual.unwrap_or(convolution.input);
    // Without a skip the input stands in for it, and is never read through it.
    let skip = convolution.skip.unwrap_or(convolution.input);
    // Without a bias the input stands in for it, and is never read through it.
    let bias = convolution.bias.unwrap_or(convolution.input);
    let quad = WRITE_WIDTH as usize;
    let load = LOAD_WIDTH as usize;

    // Safety: `check` asserts the input, output, residual and skip hold `width * height * F` halves, the
    // transformed weights `16 * F * F` and the bias `F`, and the assert above keeps every map index inside a
    // u32. Without a bias the kernel never reads the bias argument.
    // - Input reads are guarded to samples inside the frame. Output writes, residual reads and skip reads are
    //   guarded to samples inside the frame, at a channel below `F`.
    // - A weight fragment's last value is at `position * F * F + (channel + 15) * F + chunk * 16 + 15`, below
    //   `16 * F * F`, and its slice ends at most at the weights' end.
    // - Shared-memory indices stay inside the transformed tiles and scratch tiles that `shape` sizes, and
    //   fragments never read the tiles' padding.
    unsafe {
        let input = BufferArg::from_raw_parts(convolution.input.clone(), map_values / quad);
        let weights = BufferArg::from_raw_parts(transformed.clone(), weight_values / load);
        let bias = BufferArg::from_raw_parts(bias.clone(), features as usize / quad);
        let residual = BufferArg::from_raw_parts(residual.clone(), map_values / quad);
        let skip = BufferArg::from_raw_parts(skip.clone(), map_values / quad);
        let output = BufferArg::from_raw_parts(convolution.output.clone(), map_values / quad);

        winograd_kernel::launch_unchecked(
            client, cubes, unit, input, weights, bias, residual, skip, output, width, height, shape,
        );
    }

    Ok(())
}

/// The input windows of f16 maps `features` wide over a `width` by `height` region, transformed as
/// Winograd F(2x2, 3x3) does, by position, tile and channel.
///
/// The tiles lie in rows of half the width, rounded up, and there are half the height's rows, rounded up.
#[cfg(any(test, feature = "test-support"))]
pub fn transform_input(client: &Client, input: &Handle, width: u32, height: u32, features: u32) -> Handle {
    let quads = features / WRITE_WIDTH;
    let map_quads = width as usize * height as usize * quads as usize;
    let columns = width.div_ceil(2);
    let tiles = height.div_ceil(2) * columns;
    let transformed_quads = (tiles * POSITIONS * quads) as usize;
    let max_units = client.properties().hardware.max_units_per_cube;
    let units = TRANSFORM_UNITS.min(max_units);
    let cubes = CubeCount::Static((tiles * quads).div_ceil(units), 1, 1);
    let quad_bytes = WRITE_WIDTH as usize * size_of::<f16>();

    assert!(
        features.is_multiple_of(TILE),
        "the features are not a multiple of 16"
    );
    assert!(
        input.size_in_used() as usize >= map_quads * quad_bytes,
        "the input is smaller than the maps"
    );

    let transformed = client.empty(transformed_quads * quad_bytes);

    // Safety: the assert above sizes the input, and input reads are guarded to samples inside the frame. A
    // unit writes only for a tile below `tiles`, at a position below 16 and a quad below `F / 4`, which the
    // output's `16 * tiles * F / 4` quads hold.
    unsafe {
        let input = BufferArg::from_raw_parts(input.clone(), map_quads);
        let output = BufferArg::from_raw_parts(transformed.clone(), transformed_quads);

        input_transform_kernel::launch_unchecked(
            client,
            cubes,
            CubeDim::new_1d(units),
            input,
            output,
            width,
            height,
            columns,
            tiles,
            features,
        );
    }

    transformed
}
