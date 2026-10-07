use cubecl::cmma;
use cubecl::ir::DeviceProperties;
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use half::f16;

use super::BodyConvolution;
use crate::cnn::convolution::offers_cmma_16_in;
use crate::cnn::{Activation, CnnError};

/// The side of the cooperative-matrix tiles the kernel multiplies.
pub(super) const TILE: u32 = 16;

/// The values one tile holds.
pub(super) const TILE_VALUES: u32 = TILE * TILE;

/// The halves one shared-memory load moves.
pub(super) const LOAD_WIDTH: u32 = 8;

/// The halves of padding after each sample's channels in shared memory.
///
/// It moves neighbouring rows of a fragment onto different banks.
pub(super) const SAMPLE_PADDING: u32 = 8;

/// The values one output write covers.
pub(super) const WRITE_WIDTH: u32 = 4;

/// The 3x3 taps of the convolution.
const TAPS: u32 = 9;

pub(super) type Load = Vector<f16, Const<8>>;
pub(super) type Quad = Vector<f16, Const<4>>;
pub(super) type Sums = Vector<f32, Const<4>>;

/// Where a direct body convolution reads each tap's weights from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightStaging {
    /// Straight from global memory.
    Global,
    /// Through one shared-memory buffer, with a barrier before and after each tap is staged.
    Single,
    /// Through two alternating shared-memory buffers, with one barrier a tap.
    Double,
}

impl WeightStaging {
    fn buffers(self) -> u32 {
        match self {
            Self::Global => 0,
            Self::Single => 1,
            Self::Double => 2,
        }
    }
}

/// How a direct body convolution divides its work.
///
/// A workgroup computes a block of `rows` by `columns` output samples for `F / channel_groups` output
/// channels. Its planes form a grid of `planes_m` by `planes_n`, splitting the block's samples and channels
/// between them. Each plane is a row of `plane_width` units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectVariant {
    pub rows: u32,
    pub columns: u32,
    pub channel_groups: u32,
    pub planes_m: u32,
    pub planes_n: u32,
    pub weights: WeightStaging,
    pub plane_width: u32,
}

/// The fastest variant at both model widths on an AMD R9700.
impl Default for DirectVariant {
    fn default() -> Self {
        Self {
            rows: 4,
            columns: 16,
            channel_groups: 1,
            planes_m: 1,
            planes_n: 2,
            weights: WeightStaging::Global,
            plane_width: 64,
        }
    }
}

/// The block shapes, plane grids and weight staging the direct convolution is offered in, before the plane
/// width is chosen.
const LAYOUTS: [DirectVariant; 8] = [
    layout(4, 16, 1, 2, WeightStaging::Global),
    layout(8, 16, 2, 2, WeightStaging::Global),
    layout(2, 32, 1, 2, WeightStaging::Global),
    layout(4, 16, 2, 2, WeightStaging::Global),
    layout(8, 16, 4, 1, WeightStaging::Global),
    layout(6, 16, 1, 2, WeightStaging::Global),
    layout(8, 16, 4, 2, WeightStaging::Single),
    layout(8, 16, 4, 2, WeightStaging::Double),
];

/// The plane widths the layouts are offered at.
const PLANE_WIDTHS: [u32; 2] = [32, 64];

const fn layout(
    rows: u32,
    columns: u32,
    planes_m: u32,
    planes_n: u32,
    weights: WeightStaging,
) -> DirectVariant {
    DirectVariant {
        rows,
        columns,
        channel_groups: 1,
        planes_m,
        planes_n,
        weights,
        plane_width: 64,
    }
}

impl DirectVariant {
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
    ) -> Result<DirectShape, String> {
        let pixel_tiles = self.rows * self.columns / TILE;
        let groups = self.channel_groups;
        let channels = features.checked_div(groups).unwrap_or(0);
        let channel_tiles = channels / TILE;

        if self.rows == 0 {
            return Err("a block of 0 rows is empty".to_string());
        }

        if self.columns == 0 || !self.columns.is_multiple_of(TILE) {
            return Err(format!(
                "{} columns is not a positive multiple of 16",
                self.columns
            ));
        }

        if channels == 0 || channels * groups != features || !channels.is_multiple_of(TILE) {
            return Err(format!(
                "{groups} groups do not split {features} features into 16-channel tiles"
            ));
        }

        if self.planes_m == 0 || !pixel_tiles.is_multiple_of(self.planes_m) {
            return Err(format!(
                "{} planes do not split the block's {pixel_tiles} pixel tiles",
                self.planes_m
            ));
        }

        if self.planes_n == 0 || !channel_tiles.is_multiple_of(self.planes_n) {
            return Err(format!(
                "{} planes do not split the block's {channel_tiles} channel tiles",
                self.planes_n
            ));
        }

        Ok(DirectShape {
            features,
            rows: self.rows,
            columns: self.columns,
            channels,
            planes_m: self.planes_m,
            planes_n: self.planes_n,
            weights: self.weights,
            plane_width: self.plane_width,
            silu: activation == Activation::Silu,
            residual,
        })
    }
}

/// Everything about a direct convolution that is fixed when its kernel compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DirectShape {
    features: u32,
    rows: u32,
    columns: u32,
    channels: u32,
    planes_m: u32,
    planes_n: u32,
    weights: WeightStaging,
    plane_width: u32,
    silu: bool,
    residual: bool,
}

impl DirectShape {
    fn planes(&self) -> u32 {
        self.planes_m * self.planes_n
    }

    fn units(&self) -> u32 {
        self.plane_width * self.planes()
    }

    /// The halves between one sample and the next in shared memory.
    fn sample_stride(&self) -> u32 {
        self.features + SAMPLE_PADDING
    }

    /// The samples of the block with its one-sample border, as rows and columns.
    fn bordered(&self) -> (u32, u32) {
        (self.rows + 2, self.columns + 2)
    }

    fn block_loads(&self) -> u32 {
        let (rows, columns) = self.bordered();

        rows * columns * self.sample_stride() / LOAD_WIDTH
    }

    /// The loads one tap's weights for the workgroup's channels take in shared memory.
    fn tap_loads(&self) -> u32 {
        self.channels * self.sample_stride() / LOAD_WIDTH
    }

    fn pixel_tiles_per_plane(&self) -> u32 {
        self.rows * self.columns / TILE / self.planes_m
    }

    fn channel_tiles_per_plane(&self) -> u32 {
        self.channels / TILE / self.planes_n
    }

    /// The loads the staged weights take in shared memory, at least one.
    fn staged_loads(&self) -> u32 {
        let staged = self.weights.buffers() * self.tap_loads();

        staged.max(1)
    }

    /// The bytes of shared memory the kernel declares.
    pub(crate) fn shared_bytes(&self) -> usize {
        let half_loads = self.block_loads() + self.staged_loads();
        let half_bytes = half_loads as usize * LOAD_WIDTH as usize * size_of::<f16>();
        let scratch_bytes = (self.planes() * TILE_VALUES) as usize * size_of::<f32>();

        half_bytes + scratch_bytes
    }
}

/// The plane's place in the workgroup's grid of planes.
#[derive(CubeType, Clone, Copy)]
struct PlaneSpot {
    /// The scalar index of the plane's scratch tile in shared memory, in groups of four.
    scratch: u32,
    /// The first of the plane's pixel tiles in the block.
    first_pixel_tile: u32,
    /// The first of the plane's channel tiles in the workgroup's channels.
    first_channel_tile: u32,
}

#[cube]
fn plane_spot(#[comptime] shape: DirectShape) -> PlaneSpot {
    let plane = UNIT_POS_Y;
    let planes_m = comptime!(shape.planes_m);
    let pixel_tiles = comptime!(shape.pixel_tiles_per_plane());
    let channel_tiles = comptime!(shape.channel_tiles_per_plane());

    PlaneSpot {
        scratch: plane * (TILE_VALUES / WRITE_WIDTH),
        first_pixel_tile: (plane % planes_m) * pixel_tiles,
        first_channel_tile: (plane / planes_m) * channel_tiles,
    }
}

/// Loads the block of input samples the workgroup convolves, with its one-sample border, into `block`.
///
/// Samples outside the frame load as zero, the network's padding. Each unit issues all of its global loads
/// before it stores any, so the loads are in flight together. Every load reads an address inside the frame,
/// and the samples outside it are swapped for zero afterwards, so the loads need no branch.
#[cube]
fn load_block(
    input: &Array<Load>,
    block: &mut SharedMemory<Load>,
    block_x: u32,
    block_y: u32,
    width: u32,
    height: u32,
    #[comptime] shape: DirectShape,
) {
    let sample_loads = comptime!(shape.features / LOAD_WIDTH);
    let stride_loads = comptime!(shape.sample_stride() / LOAD_WIDTH);
    let bordered_columns = comptime!(shape.columns + 2);
    let total = comptime!((shape.rows + 2) * (shape.columns + 2) * shape.features / LOAD_WIDTH);
    let units = comptime!(shape.units());
    let rounds = comptime!(total.div_ceil(units) as usize);
    let last = comptime!(total - 1);
    let zero = Load::new(f16::new(0.0f32));
    let mut values = Sequence::<Load>::new();

    #[unroll]
    for round in 0..rounds {
        // Units past the block's end repeat its last load, and never store it.
        let index = min(UNIT_POS + comptime!(round as u32 * units), last);
        let sample = index / sample_loads;
        let part = index % sample_loads;
        let column = block_x + sample % bordered_columns;
        let row = block_y + sample / bordered_columns;
        // The border starts one sample before the block, so `column` and `row` are one past the frame's.
        let inside = column >= 1 && row >= 1 && column <= width && row <= height;
        let row_from_top = max(row, 1u32);
        let column_from_left = max(column, 1u32);
        let frame_row = min(row_from_top, height) - 1;
        let frame_column = min(column_from_left, width) - 1;
        let source = (frame_row * width + frame_column) * sample_loads + part;
        let loaded = input[source as usize];

        values.push(select(inside, loaded, zero));
    }

    #[unroll]
    for round in 0..rounds {
        let index = UNIT_POS + comptime!(round as u32 * units);

        if index < total {
            let sample = index / sample_loads;
            let part = index % sample_loads;

            block[(sample * stride_loads + part) as usize] = *values.index(round);
        }
    }
}

/// Stages tap `tap`'s weights for the workgroup's output channels into `taps` at `buffer`.
#[cube]
fn stage_tap(
    weights: &Array<Load>,
    taps: &mut SharedMemory<Load>,
    buffer: u32,
    tap: u32,
    first_channel: u32,
    #[comptime] shape: DirectShape,
) {
    let sample_loads = comptime!(shape.features / LOAD_WIDTH);
    let stride_loads = comptime!(shape.sample_stride() / LOAD_WIDTH);
    let total = comptime!(shape.channels * shape.features / LOAD_WIDTH);

    let mut index = UNIT_POS;
    while index < total {
        let channel = index / sample_loads;
        let part = index % sample_loads;
        let source = ((first_channel + channel) * TAPS + tap) * sample_loads + part;

        taps[(buffer + channel * stride_loads + part) as usize] = weights[source as usize];
        index += CUBE_DIM;
    }
}

/// A 16x16x16 cooperative-matrix fragment, filled before it is read.
#[cube]
pub(super) fn fragment<E: Scalar>(
    #[comptime] ident: cmma::MatrixIdent,
    layout: cmma::MatrixLayout,
) -> cmma::Matrix<E> {
    // Safety: every fragment is loaded before the kernel reads it.
    unsafe { cmma::Matrix::<E>::uninitialized(ident, TILE as usize, TILE as usize, TILE as usize, layout) }
}

/// Fills the plane's accumulators with the bias of their channels, through its scratch tile.
#[cube]
fn bias_accumulators(
    bias: &Array<Quad>,
    scratch: &mut SharedMemory<Sums>,
    spot: PlaneSpot,
    first_channel: u32,
    #[comptime] shape: DirectShape,
) -> Sequence<cmma::Matrix<f32>> {
    let pixel_tiles = comptime!(shape.pixel_tiles_per_plane() as usize);
    let channel_tiles = comptime!(shape.channel_tiles_per_plane() as usize);
    let tile_quads = comptime!(TILE_VALUES / WRITE_WIDTH);
    let row_quads = comptime!(TILE / WRITE_WIDTH);
    let mut accumulators = Sequence::<cmma::Matrix<f32>>::new();

    #[unroll]
    for _ in 0..pixel_tiles * channel_tiles {
        let matrix = fragment::<f32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
        accumulators.push(matrix);
    }

    #[unroll]
    for channel_tile in 0..channel_tiles {
        let tile_channel = first_channel + (spot.first_channel_tile + channel_tile as u32) * TILE;

        // Every plane runs the same steps, so the whole workgroup reaches each barrier.
        sync_cube();

        let mut index = UNIT_POS_X;
        while index < tile_quads {
            let quad = index % row_quads;
            let value = bias[((tile_channel + quad * WRITE_WIDTH) / WRITE_WIDTH) as usize];

            scratch[(spot.scratch + index) as usize] = Sums::cast_from(value);
            index += CUBE_DIM_X;
        }

        sync_cube();

        let start = spot.scratch as usize;
        let tile = scratch.slice(start, start + tile_quads as usize);

        #[unroll]
        for pixel_tile in 0..pixel_tiles {
            let accumulator = accumulators.index(comptime!(pixel_tile * channel_tiles + channel_tile));
            cmma::load_with_layout(accumulator, &tile, TILE, cmma::MatrixLayout::RowMajor);
        }
    }

    accumulators
}

/// Convolves one block of a body layer on cooperative matrices, with the bias, SiLU and the residual applied
/// as it writes.
///
/// Each workgroup reads its block with the one-sample border into shared memory once, and reuses it for all
/// nine taps. Each plane keeps its accumulators in f32 from the bias to the write.
#[cube(launch_unchecked)]
fn direct_kernel(
    input: &Array<Load>,
    weights: &Array<Load>,
    bias: &Array<Quad>,
    residual: &Array<Quad>,
    output: &mut Array<Quad>,
    width: u32,
    height: u32,
    #[comptime] shape: DirectShape,
) {
    let block_x = CUBE_POS_X * comptime!(shape.columns);
    let block_y = CUBE_POS_Y * comptime!(shape.rows);
    let first_channel = CUBE_POS_Z * comptime!(shape.channels);
    let spot = plane_spot(shape);

    let mut block = SharedMemory::<Load>::new(comptime!(shape.block_loads() as usize));
    let mut taps = SharedMemory::<Load>::new(comptime!(shape.staged_loads() as usize));
    let mut scratch =
        SharedMemory::<Sums>::new(comptime!((shape.planes() * TILE_VALUES / WRITE_WIDTH) as usize));

    load_block(input, &mut block, block_x, block_y, width, height, shape);
    // Every plane reads samples another plane loaded.
    sync_cube();

    let accumulators = bias_accumulators(bias, &mut scratch, spot, first_channel, shape);

    multiply_taps(
        weights,
        &block,
        &mut taps,
        &accumulators,
        spot,
        first_channel,
        shape,
    );
    write_out(
        &accumulators,
        &mut scratch,
        residual,
        output,
        spot,
        block_x,
        block_y,
        first_channel,
        width,
        height,
        shape,
    );
}

/// Adds every tap's products into the plane's accumulators.
#[cube]
fn multiply_taps(
    weights: &Array<Load>,
    block: &SharedMemory<Load>,
    taps: &mut SharedMemory<Load>,
    accumulators: &Sequence<cmma::Matrix<f32>>,
    spot: PlaneSpot,
    first_channel: u32,
    #[comptime] shape: DirectShape,
) {
    let pixel_tiles = comptime!(shape.pixel_tiles_per_plane() as usize);
    let channel_tiles = comptime!(shape.channel_tiles_per_plane() as usize);
    let chunks = comptime!((shape.features / TILE) as usize);
    let row_tiles = comptime!(shape.columns / TILE);
    let stride = comptime!(shape.sample_stride());
    let bordered_columns = comptime!(shape.columns + 2);
    let tap_loads = comptime!(shape.tap_loads());
    let fragment_loads = comptime!(((TILE - 1) * shape.sample_stride() + TILE) / LOAD_WIDTH);
    let features = comptime!(shape.features);
    let global_loads = comptime!(((TILE - 1) * shape.features * TAPS + TILE) / LOAD_WIDTH);

    for tap in 0..TAPS {
        let tap_row = tap / 3;
        let tap_column = tap % 3;
        let mut buffer = 0u32;

        if comptime!(shape.weights == WeightStaging::Double) {
            buffer = (tap % 2) * tap_loads;
            stage_tap(weights, taps, buffer, tap, first_channel, shape);
            sync_cube();
        }

        if comptime!(shape.weights == WeightStaging::Single) {
            sync_cube();
            stage_tap(weights, taps, buffer, tap, first_channel, shape);
            sync_cube();
        }

        #[unroll]
        for chunk in 0..chunks {
            let mut lhs = Sequence::<cmma::Matrix<f16>>::new();

            #[unroll]
            for pixel_tile in 0..pixel_tiles {
                let tile = spot.first_pixel_tile + pixel_tile as u32;
                let block_row = tile / row_tiles + tap_row;
                let block_column = (tile % row_tiles) * TILE + tap_column;
                let start = ((block_row * bordered_columns + block_column) * stride + chunk as u32 * TILE)
                    / LOAD_WIDTH;
                let rows = block.slice(start as usize, (start + fragment_loads) as usize);
                let pixels = fragment(cmma::MatrixIdent::A, cmma::MatrixLayout::RowMajor);
                cmma::load(&pixels, &rows, stride);
                lhs.push(pixels);
            }

            #[unroll]
            for channel_tile in 0..channel_tiles {
                let channel = (spot.first_channel_tile + channel_tile as u32) * TILE;
                let rhs = fragment(cmma::MatrixIdent::B, cmma::MatrixLayout::ColMajor);

                if comptime!(shape.weights == WeightStaging::Global) {
                    let tap_start = ((first_channel + channel) * TAPS + tap) * features;
                    let start = (tap_start + chunk as u32 * TILE) / LOAD_WIDTH;
                    let columns = weights.slice(start as usize, (start + global_loads) as usize);
                    cmma::load(&rhs, &columns, features * TAPS);
                } else {
                    let start = buffer + (channel * stride + chunk as u32 * TILE) / LOAD_WIDTH;
                    let columns = taps.slice(start as usize, (start + fragment_loads) as usize);
                    cmma::load(&rhs, &columns, stride);
                }

                #[unroll]
                for pixel_tile in 0..pixel_tiles {
                    let accumulator =
                        accumulators.index(comptime!(pixel_tile * channel_tiles + channel_tile));
                    cmma::execute::<f16, f16, f32, f32>(
                        lhs.index(pixel_tile),
                        &rhs,
                        accumulator,
                        accumulator,
                    );
                }
            }
        }
    }
}

/// Applies SiLU and the residual to the plane's accumulators, and writes them to `output` in f16.
///
/// Each accumulator passes through the plane's scratch tile. Samples outside the frame are not written.
#[cube]
fn write_out(
    accumulators: &Sequence<cmma::Matrix<f32>>,
    scratch: &mut SharedMemory<Sums>,
    residual: &Array<Quad>,
    output: &mut Array<Quad>,
    spot: PlaneSpot,
    block_x: u32,
    block_y: u32,
    first_channel: u32,
    width: u32,
    height: u32,
    #[comptime] shape: DirectShape,
) {
    let pixel_tiles = comptime!(shape.pixel_tiles_per_plane() as usize);
    let channel_tiles = comptime!(shape.channel_tiles_per_plane() as usize);
    let row_tiles = comptime!(shape.columns / TILE);
    let tile_quads = comptime!(TILE_VALUES / WRITE_WIDTH);
    let row_quads = comptime!(TILE / WRITE_WIDTH);
    let sample_quads = comptime!(shape.features / WRITE_WIDTH);

    #[unroll]
    for pixel_tile in 0..pixel_tiles {
        let tile = spot.first_pixel_tile + pixel_tile as u32;
        let row = block_y + tile / row_tiles;
        let first_column = block_x + (tile % row_tiles) * TILE;

        #[unroll]
        for channel_tile in 0..channel_tiles {
            let accumulator = accumulators.index(comptime!(pixel_tile * channel_tiles + channel_tile));
            let tile_channel = first_channel + (spot.first_channel_tile + channel_tile as u32) * TILE;
            let start = spot.scratch as usize;

            sync_cube();

            // A plane may hold several subgroups that each computed the same accumulator, so only the
            // first stores it. The condition is uniform within each subgroup.
            if UNIT_POS_X < PLANE_DIM {
                let mut tile_slice = scratch.slice_mut(start, start + tile_quads as usize);
                cmma::store(&mut tile_slice, accumulator, TILE, cmma::MatrixLayout::RowMajor);
            }

            sync_cube();

            let mut index = UNIT_POS_X;
            while index < tile_quads {
                let column = first_column + index / row_quads;
                let quad = index % row_quads;

                if row < height && column < width {
                    let position = (row * width + column) * sample_quads + tile_channel / WRITE_WIDTH + quad;
                    let sum = scratch[(spot.scratch + index) as usize];
                    let finished = finish_sums(sum, residual, position, shape.silu, shape.residual);

                    output[position as usize] = Quad::cast_from(finished);
                }

                index += CUBE_DIM_X;
            }
        }
    }
}

/// `sum` through SiLU when `silu` is set, plus the residual quad at `position` when `residual_added` is set.
#[cube]
pub(super) fn finish_sums(
    sum: Sums,
    residual: &Array<Quad>,
    position: u32,
    #[comptime] silu: bool,
    #[comptime] residual_added: bool,
) -> Sums {
    let one = Sums::new(1.0);
    let mut finished = sum;

    if comptime!(silu) {
        finished = finished / (one + Sums::exp(-finished));
    }

    if comptime!(residual_added) {
        finished += Sums::cast_from(residual[position as usize]);
    }

    finished
}

/// Whether a device with `properties` can run `variant` compiled as `shape`.
pub(crate) fn check_device(
    properties: &DeviceProperties,
    variant: &DirectVariant,
    shape: &DirectShape,
) -> Result<(), String> {
    let shared_bytes = shape.shared_bytes();
    let units = variant.units();

    check_limits(properties, variant.plane_width, units, shared_bytes)
}

/// Whether a device with `properties` can run a workgroup of `units` units in planes `plane_width` wide,
/// declaring `shared_bytes` of shared memory.
///
/// The device must offer the 16x16x16 f16 cooperative matrix accumulating in f32. A plane narrower than the
/// device's widest subgroup is refused, since the driver may run a subgroup across two planes. A wider plane
/// holds whole subgroups that repeat each other's work.
pub(super) fn check_limits(
    properties: &DeviceProperties,
    plane_width: u32,
    units: u32,
    shared_bytes: usize,
) -> Result<(), String> {
    let offers_matrix = offers_cmma_16_in(&properties.features.matmul.cmma);
    let hardware = &properties.hardware;
    let widest = hardware.plane_size_max;

    if !offers_matrix {
        return Err("the device offers no 16x16x16 f16 cooperative matrix".to_string());
    }

    if widest == 0 || !plane_width.is_multiple_of(widest) {
        return Err(format!(
            "a plane of {plane_width} units does not hold whole subgroups of up to {widest}"
        ));
    }

    if shared_bytes > hardware.max_shared_memory_size {
        let limit = hardware.max_shared_memory_size;
        return Err(format!(
            "{shared_bytes} bytes of shared memory exceed the device's {limit}"
        ));
    }

    if units > hardware.max_units_per_cube {
        let limit = hardware.max_units_per_cube;
        return Err(format!("{units} units a workgroup exceed the device's {limit}"));
    }

    Ok(())
}

/// The plane width candidates are offered at on a device with `properties`, its widest subgroup, or `None`
/// when no layout is offered at that width.
pub fn offered_plane_width(properties: &DeviceProperties) -> Option<u32> {
    let plane_width = properties.hardware.plane_size_max;

    PLANE_WIDTHS.contains(&plane_width).then_some(plane_width)
}

/// Whether a device with `properties` can run `variant` on maps of `features` channels.
pub(crate) fn check_variant(
    properties: &DeviceProperties,
    variant: &DirectVariant,
    features: u32,
) -> Result<(), String> {
    let shape = variant.shape(features, Activation::Silu, false)?;

    check_device(properties, variant, &shape)
}

/// The direct variants offered on a device with `properties` for maps of `features` channels.
///
/// A variant is offered when the device can run it and its planes are exactly the device's widest subgroup,
/// so no subgroup repeats another's work when the driver runs that width.
pub fn direct_variants(properties: &DeviceProperties, features: u32) -> Vec<DirectVariant> {
    let mut offered = Vec::new();

    let Some(plane_width) = offered_plane_width(properties) else {
        return offered;
    };

    for layout in LAYOUTS {
        let variant = DirectVariant {
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

fn missing_bias() -> CnnError {
    CnnError::Convolution("the kernel runs only layers with a bias".to_string())
}

/// Runs `convolution` as `variant` describes.
pub(crate) fn run_direct<R: Runtime>(
    client: &ComputeClient<R>,
    variant: DirectVariant,
    convolution: &BodyConvolution<'_>,
) -> Result<(), CnnError> {
    convolution.check();

    let bias = convolution.bias.ok_or_else(missing_bias)?;
    let features = convolution.features;
    let residual = convolution.residual.is_some();
    let properties = client.properties();
    let shape = variant.shape(features, convolution.activation, residual);
    let shape = shape.map_err(CnnError::Convolution)?;
    let fits = check_device(properties, &variant, &shape);
    fits.map_err(CnnError::Convolution)?;

    let width = convolution.width;
    let height = convolution.height;
    let map_values = width as usize * height as usize * features as usize;
    let weight_values = (features * TAPS * features) as usize;
    assert!(
        u32::try_from(map_values).is_ok(),
        "the maps hold more values than the kernel can index"
    );

    let column_blocks = width.div_ceil(variant.columns);
    let row_blocks = height.div_ceil(variant.rows);
    let channel_groups = variant.channel_groups;
    let cubes = CubeCount::Static(column_blocks, row_blocks, channel_groups);
    let unit = CubeDim::new_2d(variant.plane_width, variant.planes());
    // Without a residual the input stands in for it, and is never read through it.
    let residual = convolution.residual.unwrap_or(convolution.input);

    // Safety: `check` asserts the input, output and residual hold `width * height * F` halves, the weights
    // `F * 9 * F` and the bias `F`, and the assert above keeps every map index inside a u32.
    // - Input reads are clamped to samples inside the frame, which holds at least one sample whenever a
    //   workgroup runs, since the grid is empty when `width` or `height` is 0. Output writes and residual
    //   reads are guarded to samples inside the frame, at a channel below `F`.
    // - Bias reads take the workgroup's channels, which `shape` keeps below `F`.
    // - A weight fragment's last value is at `((channel + 15) * 9 + tap) * F + chunk * 16 + 15`, below
    //   `9 * F * F`, and its slice ends at most at the weights' end.
    // - Shared-memory indices stay inside the bordered block, the staged taps and the scratch tiles that
    //   `shape` sizes. Fragments read only the channel halves each sample's load wrote, never its padding.
    unsafe {
        let input = ArrayArg::from_raw_parts(convolution.input.clone(), map_values / LOAD_WIDTH as usize);
        let weights =
            ArrayArg::from_raw_parts(convolution.weights.clone(), weight_values / LOAD_WIDTH as usize);
        let bias = ArrayArg::from_raw_parts(bias.clone(), (features / WRITE_WIDTH) as usize);
        let residual = ArrayArg::from_raw_parts(residual.clone(), map_values / WRITE_WIDTH as usize);
        let output = ArrayArg::from_raw_parts(convolution.output.clone(), map_values / WRITE_WIDTH as usize);

        direct_kernel::launch_unchecked::<R>(
            client, cubes, unit, input, weights, bias, residual, output, width, height, shape,
        );
    }

    Ok(())
}
