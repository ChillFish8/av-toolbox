use cubecl::cmma;
use cubecl::ir::DeviceProperties;
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

use super::add::spread_cubes;
use super::direct::{
    LOAD_WIDTH,
    Load,
    Quad,
    SAMPLE_PADDING,
    Sums,
    TILE,
    TILE_VALUES,
    WRITE_WIDTH,
    check_limits,
    fragment,
    offered_plane_width,
};
use super::same_buffer;
use crate::cnn::CnnError;

/// The output columns a down block spans, one pixel tile.
const DOWN_COLUMNS: u32 = TILE;

/// Which way a strided layer resamples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stride {
    /// A 2x2 stride 2 convolution, halving the map and doubling its channels.
    Down,
    /// A 2x2 stride 2 transposed convolution, doubling the map and halving its channels.
    Up,
}

/// How a down or up kernel divides its work.
///
/// A workgroup computes `pixel_tiles` tiles of 16 samples for `channel_tiles` tiles of 16 output columns. A
/// down tile is 16 output samples of one row, and an up tile is 16 input samples in row-major order. Its
/// planes form a grid of `planes_m` by `planes_n`, splitting the tiles and the columns between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StridedVariant {
    pub pixel_tiles: u32,
    pub channel_tiles: u32,
    pub planes_m: u32,
    pub planes_n: u32,
    pub plane_width: u32,
}

/// Everything about a strided kernel that is fixed when it compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StridedShape {
    stride: Stride,
    /// The channels of each staged input sample.
    in_channels: u32,
    /// The columns of the weight matrix, twice the input's channels for both strides.
    matrix_columns: u32,
    pixel_tiles: u32,
    channel_tiles: u32,
    planes_m: u32,
    planes_n: u32,
    plane_width: u32,
}

impl StridedShape {
    fn planes(&self) -> u32 {
        self.planes_m * self.planes_n
    }

    fn units(&self) -> u32 {
        self.plane_width * self.planes()
    }

    fn sample_stride(&self) -> u32 {
        self.in_channels + SAMPLE_PADDING
    }

    /// The input samples a workgroup stages, `2R` rows of 32 for a down block and `16P` for an up block.
    fn staged_samples(&self) -> u32 {
        match self.stride {
            Stride::Down => 2 * self.pixel_tiles * 2 * DOWN_COLUMNS,
            Stride::Up => self.pixel_tiles * TILE,
        }
    }

    fn staged_loads(&self) -> u32 {
        self.staged_samples() * self.sample_stride() / LOAD_WIDTH
    }

    fn taps(&self) -> u32 {
        match self.stride {
            Stride::Down => 4,
            Stride::Up => 1,
        }
    }

    /// The f16 values between one row of a pixel fragment and the next, every other staged sample for a down
    /// layer.
    fn row_step(&self) -> u32 {
        match self.stride {
            Stride::Down => 2 * self.sample_stride(),
            Stride::Up => self.sample_stride(),
        }
    }

    fn pixel_tiles_per_plane(&self) -> u32 {
        self.pixel_tiles / self.planes_m
    }

    fn channel_tiles_per_plane(&self) -> u32 {
        self.channel_tiles / self.planes_n
    }

    /// The bytes of shared memory the kernel declares.
    pub(crate) fn shared_bytes(&self) -> usize {
        let staged_bytes = (self.staged_loads() * LOAD_WIDTH) as usize * size_of::<f16>();
        let scratch_bytes = (self.planes() * TILE_VALUES) as usize * size_of::<f32>();

        staged_bytes + scratch_bytes
    }
}

/// The columns of a strided layer's weight matrix when its input holds `channels` channels.
///
/// A down layer from `C` channels has `[4C, 2C]` weights, and an up layer to `C` channels reads `2C` and has
/// `[2C, 4C]` weights, so both have twice the input's channels as columns.
fn matrix_columns(channels: u32) -> u32 {
    2 * channels
}

/// The shape `variant` compiles to for a `stride` layer reading `channels` channels.
///
/// Gives an error naming the first constraint the variant breaks.
pub(crate) fn strided_shape(
    variant: StridedVariant,
    stride: Stride,
    channels: u32,
) -> Result<StridedShape, String> {
    let columns = matrix_columns(channels);
    let column_tiles = columns / TILE;

    if channels == 0 || !channels.is_multiple_of(TILE) {
        return Err(format!("{channels} channels is not a positive multiple of 16"));
    }

    if variant.channel_tiles == 0 || !column_tiles.is_multiple_of(variant.channel_tiles) {
        return Err(format!(
            "{} channel tiles do not split the {column_tiles} column tiles",
            variant.channel_tiles
        ));
    }

    if variant.planes_m == 0 || !variant.pixel_tiles.is_multiple_of(variant.planes_m) {
        return Err(format!(
            "{} planes do not split {} pixel tiles",
            variant.planes_m, variant.pixel_tiles
        ));
    }

    if variant.planes_n == 0 || !variant.channel_tiles.is_multiple_of(variant.planes_n) {
        return Err(format!(
            "{} planes do not split {} channel tiles",
            variant.planes_n, variant.channel_tiles
        ));
    }

    Ok(StridedShape {
        stride,
        in_channels: channels,
        matrix_columns: columns,
        pixel_tiles: variant.pixel_tiles,
        channel_tiles: variant.channel_tiles,
        planes_m: variant.planes_m,
        planes_n: variant.planes_n,
        plane_width: variant.plane_width,
    })
}

/// The fixed variant of a `stride` layer reading `channels` channels, before the plane width is chosen.
///
/// Measured on an AMD R9700.
fn fixed_layout(stride: Stride, channels: u32) -> StridedVariant {
    let pixel_tiles = match (stride, channels) {
        (Stride::Down, 256) => 1,
        (Stride::Down, _) => 2,
        (Stride::Up, 256) => 4,
        (Stride::Up, _) => 2,
    };
    // A down layer from 64 channels has only 8 column tiles.
    let channel_tiles = match (stride, channels) {
        (Stride::Down, 64) => 8,
        _ => 16,
    };

    StridedVariant {
        pixel_tiles,
        channel_tiles,
        planes_m: 1,
        planes_n: 4,
        plane_width: 64,
    }
}

/// The variant a `stride` layer reading `channels` channels runs with on a device with `properties`, or
/// `None` when the device cannot run it.
pub fn strided_variant(
    properties: &DeviceProperties,
    stride: Stride,
    channels: u32,
) -> Option<StridedVariant> {
    let plane_width = offered_plane_width(properties)?;
    let layout = fixed_layout(stride, channels);
    let variant = StridedVariant {
        plane_width,
        ..layout
    };
    let shape = strided_shape(variant, stride, channels).ok()?;
    let fits = check_limits(properties, plane_width, shape.units(), shape.shared_bytes());

    fits.ok().map(|_| variant)
}

/// The plane's place in the workgroup's grid of planes.
#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct PlaneSpot {
    /// The quad index of the plane's scratch tile in shared memory.
    scratch: u32,
    first_pixel_tile: u32,
    /// The first of the plane's column tiles in the whole matrix.
    first_column_tile: u32,
}

#[cube]
fn plane_spot(#[comptime] shape: StridedShape) -> PlaneSpot {
    let plane = UNIT_POS_Y;
    let planes_m = comptime!(shape.planes_m);
    let pixel_tiles = comptime!(shape.pixel_tiles_per_plane());
    let channel_tiles = comptime!(shape.channel_tiles_per_plane());
    let workgroup_tiles = CUBE_POS_Z * comptime!(shape.channel_tiles);

    PlaneSpot {
        scratch: plane * (TILE_VALUES / WRITE_WIDTH),
        first_pixel_tile: (plane % planes_m) * pixel_tiles,
        first_column_tile: workgroup_tiles + (plane / planes_m) * channel_tiles,
    }
}

/// The first input sample of the workgroup's up block, numbering the blocks row by row over the grid's first
/// two axes.
#[cube]
fn first_up_sample(#[comptime] shape: StridedShape) -> u32 {
    let block = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;

    block * comptime!(shape.pixel_tiles * TILE)
}

/// Stages the workgroup's input samples into `staged`, clamping each read to a sample inside the map.
///
/// A down block stages `2R` rows of 32 samples from twice its output position. An up block stages `16P`
/// samples in row-major order.
#[cube]
fn stage_samples(
    input: &[Load],
    staged: &mut Shared<[Load]>,
    width: u32,
    height: u32,
    #[comptime] shape: StridedShape,
) {
    let sample_loads = comptime!(shape.in_channels / LOAD_WIDTH);
    let stride_loads = comptime!(shape.sample_stride() / LOAD_WIDTH);
    let total = comptime!(shape.staged_samples() * shape.in_channels / LOAD_WIDTH);
    let last_sample = width * height - 1;

    let mut index = UNIT_POS;
    while index < total {
        let sample = index / sample_loads;
        let part = index % sample_loads;
        let source_sample = if comptime!(shape.stride == Stride::Down) {
            let staged_columns = comptime!(2 * DOWN_COLUMNS);
            let row = CUBE_POS_Y * comptime!(2 * shape.pixel_tiles) + sample / staged_columns;
            let column = CUBE_POS_X * staged_columns + sample % staged_columns;
            let frame_row = min(row, height - 1);
            let frame_column = min(column, width - 1);
            frame_row * width + frame_column
        } else {
            let first_sample = first_up_sample(shape);
            min(first_sample + sample, last_sample)
        };

        let source = source_sample * sample_loads + part;
        staged[(sample * stride_loads + part) as usize] = input[source as usize];
        index += CUBE_DIM;
    }
}

/// Multiplies the staged samples by the weight matrix into the plane's accumulators.
#[cube]
fn multiply(
    staged: &Shared<[Load]>,
    weights: &[Load],
    accumulators: &Sequence<cmma::Matrix<f32>>,
    spot: PlaneSpot,
    #[comptime] shape: StridedShape,
) {
    let pixel_tiles = comptime!(shape.pixel_tiles_per_plane() as usize);
    let channel_tiles = comptime!(shape.channel_tiles_per_plane() as usize);
    let chunks = comptime!((shape.in_channels / TILE) as usize);
    let taps = comptime!(shape.taps() as usize);
    let stride = comptime!(shape.sample_stride());
    let columns = comptime!(shape.matrix_columns);
    let row_step = comptime!(shape.row_step());
    let fragment_loads = comptime!(((TILE - 1) * shape.row_step() + TILE) / LOAD_WIDTH);
    let weight_loads = comptime!(((TILE - 1) * shape.matrix_columns + TILE) / LOAD_WIDTH);

    #[unroll]
    for tap in 0..taps {
        let tap_row = comptime!(tap as u32 / 2);
        let tap_column = comptime!(tap as u32 % 2);

        #[unroll]
        for chunk in 0..chunks {
            let mut lhs = Sequence::<cmma::Matrix<f16>>::new();

            #[unroll]
            for pixel_tile in 0..pixel_tiles {
                let tile = spot.first_pixel_tile + pixel_tile as u32;
                let first_sample = if comptime!(shape.stride == Stride::Down) {
                    let staged_columns = comptime!(2 * DOWN_COLUMNS);
                    (tile * 2 + tap_row) * staged_columns + tap_column
                } else {
                    tile * TILE
                };

                let start = (first_sample * stride + chunk as u32 * TILE) / LOAD_WIDTH;
                let rows = staged.slice(start as usize, (start + fragment_loads) as usize);
                let mut pixels = fragment(cmma::MatrixIdent::A, cmma::MatrixLayout::RowMajor);
                cmma::load(&mut pixels, rows, row_step);
                lhs.push(pixels);
            }

            let depth_row = comptime!(tap as u32 * shape.in_channels + chunk as u32 * TILE);

            #[unroll]
            for channel_tile in 0..channel_tiles {
                let column = (spot.first_column_tile + channel_tile as u32) * TILE;
                let start = (depth_row * columns + column) / LOAD_WIDTH;
                let matrix_rows = weights.slice(start as usize, (start + weight_loads) as usize);
                let mut rhs = fragment(cmma::MatrixIdent::B, cmma::MatrixLayout::RowMajor);
                cmma::load(&mut rhs, matrix_rows, columns);

                #[unroll]
                for pixel_tile in 0..pixel_tiles {
                    let accumulator =
                        accumulators.index(comptime!(pixel_tile * channel_tiles + channel_tile));
                    cmma::execute::<f16, f16, f32, f32, cmma::Plane>(
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

/// Where a value lands in the output, in quads, and whether that sample lies inside the output map.
#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct OutputQuad {
    position: u32,
    inside: bool,
}

/// Where row `row_in_tile` of pixel tile `tile`, at matrix column `column`, lands in the output.
///
/// A down column is the output channel. An up column `(dy * 2 + dx) * C + c` is channel `c` of output sample
/// `(2y + dy, 2x + dx)`.
#[cube]
fn output_quad(
    tile: u32,
    row_in_tile: u32,
    column: u32,
    width: u32,
    height: u32,
    #[comptime] shape: StridedShape,
) -> OutputQuad {
    if comptime!(shape.stride == Stride::Down) {
        let out_width = width / 2;
        let out_height = height / 2;
        let out_row = CUBE_POS_Y * comptime!(shape.pixel_tiles) + tile;
        let out_column = CUBE_POS_X * DOWN_COLUMNS + row_in_tile;
        let sample_quads = comptime!(shape.matrix_columns / WRITE_WIDTH);
        let inside = out_row < out_height && out_column < out_width;
        let position = (out_row * out_width + out_column) * sample_quads + column / WRITE_WIDTH;

        OutputQuad { position, inside }
    } else {
        let out_channels = comptime!(shape.matrix_columns / 4);
        let sample_quads = comptime!(shape.matrix_columns / 4 / WRITE_WIDTH);
        let sample = first_up_sample(shape) + tile * TILE + row_in_tile;
        let row = sample / width;
        let column_in_row = sample % width;
        let tap = column / out_channels;
        let channel = column % out_channels;
        let out_row = row * 2 + tap / 2;
        let out_column = column_in_row * 2 + tap % 2;
        let inside = sample < width * height;
        let position = (out_row * width * 2 + out_column) * sample_quads + channel / WRITE_WIDTH;

        OutputQuad { position, inside }
    }
}

/// Writes the plane's accumulators to `output` in f16 through its scratch tile, skipping samples outside the
/// map.
#[cube]
fn write_strided(
    accumulators: &Sequence<cmma::Matrix<f32>>,
    scratch: &mut Shared<[Sums]>,
    output: &mut [Quad],
    spot: PlaneSpot,
    width: u32,
    height: u32,
    #[comptime] shape: StridedShape,
) {
    let pixel_tiles = comptime!(shape.pixel_tiles_per_plane() as usize);
    let channel_tiles = comptime!(shape.channel_tiles_per_plane() as usize);
    let tile_quads = comptime!(TILE_VALUES / WRITE_WIDTH);
    let row_quads = comptime!(TILE / WRITE_WIDTH);

    #[unroll]
    for pixel_tile in 0..pixel_tiles {
        let tile = spot.first_pixel_tile + pixel_tile as u32;

        #[unroll]
        for channel_tile in 0..channel_tiles {
            let accumulator = accumulators.index(comptime!(pixel_tile * channel_tiles + channel_tile));
            let first_column = (spot.first_column_tile + channel_tile as u32) * TILE;
            let start = spot.scratch as usize;

            sync_cube();

            // A plane may hold several subgroups that each computed the same accumulator, so only the first
            // stores it. The condition is uniform within each subgroup.
            if UNIT_POS_X < PLANE_DIM {
                let tile_slice = scratch.slice_mut(start, start + tile_quads as usize);
                cmma::store(tile_slice, accumulator, TILE, cmma::MatrixLayout::RowMajor);
            }

            sync_cube();

            let mut index = UNIT_POS_X;
            while index < tile_quads {
                let row_in_tile = index / row_quads;
                let column = first_column + (index % row_quads) * WRITE_WIDTH;
                let sum = scratch[(spot.scratch + index) as usize];
                let landing = output_quad(tile, row_in_tile, column, width, height, shape);

                if landing.inside {
                    output[landing.position as usize] = Quad::cast_from(sum);
                }

                index += CUBE_DIM_X;
            }
        }
    }
}

/// Runs one block of a down or up layer on cooperative matrices accumulating in f32, writing f16.
#[cube(launch_unchecked)]
fn strided_kernel(
    input: &[Load],
    weights: &[Load],
    output: &mut [Quad],
    width: u32,
    height: u32,
    #[comptime] shape: StridedShape,
) {
    let spot = plane_spot(shape);
    let pixel_tiles = comptime!(shape.pixel_tiles_per_plane() as usize);
    let channel_tiles = comptime!(shape.channel_tiles_per_plane() as usize);

    let mut staged = Shared::<[Load]>::new_slice(comptime!(shape.staged_loads() as usize));
    let mut scratch =
        Shared::<[Sums]>::new_slice(comptime!((shape.planes() * TILE_VALUES / WRITE_WIDTH) as usize));

    stage_samples(input, &mut staged, width, height, shape);
    // Every plane reads samples another plane staged.
    sync_cube();

    let mut accumulators = Sequence::<cmma::Matrix<f32>>::new();

    #[unroll]
    for _ in 0..pixel_tiles * channel_tiles {
        let mut matrix = fragment::<f32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
        cmma::fill(&mut matrix, 0.0);
        accumulators.push(matrix);
    }

    multiply(&staged, weights, &accumulators, spot, shape);
    write_strided(&accumulators, &mut scratch, output, spot, width, height, shape);
}

/// A down or up layer over f16 NHWC maps. `width`, `height` and `channels` are the input map's.
///
/// The weights are the layer's row-major matrix, `[4C, 2C]` for a down layer and `[2C, 4C]` for an up layer.
#[derive(Clone, Copy)]
pub struct StridedLayer<'a> {
    pub input: &'a Handle,
    pub weights: &'a Handle,
    pub output: &'a Handle,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stride: Stride,
}

/// Runs `layer` with `variant`.
///
/// Gives an error if the variant cannot run on the device or with this layer's channels.
pub fn run_strided(
    client: &Client,
    variant: StridedVariant,
    layer: &StridedLayer<'_>,
) -> Result<(), CnnError> {
    assert!(!same_buffer(layer.input, layer.output), "the output is the input");

    let shape = strided_shape(variant, layer.stride, layer.channels);
    let shape = shape.map_err(CnnError::Convolution)?;
    let properties = client.properties();
    let fits = check_limits(
        properties,
        variant.plane_width,
        shape.units(),
        shape.shared_bytes(),
    );
    fits.map_err(CnnError::Convolution)?;

    let input_values = layer.width as usize * layer.height as usize * layer.channels as usize;
    let output_values = match layer.stride {
        Stride::Down => input_values / 2,
        Stride::Up => input_values * 2,
    };
    let weight_rows = shape.taps() as usize * layer.channels as usize;
    let weight_values = weight_rows * shape.matrix_columns as usize;
    let half = size_of::<f16>() as u64;
    let splits = layer.width.is_multiple_of(2) && layer.height.is_multiple_of(2);
    let even = layer.stride == Stride::Up || splits;
    assert!(even, "a down layer's map does not split into 2x2 blocks");
    assert!(layer.width > 0 && layer.height > 0, "the map is empty");
    assert!(
        layer.input.size_in_used() >= input_values as u64 * half,
        "the input is smaller than the map"
    );
    assert!(
        layer.output.size_in_used() >= output_values as u64 * half,
        "the output is smaller than the map"
    );
    assert!(
        layer.weights.size_in_used() >= weight_values as u64 * half,
        "the weights are smaller than the layer's"
    );
    assert!(
        u32::try_from(output_values).is_ok(),
        "the maps hold more values than the kernel can index"
    );

    let column_groups = shape.matrix_columns / TILE / variant.channel_tiles;
    let cubes = match layer.stride {
        Stride::Down => {
            let column_blocks = (layer.width / 2).div_ceil(DOWN_COLUMNS);
            let row_blocks = (layer.height / 2).div_ceil(variant.pixel_tiles);
            CubeCount::Static(column_blocks, row_blocks, column_groups)
        },
        Stride::Up => {
            let samples = layer.width * layer.height;
            let sample_blocks = samples.div_ceil(variant.pixel_tiles * TILE);
            let (max_x, max_y, _) = properties.hardware.max_cube_count;
            let spread = spread_cubes(sample_blocks as usize, (max_x, max_y, 1));
            let CubeCount::Static(x, y, _) = spread else {
                unreachable!("blocks spread into a static count");
            };
            assert!(
                x as usize * y as usize >= sample_blocks as usize,
                "the map holds more blocks than the device can launch"
            );
            CubeCount::Static(x, y, column_groups)
        },
    };
    let unit = CubeDim::new_2d(variant.plane_width, variant.planes_m * variant.planes_n);

    // Safety: the asserts above size the input, output and weights, and keep every output index inside a
    // u32.
    // - Input reads are clamped to samples inside the map, which holds at least one sample. Output writes are
    //   guarded to samples inside the output map, so the up blocks a spread grid adds past the map write
    //   nothing.
    // - A weight fragment's last value is at `(depth_row + 15) * columns + column + 15`, below the matrix's
    //   `taps * channels * columns` values, and its slice ends at most at the weights' end.
    // - Shared-memory indices stay inside the staged samples and the scratch tiles `shape` sizes. Fragments
    //   read only the channel halves each staged sample holds, never its padding.
    // - The launch takes each length in f16 values, and the kernel sees it in loads or quads.
    unsafe {
        let input = BufferArg::from_raw_parts(layer.input.clone(), input_values);
        let weights = BufferArg::from_raw_parts(layer.weights.clone(), weight_values);
        let output = BufferArg::from_raw_parts(layer.output.clone(), output_values);

        strided_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input,
            weights,
            output,
            layer.width,
            layer.height,
            shape,
        );
    }

    Ok(())
}
