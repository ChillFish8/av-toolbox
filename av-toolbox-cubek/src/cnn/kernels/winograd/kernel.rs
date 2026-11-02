use cubecl::cmma;
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use half::f16;

use super::{INPUT_TRANSFORM, OUTPUT_TRANSFORM, TILE_OUTPUTS, TILE_QUADS, WINDOW, WinogradShape};
use crate::cnn::kernels::direct::{LOAD_WIDTH, Load, Quad, Sums, TILE, WRITE_WIDTH, finish_sums, fragment};

/// The plane's place in the workgroup's grid of planes.
#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct PlaneSpot {
    /// The index of the plane's first scratch quad in shared memory.
    scratch: u32,
    /// The first of the plane's groups of 16 tiles in the block.
    first_group: u32,
    /// The first of the plane's channel tiles.
    first_channel_tile: u32,
}

#[cube]
fn plane_spot(#[comptime] shape: WinogradShape) -> PlaneSpot {
    let plane = UNIT_POS_Y;
    let planes_m = comptime!(shape.planes_m);
    let groups = comptime!(shape.groups_per_plane());
    let channel_tiles = comptime!(shape.channel_tiles_per_plane());
    let scratch_quads = comptime!(shape.plane_scratch_quads());

    PlaneSpot {
        scratch: plane * scratch_quads,
        first_group: (plane % planes_m) * groups,
        first_channel_tile: (plane / planes_m) * channel_tiles,
    }
}

/// `sum` plus `coefficient` times `value`, leaving out a zero coefficient.
#[cube]
fn add_scaled(sum: Sums, value: Sums, #[comptime] coefficient: i32) -> Sums {
    let mut result = sum;

    if comptime!(coefficient == 1) {
        result = sum + value;
    }

    if comptime!(coefficient == -1) {
        result = sum - value;
    }

    if comptime!(coefficient != 0 && coefficient != 1 && coefficient != -1) {
        let scale = comptime!(coefficient as f32);
        result = sum + value * Sums::new(scale);
    }

    result
}

/// Position `position_row`, `position_column` of `Bᵀ d B`, where `d` is the 4x4 input window of tile
/// `tile_row`, `tile_column`, for the four channels at quad `part`.
///
/// The transform runs in f32 and is rounded to f16 once. Samples outside the frame are zero, the network's
/// padding.
#[cube]
fn transform_position(
    input: &[Quad],
    tile_row: u32,
    tile_column: u32,
    part: u32,
    width: u32,
    height: u32,
    #[comptime] position_row: usize,
    #[comptime] position_column: usize,
    #[comptime] features: u32,
) -> Quad {
    let quads = comptime!(features / WRITE_WIDTH);
    let window = comptime!(WINDOW as usize);
    let mut sum = Sums::new(0.0);

    #[unroll]
    #[expect(
        clippy::needless_range_loop,
        reason = "the unrolled loops read the transform matrices when the kernel compiles"
    )]
    for window_row in 0..window {
        let row_coefficient = comptime!(INPUT_TRANSFORM[position_row][window_row]);

        if comptime!(row_coefficient != 0) {
            // The window starts one sample before the tile, so `row` and `column` are one past the frame's.
            let row = tile_row * 2 + window_row as u32;
            let mut across = Sums::new(0.0);

            #[unroll]
            #[expect(
                clippy::needless_range_loop,
                reason = "the unrolled loops read the transform matrices when the kernel compiles"
            )]
            for window_column in 0..window {
                let column_coefficient = comptime!(INPUT_TRANSFORM[position_column][window_column]);

                if comptime!(column_coefficient != 0) {
                    let column = tile_column * 2 + window_column as u32;
                    let inside = row >= 1 && column >= 1 && row <= height && column <= width;
                    let mut sample = Sums::new(0.0);

                    if inside {
                        let source = ((row - 1) * width + column - 1) * quads + part;
                        sample = Sums::cast_from(input[source as usize]);
                    }

                    across = add_scaled(across, sample, column_coefficient);
                }
            }

            sum = add_scaled(sum, across, row_coefficient);
        }
    }

    Quad::cast_from(sum)
}

/// Transforms the input windows of the workgroup's tiles at position `position_row`, `position_column` into
/// `transformed`, by tile and channel.
#[cube]
fn stage_position(
    input: &[Quad],
    transformed: &mut Shared<[Quad]>,
    block_row: u32,
    block_column: u32,
    width: u32,
    height: u32,
    #[comptime] position_row: usize,
    #[comptime] position_column: usize,
    #[comptime] shape: WinogradShape,
) {
    let quads = comptime!(shape.features / WRITE_WIDTH);
    let tile_columns = comptime!(shape.tile_columns);
    let stride_quads = comptime!(shape.tile_stride() / WRITE_WIDTH);
    let total = comptime!(shape.tiles() * shape.features / WRITE_WIDTH);

    let mut index = UNIT_POS;
    while index < total {
        let tile = index / quads;
        let part = index % quads;
        let tile_row = block_row + tile / tile_columns;
        let tile_column = block_column + tile % tile_columns;

        transformed[(tile * stride_quads + part) as usize] = transform_position(
            input,
            tile_row,
            tile_column,
            part,
            width,
            height,
            position_row,
            position_column,
            shape.features,
        );

        index += CUBE_DIM;
    }
}

/// Transforms the input windows of all `tiles` tiles, `columns` to a row, into `transformed`, by position,
/// tile and channel.
#[cfg(any(test, feature = "test-support"))]
#[cube(launch_unchecked)]
pub(super) fn input_transform_kernel(
    input: &[Quad],
    transformed: &mut [Quad],
    width: u32,
    height: u32,
    columns: u32,
    tiles: u32,
    #[comptime] features: u32,
) {
    let quads = comptime!(features / WRITE_WIDTH);
    let window = comptime!(WINDOW as usize);
    let index = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let tile = index / quads;
    let part = index % quads;

    if tile < tiles {
        let tile_row = tile / columns;
        let tile_column = tile % columns;

        #[unroll]
        for position_row in 0..window {
            #[unroll]
            for position_column in 0..window {
                let position = (position_row * window + position_column) as u32;
                let value = transform_position(
                    input,
                    tile_row,
                    tile_column,
                    part,
                    width,
                    height,
                    position_row,
                    position_column,
                    features,
                );

                transformed[((position * tiles + tile) * quads + part) as usize] = value;
            }
        }
    }
}

/// Computes a block of 2x2 output tiles of a body layer as Winograd F(2x2, 3x3), with the bias, the
/// activation, the residual and the skip applied as it writes.
///
/// For each of the 16 positions in turn, the transformed input and transformed weights are multiplied on
/// cooperative matrices accumulating in f32, and each unit folds the products into its output tiles through
/// `Aᵀ m A` in f32. The workgroup transforms its own windows into shared memory one position at a time.
#[cube(launch_unchecked)]
pub(super) fn winograd_kernel(
    input: &[Quad],
    weights: &[Load],
    bias: &[Quad],
    residual: &[Quad],
    skip: &[Quad],
    output: &mut [Quad],
    width: u32,
    height: u32,
    #[comptime] shape: WinogradShape,
) {
    let block_column = CUBE_POS_X * comptime!(shape.tile_columns);
    let block_row = CUBE_POS_Y * comptime!(shape.tile_rows);
    let spot = plane_spot(shape);
    let window = comptime!(WINDOW as usize);
    let sums_per_unit = comptime!(shape.sums_per_unit() as usize);

    let mut transformed = Shared::<[Quad]>::new_slice(comptime!(shape.transformed_quads() as usize));
    let mut scratch = Shared::<[Sums]>::new_slice(comptime!(shape.scratch_quads() as usize));
    let mut sums = Array::<Sums>::new(sums_per_unit);

    #[unroll]
    for slot in 0..sums_per_unit {
        sums[slot] = Sums::new(0.0);
    }

    #[unroll]
    for position_row in 0..window {
        #[unroll]
        for position_column in 0..window {
            let position = comptime!((position_row * window + position_column) as u32);

            // Every unit has finished with the previous position's windows and products before they are
            // replaced.
            sync_cube();

            stage_position(
                input,
                &mut transformed,
                block_row,
                block_column,
                width,
                height,
                position_row,
                position_column,
                shape,
            );
            // Every plane reads windows other planes transformed.
            sync_cube();

            let products = multiply_position(&transformed, weights, spot, position, shape);

            store_products(&products, &mut scratch, spot, shape);
            // Every unit reads products its plane's first subgroup stored.
            sync_cube();
            fold_products(
                &scratch,
                sums.as_mut_slice(),
                spot,
                position_row,
                position_column,
                shape,
            );
        }
    }

    write_out(
        sums.as_slice(),
        bias,
        residual,
        skip,
        output,
        spot,
        block_row,
        block_column,
        width,
        height,
        shape,
    );
}

/// Multiplies the plane's transformed tiles at `position` by that position's transformed weights.
///
/// Gives a product for each of the plane's groups of 16 tiles and channel tiles, the groups' first.
#[cube]
fn multiply_position(
    transformed: &Shared<[Quad]>,
    weights: &[Load],
    spot: PlaneSpot,
    #[comptime] position: u32,
    #[comptime] shape: WinogradShape,
) -> Sequence<cmma::Matrix<f32>> {
    let groups = comptime!(shape.groups_per_plane() as usize);
    let channel_tiles = comptime!(shape.channel_tiles_per_plane() as usize);
    let chunks = comptime!((shape.features / TILE) as usize);
    let features = comptime!(shape.features);
    let shared_stride = comptime!(shape.tile_stride());
    let shared_quads = comptime!(((TILE - 1) * shape.tile_stride() + TILE) / WRITE_WIDTH);
    let weight_loads = comptime!(((TILE - 1) * shape.features + TILE) / LOAD_WIDTH);
    let weights_start = comptime!(position * shape.features * shape.features);
    let mut products = Sequence::<cmma::Matrix<f32>>::new();

    #[unroll]
    for _ in 0..groups * channel_tiles {
        let mut matrix = fragment::<f32>(cmma::MatrixIdent::Accumulator, cmma::MatrixLayout::Undefined);
        cmma::fill(&mut matrix, 0.0);
        products.push(matrix);
    }

    #[unroll]
    for chunk in 0..chunks {
        let mut lhs = Sequence::<cmma::Matrix<f16>>::new();

        #[unroll]
        for group in 0..groups {
            let first_tile = (spot.first_group + group as u32) * TILE;
            let mut windows = fragment(cmma::MatrixIdent::A, cmma::MatrixLayout::RowMajor);

            let start = (first_tile * shared_stride + chunk as u32 * TILE) / WRITE_WIDTH;
            let rows = transformed.slice(start as usize, (start + shared_quads) as usize);
            cmma::load(&mut windows, rows, shared_stride);

            lhs.push(windows);
        }

        #[unroll]
        for channel_tile in 0..channel_tiles {
            let channel = (spot.first_channel_tile + channel_tile as u32) * TILE;
            let start = (weights_start + channel * features + chunk as u32 * TILE) / LOAD_WIDTH;
            let columns = weights.slice(start as usize, (start + weight_loads) as usize);
            let mut rhs = fragment(cmma::MatrixIdent::B, cmma::MatrixLayout::ColMajor);
            cmma::load(&mut rhs, columns, features);

            #[unroll]
            for group in 0..groups {
                let product = products.index(comptime!(group * channel_tiles + channel_tile));
                cmma::execute::<f16, f16, f32, f32, cmma::Plane>(lhs.index(group), &rhs, product, product);
            }
        }
    }

    products
}

/// Stores a position's products in the plane's scratch tiles.
#[cube]
fn store_products(
    products: &Sequence<cmma::Matrix<f32>>,
    scratch: &mut Shared<[Sums]>,
    spot: PlaneSpot,
    #[comptime] shape: WinogradShape,
) {
    let fragments = comptime!(shape.fragments_per_plane() as usize);

    // A plane may hold several subgroups that each computed the same products, so only the first stores
    // them. The condition is uniform within each subgroup.
    if UNIT_POS_X < PLANE_DIM {
        #[unroll]
        for fragment_index in 0..fragments {
            let start = (spot.scratch + fragment_index as u32 * TILE_QUADS) as usize;
            let tile = scratch.slice_mut(start, start + TILE_QUADS as usize);
            let product = products.index(fragment_index);
            cmma::store(tile, product, TILE, cmma::MatrixLayout::RowMajor);
        }
    }
}

/// Adds `Aᵀ m A`'s share of the products at position `position_row`, `position_column` to the unit's output
/// sums, from the plane's scratch tiles.
#[cube]
fn fold_products(
    scratch: &Shared<[Sums]>,
    sums: &mut [Sums],
    spot: PlaneSpot,
    #[comptime] position_row: usize,
    #[comptime] position_column: usize,
    #[comptime] shape: WinogradShape,
) {
    let fragments = comptime!(shape.fragments_per_plane() as usize);
    let quads_per_unit = comptime!(shape.quads_per_unit() as usize);
    let plane_width = comptime!(shape.plane_width);
    let outputs = comptime!(TILE_OUTPUTS as usize);

    #[unroll]
    for fragment_index in 0..fragments {
        #[unroll]
        for quad in 0..quads_per_unit {
            let index = UNIT_POS_X + quad as u32 * plane_width;
            let product = scratch[(spot.scratch + fragment_index as u32 * TILE_QUADS + index) as usize];

            #[unroll]
            #[expect(
                clippy::needless_range_loop,
                reason = "the unrolled loops read the transform matrices when the kernel compiles"
            )]
            for output_row in 0..2usize {
                #[unroll]
                #[expect(
                    clippy::needless_range_loop,
                    reason = "the unrolled loops read the transform matrices when the kernel compiles"
                )]
                for output_column in 0..2usize {
                    let row_coefficient = comptime!(OUTPUT_TRANSFORM[output_row][position_row]);
                    let column_coefficient = comptime!(OUTPUT_TRANSFORM[output_column][position_column]);
                    let coefficient = comptime!(row_coefficient * column_coefficient);
                    let slot = comptime!(
                        ((fragment_index * quads_per_unit + quad) * outputs) + output_row * 2 + output_column
                    );

                    sums[slot] = add_scaled(sums[slot], product, coefficient);
                }
            }
        }
    }
}

/// Adds the bias, if the layer has one, to the unit's output sums, applies the activation, the residual
/// and the skip, and writes them to `output` in f16.
///
/// Samples outside the frame are not written.
#[cube]
fn write_out(
    sums: &[Sums],
    bias: &[Quad],
    residual: &[Quad],
    skip: &[Quad],
    output: &mut [Quad],
    spot: PlaneSpot,
    block_row: u32,
    block_column: u32,
    width: u32,
    height: u32,
    #[comptime] shape: WinogradShape,
) {
    let groups = comptime!(shape.groups_per_plane() as usize);
    let channel_tiles = comptime!(shape.channel_tiles_per_plane() as usize);
    let quads_per_unit = comptime!(shape.quads_per_unit() as usize);
    let plane_width = comptime!(shape.plane_width);
    let tile_columns = comptime!(shape.tile_columns);
    let row_quads = comptime!(TILE / WRITE_WIDTH);
    let sample_quads = comptime!(shape.features / WRITE_WIDTH);
    let outputs = comptime!(TILE_OUTPUTS as usize);

    #[unroll]
    for group in 0..groups {
        #[unroll]
        for channel_tile in 0..channel_tiles {
            let fragment_index = comptime!(group * channel_tiles + channel_tile);
            let channel = (spot.first_channel_tile + channel_tile as u32) * TILE;

            #[unroll]
            for quad in 0..quads_per_unit {
                let index = UNIT_POS_X + quad as u32 * plane_width;
                let block_tile = (spot.first_group + group as u32) * TILE + index / row_quads;
                let tile_row = block_row + block_tile / tile_columns;
                let tile_column = block_column + block_tile % tile_columns;
                let channel_quad = (channel + (index % row_quads) * WRITE_WIDTH) / WRITE_WIDTH;
                let mut bias_sums = Sums::new(0.0);
                if comptime!(shape.bias) {
                    bias_sums = Sums::cast_from(bias[channel_quad as usize]);
                }

                #[unroll]
                for output_row in 0..2usize {
                    #[unroll]
                    for output_column in 0..2usize {
                        let row = tile_row * 2 + output_row as u32;
                        let column = tile_column * 2 + output_column as u32;
                        let slot = comptime!(
                            ((fragment_index * quads_per_unit + quad) * outputs)
                                + output_row * 2
                                + output_column
                        );

                        if row < height && column < width {
                            let position = (row * width + column) * sample_quads + channel_quad;
                            let sum = sums[slot] + bias_sums;
                            let finished = finish_sums(
                                sum,
                                residual,
                                skip,
                                position,
                                shape.activation,
                                shape.residual,
                                shape.skip,
                            );

                            output[position as usize] = Quad::cast_from(finished);
                        }
                    }
                }
            }
        }
    }
}
