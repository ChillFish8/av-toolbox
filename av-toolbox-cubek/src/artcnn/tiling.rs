use cubecl::prelude::{ComputeClient, CubeElement, Runtime};

use super::network::conv0_sample_bytes;
use crate::artcnn::{Maps, Network, run_region};
use crate::cnn::{BodyPath, CnnError, LayerRun, Precision, tiles};
use crate::{DevicePlane, Dispatch};

/// How many bytes of scratch a tile may use.
pub const TILE_BUDGET: usize = 1 << 30;

/// The scratch buffers one tile needs, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchSizes {
    /// The size of the `first`, `second` and `third` feature maps.
    pub map_bytes: usize,
    /// The size of `conv0`, which the last layer also writes into on the convolution paths.
    pub conv0_bytes: usize,
    /// The size of the im2col columns, or 0 on the convolution paths.
    pub column_bytes: usize,
}

/// The bytes one sample of a haloed tile needs, across the four maps and, on the im2col path, the columns.
fn bytes_per_sample(features: u32, precision: Precision, path: BodyPath) -> usize {
    let value_bytes = precision.bytes();
    let map_bytes = 3 * features as usize * value_bytes;
    let conv0_bytes = conv0_sample_bytes(features, precision, path);
    let column_bytes = match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => 0,
        BodyPath::Im2col => 9 * features as usize * value_bytes,
    };

    map_bytes + conv0_bytes + column_bytes
}

/// The bytes one sample of a haloed tile needs in its largest buffer.
///
/// That is the columns on the im2col path, and `conv0` otherwise.
fn largest_buffer_bytes_per_sample(features: u32, precision: Precision, path: BodyPath) -> usize {
    let map_bytes = features as usize * precision.bytes();

    match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => conv0_sample_bytes(features, precision, path),
        BodyPath::Im2col => 9 * map_bytes,
    }
}

/// The largest tile centre side whose haloed square fits `budget` bytes of scratch.
///
/// `max_allocation` is the device's largest single buffer, which the haloed square's largest buffer must also
/// fit. A tile is never smaller than 1, even when the budget is too small for it.
pub fn tile_side(
    features: u32,
    halo: u32,
    precision: Precision,
    path: BodyPath,
    budget: usize,
    max_allocation: usize,
) -> u32 {
    let per_sample = bytes_per_sample(features, precision, path);
    let largest_per_sample = largest_buffer_bytes_per_sample(features, precision, path);
    let budget_samples = budget / per_sample;
    let allocation_samples = max_allocation / largest_per_sample;
    let samples = budget_samples.min(allocation_samples);
    let haloed = samples.isqrt() as u32;
    let centre = haloed.saturating_sub(2 * halo);

    centre.max(1)
}

/// Whether a `width` by `height` frame runs as one tile within `budget` bytes of scratch.
///
/// `max_allocation` is the device's largest single buffer, which the frame's largest buffer must also fit.
pub fn fits_whole(
    features: u32,
    precision: Precision,
    path: BodyPath,
    width: u32,
    height: u32,
    budget: usize,
    max_allocation: usize,
) -> bool {
    let samples = width as usize * height as usize;
    let per_sample = bytes_per_sample(features, precision, path);
    let largest_per_sample = largest_buffer_bytes_per_sample(features, precision, path);
    let scratch_bytes = samples.checked_mul(per_sample);
    let largest_bytes = samples.checked_mul(largest_per_sample);

    match (scratch_bytes, largest_bytes) {
        (Some(scratch), Some(largest)) => scratch <= budget && largest <= max_allocation,
        _ => false,
    }
}

/// The scratch the largest region of a `width` by `height` frame needs, split into tiles of `tile`.
///
/// A region is at most `tile + 2 * halo` on a side, and never larger than the frame.
pub fn scratch_sizes(network: &Network, path: BodyPath, tile: u32, width: u32, height: u32) -> ScratchSizes {
    let features = network.features();
    let halo = network.halo();
    let precision = network.precision();

    haloed_scratch(features, halo, precision, path, tile, width, height)
}

/// The same as [scratch_sizes](crate::artcnn::scratch_sizes), from the network's width, halo and precision.
pub(crate) fn haloed_scratch(
    features: u32,
    halo: u32,
    precision: Precision,
    path: BodyPath,
    tile: u32,
    width: u32,
    height: u32,
) -> ScratchSizes {
    let haloed = tile + 2 * halo;
    let region_width = haloed.min(width) as usize;
    let region_height = haloed.min(height) as usize;
    let samples = region_width * region_height;
    let value_bytes = precision.bytes();
    let map_bytes = samples * features as usize * value_bytes;
    let conv0_bytes = samples * conv0_sample_bytes(features, precision, path);
    let column_bytes = match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => 0,
        BodyPath::Im2col => map_bytes * 9,
    };

    ScratchSizes {
        map_bytes,
        conv0_bytes,
        column_bytes,
    }
}

/// Runs the network over the whole of `input`, divided by `peak`, one tile at a time.
///
/// The result is written into `output`, which must be twice the input on each side. The maps must be sized
/// by [scratch_sizes](crate::artcnn::scratch_sizes) for the same `tile` and frame.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn artcnn<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
    input: DevicePlane<'_>,
    peak: f32,
    tile: u32,
    maps: Maps<'_>,
    output: DevicePlane<'_>,
) -> Result<(), CnnError> {
    run_tiles(client, dispatch, network, path, input, peak, tile, maps, output)?;

    Ok(())
}

/// The same as [artcnn](crate::artcnn::artcnn), giving how each body layer of each tile ran, in order.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
fn run_tiles<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
    input: DevicePlane<'_>,
    peak: f32,
    tile: u32,
    maps: Maps<'_>,
    output: DevicePlane<'_>,
) -> Result<Vec<LayerRun>, CnnError> {
    let width = input.width();
    let height = input.height();
    assert!(
        output.width() == width * 2 && output.height() == height * 2,
        "the output is not twice the input on each side"
    );

    let sizes = scratch_sizes(network, path, tile, width, height);
    let map_bytes = sizes.map_bytes as u64;
    let conv0_bytes = sizes.conv0_bytes as u64;
    let every_map = [maps.first, maps.second, maps.third];
    for map in every_map {
        assert!(
            map.size_in_used() >= map_bytes,
            "a map is smaller than the tile's scratch"
        );
    }

    assert!(
        maps.conv0.size_in_used() >= conv0_bytes,
        "conv0 is smaller than the tile's scratch"
    );

    if let Some(columns) = maps.columns {
        let column_bytes = sizes.column_bytes as u64;
        assert!(
            columns.size_in_used() >= column_bytes,
            "the columns are smaller than the tile's scratch"
        );
    }

    let halo = network.halo();
    let list = tiles(width, height, tile, halo);
    let mut runs = Vec::new();

    for (region, core) in list {
        let tile_maps = Maps {
            conv0: maps.conv0,
            first: maps.first,
            second: maps.second,
            third: maps.third,
            columns: maps.columns,
        };

        let region_runs = run_region(
            client, dispatch, network, path, input, peak, region, core, tile_maps, output,
        )?;
        runs.extend(region_runs);
    }

    Ok(runs)
}

/// Runs one frame of zeros at the stage's size and waits for the device, so every kernel a frame uses is
/// compiled and tuned before the first real frame. Gives how each body layer of each tile ran, in order.
///
/// The maps must be sized by [scratch_sizes](crate::artcnn::scratch_sizes) for the same `tile` and frame. The
/// input and output planes are allocated here.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn warm_up<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
    width: u32,
    height: u32,
    tile: u32,
    maps: Maps<'_>,
) -> Result<Vec<LayerRun>, CnnError> {
    let samples = width as usize * height as usize;
    let zeros = vec![0.0f32; samples];
    let output_bytes = samples * 4 * size_of::<f32>();

    let input_bytes = f32::as_bytes(&zeros);
    let input = client.create_from_slice(input_bytes);
    let output = client.empty(output_bytes);

    // Safety: the input holds `width * height` f32 values, which the kernels read it as.
    let input_plane = unsafe { DevicePlane::new(&input, width, height) };
    // Safety: the output holds four times the input's f32 values, one per sample of the doubled plane.
    let output_plane = unsafe { DevicePlane::new(&output, width * 2, height * 2) };

    let runs = run_tiles(
        client,
        dispatch,
        network,
        path,
        input_plane,
        1.0,
        tile,
        maps,
        output_plane,
    )?;

    let finished = cubecl::future::block_on(client.sync());
    finished?;

    Ok(runs)
}
