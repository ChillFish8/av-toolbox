use cubecl::prelude::{ComputeClient, CubeElement, Runtime};
use cubecl::server::ServerError;

use crate::artcnn::{ArtcnnError, BodyPath, Maps, Network, Precision, Region, run_region};
use crate::{DevicePlane, Dispatch};

/// How many bytes of scratch a tile may use.
pub const TILE_BUDGET: usize = 1 << 30;

/// The side of the square plane [warm_up](crate::artcnn::warm_up) runs.
const WARM_UP_SIDE: u32 = 8;

/// The scratch buffers one tile needs, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchSizes {
    /// The size of each of the four feature maps.
    pub map_bytes: usize,
    /// The size of the im2col columns, or 0 on [BodyPath::Convolution](crate::artcnn::BodyPath::Convolution).
    pub column_bytes: usize,
}

/// The bytes one sample of a haloed tile needs, across the four maps and, on the im2col path, the columns.
fn bytes_per_sample(features: u32, precision: Precision, path: BodyPath) -> usize {
    let value_bytes = precision.bytes();
    let map_bytes = 4 * features as usize * value_bytes;
    let column_bytes = match path {
        BodyPath::Convolution => 0,
        BodyPath::Im2col => 9 * features as usize * value_bytes,
    };

    map_bytes + column_bytes
}

/// The largest tile centre side whose haloed square fits `budget` bytes of scratch.
///
/// A tile is never smaller than 1, even when the budget is too small for it.
pub fn tile_side(features: u32, halo: u32, precision: Precision, path: BodyPath, budget: usize) -> u32 {
    let per_sample = bytes_per_sample(features, precision, path);
    let samples = budget / per_sample;
    let haloed = samples.isqrt() as u32;
    let centre = haloed.saturating_sub(2 * halo);

    centre.max(1)
}

/// The scratch a haloed tile of `tile + 2 * halo` on a side needs.
pub fn scratch_sizes(network: &Network, path: BodyPath, tile: u32) -> ScratchSizes {
    let haloed = (tile + 2 * network.halo()) as usize;
    let samples = haloed * haloed;
    let value_bytes = network.precision().bytes();
    let features = network.features() as usize;
    let map_bytes = samples * features * value_bytes;
    let column_bytes = match path {
        BodyPath::Convolution => 0,
        BodyPath::Im2col => map_bytes * 9,
    };

    ScratchSizes {
        map_bytes,
        column_bytes,
    }
}

/// Splits a plane into tiles, giving each tile's region and core.
///
/// The cores are at most `tile` on a side and cover the plane exactly once. Each region is its core widened by
/// `halo` on every side, clipped to the plane.
pub fn tiles(width: u32, height: u32, tile: u32, halo: u32) -> Vec<(Region, Region)> {
    let mut list = Vec::new();

    for core_y in (0..height).step_by(tile as usize) {
        for core_x in (0..width).step_by(tile as usize) {
            let core_width = tile.min(width - core_x);
            let core_height = tile.min(height - core_y);
            let region_x = core_x.saturating_sub(halo);
            let region_y = core_y.saturating_sub(halo);
            let region_right = (core_x + core_width + halo).min(width);
            let region_bottom = (core_y + core_height + halo).min(height);

            let core = Region {
                x: core_x,
                y: core_y,
                width: core_width,
                height: core_height,
            };
            let region = Region {
                x: region_x,
                y: region_y,
                width: region_right - region_x,
                height: region_bottom - region_y,
            };
            list.push((region, core));
        }
    }

    list
}

/// Runs the network over the whole of `input`, divided by `peak`, one tile at a time, into the doubled `output`.
///
/// The maps must be sized by [scratch_sizes](crate::artcnn::scratch_sizes) for the same `tile`.
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
) -> Result<(), ArtcnnError> {
    assert!(tile >= 1, "a tile must be at least one sample on a side");

    let sizes = scratch_sizes(network, path, tile);
    let map_bytes = sizes.map_bytes as u64;
    let every_map = [maps.conv0, maps.first, maps.second, maps.third];
    for map in every_map {
        assert!(
            map.size_in_used() >= map_bytes,
            "a map is smaller than the tile's scratch"
        );
    }

    if let Some(columns) = maps.columns {
        let column_bytes = sizes.column_bytes as u64;
        assert!(
            columns.size_in_used() >= column_bytes,
            "the columns are smaller than the tile's scratch"
        );
    }

    let width = input.width();
    let height = input.height();
    let halo = network.halo();
    let list = tiles(width, height, tile, halo);

    for (region, core) in list {
        let tile_maps = Maps {
            conv0: maps.conv0,
            first: maps.first,
            second: maps.second,
            third: maps.third,
            columns: maps.columns,
        };

        run_region(
            client, dispatch, network, path, input, peak, region, core, tile_maps, output,
        )?;
    }

    Ok(())
}

/// Runs the network once over a small plane and waits for the device, so a kernel that fails to compile is an
/// error before any frame.
///
/// It allocates its own buffers, since it runs before any frame exists.
pub fn warm_up<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
) -> Result<(), ArtcnnError> {
    let samples = (WARM_UP_SIDE * WARM_UP_SIDE) as usize;
    let zeros = vec![0.0f32; samples];
    let untouched = vec![0.0f32; samples * 4];
    let sizes = scratch_sizes(network, path, WARM_UP_SIDE);

    let input_bytes = f32::as_bytes(&zeros);
    let output_bytes = f32::as_bytes(&untouched);
    let input = client.create_from_slice(input_bytes);
    let output = client.create_from_slice(output_bytes);
    let conv0 = client.empty(sizes.map_bytes);
    let first = client.empty(sizes.map_bytes);
    let second = client.empty(sizes.map_bytes);
    let third = client.empty(sizes.map_bytes);
    let columns = match path {
        BodyPath::Convolution => None,
        BodyPath::Im2col => Some(client.empty(sizes.column_bytes)),
    };

    // Safety: the input holds `WARM_UP_SIDE` squared f32 values, which the kernels read it as.
    let input_plane = unsafe { DevicePlane::new(&input, WARM_UP_SIDE, WARM_UP_SIDE) };
    // Safety: the output holds four times the input's f32 values, one per sample of the doubled plane.
    let output_plane = unsafe { DevicePlane::new(&output, WARM_UP_SIDE * 2, WARM_UP_SIDE * 2) };
    let maps = Maps {
        conv0: &conv0,
        first: &first,
        second: &second,
        third: &third,
        columns: columns.as_ref(),
    };

    artcnn(
        client,
        dispatch,
        network,
        path,
        input_plane,
        1.0,
        WARM_UP_SIDE,
        maps,
        output_plane,
    )?;

    let finished = cubecl::future::block_on(client.sync());

    finished.map_err(device_failed)
}

fn device_failed(error: ServerError) -> ArtcnnError {
    ArtcnnError::Device(error.to_string())
}
