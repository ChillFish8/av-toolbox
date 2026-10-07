use cubecl::prelude::{ComputeClient, CubeElement, Runtime};

use super::network::ALIGNMENT;
use crate::cnn::{BodyPath, CnnError, LayerRun, Precision, tiles};
use crate::dpir::{Dpir, DpirMaps, DpirPlanes, Placement, column_bytes, level_bytes, run_region};
use crate::plane::{are_distinct, assert_plane};
use crate::{DevicePlane, Dispatch};

/// How many bytes of scratch one region's pass may use.
pub const DPIR_BUDGET: usize = 4 << 30;

/// How far each tile's region reaches past its core on every side.
pub const OVERLAP: u32 = 16;

/// The strength [warm_up](crate::dpir::warm_up) runs at.
const WARM_UP_STRENGTH: f32 = 10.0;

/// How a plane of `width` by `height` is padded to multiples of 8: the padded size and the padding before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padded {
    pub width: u32,
    pub height: u32,
    pub left: u32,
    pub top: u32,
}

impl Padded {
    /// Grows each side by the samples it lacks of a multiple of 8, half of them (rounded down) before it.
    pub fn of(width: u32, height: u32) -> Padded {
        let (padded_width, left) = padded_side(width);
        let (padded_height, top) = padded_side(height);

        Padded {
            width: padded_width,
            height: padded_height,
            left,
            top,
        }
    }
}

/// The padded length of a side `length` long, and the padding before it.
fn padded_side(length: u32) -> (u32, u32) {
    let padding = (ALIGNMENT - length % ALIGNMENT) % ALIGNMENT;

    (length + padding, padding / 2)
}

/// The scratch one region's pass needs, in bytes, and the tile the frame is split into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchSizes {
    /// The size of each of the four maps of each level.
    pub level_bytes: [usize; 4],
    /// The size of the im2col columns, or 0 off [BodyPath::Im2col](crate::cnn::BodyPath).
    pub column_bytes: usize,
    /// The side of each tile's core.
    pub tile: u32,
}

impl ScratchSizes {
    /// The bytes of every buffer together, four maps a level and the columns.
    fn total_bytes(&self) -> usize {
        let maps: usize = self.level_bytes.iter().sum();

        4 * maps + self.column_bytes
    }

    /// The bytes of the largest single buffer.
    fn largest_bytes(&self) -> usize {
        self.level_bytes[0].max(self.column_bytes)
    }

    /// Whether every buffer together fits [DPIR_BUDGET](crate::dpir::DPIR_BUDGET) and the largest one fits
    /// `allocation_limit`.
    fn fits(&self, allocation_limit: usize) -> bool {
        self.total_bytes() <= DPIR_BUDGET && self.largest_bytes() <= allocation_limit
    }
}

/// The scratch a `width` by `height` frame needs, and the tile it runs in.
///
/// The frame runs whole when its scratch fits [DPIR_BUDGET](crate::dpir::DPIR_BUDGET) and its largest buffer
/// fits `max_allocation`, where 0 means no limit. Otherwise the tile is the largest multiple of 8 whose
/// region fits both, and never smaller than 8.
pub fn scratch_sizes(
    precision: Precision,
    path: BodyPath,
    width: u32,
    height: u32,
    max_allocation: usize,
) -> ScratchSizes {
    let padded = Padded::of(width, height);
    let longer = padded.width.max(padded.height);
    let allocation_limit = match max_allocation {
        0 => usize::MAX,
        limit => limit,
    };

    let whole = tile_sizes(precision, path, width, height, longer);
    if whole.fits(allocation_limit) {
        return whole;
    }

    let mut tile = longer.saturating_sub(ALIGNMENT);
    while tile > ALIGNMENT {
        let sizes = tile_sizes(precision, path, width, height, tile);

        if sizes.fits(allocation_limit) {
            return sizes;
        }

        tile -= ALIGNMENT;
    }

    tile_sizes(precision, path, width, height, ALIGNMENT)
}

/// The scratch the largest region of a `width` by `height` frame needs, split into tiles of `tile`.
///
/// A region is at most `tile + 2 * OVERLAP` on a side, and never larger than the padded frame.
pub fn tile_sizes(precision: Precision, path: BodyPath, width: u32, height: u32, tile: u32) -> ScratchSizes {
    let padded = Padded::of(width, height);
    let region = tile + 2 * OVERLAP;
    let region_width = region.min(padded.width);
    let region_height = region.min(padded.height);
    let level_bytes = level_bytes(precision, region_width, region_height);
    let column_bytes = match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => 0,
        BodyPath::Im2col => column_bytes(precision, region_width, region_height),
    };

    ScratchSizes {
        level_bytes,
        column_bytes,
        tile,
    }
}

/// Deblocks every input plane into its output plane, one tile at a time. Gives how each ResBlock convolution
/// of each tile ran, in order.
///
/// The frame is padded to multiples of 8 by mirroring and split into tiles of `tile`, a multiple of 8. The
/// inputs are divided by `peak`, a finite value above 0, and `strength` is on the 8-bit scale, between 0 and
/// 255. Each output sample is clamped to `0..=1` and scaled by `peak`. No output may share a buffer with an
/// input. The maps must hold [tile_sizes](crate::dpir::tile_sizes) for the same frame and `tile`, as
/// [scratch_sizes](crate::dpir::scratch_sizes) gives them.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn deblock<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    dpir: &Dpir,
    path: BodyPath,
    planes: DpirPlanes<'_>,
    strength: f32,
    peak: f32,
    tile: u32,
    maps: DpirMaps<'_>,
) -> Result<Vec<LayerRun>, CnnError> {
    let (width, height) = check_planes(dpir, &planes);
    debug_assert!(
        outputs_are_apart(&planes),
        "an output plane shares a buffer with an input plane, which later tiles read past their core"
    );

    assert!(
        (0.0..=255.0).contains(&strength),
        "the strength {strength} is not between 0 and 255"
    );
    assert!(
        peak.is_finite() && peak > 0.0,
        "the peak {peak} is not a finite value above 0"
    );
    assert!(
        tile >= ALIGNMENT && tile.is_multiple_of(ALIGNMENT),
        "the tile {tile} is not a multiple of 8"
    );

    let padded = Padded::of(width, height);
    let list = tiles(padded.width, padded.height, tile, OVERLAP);
    let mut runs = Vec::new();

    for (region, core) in list {
        let placement = Placement {
            region,
            left: padded.left,
            top: padded.top,
        };

        let region_runs = run_region(
            client, dispatch, dpir, path, &planes, strength, peak, placement, core, maps,
        )?;
        runs.extend(region_runs);
    }

    Ok(runs)
}

/// Asserts there is one input and one output plane per image channel, all the first input's size, and
/// gives that size.
fn check_planes(dpir: &Dpir, planes: &DpirPlanes<'_>) -> (u32, u32) {
    let channels = dpir.variant().image_channels() as usize;
    assert!(
        planes.inputs.len() == channels && planes.outputs.len() == channels,
        "the network takes {channels} input and output planes"
    );

    let width = planes.inputs[0].width();
    let height = planes.inputs[0].height();

    for input in planes.inputs {
        assert_plane(input, width, height, "input");
    }

    for output in planes.outputs {
        assert_plane(output, width, height, "output");
    }

    (width, height)
}

/// Whether no output plane shares a buffer with an input plane.
fn outputs_are_apart(planes: &DpirPlanes<'_>) -> bool {
    for output in planes.outputs {
        for input in planes.inputs {
            let pair = [output.handle(), input.handle()];

            if !are_distinct(&pair) {
                return false;
            }
        }
    }

    true
}

/// Deblocks one frame of zeros at the stage's size and waits for the device, so every kernel a frame uses is
/// compiled and tuned before the first real frame. Gives how each ResBlock convolution of each tile ran, in
/// order.
///
/// The maps must hold [tile_sizes](crate::dpir::tile_sizes) for the same frame and `tile`, as
/// [scratch_sizes](crate::dpir::scratch_sizes) gives them. The input and output planes are allocated here.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn warm_up<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    dpir: &Dpir,
    path: BodyPath,
    width: u32,
    height: u32,
    tile: u32,
    maps: DpirMaps<'_>,
) -> Result<Vec<LayerRun>, CnnError> {
    let channels = dpir.variant().image_channels() as usize;
    let samples = width as usize * height as usize;
    let zeros = vec![0.0f32; samples];
    let output_bytes = samples * size_of::<f32>();

    let input_bytes = f32::as_bytes(&zeros);
    let input = client.create_from_slice(input_bytes);
    let outputs: Vec<_> = (0..channels).map(|_| client.empty(output_bytes)).collect();

    // Safety: the input holds `width * height` f32 values, which the kernels read it as.
    let input_plane = unsafe { DevicePlane::new(&input, width, height) };
    let inputs = vec![input_plane; channels];
    // Safety: each output holds `width * height` f32 values, which the kernels write it as.
    let output_planes: Vec<_> = outputs
        .iter()
        .map(|output| unsafe { DevicePlane::new(output, width, height) })
        .collect();
    let planes = DpirPlanes {
        inputs: &inputs,
        outputs: &output_planes,
    };

    let runs = deblock(
        client,
        dispatch,
        dpir,
        path,
        planes,
        WARM_UP_STRENGTH,
        1.0,
        tile,
        maps,
    )?;

    let finished = cubecl::future::block_on(client.sync());
    finished?;

    Ok(runs)
}
