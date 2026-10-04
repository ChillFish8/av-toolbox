// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

#[cube(launch_unchecked)]
fn clear_u32_kernel(values: &mut Array<u32>) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    values[ABSOLUTE_POS] = 0u32;
}

#[cube]
pub(super) fn tap(input: &Array<f32>, width: u32, x: u32, y: u32) -> i32 {
    let index = (y * width + x) as usize;
    i32::cast_from(input[index])
}

/// Whether sample (`x`, `y`) is combed, the way `calcMI` marks it.
#[cube]
fn combed_at(input: &Array<f32>, width: u32, height: u32, x: u32, y: u32, cthresh: i32) -> bool {
    let cthresh6 = cthresh * 6;
    let last = height - 1;
    let here = tap(input, width, x, y);
    let mut combed = false;

    if y == 0 {
        let below = tap(input, width, x, 1);
        let first = here - below;
        if first > cthresh || first < -cthresh {
            let two_below = tap(input, width, x, 2);
            let curve = two_below + here * 4 + two_below - below * 6;
            combed = i32::abs(curve) > cthresh6;
        }
    } else if y == last {
        let above = tap(input, width, x, y - 1);
        let first = here - above;
        if first > cthresh || first < -cthresh {
            let two_above = tap(input, width, x, y - 2);
            let curve = two_above * 2 + here * 4 - above * 6;
            combed = i32::abs(curve) > cthresh6;
        }
    } else {
        let above = tap(input, width, x, y - 1);
        let below = tap(input, width, x, y + 1);
        let first = here - above;
        let second = here - below;
        let peak = first > cthresh && second > cthresh;
        let trough = first < -cthresh && second < -cthresh;

        if peak || trough {
            // Row 1 mirrors its missing upper tap from below, and row `height - 2` its lower tap from above.
            let mut outer_above_row = 3u32;
            let mut outer_below_row = 3u32;
            if y > 1 {
                outer_above_row = y - 2;
                outer_below_row = y + 2;
                if y == last - 1 {
                    outer_below_row = y - 2;
                }
            }

            let outer_above = tap(input, width, x, outer_above_row);
            let outer_below = tap(input, width, x, outer_below_row);
            let curve = outer_above + here * 4 + outer_below - (above + below) * 3;
            combed = i32::abs(curve) > cthresh6;
        }
    }

    combed
}

#[cube(launch_unchecked)]
fn comb_mask_kernel(input: &Array<f32>, mask: &mut Array<u32>, width: u32, height: u32, cthresh: i32) {
    if ABSOLUTE_POS >= mask.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % width;
    let y = position / width;

    let mut combed = true;
    if cthresh >= 0 {
        combed = combed_at(input, width, height, x, y, cthresh);
    }

    if combed {
        mask[ABSOLUTE_POS] = 255u32;
    } else {
        mask[ABSOLUTE_POS] = 0u32;
    }
}

/// Whether sample (`x`, `y`) of `mask` is set along with at least one of its 8 neighbours.
#[cube]
fn clustered(mask: &Array<u32>, width: u32, x: u32, y: u32) -> bool {
    let centre = mask[(y * width + x) as usize] == 255u32;

    let mut set_count: u32 = 0u32;
    for row_offset in 0..3u32 {
        for column_offset in 0..3u32 {
            let row = y + row_offset - 1u32;
            let column = x + column_offset - 1u32;
            let index = (row * width + column) as usize;
            if mask[index] == 255u32 {
                set_count += 1u32;
            }
        }
    }

    // The count includes the centre, so a neighbour is set once it passes one.
    let neighboured = set_count > 1u32;
    centre && neighboured
}

#[cube]
fn mark(luma: &mut Array<u32>, luma_width: u32, row: u32, column: u32, shift_x: u32) {
    let start = row * luma_width + column;
    luma[start as usize] = 255u32;
    if shift_x > 0 {
        luma[(start + 1) as usize] = 255u32;
    }
}

#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
fn chroma_into_luma_kernel(
    u: &Array<u32>,
    v: &Array<u32>,
    luma: &mut Array<u32>,
    chroma_width: u32,
    chroma_height: u32,
    luma_width: u32,
    shift_x: u32,
    shift_y: u32,
) {
    if ABSOLUTE_POS >= u.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % chroma_width;
    let y = position / chroma_width;
    let inside = x >= 1 && x < chroma_width - 1 && y >= 1 && y < chroma_height - 1;
    if !inside {
        terminate!();
    }

    let combed = clustered(u, chroma_width, x, y) || clustered(v, chroma_width, x, y);
    if combed {
        let row = y << shift_y;
        let column = x << shift_x;
        mark(luma, luma_width, row, column, shift_x);

        if shift_y > 0 {
            mark(luma, luma_width, row + 1, column, shift_x);
            if y % 2 == 1 {
                mark(luma, luma_width, row - 1, column, shift_x);
            } else {
                mark(luma, luma_width, row + 2, column, shift_x);
            }
        }
    }
}

#[cube(launch_unchecked)]
fn block_counts_kernel(
    mask: &Array<u32>,
    counts: &mut Array<Atomic<u32>>,
    width: u32,
    height: u32,
    block_x: u32,
    block_y: u32,
    blocks_across: u32,
) {
    let position = ABSOLUTE_POS as u32;
    if position >= width * (height - 2) {
        terminate!();
    }

    let x = position % width;
    let y = position / width + 1;
    let above = mask[((y - 1) * width + x) as usize] == 255u32;
    let here = mask[(y * width + x) as usize] == 255u32;
    let below = mask[((y + 1) * width + x) as usize] == 255u32;

    if above && here && below {
        let half_x = block_x / 2;
        let half_y = block_y / 2;
        let row_stride = blocks_across * 4;
        let first_row = (y / block_y) * row_stride;
        let second_row = ((y + half_y) / block_y) * row_stride;
        let first_column = (x / block_x) * 4;
        let second_column = ((x + half_x) / block_x) * 4;

        counts[(first_row + first_column) as usize].fetch_add(1u32);
        counts[(first_row + second_column + 1) as usize].fetch_add(1u32);
        counts[(second_row + first_column + 2) as usize].fetch_add(1u32);
        counts[(second_row + second_column + 3) as usize].fetch_add(1u32);
    }
}

#[cube(launch_unchecked)]
fn max_count_kernel(counts: &Array<u32>, results: &mut Array<Atomic<u32>>, slot: u32) {
    if ABSOLUTE_POS >= counts.len() {
        terminate!();
    }

    results[slot as usize].fetch_max(counts[ABSOLUTE_POS]);
}

pub(super) fn clear<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, plane: DevicePlane<'_>) {
    let values = plane.array::<R>();
    let samples = plane.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the plane holds `samples` u32 values and the kernel writes nothing past them.
    unsafe {
        clear_u32_kernel::launch_unchecked::<R>(client, cubes, unit, values);
    }
}

/// A plane of f32 code values and the u32 comb mask it builds, both of one size.
pub struct CombPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub mask: DevicePlane<'a>,
}

/// Writes 255 into `planes.mask` where `planes.input` is combed and 0 elsewhere, as VIVTC's `calcMI` does.
///
/// `cthresh` is already scaled to the input's bit depth. A negative `cthresh` marks every sample. The plane
/// must be at least 4 rows tall.
pub fn comb_mask<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: CombPlanes<'_>,
    cthresh: i32,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    assert!(height >= 4, "the comb mask needs 4 rows, the plane has {height}");
    assert_plane(&planes.input, width, height, "input");
    assert_plane(&planes.mask, width, height, "mask");

    let samples = planes.mask.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let input = planes.input.array::<R>();
    let mask = planes.mask.array::<R>();

    // Safety: both planes hold `samples` values, every tap stays inside the plane's 4 or more rows, and each
    // unit writes only its own sample.
    unsafe {
        comb_mask_kernel::launch_unchecked::<R>(client, cubes, unit, input, mask, width, height, cthresh);
    }
}

/// The U and V comb masks and the luma comb mask they mark into.
pub struct ChromaMasks<'a> {
    pub u: DevicePlane<'a>,
    pub v: DevicePlane<'a>,
    pub luma: DevicePlane<'a>,
}

/// Marks 255 into `masks.luma` under every chroma sample set in U or V along with one of its 8 neighbours.
///
/// `shift_x` and `shift_y` are the chroma subsampling, so the chroma planes are the luma plane shifted down
/// by them. Chroma samples on the plane's edge are never tested, though they still count as neighbours.
pub fn chroma_into_luma<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    masks: ChromaMasks<'_>,
    shift_x: u32,
    shift_y: u32,
) {
    let luma_width = masks.luma.width();
    let luma_height = masks.luma.height();
    let chroma_width = masks.u.width();
    let chroma_height = masks.u.height();
    assert!(
        chroma_width == luma_width >> shift_x && chroma_height == luma_height >> shift_y,
        "the {chroma_width}x{chroma_height} chroma masks do not subsample the \
         {luma_width}x{luma_height} luma mask",
    );
    assert_plane(&masks.u, chroma_width, chroma_height, "U mask");
    assert_plane(&masks.v, chroma_width, chroma_height, "V mask");
    assert_plane(&masks.luma, luma_width, luma_height, "luma mask");

    let samples = masks.u.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let u = masks.u.array::<R>();
    let v = masks.v.array::<R>();
    let luma = masks.luma.array::<R>();

    // Safety: both chroma masks hold `samples` values and only units inside their edge read neighbours. The
    // luma rows and columns marked stay inside the luma mask the chroma subsamples. Units that mark one luma
    // sample all write 255.
    unsafe {
        chroma_into_luma_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            u,
            v,
            luma,
            chroma_width,
            chroma_height,
            luma_width,
            shift_x,
            shift_y,
        );
    }
}

/// The width and height of one block a mic counts.
#[derive(Debug, Clone, Copy)]
pub struct MicBlocks {
    pub x: u32,
    pub y: u32,
}

fn blocks_across(width: u32, blocks: MicBlocks) -> u32 {
    (width + blocks.x / 2) / blocks.x + 1
}

/// How many `u32` counts one mic of a `width` by `height` mask needs.
///
/// The counts hold four offset grids of blocks, interleaved.
pub fn mic_count_len(width: u32, height: u32, blocks: MicBlocks) -> usize {
    let across = blocks_across(width, blocks) as usize;
    let down = (height + blocks.y / 2) / blocks.y + 1;

    across * down as usize * 4
}

/// The luma comb mask, the counts one mic works in and the results it writes into.
///
/// `counts` holds [mic_count_len] `u32` values and `results` one `u32` per slot.
pub struct MicPlanes<'a> {
    pub mask: DevicePlane<'a>,
    pub counts: DevicePlane<'a>,
    pub results: DevicePlane<'a>,
}

/// Zeroes every slot of `results`.
pub fn clear_mics<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, results: DevicePlane<'_>) {
    clear(client, dispatch, results);
}

/// Raises `results[slot]` to the mic of `planes.mask`, as VIVTC's `calcMI` counts it.
///
/// The mic is the highest count of samples in any block of four offset grids whose mask is set on their own
/// row and the rows above and below. Each side of a block must be a power of 2. `results[slot]` must be
/// cleared first, with [clear_mics].
pub fn mic<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MicPlanes<'_>,
    blocks: MicBlocks,
    slot: u32,
) {
    let width = planes.mask.width();
    let height = planes.mask.height();
    assert!(height >= 3, "a mic needs 3 rows, the mask has {height}");
    assert!(
        blocks.x.is_power_of_two() && blocks.y.is_power_of_two(),
        "a mic counts as calcMI does only with power of 2 blocks, not {} by {}",
        blocks.x,
        blocks.y,
    );
    assert_plane(&planes.mask, width, height, "mask");

    let count_len = mic_count_len(width, height, blocks);
    assert!(
        planes.counts.samples() == count_len,
        "the counts hold {} values, expected {count_len}",
        planes.counts.samples(),
    );
    assert!(
        (slot as usize) < planes.results.samples(),
        "the results hold {} slots, not slot {slot}",
        planes.results.samples(),
    );

    clear(client, dispatch, planes.counts);

    let units = width as usize * (height as usize - 2);
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();
    let across = blocks_across(width, blocks);
    let mask = planes.mask.array::<R>();
    let counts = planes.counts.array::<R>();

    // Safety: each unit reads its own mask sample and the ones above and below inside the plane, and adds
    // into counts of blocks that hold its sample, which `mic_count_len` sizes.
    unsafe {
        block_counts_kernel::launch_unchecked::<R>(
            client, cubes, unit, mask, counts, width, height, blocks.x, blocks.y, across,
        );
    }

    let cubes = dispatch.cube_count(count_len);
    let unit = dispatch.cube_dim();
    let counts = planes.counts.array::<R>();
    let results = planes.results.array::<R>();

    // Safety: each unit reads its own count, and `slot` is inside the results.
    unsafe {
        max_count_kernel::launch_unchecked::<R>(client, cubes, unit, counts, results, slot);
    }
}
