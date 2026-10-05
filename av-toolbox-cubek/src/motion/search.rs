// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::confidence::confidence_of;
use super::subpel::refine_half_pel;
use super::{
    BlockGrid,
    ConfidenceScores,
    MAX_SEARCH_RADIUS,
    MotionField,
    MotionSettings,
    Pyramid,
    assert_field,
    assert_scores_match,
    assert_whole_pel_field,
    clamp_i32,
};
use crate::{DevicePlane, Dispatch, assert_plane};

/// Loads the block at `origin_x`, `origin_y` of `centre`, clamped to the level, into `tile`.
#[cube]
fn load_centre(
    centre: &Array<f32>,
    tile: &mut SharedMemory<f32>,
    origin_x: i32,
    origin_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
) {
    let block_pixels = comptime!(blksize * blksize);

    let mut pixel = UNIT_POS;
    while pixel < block_pixels {
        let px = pixel % blksize;
        let py = pixel / blksize;
        let cx_c = clamp_i32(origin_x + px as i32, width as i32);
        let cy_c = clamp_i32(origin_y + py as i32, height as i32);
        tile[pixel as usize] = centre[(cy_c * width as i32 + cx_c) as usize];
        pixel += CUBE_DIM;
    }
}

/// Scores every candidate of the window centred on `seed_x`, `seed_y` into `scratch`, the candidates
/// shared between the cube's units.
///
/// Each candidate sums its absolute differences row by row, in one unit.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel helper takes each plane and setting as its own argument"
)]
#[cube]
fn score_candidates(
    tile: &SharedMemory<f32>,
    neighbour: &Array<f32>,
    scratch: &mut SharedMemory<f32>,
    origin_x: i32,
    origin_y: i32,
    seed_x: i32,
    seed_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] search_radius: u32,
) {
    let window_side = comptime!(2 * search_radius + 1);
    let candidates = comptime!(window_side * window_side);

    let mut candidate = UNIT_POS;
    while candidate < candidates {
        let dy = candidate / window_side;
        let dx = candidate % window_side;
        let mvx = seed_x + (dx as i32 - search_radius as i32);
        let mvy = seed_y + (dy as i32 - search_radius as i32);

        let mut sad = 0.0f32;
        for iy in 0..blksize {
            for ix in 0..blksize {
                let cx = origin_x + ix as i32;
                let cy = origin_y + iy as i32;
                let centre_value = tile[(iy * blksize + ix) as usize];
                let nx = clamp_i32(cx + mvx, width as i32);
                let ny = clamp_i32(cy + mvy, height as i32);
                let neighbour_value = neighbour[(ny * width as i32 + nx) as usize];
                let diff = centre_value - neighbour_value;
                let abs_diff = if diff < 0.0f32 { -diff } else { diff };
                sad += abs_diff;
            }
        }

        scratch[candidate as usize] = sad;
        candidate += CUBE_DIM;
    }
}

/// Searches one block of the coarsest level per cube, and seeds every full resolution block inside
/// it with the winner scaled by `level_scale`.
///
/// An exact tie goes to zero motion, so a flat block never seeds a shift no comparison preferred.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes each plane and setting as its own argument"
)]
#[cube(launch_unchecked)]
fn block_match_coarse_kernel(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    vectors: &mut Array<i32>,
    #[comptime] level_width: u32,
    #[comptime] level_height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] search_radius: u32,
    #[comptime] level_scale: u32,
    #[comptime] fine_across: u32,
    #[comptime] fine_down: u32,
    #[comptime] fine_step: u32,
) {
    let bx = CUBE_POS_X;
    let by = CUBE_POS_Y;
    let origin_x = bx as i32 * step as i32;
    let origin_y = by as i32 * step as i32;

    let window_side = comptime!(2 * search_radius + 1);
    let candidates = comptime!(window_side * window_side);
    let block_pixels = comptime!(blksize * blksize);
    let mut scratch = SharedMemory::<f32>::new(candidates as usize);
    let mut tile = SharedMemory::<f32>::new(block_pixels as usize);

    load_centre(
        centre,
        &mut tile,
        origin_x,
        origin_y,
        level_width,
        level_height,
        blksize,
    );
    sync_cube();

    score_candidates(
        &tile,
        neighbour,
        &mut scratch,
        origin_x,
        origin_y,
        0i32,
        0i32,
        level_width,
        level_height,
        blksize,
        search_radius,
    );
    sync_cube();

    if UNIT_POS != 0 {
        terminate!();
    }

    let mut best_sad = 1.0e30f32;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    for dy in 0..window_side {
        for dx in 0..window_side {
            let sad = scratch[(dy * window_side + dx) as usize];
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx as i32 - search_radius as i32;
                best_dy = dy as i32 - search_radius as i32;
            }
        }
    }

    let zero_sad = scratch[(search_radius * window_side + search_radius) as usize];
    if zero_sad <= best_sad {
        best_dx = 0i32;
        best_dy = 0i32;
    }

    // A coarse block covers the fine blocks whose origin falls inside its own span, scaled to full
    // resolution. The last block of each axis reaches the fine grid's edge, which the two grids'
    // rounding can leave short.
    let mvx_fine = best_dx * level_scale as i32;
    let mvy_fine = best_dy * level_scale as i32;

    let fbx_start = u32::min(bx * step * level_scale / fine_step, fine_across);
    let mut fbx_end = u32::min((bx + 1) * step * level_scale / fine_step, fine_across);
    if bx == CUBE_COUNT_X - 1 {
        fbx_end = fine_across;
    }

    let fby_start = u32::min(by * step * level_scale / fine_step, fine_down);
    let mut fby_end = u32::min((by + 1) * step * level_scale / fine_step, fine_down);
    if by == CUBE_COUNT_Y - 1 {
        fby_end = fine_down;
    }

    let mut fby = fby_start;
    while fby < fby_end {
        let mut fbx = fbx_start;
        while fbx < fbx_end {
            let index = ((fby * fine_across + fbx) * 2) as usize;
            vectors[index] = mvx_fine;
            vectors[index + 1] = mvy_fine;
            fbx += 1;
        }
        fby += 1;
    }
}

/// Searches one full resolution block per cube, around its seed in `vectors` when `use_seed` is 1 and
/// around zero motion otherwise.
///
/// An exact tie goes to the seed. The winner and its SAD replace the block's entries, and its
/// confidence is written when `write_confidence` is set.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes each plane and setting as its own argument"
)]
#[cube(launch_unchecked)]
fn block_match_fine_kernel(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    vectors: &mut Array<i32>,
    sads: &mut Array<f32>,
    confidence: &mut Array<f32>,
    noise_floor: f32,
    thsad: f32,
    use_seed: u32,
    #[comptime] write_confidence: bool,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] search_radius: u32,
    #[comptime] across: u32,
) {
    let bx = CUBE_POS_X;
    let by = CUBE_POS_Y;
    let block = by * across + bx;
    let slot = (block * 2) as usize;

    let mut seed_dx = 0i32;
    let mut seed_dy = 0i32;
    if use_seed == 1u32 {
        seed_dx = vectors[slot];
        seed_dy = vectors[slot + 1];
    }

    let origin_x = bx as i32 * step as i32;
    let origin_y = by as i32 * step as i32;

    let window_side = comptime!(2 * search_radius + 1);
    let candidates = comptime!(window_side * window_side);
    let block_pixels = comptime!(blksize * blksize);
    let mut scratch = SharedMemory::<f32>::new(candidates as usize);
    let mut tile = SharedMemory::<f32>::new(block_pixels as usize);

    load_centre(centre, &mut tile, origin_x, origin_y, width, height, blksize);
    sync_cube();

    score_candidates(
        &tile,
        neighbour,
        &mut scratch,
        origin_x,
        origin_y,
        seed_dx,
        seed_dy,
        width,
        height,
        blksize,
        search_radius,
    );
    sync_cube();

    if UNIT_POS != 0 {
        terminate!();
    }

    let mut best_sad = 1.0e30f32;
    let mut best_dx = seed_dx;
    let mut best_dy = seed_dy;
    for dy in 0..window_side {
        for dx in 0..window_side {
            let sad = scratch[(dy * window_side + dx) as usize];
            if sad < best_sad {
                best_sad = sad;
                best_dx = seed_dx + (dx as i32 - search_radius as i32);
                best_dy = seed_dy + (dy as i32 - search_radius as i32);
            }
        }
    }

    let seed_sad = scratch[(search_radius * window_side + search_radius) as usize];
    if seed_sad <= best_sad {
        best_sad = seed_sad;
        best_dx = seed_dx;
        best_dy = seed_dy;
    }

    vectors[slot] = best_dx;
    vectors[slot + 1] = best_dy;
    sads[block as usize] = best_sad;

    if write_confidence {
        confidence[block as usize] = confidence_of(best_sad, noise_floor, thsad);
    }
}

/// One full resolution search over the grid of `field`.
struct FineSearch<'a, 'b> {
    centre: DevicePlane<'a>,
    neighbour: DevicePlane<'a>,
    field: &'b MotionField<'a>,
    search_radius: u32,
    use_seed: bool,
    scores: Option<ConfidenceScores>,
}

/// The scores a search passes when it writes no confidence, which the kernel never reads.
const UNUSED_SCORES: ConfidenceScores = ConfidenceScores {
    noise_floor: 0.0,
    thsad: 1.0,
};

fn launch_fine<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, search: FineSearch<'_, '_>) {
    let grid = &search.field.grid;
    let cubes = CubeCount::Static(grid.across, grid.down, 1);
    let unit = dispatch.cube_dim();
    let use_seed = u32::from(search.use_seed);

    let write_confidence = search.scores.is_some();
    let scores = search.scores.unwrap_or(UNUSED_SCORES);
    let confidence = match &search.field.planes.confidence {
        Some(plane) => plane.array::<R>(),
        None => {
            // Without confidence the kernel never reads its confidence argument, so a single value
            // stands in.
            let placeholder = client.empty(size_of::<f32>());

            // Safety: the placeholder holds one f32 and is never indexed.
            unsafe { ArrayArg::from_raw_parts(placeholder, 1) }
        },
    };

    let centre = search.centre.array::<R>();
    let neighbour = search.neighbour.array::<R>();
    let vectors = search.field.planes.vectors.array::<R>();
    let sads = search.field.planes.sads.array::<R>();
    let width = search.centre.width();
    let height = search.centre.height();

    // Safety: one cube per block of `grid`, each reading its block and window clamped to the planes, and
    // writing only its own block's entries.
    unsafe {
        block_match_fine_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            centre,
            neighbour,
            vectors,
            sads,
            confidence,
            scores.noise_floor,
            scores.thsad,
            use_seed,
            write_confidence,
            width,
            height,
            grid.blksize,
            grid.step,
            search.search_radius,
            grid.across,
        );
    }
}

/// The coarse search over the coarsest level, seeding every block of `field`.
fn launch_coarse<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    centre: &DevicePlane<'_>,
    neighbour: &DevicePlane<'_>,
    field: &MotionField<'_>,
    settings: &MotionSettings,
) {
    let grid = &field.grid;
    let coarse_level = settings.pyramid_levels - 1;
    let level_scale = 1u32 << coarse_level;
    let blksize = (settings.blksize / level_scale).max(2);
    let step = (grid.step / level_scale).max(1);
    let across = centre.width().div_ceil(step).max(1);
    let down = centre.height().div_ceil(step).max(1);

    let cubes = CubeCount::Static(across, down, 1);
    let unit = dispatch.cube_dim();
    let centre_array = centre.array::<R>();
    let neighbour_array = neighbour.array::<R>();
    let vectors = field.planes.vectors.array::<R>();

    // Safety: one cube per coarse block, each reading its block and window clamped to the level, and
    // writing the fine blocks inside its span, which no other coarse block reaches.
    unsafe {
        block_match_coarse_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            centre_array,
            neighbour_array,
            vectors,
            centre.width(),
            centre.height(),
            blksize,
            step,
            settings.search_radius,
            level_scale,
            grid.across,
            grid.down,
            grid.step,
        );
    }
}

/// Asserts that `scores`, when given, can score a block.
fn assert_scores(scores: Option<ConfidenceScores>) {
    let Some(scores) = scores else {
        return;
    };

    assert!(
        scores.thsad.is_finite() && scores.thsad > 0.0,
        "thsad is {}, expected a finite value above 0",
        scores.thsad
    );
}

/// Asserts that `centre` and `neighbour` have the same levels, at least `levels` of them.
pub(super) fn assert_pyramids(centre: &Pyramid<'_>, neighbour: &Pyramid<'_>, levels: u32) {
    let held = centre.levels.len().min(neighbour.levels.len());
    assert!(
        held >= levels as usize,
        "the pyramids hold {held} levels, expected at least {levels}"
    );

    for level in 0..levels as usize {
        let plane = &centre.levels[level];
        let name = format!("neighbour level {level}");
        assert_plane(&neighbour.levels[level], plane.width(), plane.height(), &name);
    }
}

/// Searches where each block of `centre` is found in `neighbour`, writing `field`.
///
/// With more than one level, a search of `search_radius` on the coarsest level seeds a search of the
/// same radius at full resolution. With one level, the full resolution search starts from zero motion.
///
/// `field.pel` must equal `settings.pel`. With `pel` 2 the result is then refined to half pixels, as
/// [refine_half_pel] does. `scores` is given exactly when `field` has a confidence plane.
pub fn search<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    centre: &Pyramid<'_>,
    neighbour: &Pyramid<'_>,
    field: &MotionField<'_>,
    settings: &MotionSettings,
    scores: Option<ConfidenceScores>,
) {
    debug_assert_eq!(settings.validate(), Ok(()));
    assert_pyramids(centre, neighbour, settings.pyramid_levels);
    assert_field(field);
    assert_scores_match(field, scores);
    assert_scores(scores);
    assert_eq!(
        field.pel, settings.pel,
        "the field's pel does not match the settings' pel"
    );

    let expected = BlockGrid::new(settings, centre.width(), centre.height());
    assert_eq!(field.grid, expected, "the grid does not match the settings");

    let seeded = settings.pyramid_levels > 1;
    if seeded {
        let coarse_level = settings.pyramid_levels as usize - 1;
        let coarse_centre = &centre.levels[coarse_level];
        let coarse_neighbour = &neighbour.levels[coarse_level];
        launch_coarse(client, dispatch, coarse_centre, coarse_neighbour, field, settings);
    }

    let whole_pixels = MotionField { pel: 1, ..*field };
    let fine = FineSearch {
        centre: centre.levels[0],
        neighbour: neighbour.levels[0],
        field: &whole_pixels,
        search_radius: settings.search_radius,
        use_seed: seeded,
        scores,
    };
    launch_fine(client, dispatch, fine);

    if settings.pel == 2 {
        refine_half_pel(client, dispatch, centre, neighbour, &whole_pixels);
    }
}

/// Searches `refine_radius` pixels around the seed already in `field` at full resolution, writing the
/// refined field over it.
///
/// `field` must have `pel` 1 and keeps it. For half pixels, follow it with [refine_half_pel]. `scores` is
/// given exactly when `field` has a confidence plane.
pub fn refine_seed<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    centre: &Pyramid<'_>,
    neighbour: &Pyramid<'_>,
    field: &MotionField<'_>,
    refine_radius: u32,
    scores: Option<ConfidenceScores>,
) {
    assert_pyramids(centre, neighbour, 1);
    assert_field(field);
    assert_whole_pel_field(field, "refine_seed");
    assert_scores_match(field, scores);
    assert_scores(scores);
    assert!(
        refine_radius <= MAX_SEARCH_RADIUS,
        "refine_radius is {refine_radius}, expected at most {MAX_SEARCH_RADIUS}"
    );

    let fine = FineSearch {
        centre: centre.levels[0],
        neighbour: neighbour.levels[0],
        field,
        search_radius: refine_radius,
        use_seed: true,
        scores,
    };
    launch_fine(client, dispatch, fine);
}
