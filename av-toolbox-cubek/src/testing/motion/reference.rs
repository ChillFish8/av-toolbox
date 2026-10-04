use super::{MotionClip, normalised};
use crate::motion::{BlockGrid, ConfidenceScores, MotionSettings, level_dims};

/// A field worked out on the CPU, laid out as [FieldPlanes](crate::motion::FieldPlanes) lays it out.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceField {
    pub vectors: Vec<i32>,
    pub sads: Vec<f32>,
    pub confidence: Vec<f32>,
}

fn clamp_i32(value: i32, limit: i32) -> i32 {
    if value < 0 {
        0
    } else if value >= limit {
        limit - 1
    } else {
        value
    }
}

/// Halves `source`, `width` by `height`, on the CPU, mirroring [build_pyramid](crate::motion::build_pyramid).
pub fn reference_downscale(source: &[f32], width: u32, height: u32) -> Vec<f32> {
    let (level_width, level_height) = level_dims(width, height, 1);

    let mut output = Vec::with_capacity((level_width * level_height) as usize);
    for y in 0..level_height {
        for x in 0..level_width {
            let sx = x * 2;
            let sy = y * 2;
            let sx1 = if sx + 1 < width { sx + 1 } else { sx };
            let sy1 = if sy + 1 < height { sy + 1 } else { sy };

            let s00 = source[(sy * width + sx) as usize];
            let s10 = source[(sy * width + sx1) as usize];
            let s01 = source[(sy1 * width + sx) as usize];
            let s11 = source[(sy1 * width + sx1) as usize];

            let average = (s00 + s10 + s01 + s11) * 0.25f32;
            output.push(average);
        }
    }

    output
}

/// Every level of the pyramid over `luma`, `width` by `height`, level 0 being `luma` itself.
pub fn reference_pyramid(luma: &[f32], width: u32, height: u32, levels: u32) -> Vec<Vec<f32>> {
    let mut pyramid = vec![luma.to_vec()];
    for level in 1..levels {
        let (source_width, source_height) = level_dims(width, height, level - 1);
        let source = &pyramid[level as usize - 1];
        let halved = reference_downscale(source, source_width, source_height);
        pyramid.push(halved);
    }

    pyramid
}

/// Every frame's pyramid of `clip`'s normalised luma, `levels` deep.
pub fn reference_clip_pyramids(clip: &MotionClip, levels: u32) -> Vec<Vec<Vec<f32>>> {
    let width = clip.width as u32;
    let height = clip.height as u32;

    clip.frames
        .iter()
        .map(|planes| {
            let luma = normalised(&planes[0], clip.bits);
            reference_pyramid(&luma, width, height, levels)
        })
        .collect()
}

/// The confidence of a block whose best SAD is `best_sad`, mirroring the search's own score.
pub fn reference_confidence(best_sad: f32, scores: ConfidenceScores) -> f32 {
    let mut excess = best_sad - scores.noise_floor;
    if excess < 0.0f32 {
        excess = 0.0f32;
    }

    let thsad_sq = scores.thsad * scores.thsad;
    let excess_sq = excess * excess;
    let confidence = (thsad_sq - excess_sq) / (thsad_sq + excess_sq);
    if confidence < 0.0f32 { 0.0f32 } else { confidence }
}

/// One level searched for the block at `origin_x`, `origin_y`, `blksize` pixels square.
struct Window<'a> {
    centre: &'a [f32],
    neighbour: &'a [f32],
    width: u32,
    height: u32,
    blksize: u32,
    search_radius: u32,
}

impl Window<'_> {
    /// The SAD of every candidate around `seed`, row of candidates by row, each summed row by row.
    fn scores(&self, origin_x: i32, origin_y: i32, seed: (i32, i32)) -> Vec<f32> {
        let width = self.width as i32;
        let height = self.height as i32;
        let radius = self.search_radius as i32;
        let blksize = self.blksize as i32;

        let mut tile = Vec::with_capacity((blksize * blksize) as usize);
        for py in 0..blksize {
            for px in 0..blksize {
                let cx_c = clamp_i32(origin_x + px, width);
                let cy_c = clamp_i32(origin_y + py, height);
                tile.push(self.centre[(cy_c * width + cx_c) as usize]);
            }
        }

        let mut scores = Vec::new();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let mvx = seed.0 + dx;
                let mvy = seed.1 + dy;

                let mut sad = 0.0f32;
                for iy in 0..blksize {
                    for ix in 0..blksize {
                        let cx = origin_x + ix;
                        let cy = origin_y + iy;
                        let centre_value = tile[(iy * blksize + ix) as usize];
                        let nx = clamp_i32(cx + mvx, width);
                        let ny = clamp_i32(cy + mvy, height);
                        let neighbour_value = self.neighbour[(ny * width + nx) as usize];
                        let diff = centre_value - neighbour_value;
                        let abs_diff = if diff < 0.0f32 { -diff } else { diff };
                        sad += abs_diff;
                    }
                }

                scores.push(sad);
            }
        }

        scores
    }

    /// The lowest scoring candidate around `seed` and its SAD, an exact tie going to `seed` itself.
    fn best(&self, origin_x: i32, origin_y: i32, seed: (i32, i32)) -> ((i32, i32), f32) {
        let scores = self.scores(origin_x, origin_y, seed);
        let radius = self.search_radius as i32;
        let side = 2 * radius + 1;

        let mut best_sad = 1.0e30f32;
        let mut best = seed;
        for (index, sad) in scores.iter().enumerate() {
            if *sad < best_sad {
                let dx = index as i32 % side - radius;
                let dy = index as i32 / side - radius;
                best_sad = *sad;
                best = (seed.0 + dx, seed.1 + dy);
            }
        }

        let seed_sad = scores[(radius * side + radius) as usize];
        if seed_sad <= best_sad {
            best_sad = seed_sad;
            best = seed;
        }

        (best, best_sad)
    }
}

/// The coarse search on `pyramid`'s coarsest level, giving every block of `grid` its scaled seed.
fn coarse_seeds(
    centre: &[Vec<f32>],
    neighbour: &[Vec<f32>],
    width: u32,
    height: u32,
    grid: &BlockGrid,
    settings: &MotionSettings,
) -> Vec<i32> {
    let coarse_level = settings.pyramid_levels - 1;
    let (level_width, level_height) = level_dims(width, height, coarse_level);
    let level_scale = 1u32 << coarse_level;
    let step = (grid.step / level_scale).max(1);
    let across = level_width.div_ceil(step).max(1);
    let down = level_height.div_ceil(step).max(1);
    let window = Window {
        centre: &centre[coarse_level as usize],
        neighbour: &neighbour[coarse_level as usize],
        width: level_width,
        height: level_height,
        blksize: (settings.blksize / level_scale).max(2),
        search_radius: settings.search_radius,
    };

    let mut seeds = vec![0i32; 2 * grid.blocks()];
    for by in 0..down {
        for bx in 0..across {
            let origin_x = (bx * step) as i32;
            let origin_y = (by * step) as i32;
            let ((best_dx, best_dy), _) = window.best(origin_x, origin_y, (0, 0));
            let mvx_fine = best_dx * level_scale as i32;
            let mvy_fine = best_dy * level_scale as i32;

            let fbx_start = (bx * step * level_scale / grid.step).min(grid.across);
            let fbx_end = if bx == across - 1 {
                grid.across
            } else {
                ((bx + 1) * step * level_scale / grid.step).min(grid.across)
            };
            let fby_start = (by * step * level_scale / grid.step).min(grid.down);
            let fby_end = if by == down - 1 {
                grid.down
            } else {
                ((by + 1) * step * level_scale / grid.step).min(grid.down)
            };

            for fby in fby_start..fby_end {
                for fbx in fbx_start..fbx_end {
                    let index = ((fby * grid.across + fbx) * 2) as usize;
                    seeds[index] = mvx_fine;
                    seeds[index + 1] = mvy_fine;
                }
            }
        }
    }

    seeds
}

/// The full resolution search of every block of `grid`, around `seeds` or zero motion.
fn fine_search(
    window: &Window<'_>,
    grid: &BlockGrid,
    seeds: Option<&[i32]>,
    scores: ConfidenceScores,
) -> ReferenceField {
    let mut field = ReferenceField {
        vectors: Vec::with_capacity(2 * grid.blocks()),
        sads: Vec::with_capacity(grid.blocks()),
        confidence: Vec::with_capacity(grid.blocks()),
    };

    for by in 0..grid.down {
        for bx in 0..grid.across {
            let block = (by * grid.across + bx) as usize;
            let seed = match seeds {
                Some(seeds) => (seeds[2 * block], seeds[2 * block + 1]),
                None => (0, 0),
            };

            let origin_x = (bx * grid.step) as i32;
            let origin_y = (by * grid.step) as i32;
            let ((best_dx, best_dy), best_sad) = window.best(origin_x, origin_y, seed);
            let confidence = reference_confidence(best_sad, scores);

            field.vectors.push(best_dx);
            field.vectors.push(best_dy);
            field.sads.push(best_sad);
            field.confidence.push(confidence);
        }
    }

    field
}

/// Searches where each block of `centre` is found in `neighbour` on the CPU, mirroring
/// [search](crate::motion::search).
///
/// Both pyramids hold `settings.pyramid_levels` levels of a frame `width` by `height`.
pub fn reference_search(
    centre: &[Vec<f32>],
    neighbour: &[Vec<f32>],
    width: u32,
    height: u32,
    settings: &MotionSettings,
    scores: ConfidenceScores,
) -> ReferenceField {
    let grid = BlockGrid::new(settings, width, height);
    let seeded = settings.pyramid_levels > 1;
    let seeds = seeded.then(|| coarse_seeds(centre, neighbour, width, height, &grid, settings));

    let window = Window {
        centre: &centre[0],
        neighbour: &neighbour[0],
        width,
        height,
        blksize: grid.blksize,
        search_radius: settings.search_radius,
    };

    fine_search(&window, &grid, seeds.as_deref(), scores)
}

/// Searches `refine_radius` pixels around `seeds` on the CPU, mirroring
/// [refine_seed](crate::motion::refine_seed).
///
/// `centre` and `neighbour` are full resolution luma, `width` by `height`.
#[expect(
    clippy::too_many_arguments,
    reason = "a search takes both frames, its seed and each setting"
)]
pub fn reference_refine_seed(
    centre: &[f32],
    neighbour: &[f32],
    width: u32,
    height: u32,
    seeds: &[i32],
    grid: &BlockGrid,
    refine_radius: u32,
    scores: ConfidenceScores,
) -> ReferenceField {
    let window = Window {
        centre,
        neighbour,
        width,
        height,
        blksize: grid.blksize,
        search_radius: refine_radius,
    };

    fine_search(&window, grid, Some(seeds), scores)
}

/// The block of a field nearest `position`, a frame `limit` pixels long and `count` blocks `step` apart.
fn block_at(position: i32, limit: u32, step: u32, count: u32) -> u32 {
    let clamped = clamp_i32(position, limit as i32) as u32;

    (clamped / step).min(count - 1)
}

/// Joins `first`, from frame `a` to `b`, with `second`, from `b` to `c`, on the CPU, mirroring
/// [chain](crate::motion::chain) over a frame `width` by `height`.
pub fn reference_chain_seed(
    first: &[i32],
    second: &[i32],
    grid: &BlockGrid,
    width: u32,
    height: u32,
) -> Vec<i32> {
    let mut seeds = Vec::with_capacity(2 * grid.blocks());
    for by in 0..grid.down {
        for bx in 0..grid.across {
            let mut pos_x = (bx * grid.step + grid.step / 2) as i32;
            let mut pos_y = (by * grid.step + grid.step / 2) as i32;
            let mut acc_x = 0i32;
            let mut acc_y = 0i32;

            for field in [first, second] {
                let block_x = block_at(pos_x, width, grid.step, grid.across);
                let block_y = block_at(pos_y, height, grid.step, grid.down);
                let base = ((block_y * grid.across + block_x) * 2) as usize;
                let dx = field[base];
                let dy = field[base + 1];

                acc_x += dx;
                acc_y += dy;
                pos_x += dx;
                pos_y += dy;
            }

            seeds.push(acc_x);
            seeds.push(acc_y);
        }
    }

    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::motion::{
        MOTION_FIXTURES,
        MotionFixture,
        SettingFixture,
        fixture_f32s,
        fixture_i32s,
        motion_clips,
    };
    use crate::testing::within_ulps;

    /// The deepest pyramid any fixture setting searches.
    const LEVELS: u32 = 3;

    /// Every level of every frame of a clip, frame then level.
    type ClipPyramids = Vec<Vec<Vec<f32>>>;

    /// Every clip with its fixtures and each frame's pyramid.
    fn fixture_clips() -> Vec<(MotionClip, &'static MotionFixture, ClipPyramids)> {
        motion_clips()
            .into_iter()
            .zip(MOTION_FIXTURES.iter())
            .map(|(clip, fixture)| {
                assert_eq!(clip.name, fixture.clip, "the fixtures follow the clips' order");
                let pyramids = reference_clip_pyramids(&clip, LEVELS);
                (clip, fixture, pyramids)
            })
            .collect()
    }

    /// Panics at the first value where `found` differs from `expected`, naming the field and block.
    fn assert_fields<T: PartialEq + std::fmt::Debug>(
        found: &[T],
        expected: &[T],
        per_field: usize,
        label: &str,
    ) {
        assert_eq!(found.len(), expected.len(), "{label} has the wrong length");

        let differing = found
            .iter()
            .zip(expected)
            .position(|(actual, wanted)| actual != wanted);
        if let Some(index) = differing {
            let field = index / per_field;
            let within = index % per_field;
            panic!(
                "{label} differs first in field {field} at value {within}: found {:?}, expected {:?}",
                found[index], expected[index],
            );
        }
    }

    fn grid_of(clip: &MotionClip, setting: &SettingFixture) -> BlockGrid {
        BlockGrid::new(&setting.settings, clip.width as u32, clip.height as u32)
    }

    /// The direct field from frame `centre` to frame `neighbour`.
    fn direct(
        clip: &MotionClip,
        pyramids: &[Vec<Vec<f32>>],
        setting: &SettingFixture,
        centre: usize,
        neighbour: usize,
    ) -> ReferenceField {
        let width = clip.width as u32;
        let height = clip.height as u32;

        reference_search(
            &pyramids[centre],
            &pyramids[neighbour],
            width,
            height,
            &setting.settings,
            setting.scores,
        )
    }

    /// The chained field from frame `centre` to `neighbour`, two frames away either way.
    fn chained(
        clip: &MotionClip,
        pyramids: &[Vec<Vec<f32>>],
        setting: &SettingFixture,
        centre: usize,
        neighbour: usize,
    ) -> ReferenceField {
        let width = clip.width as u32;
        let height = clip.height as u32;
        let grid = grid_of(clip, setting);
        let middle = (centre + neighbour) / 2;

        let first = direct(clip, pyramids, setting, centre, middle);
        let second = direct(clip, pyramids, setting, middle, neighbour);
        let seeds = reference_chain_seed(&first.vectors, &second.vectors, &grid, width, height);

        reference_refine_seed(
            &pyramids[centre][0],
            &pyramids[neighbour][0],
            width,
            height,
            &seeds,
            &grid,
            setting.refine_radius,
            setting.scores,
        )
    }

    #[test]
    fn every_pyramid_matches_av_denoise() {
        for (clip, fixture, pyramids) in fixture_clips() {
            for (level, bytes) in [(1, fixture.pyramid_1), (2, fixture.pyramid_2)] {
                let found: Vec<f32> = pyramids
                    .iter()
                    .flat_map(|pyramid| pyramid[level].iter().copied())
                    .collect();
                let expected = fixture_f32s(bytes);
                let per_frame = pyramids[0][level].len();
                let label = format!("{} pyramid_{level}", clip.name);

                assert_fields(&found, &expected, per_frame, &label);
            }
        }
    }

    #[test]
    fn every_direct_field_matches_av_denoise() {
        for (clip, fixture, pyramids) in fixture_clips() {
            for setting in &fixture.settings {
                let per_field = 2 * grid_of(&clip, setting).blocks();
                let frames = pyramids.len();

                for (distance, bytes) in [(1, setting.direct_1), (2, setting.direct_2)] {
                    let found: Vec<i32> = (0..frames - distance)
                        .flat_map(|centre| {
                            direct(&clip, &pyramids, setting, centre, centre + distance).vectors
                        })
                        .collect();
                    let expected = fixture_i32s(bytes);
                    let label = format!("{} {} direct_{distance}", clip.name, setting.label);

                    assert_fields(&found, &expected, per_field, &label);
                }
            }
        }
    }

    #[test]
    fn every_chained_field_matches_av_denoise() {
        for (clip, fixture, pyramids) in fixture_clips() {
            for setting in &fixture.settings {
                let per_field = 2 * grid_of(&clip, setting).blocks();
                let firsts = 0..pyramids.len() - 2;

                let forward: Vec<i32> = firsts
                    .clone()
                    .flat_map(|first| chained(&clip, &pyramids, setting, first, first + 2).vectors)
                    .collect();
                let expected_forward = fixture_i32s(setting.chained_2);
                let forward_label = format!("{} {} chained_2", clip.name, setting.label);
                assert_fields(&forward, &expected_forward, per_field, &forward_label);

                let backward: Vec<i32> = firsts
                    .flat_map(|first| chained(&clip, &pyramids, setting, first + 2, first).vectors)
                    .collect();
                let expected_backward = fixture_i32s(setting.chained_back_2);
                let backward_label = format!("{} {} chained_back_2", clip.name, setting.label);
                assert_fields(&backward, &expected_backward, per_field, &backward_label);
            }
        }
    }

    /// How far a confidence may sit from av-denoise's, as its division rounds differently by device.
    const CONFIDENCE_ULPS: u32 = 2;

    /// Panics at the first confidence further than [CONFIDENCE_ULPS] from `expected`.
    fn assert_confidences(found: &[f32], expected: &[f32], per_field: usize, label: &str) {
        assert_eq!(found.len(), expected.len(), "{label} has the wrong length");

        let differing = found
            .iter()
            .zip(expected)
            .position(|(actual, wanted)| !within_ulps(*actual, *wanted, CONFIDENCE_ULPS));
        if let Some(index) = differing {
            let field = index / per_field;
            let within = index % per_field;
            panic!(
                "{label} differs first in field {field} at block {within}: found {}, expected {}",
                found[index], expected[index],
            );
        }
    }

    #[test]
    fn every_confidence_matches_av_denoise() {
        for (clip, fixture, pyramids) in fixture_clips() {
            for setting in &fixture.settings {
                let blocks = grid_of(&clip, setting).blocks();
                let frames = pyramids.len();

                let direct_fields: Vec<ReferenceField> = (0..frames - 1)
                    .map(|centre| direct(&clip, &pyramids, setting, centre, centre + 1))
                    .collect();
                let chained_fields: Vec<ReferenceField> = (0..frames - 2)
                    .map(|first| chained(&clip, &pyramids, setting, first, first + 2))
                    .collect();

                let cases = [
                    (
                        "confidence_1",
                        direct_fields,
                        setting.direct_1,
                        setting.confidence_1,
                    ),
                    (
                        "confidence_chained_2",
                        chained_fields,
                        setting.chained_2,
                        setting.confidence_chained_2,
                    ),
                ];
                for (name, fields, vector_bytes, confidence_bytes) in cases {
                    let label = format!("{} {} {name}", clip.name, setting.label);
                    let vectors: Vec<i32> = fields.iter().flat_map(|field| field.vectors.clone()).collect();
                    let scores: Vec<f32> = fields.iter().flat_map(|field| field.confidence.clone()).collect();
                    let expected_vectors = fixture_i32s(vector_bytes);
                    let expected_scores = fixture_f32s(confidence_bytes);

                    assert_fields(&vectors, &expected_vectors, 2 * blocks, &label);
                    assert_confidences(&scores, &expected_scores, blocks, &label);
                }
            }
        }
    }

    #[test]
    fn a_pyramid_level_never_drops_under_one_sample() {
        let pyramid = reference_pyramid(&[0.25, 0.75, 0.5], 3, 1, 3);

        assert_eq!(pyramid[1], vec![(0.25 + 0.75 + 0.25 + 0.75) * 0.25]);
        assert_eq!(pyramid[2].len(), 1);
    }
}
