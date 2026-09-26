use std::sync::OnceLock;

/// The side of the Bayer matrix.
const ORDERED_SIZE: usize = 16;

/// The side of the blue-noise tile.
const TILE: usize = 64;

const TILE_SAMPLES: usize = TILE * TILE;

/// The spread of the energy each set point gives its neighbours, in samples.
const SIGMA: f32 = 1.5;

/// One point in this many is set before the starting pattern is relaxed.
const STARTING_SPARSITY: usize = 10;

const SEED: u64 = 0x005e_ed0f_b100_d5e5;

/// How far each plane's pattern is moved along each axis, per plane index.
const PLANE_STEP_ACROSS: u32 = 23;
const PLANE_STEP_DOWN: u32 = 41;

static BLUE_NOISE: OnceLock<Vec<u32>> = OnceLock::new();

/// How far a plane's pattern is moved, so planes do not share their noise.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Shift {
    pub across: u32,
    pub down: u32,
}

pub fn plane_shift(plane: u32) -> Shift {
    Shift {
        across: PLANE_STEP_ACROSS * plane,
        down: PLANE_STEP_DOWN * plane,
    }
}

/// A square grid of rounding thresholds, each between 0 and 1, repeated across a plane.
#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    pub size: u32,
    pub values: Vec<f32>,
}

impl Thresholds {
    /// Plain rounding to the nearest value.
    pub fn rounding() -> Self {
        Self {
            size: 1,
            values: vec![0.5],
        }
    }

    pub fn ordered() -> Self {
        let ranks = bayer_ranks();

        from_ranks(&ranks, ORDERED_SIZE as u32)
    }

    pub fn blue_noise() -> Self {
        let ranks = blue_noise_ranks();

        from_ranks(ranks, TILE as u32)
    }

    /// The threshold for the sample at `column` and `row` of a plane moved by `shift`.
    pub fn at(&self, column: u32, row: u32, shift: Shift) -> f32 {
        let pattern_column = (column + shift.across) % self.size;
        let pattern_row = (row + shift.down) % self.size;
        let index = pattern_row * self.size + pattern_column;

        self.values[index as usize]
    }
}

/// Spreads each rank to the middle of its own step between 0 and 1.
fn from_ranks(ranks: &[u32], size: u32) -> Thresholds {
    let count = ranks.len() as f32;
    let values = ranks.iter().map(|rank| (*rank as f32 + 0.5) / count).collect();

    Thresholds { size, values }
}

/// The 16x16 Bayer matrix, built by doubling from a single rank.
pub fn bayer_ranks() -> Vec<u32> {
    let mut ranks = vec![0u32];
    let mut size = 1;

    while size < ORDERED_SIZE {
        let doubled = size * 2;
        let mut next = vec![0u32; doubled * doubled];

        for row in 0..doubled {
            for column in 0..doubled {
                let base = ranks[(row % size) * size + column % size];
                let quadrant = match (column / size, row / size) {
                    (0, 0) => 0,
                    (1, 0) => 2,
                    (0, 1) => 3,
                    _ => 1,
                };

                next[row * doubled + column] = 4 * base + quadrant;
            }
        }

        ranks = next;
        size = doubled;
    }

    ranks
}

/// The 64x64 blue-noise tile, made once per process.
pub fn blue_noise_ranks() -> &'static [u32] {
    BLUE_NOISE.get_or_init(generate_blue_noise)
}

/// A small fixed-seed generator, so the tile is the same on every run.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

        mixed ^ (mixed >> 31)
    }
}

/// The energy one set point gives each other point, by wrapped offset.
fn gaussian_table() -> Vec<f32> {
    let mut table = vec![0.0f32; TILE_SAMPLES];

    for offset_y in 0..TILE {
        for offset_x in 0..TILE {
            let wrapped_x = offset_x.min(TILE - offset_x) as f32;
            let wrapped_y = offset_y.min(TILE - offset_y) as f32;
            let distance = wrapped_x * wrapped_x + wrapped_y * wrapped_y;

            table[offset_y * TILE + offset_x] = (-distance / (2.0 * SIGMA * SIGMA)).exp();
        }
    }

    table
}

/// A binary pattern on a torus, with the energy every point receives from the set points.
#[derive(Clone)]
struct Field<'a> {
    set: Vec<bool>,
    energy: Vec<f32>,
    gaussian: &'a [f32],
}

impl<'a> Field<'a> {
    fn empty(gaussian: &'a [f32]) -> Self {
        Self {
            set: vec![false; TILE_SAMPLES],
            energy: vec![0.0; TILE_SAMPLES],
            gaussian,
        }
    }

    fn toggle(&mut self, index: usize, on: bool) {
        self.set[index] = on;
        let sign = if on { 1.0 } else { -1.0 };
        let origin_x = index % TILE;
        let origin_y = index / TILE;

        for y in 0..TILE {
            for x in 0..TILE {
                let offset_x = (x + TILE - origin_x) % TILE;
                let offset_y = (y + TILE - origin_y) % TILE;
                let contribution = self.gaussian[offset_y * TILE + offset_x];

                self.energy[y * TILE + x] += sign * contribution;
            }
        }
    }

    /// The set point with the most energy, the lowest index winning a tie.
    fn tightest_cluster(&self) -> usize {
        let mut best = None;

        for index in 0..TILE_SAMPLES {
            if !self.set[index] {
                continue;
            }

            let energy = self.energy[index];
            let is_better = best.is_none_or(|(_, current)| energy > current);
            if is_better {
                best = Some((index, energy));
            }
        }

        let (index, _) = best.expect("the field has a set point");

        index
    }

    /// The unset point with the least energy, the lowest index winning a tie.
    fn largest_void(&self) -> usize {
        let mut best = None;

        for index in 0..TILE_SAMPLES {
            if self.set[index] {
                continue;
            }

            let energy = self.energy[index];
            let is_better = best.is_none_or(|(_, current)| energy < current);
            if is_better {
                best = Some((index, energy));
            }
        }

        let (index, _) = best.expect("the field has an unset point");

        index
    }

    /// The same field with every point flipped, so its clusters are this field's voids.
    fn inverted(&self) -> Self {
        let mut flipped = Self::empty(self.gaussian);

        for index in 0..TILE_SAMPLES {
            if !self.set[index] {
                flipped.toggle(index, true);
            }
        }

        flipped
    }
}

/// Places sparse points by the fixed seed, then moves points from clusters into voids until settled.
fn starting_pattern(gaussian: &[f32]) -> Field<'_> {
    let mut field = Field::empty(gaussian);
    let mut random = SplitMix(SEED);
    let wanted = TILE_SAMPLES / STARTING_SPARSITY;
    let mut placed = 0;

    while placed < wanted {
        let index = (random.next() % TILE_SAMPLES as u64) as usize;
        if field.set[index] {
            continue;
        }

        field.toggle(index, true);
        placed += 1;
    }

    loop {
        let cluster = field.tightest_cluster();
        field.toggle(cluster, false);

        let void = field.largest_void();
        field.toggle(void, true);

        if void == cluster {
            return field;
        }
    }
}

/// Ranks every point of the tile by void-and-cluster.
pub(super) fn generate_blue_noise() -> Vec<u32> {
    let gaussian = gaussian_table();
    let start = starting_pattern(&gaussian);
    let starting_count = start.set.iter().filter(|set| **set).count();
    let mut ranks = vec![0u32; TILE_SAMPLES];

    let mut removing = start.clone();
    for rank in (0..starting_count).rev() {
        let cluster = removing.tightest_cluster();
        removing.toggle(cluster, false);
        ranks[cluster] = rank as u32;
    }

    let mut filling = start;
    for rank in starting_count..TILE_SAMPLES / 2 {
        let void = filling.largest_void();
        filling.toggle(void, true);
        ranks[void] = rank as u32;
    }

    let mut remaining = filling.inverted();
    for rank in TILE_SAMPLES / 2..TILE_SAMPLES {
        let cluster = remaining.tightest_cluster();
        remaining.toggle(cluster, false);
        ranks[cluster] = rank as u32;
    }

    ranks
}
