use super::Eedi3Parameters;

/// A small fixed-seed generator, so every run draws the same planes.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn draw(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

        mixed ^ (mixed >> 31)
    }

    pub fn below(&mut self, bound: u32) -> u32 {
        let drawn = self.draw() % u64::from(bound);
        drawn as u32
    }

    /// A value between 0 and 1, in steps of 2^-24.
    pub fn unit(&mut self) -> f32 {
        let drawn = self.draw() >> 40;
        drawn as f32 / (1u32 << 24) as f32
    }
}

/// EEDI3 arguments drawn over their whole ranges, for a plane of `bits` per sample.
pub fn random_eedi3_parameters(random: &mut SplitMix, bits: u32) -> Eedi3Parameters {
    let alpha = random.unit();
    let mut beta = random.unit() * (1.0 - alpha);
    while alpha + beta > 1.0 {
        beta *= 0.999;
    }

    Eedi3Parameters {
        alpha,
        beta,
        gamma: random.unit() * 100.0,
        nrad: random.below(4),
        mdis: 1 + random.below(40),
        vcheck: random.below(4),
        vthresh: [
            0.5 + random.unit() * 100.0,
            0.5 + random.unit() * 100.0,
            0.5 + random.unit() * 10.0,
        ],
        bits,
    }
}

/// The content a random plane can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eedi3PlaneKind {
    /// Every sample drawn over the whole range.
    Noise,
    /// A hard sloped edge between two levels.
    Edge,
    /// The edge with light noise.
    NoisyEdge,
    /// Low noise around one level.
    Level,
}

/// A random plane of one of `kinds`.
///
/// The kind is drawn only when `kinds` holds more than one.
pub fn random_eedi3_plane(
    random: &mut SplitMix,
    width: u32,
    height: u32,
    bits: u32,
    kinds: &[Eedi3PlaneKind],
) -> Vec<f32> {
    assert!(!kinds.is_empty(), "a random plane needs at least one kind");

    let mut kind = kinds[0];
    if kinds.len() > 1 {
        let count = kinds.len() as u32;
        let index = random.below(count);
        kind = kinds[index as usize];
    }

    let peak = (1u32 << bits) - 1;
    let low = random.below(peak + 1) as i32;
    let high = random.below(peak + 1) as i32;
    let slope = (random.unit() - 0.5) * 16.0;
    let shift = random.unit() * width as f32;
    let spread = match kind {
        Eedi3PlaneKind::NoisyEdge => peak as i32 / 40,
        Eedi3PlaneKind::Level => 1 + random.below(3) as i32,
        Eedi3PlaneKind::Noise | Eedi3PlaneKind::Edge => 0,
    };

    let mut plane = Vec::with_capacity(width as usize * height as usize);
    for row in 0..height {
        for column in 0..width {
            if kind == Eedi3PlaneKind::Noise {
                let value = random.below(peak + 1);
                plane.push(value as f32);
                continue;
            }

            let beyond_edge = column as f32 > slope * row as f32 + shift;
            let level = if beyond_edge && kind != Eedi3PlaneKind::Level {
                high
            } else {
                low
            };
            let noise = random.below(2 * spread as u32 + 1) as i32 - spread;
            let value = (level + noise).clamp(0, peak as i32);
            plane.push(value as f32);
        }
    }

    plane
}

/// A random 8-bit mask of sparse dots, a few blocks, or every sample set.
pub fn random_eedi3_mask(random: &mut SplitMix, width: u32, height: u32) -> Vec<f32> {
    let samples = width as usize * height as usize;
    let kind = random.below(3);

    let mut mask = vec![0.0; samples];
    match kind {
        0 => {
            let odds = 20 + random.below(200);
            for value in &mut mask {
                if random.below(odds) == 0 {
                    *value = (1 + random.below(255)) as f32;
                }
            }
        },
        1 => {
            let blocks = 1 + random.below(4);
            for _ in 0..blocks {
                let left = random.below(width);
                let top = random.below(height);
                let right = (left + 1 + random.below(12)).min(width);
                let bottom = (top + 1 + random.below(12)).min(height);
                for row in top..bottom {
                    for column in left..right {
                        mask[(row * width + column) as usize] = 255.0;
                    }
                }
            }
        },
        _ => mask.fill(255.0),
    }

    mask
}
