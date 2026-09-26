use crate::depth::{Shift, Thresholds, bayer_ranks, blue_noise_ranks, plane_shift};

const TILE: usize = 64;

fn holds_every_rank_once(ranks: &[u32]) -> bool {
    let mut sorted = ranks.to_vec();
    sorted.sort_unstable();

    sorted
        .iter()
        .enumerate()
        .all(|(index, rank)| *rank == index as u32)
}

/// Power per frequency of the tile's ranks, centred on zero, in row order of frequency.
fn power_spectrum(ranks: &[u32]) -> Vec<f64> {
    let count = ranks.len() as f64;
    let centred: Vec<f64> = ranks.iter().map(|rank| *rank as f64 / count - 0.5).collect();

    let mut power = vec![0.0; TILE * TILE];
    for frequency_y in 0..TILE {
        for frequency_x in 0..TILE {
            let mut real = 0.0;
            let mut imaginary = 0.0;

            for y in 0..TILE {
                for x in 0..TILE {
                    let phase_x = (frequency_x * x) as f64 / TILE as f64;
                    let phase_y = (frequency_y * y) as f64 / TILE as f64;
                    let angle = -2.0 * std::f64::consts::PI * (phase_x + phase_y);
                    let value = centred[y * TILE + x];
                    real += value * angle.cos();
                    imaginary += value * angle.sin();
                }
            }

            power[frequency_y * TILE + frequency_x] = real * real + imaginary * imaginary;
        }
    }

    power
}

/// The wrapped distance of a frequency index from zero.
fn wrapped(frequency: usize) -> f64 {
    frequency.min(TILE - frequency) as f64
}

#[test]
fn the_bayer_matrix_holds_every_rank_once() {
    let ranks = bayer_ranks();

    assert_eq!(ranks.len(), 256);
    assert!(holds_every_rank_once(&ranks));
}

#[test]
fn the_bayer_matrix_follows_the_standard_recursion() {
    let ranks = bayer_ranks();

    assert_eq!(&ranks[0..2], &[0, 128]);
    assert_eq!(&ranks[16..18], &[192, 64]);
}

#[test]
fn the_blue_noise_tile_holds_every_rank_once() {
    let ranks = blue_noise_ranks();

    assert_eq!(ranks.len(), TILE * TILE);
    assert!(holds_every_rank_once(ranks));
}

#[test]
fn the_blue_noise_tile_is_the_same_every_time() {
    let first = crate::depth::pattern::generate_blue_noise();
    let second = crate::depth::pattern::generate_blue_noise();

    assert_eq!(first, second);
}

/// White noise spreads its power evenly, so blue noise should hold far less of it near zero frequency.
#[test]
fn the_blue_noise_tile_has_little_low_frequency_energy() {
    let ranks = blue_noise_ranks();
    let power = power_spectrum(ranks);

    let mut low_total = 0.0;
    let mut low_count = 0.0;
    let mut all_total = 0.0;
    let mut all_count = 0.0;
    for frequency_y in 0..TILE {
        for frequency_x in 0..TILE {
            if frequency_x == 0 && frequency_y == 0 {
                continue;
            }

            let radius = wrapped(frequency_x).hypot(wrapped(frequency_y));
            let value = power[frequency_y * TILE + frequency_x];
            all_total += value;
            all_count += 1.0;

            if radius <= 8.0 {
                low_total += value;
                low_count += 1.0;
            }
        }
    }

    let low_mean = low_total / low_count;
    let all_mean = all_total / all_count;
    assert!(low_mean < 0.3 * all_mean, "low {low_mean} against {all_mean}");
}

#[test]
fn rounding_is_a_half_everywhere() {
    let rounding = Thresholds::rounding();
    let shift = plane_shift(2);

    assert_eq!(rounding.at(0, 0, shift), 0.5);
    assert_eq!(rounding.at(37, 11, shift), 0.5);
}

#[test]
fn a_threshold_sits_inside_its_rank_step() {
    let ordered = Thresholds::ordered();
    let none = Shift { across: 0, down: 0 };

    assert_eq!(ordered.at(0, 0, none), 0.5 / 256.0);
    assert_eq!(ordered.at(1, 0, none), 128.5 / 256.0);
}

#[test]
fn the_pattern_wraps_and_shifts_per_plane() {
    let ordered = Thresholds::ordered();
    let none = Shift { across: 0, down: 0 };
    let second = plane_shift(1);

    assert_eq!(ordered.at(16, 16, none), ordered.at(0, 0, none));
    assert_eq!(second, Shift { across: 23, down: 41 });
    assert_eq!(ordered.at(0, 0, second), ordered.at(23, 41, none));
}
