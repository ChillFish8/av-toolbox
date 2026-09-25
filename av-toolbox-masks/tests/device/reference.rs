use av_toolbox_cubek::blur::{gaussian_weights, radius_for};

pub const BINS: usize = 4096;

pub fn normalise_u8(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample) / 255.0).collect()
}

pub fn reflect(index: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }

    let period = 2 * length - 2;
    let folded = index.abs() % period;
    if folded >= length { period - folded } else { folded }
}

/// A separable gaussian blur with reflect-101 borders.
pub fn blur_with_radius(input: &[f32], width: u32, height: u32, sigma: f32, radius: u32) -> Vec<f32> {
    let weights = gaussian_weights(sigma, radius);
    let radius = radius as i32;
    let width = width as i32;
    let height = height as i32;

    let mut rows = vec![0.0f32; input.len()];
    for row in 0..height {
        for column in 0..width {
            let mut total = 0.0f32;
            for (tap, weight) in weights.iter().enumerate() {
                let source = reflect(column + tap as i32 - radius, width);
                total += input[(row * width + source) as usize] * weight;
            }

            rows[(row * width + column) as usize] = total;
        }
    }

    let mut output = vec![0.0f32; input.len()];
    for row in 0..height {
        for column in 0..width {
            let mut total = 0.0f32;
            for (tap, weight) in weights.iter().enumerate() {
                let source = reflect(row + tap as i32 - radius, height);
                total += rows[(source * width + column) as usize] * weight;
            }

            output[(row * width + column) as usize] = total;
        }
    }

    output
}

pub fn blur(input: &[f32], width: u32, height: u32, sigma: f32) -> Vec<f32> {
    let radius = radius_for(sigma);

    blur_with_radius(input, width, height, sigma, radius)
}

/// The value `percentile` percent of the way through `values`, the centre of the first of 4096
/// bins whose running count reaches it.
pub fn percentile_by_bins(values: &[f32], percentile: f32) -> f32 {
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = high - low;

    let mut bins = vec![0u32; BINS];
    for value in values {
        let mut bin = 0usize;
        if span > 0.0 {
            let scaled = (value - low) / span * BINS as f32;
            bin = (scaled.max(0.0) as usize).min(BINS - 1);
        }

        bins[bin] += 1;
    }

    let target = percentile / 100.0 * values.len() as f32;
    let bin_width = span / BINS as f32;
    let mut running = 0u32;
    for (bin, count) in bins.iter().enumerate() {
        running += count;
        if running as f32 >= target {
            return low + (bin as f32 + 0.5) * bin_width;
        }
    }

    high
}

pub struct XdogSettings {
    pub sigma: f32,
    pub k: f32,
    pub tau: f32,
    pub eps: Option<f32>,
    pub phi: Option<f32>,
    pub eps_percentile: f32,
}

impl Default for XdogSettings {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            k: 1.6,
            tau: 0.95,
            eps: None,
            phi: None,
            eps_percentile: 8.0,
        }
    }
}

/// Finds line art with an extended difference of gaussians, a narrow blur minus a wider one
/// softly thresholded and stretched across the full 0 to 255 range.
pub fn xdog(luma: &[f32], width: u32, height: u32, settings: &XdogSettings) -> Vec<u8> {
    let near = blur(luma, width, height, settings.sigma);
    let far_sigma = settings.sigma * settings.k;
    let far = blur(luma, width, height, far_sigma);

    let difference: Vec<f32> = near
        .iter()
        .zip(&far)
        .map(|(near, far)| near - settings.tau * far)
        .collect();

    let eps = settings
        .eps
        .unwrap_or_else(|| percentile_by_bins(&difference, settings.eps_percentile));
    let phi = settings.phi.unwrap_or_else(|| {
        let count = difference.len() as f32;
        let mean = difference.iter().sum::<f32>() / count;
        let squares = difference.iter().map(|value| value * value).sum::<f32>() / count;
        let deviation = (squares - mean * mean).max(0.0).sqrt();
        4.0 / (deviation + 1e-6)
    });

    let edges: Vec<f32> = difference
        .iter()
        .map(|value| {
            let edge = if *value >= eps {
                1.0
            } else {
                1.0 + (phi * (value - eps)).tanh()
            };
            edge.clamp(0.0, 1.0)
        })
        .collect();

    let low = edges.iter().copied().fold(f32::INFINITY, f32::min);
    let high = edges.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    edges
        .iter()
        .map(|edge| {
            let stretched = if high > low {
                (edge - low) / (high - low)
            } else {
                *edge
            };
            ((1.0 - stretched) * 255.0).round() as u8
        })
        .collect()
}

/// The 8-connected component of every set mark, numbered in the order they are found.
pub fn components(marks: &[bool], width: u32, height: u32) -> Vec<Option<usize>> {
    let width = width as i32;
    let height = height as i32;
    let mut found: Vec<Option<usize>> = vec![None; marks.len()];
    let mut next = 0;

    for start in 0..marks.len() {
        if !marks[start] || found[start].is_some() {
            continue;
        }

        found[start] = Some(next);
        let mut pending = vec![start];
        while let Some(index) = pending.pop() {
            let column = index as i32 % width;
            let row = index as i32 / width;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let x = column + dx;
                    let y = row + dy;
                    if x < 0 || x >= width || y < 0 || y >= height {
                        continue;
                    }

                    let neighbour = (y * width + x) as usize;
                    if marks[neighbour] && found[neighbour].is_none() {
                        found[neighbour] = Some(next);
                        pending.push(neighbour);
                    }
                }
            }
        }

        next += 1;
    }

    found
}

fn sobel(input: &[f32], width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let width = width as i32;
    let height = height as i32;
    let sample = |column: i32, row: i32| {
        let x = reflect(column, width);
        let y = reflect(row, height);
        input[(y * width + x) as usize]
    };

    let mut horizontal = Vec::with_capacity(input.len());
    let mut vertical = Vec::with_capacity(input.len());
    for row in 0..height {
        for column in 0..width {
            let right =
                sample(column + 1, row - 1) + 2.0 * sample(column + 1, row) + sample(column + 1, row + 1);
            let left =
                sample(column - 1, row - 1) + 2.0 * sample(column - 1, row) + sample(column - 1, row + 1);
            let below =
                sample(column - 1, row + 1) + 2.0 * sample(column, row + 1) + sample(column + 1, row + 1);
            let above =
                sample(column - 1, row - 1) + 2.0 * sample(column, row - 1) + sample(column + 1, row - 1);
            horizontal.push(right - left);
            vertical.push(below - above);
        }
    }

    (horizontal, vertical)
}

/// Blurs, finds gradients, suppresses non-maxima, and links weak edges to strong ones by
/// connected component, with `low` and `high` in U8 units.
pub fn canny(luma: &[f32], width: u32, height: u32, low: f32, high: f32) -> Vec<u8> {
    let smoothed = blur_with_radius(luma, width, height, 0.8, 1);
    let (horizontal, vertical) = sobel(&smoothed, width, height);
    let magnitude: Vec<f32> = horizontal
        .iter()
        .zip(&vertical)
        .map(|(across, down)| across.abs() + down.abs())
        .collect();

    let width_signed = width as i32;
    let height_signed = height as i32;
    let magnitude_at = |column: i32, row: i32| {
        if column < 0 || column >= width_signed || row < 0 || row >= height_signed {
            return 0.0;
        }

        magnitude[(row * width_signed + column) as usize]
    };

    let mut candidates = vec![false; luma.len()];
    let mut strong = vec![false; luma.len()];
    for index in 0..luma.len() {
        let column = index as i32 % width_signed;
        let row = index as i32 / width_signed;
        let across = horizontal[index].abs();
        let down = vertical[index].abs();

        let (step_x, step_y) = if down > across * 2.414_213_5 {
            (0, 1)
        } else if down > across * 0.414_213_57 {
            if horizontal[index] * vertical[index] > 0.0 {
                (1, 1)
            } else {
                (1, -1)
            }
        } else {
            (1, 0)
        };

        let ahead = magnitude_at(column + step_x, row + step_y);
        let behind = magnitude_at(column - step_x, row - step_y);
        let kept = magnitude[index] > behind && magnitude[index] >= ahead;

        candidates[index] = kept && magnitude[index] > low / 255.0;
        strong[index] = kept && magnitude[index] > high / 255.0;
    }

    let labels = components(&candidates, width, height);
    let mut anchored = std::collections::HashSet::new();
    for (index, label) in labels.iter().enumerate() {
        if strong[index]
            && let Some(label) = label
        {
            anchored.insert(*label);
        }
    }

    labels
        .iter()
        .map(|label| match label {
            Some(label) if anchored.contains(label) => 255,
            _ => 0,
        })
        .collect()
}

/// A light square on a darker background.
pub fn square(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|row| {
            (0..width).map(move |column| {
                let inside = (8..16).contains(&column) && (8..16).contains(&row);
                if inside { 200 } else { 40 }
            })
        })
        .collect()
}

/// A left to right gradient with a dark vertical stroke three samples wide down the middle.
pub fn stroke_on_gradient(width: u32, height: u32) -> Vec<u8> {
    let middle = width / 2;

    (0..height)
        .flat_map(|_| {
            (0..width).map(move |column| {
                if column + 1 >= middle && column <= middle + 1 {
                    10
                } else {
                    (60 + column * 150 / width) as u8
                }
            })
        })
        .collect()
}

pub fn block_sigma(block_size: u32) -> f32 {
    0.3 * ((block_size as f32 - 1.0) * 0.5 - 1.0) + 0.8
}

/// The threshold step `lines/adaptive` runs, with `offset` in U8 units.
pub fn adaptive(plane: &[f32], width: u32, height: u32, block_size: u32, offset: f32) -> Vec<u8> {
    let smoothed = blur_with_radius(plane, width, height, 0.8, 1);
    let sigma = block_sigma(block_size);
    let mean = blur_with_radius(&smoothed, width, height, sigma, block_size / 2);
    let offset = offset / 255.0;

    smoothed
        .iter()
        .zip(&mean)
        .map(|(sample, local)| if *sample <= local - offset { 255 } else { 0 })
        .collect()
}

/// A bilateral filter, radius 3 and spatial sigma 7, with `colour_sigma` in 0 to 1 units.
pub fn bilateral(input: &[f32], width: u32, height: u32, colour_sigma: f32) -> Vec<f32> {
    let width = width as i32;
    let height = height as i32;
    let mut output = Vec::with_capacity(input.len());

    for row in 0..height {
        for column in 0..width {
            let centre = input[(row * width + column) as usize];
            let mut total = 0.0f32;
            let mut weights = 0.0f32;

            for offset_y in -3..=3i32 {
                for offset_x in -3..=3i32 {
                    let distance = (offset_x * offset_x + offset_y * offset_y) as f32;
                    if distance > 9.0 {
                        continue;
                    }

                    let source_column = reflect(column + offset_x, width);
                    let source_row = reflect(row + offset_y, height);
                    let value = input[(source_row * width + source_column) as usize];
                    let change = value - centre;
                    let spatial = (-distance / (2.0 * 49.0)).exp();
                    let colour = (-(change * change) / (2.0 * colour_sigma * colour_sigma)).exp();
                    let weight = spatial * colour;

                    total += value * weight;
                    weights += weight;
                }
            }

            output.push(total / weights);
        }
    }

    output
}

pub fn flatten(denoised: &[f32], blurred: &[f32]) -> Vec<f32> {
    denoised
        .iter()
        .zip(blurred)
        .map(|(value, light)| (value / (light + 1.0 / 255.0) * (128.0 / 255.0)).clamp(0.0, 1.0))
        .collect()
}

const TILES: u32 = 8;
const LEVELS: usize = 256;

fn level_of(value: f32) -> usize {
    (value * 255.0).round().clamp(0.0, 255.0) as usize
}

/// CLAHE over an 8x8 grid of tiles.
pub fn clahe(input: &[f32], width: u32, height: u32, clip: f32) -> Vec<f32> {
    let tile_width = width.div_ceil(TILES);
    let tile_height = height.div_ceil(TILES);
    let tiles_across = width.div_ceil(tile_width);
    let tiles_down = height.div_ceil(tile_height);
    let limit = ((clip * (tile_width * tile_height) as f32 / 256.0).floor()).max(1.0);

    let tile_count = (TILES * TILES) as usize;
    let mut histograms = vec![0u32; tile_count * LEVELS];
    for (index, value) in input.iter().enumerate() {
        let column = index as u32 % width;
        let row = index as u32 / width;
        let tile = ((row / tile_height) * TILES + column / tile_width) as usize;
        let level = level_of(*value);
        histograms[tile * LEVELS + level] += 1;
    }

    let mut tables = vec![0.0f32; tile_count * LEVELS];
    for tile in 0..tile_count {
        let counts = &histograms[tile * LEVELS..(tile + 1) * LEVELS];
        let total: f32 = counts.iter().map(|count| *count as f32).sum();
        let excess: f32 = counts.iter().map(|count| (*count as f32 - limit).max(0.0)).sum();
        let share = excess / 256.0;

        let mut running = 0.0f32;
        for (level, count) in counts.iter().enumerate() {
            running += (*count as f32).min(limit) + share;
            tables[tile * LEVELS + level] = if total > 0.0 { running / total } else { 0.0 };
        }
    }

    let table = |tile_x: u32, tile_y: u32, level: usize| {
        tables[((tile_y * TILES + tile_x) as usize) * LEVELS + level]
    };

    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let column = index as u32 % width;
            let row = index as u32 / width;
            let level = level_of(*value);

            let across = (column as f32 + 0.5) / tile_width as f32 - 0.5;
            let down = (row as f32 + 0.5) / tile_height as f32 - 0.5;
            let left = (across.floor().max(0.0) as u32).min(tiles_across - 1);
            let top = (down.floor().max(0.0) as u32).min(tiles_down - 1);
            let right = (left + 1).min(tiles_across - 1);
            let bottom = (top + 1).min(tiles_down - 1);
            let blend_x = (across - left as f32).clamp(0.0, 1.0);
            let blend_y = (down - top as f32).clamp(0.0, 1.0);

            let top_left = table(left, top, level);
            let top_right = table(right, top, level);
            let bottom_left = table(left, bottom, level);
            let bottom_right = table(right, bottom, level);
            let upper = (1.0 - blend_x) * top_left + blend_x * top_right;
            let lower = (1.0 - blend_x) * bottom_left + blend_x * bottom_right;

            (1.0 - blend_y) * upper + blend_y * lower
        })
        .collect()
}

pub fn remove_specks(lines: &[u8], width: u32, height: u32, min_speck: u32) -> Vec<u8> {
    let marks: Vec<bool> = lines.iter().map(|sample| *sample != 0).collect();
    let labels = components(&marks, width, height);

    let mut sizes = std::collections::HashMap::new();
    for label in labels.iter().flatten() {
        *sizes.entry(*label).or_insert(0u32) += 1;
    }

    labels
        .iter()
        .map(|label| match label {
            Some(label) if sizes[label] >= min_speck => 255,
            _ => 0,
        })
        .collect()
}

pub struct ShadowSettings {
    pub block_size: u32,
    pub c: f32,
    pub blur_sigma: f32,
    pub clahe_clip: f32,
    pub denoise: f32,
    pub min_speck: u32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            block_size: 15,
            c: 9.0,
            blur_sigma: 25.0,
            clahe_clip: 1.8,
            denoise: 50.0,
            min_speck: 6,
        }
    }
}

/// Evens out the lighting, equalises, runs the threshold step, then drops small specks.
pub fn recover_shadows(luma: &[f32], width: u32, height: u32, settings: &ShadowSettings) -> Vec<u8> {
    let denoised = bilateral(luma, width, height, settings.denoise / 255.0);
    let blurred = blur(&denoised, width, height, settings.blur_sigma);
    let flattened = flatten(&denoised, &blurred);
    let equalised = clahe(&flattened, width, height, settings.clahe_clip);
    let lines = adaptive(&equalised, width, height, settings.block_size, settings.c);

    remove_specks(&lines, width, height, settings.min_speck)
}

/// A bright left half and a dark right half, each with a vertical stroke and a speck.
///
/// Each stroke is 3 samples wide and runs the full height. The bright half's speck is 2 by 2
/// samples, and the dark half's is 3 by 3 and sits clear of the dark stroke.
pub fn shadowed_scene(width: u32, height: u32) -> Vec<u8> {
    let half = width / 2;

    (0..height)
        .flat_map(|row| {
            (0..width).map(move |column| {
                let bright = column < half;
                let stroke = (10..13).contains(&column) || (half + 10..half + 13).contains(&column);
                let small_speck = (20..22).contains(&row) && (5..7).contains(&column);
                let large_speck = (20..23).contains(&row) && (half + 17..half + 20).contains(&column);

                match (bright, stroke || small_speck || large_speck) {
                    (true, false) => 200,
                    (true, true) => 120,
                    (false, false) => 30,
                    (false, true) => 12,
                }
            })
        })
        .collect()
}
