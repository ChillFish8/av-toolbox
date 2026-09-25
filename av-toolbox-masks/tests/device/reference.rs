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
