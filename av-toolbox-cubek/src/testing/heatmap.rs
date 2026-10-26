use crate::Edge;
use crate::blur::{gaussian_weights, radius_for};
use crate::heatmap::{FLOOR, PaintLayout, Painting, Style, lift, ramp_stops, sigma_for};
use crate::matrix::scaling;
use crate::testing::reference_blur_with_edge;

const LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// How strongly a smoothed map `value` paints, between 0 and 1. A NaN paints nothing.
pub fn reference_strength(value: f32, scale: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }

    ((value / scale - FLOOR) / (1.0 - FLOOR)).clamp(0.0, 1.0)
}

/// The red, green and blue a pixel of brightness `grey` paints with at `strength`.
pub fn reference_colour(grey: f32, strength: f32, style: Style) -> [f32; 3] {
    let stops = ramp_stops(style);
    let (base, gain) = lift(style);
    let brightness = base + gain * grey;

    let (from, to, fraction) = if strength <= 0.5 {
        (stops[0], stops[1], strength * 2.0)
    } else {
        (stops[1], stops[2], (strength - 0.5) * 2.0)
    };

    [0, 1, 2].map(|channel| {
        let tint = from[channel] + (to[channel] - from[channel]) * fraction;
        brightness * tint
    })
}

fn normalised_plane(plane: &[f32], painting: &Painting, chroma: bool) -> Vec<f32> {
    let (offset, scale) = scaling(painting.samples, painting.range, chroma);
    let (offset, scale) = (offset as f32, scale as f32);

    plane
        .iter()
        .map(|code| ((code - offset) / scale).clamp(0.0, 1.0))
        .collect()
}

fn grey_plane(picture: [&[f32]; 3], painting: &Painting) -> Vec<f32> {
    match painting.layout {
        PaintLayout::Yuv { .. } => normalised_plane(picture[0], painting, false),
        PaintLayout::Rgb => {
            let red = normalised_plane(picture[0], painting, false);
            let green = normalised_plane(picture[1], painting, false);
            let blue = normalised_plane(picture[2], painting, false);

            (0..red.len())
                .map(|index| {
                    LUMA_WEIGHTS[0] * red[index]
                        + LUMA_WEIGHTS[1] * green[index]
                        + LUMA_WEIGHTS[2] * blue[index]
                })
                .collect()
        },
    }
}

fn smoothed_strength(map: &[f32], width: u32, height: u32, painting: &Painting) -> Vec<f32> {
    let sigma = sigma_for(height);
    let radius = radius_for(sigma);
    let weights = gaussian_weights(sigma, radius);
    let smoothed = reference_blur_with_edge(map, width, height, &weights, Edge::Clamp);

    smoothed
        .iter()
        .map(|value| reference_strength(*value, painting.scale))
        .collect()
}

fn encode(values: &[f32], painting: &Painting, chroma: bool) -> Vec<f32> {
    let (offset, scale) = scaling(painting.samples, painting.range, chroma);
    let (offset, scale) = (offset as f32, scale as f32);

    values.iter().map(|value| offset + scale * value).collect()
}

/// Averages `differences` over the pixels each chroma sample covers that lie inside the frame.
fn average_chroma(differences: &[f32], width: u32, height: u32, shifts: (u32, u32)) -> Vec<f32> {
    let (width_shift, height_shift) = shifts;
    let chroma_width = width.div_ceil(1 << width_shift);
    let chroma_height = height.div_ceil(1 << height_shift);
    let mut plane = Vec::with_capacity((chroma_width * chroma_height) as usize);

    for chroma_row in 0..chroma_height {
        let rows = (chroma_row << height_shift)..((chroma_row + 1) << height_shift).min(height);

        for chroma_column in 0..chroma_width {
            let columns = (chroma_column << width_shift)..((chroma_column + 1) << width_shift).min(width);
            let mut total = 0.0;
            let mut count = 0.0;

            for row in rows.clone() {
                for column in columns.clone() {
                    total += differences[(row * width + column) as usize];
                    count += 1.0;
                }
            }

            plane.push(total / count);
        }
    }

    plane
}

/// Paints `map` over `picture` on the CPU and returns the output planes as unrounded code values.
///
/// `picture` holds the clip's planes as code values, chroma at its subsampled size. The map is smoothed
/// with an exact clamp-edged gaussian before it is painted.
pub fn reference_heatmap(
    picture: [&[f32]; 3],
    map: &[f32],
    width: u32,
    height: u32,
    painting: &Painting,
) -> [Vec<f32>; 3] {
    let grey = grey_plane(picture, painting);
    let strength = smoothed_strength(map, width, height, painting);
    let colours: Vec<[f32; 3]> = grey
        .iter()
        .zip(&strength)
        .map(|(grey, strength)| reference_colour(*grey, *strength, painting.style))
        .collect();

    let PaintLayout::Yuv {
        width_shift,
        height_shift,
    } = painting.layout
    else {
        return [0, 1, 2].map(|channel| {
            let channel_values: Vec<f32> = colours.iter().map(|colour| colour[channel]).collect();
            encode(&channel_values, painting, false)
        });
    };

    let (weight_red, weight_blue) = painting.coefficients.weights();
    let (weight_red, weight_blue) = (weight_red as f32, weight_blue as f32);
    let weight_green = 1.0 - weight_red - weight_blue;

    let luma: Vec<f32> = colours
        .iter()
        .map(|[red, green, blue]| weight_red * red + weight_green * green + weight_blue * blue)
        .collect();
    let blue_difference: Vec<f32> = colours
        .iter()
        .zip(&luma)
        .map(|(colour, luma)| (colour[2] - luma) / (2.0 * (1.0 - weight_blue)))
        .collect();
    let red_difference: Vec<f32> = colours
        .iter()
        .zip(&luma)
        .map(|(colour, luma)| (colour[0] - luma) / (2.0 * (1.0 - weight_red)))
        .collect();

    let shifts = (width_shift, height_shift);
    let blue_plane = average_chroma(&blue_difference, width, height, shifts);
    let red_plane = average_chroma(&red_difference, width, height, shifts);

    [
        encode(&luma, painting, false),
        encode(&blue_plane, painting, true),
        encode(&red_plane, painting, true),
    ]
}
