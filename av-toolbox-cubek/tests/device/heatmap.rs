use av_toolbox_cubek::heatmap::{FLOOR, Heatmap, HeatmapPlanes, PaintLayout, Painting, Style};
use av_toolbox_cubek::matrix::{Coefficients, MatrixSamples, SampleRange, scaling};
use av_toolbox_cubek::testing::{decode_samples, encode_samples, reference_heatmap};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, backend, empty, plane, read_u8, seeded_plane, upload_f32, upload_u8};

const SCALE: f32 = 0.003;

/// Paints a map over a picture, given and returned as little-endian sample bytes, on the device.
struct Paint {
    painting: Painting,
    picture: [Vec<u8>; 3],
    map: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Paint {
    type Output = [Vec<u8>; 3];

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let (width, height) = (self.width, self.height);
        let [luma_size, chroma_size, _] = plane_sizes(width, height, self.painting.layout);
        let map_bytes = (width * height) as usize * size_of::<f32>();

        let picture = self.picture.each_ref().map(|bytes| upload_u8(client, bytes));
        let output = self.picture.each_ref().map(|bytes| empty(client, bytes.len()));
        let map = upload_f32(client, &self.map);
        let smooths = self.painting.smooth;
        let scratch = smooths.then(|| empty(client, map_bytes));
        let smoothed = smooths.then(|| empty(client, map_bytes));
        let sizes = [luma_size, chroma_size, chroma_size];

        let picture_planes = [0, 1, 2].map(|index| plane(&picture[index], sizes[index].0, sizes[index].1));
        let output_planes = [0, 1, 2].map(|index| plane(&output[index], sizes[index].0, sizes[index].1));
        let planes = HeatmapPlanes {
            picture: picture_planes,
            map: plane(&map, width, height),
            scratch: scratch.as_ref().map(|handle| plane(handle, width, height)),
            smoothed: smoothed.as_ref().map(|handle| plane(handle, width, height)),
            output: output_planes,
        };
        let heatmap = Heatmap::new(client, self.painting, height);
        heatmap.apply(client, &sizing, planes);

        output.each_ref().map(|handle| read_u8(client, handle))
    }
}

/// The width and height of each plane, luma first.
fn plane_sizes(width: u32, height: u32, layout: PaintLayout) -> [(u32, u32); 3] {
    let PaintLayout::Yuv {
        width_shift,
        height_shift,
    } = layout
    else {
        return [(width, height); 3];
    };

    let chroma_width = width.div_ceil(1 << width_shift);
    let chroma_height = height.div_ceil(1 << height_shift);

    [
        (width, height),
        (chroma_width, chroma_height),
        (chroma_width, chroma_height),
    ]
}

fn largest(samples: MatrixSamples) -> Option<f32> {
    match samples {
        MatrixSamples::U8 => Some(255.0),
        MatrixSamples::U16 { bits } => Some(((1u32 << bits) - 1) as f32),
        MatrixSamples::F32 => None,
    }
}

fn stored(value: f32, samples: MatrixSamples) -> f32 {
    match largest(samples) {
        Some(peak) => (value + 0.5).floor().clamp(0.0, peak),
        None => value,
    }
}

/// A picture of seeded samples running a little past both ends of each plane's range.
fn patterned_picture(width: u32, height: u32, painting: &Painting) -> [Vec<f32>; 3] {
    let sizes = plane_sizes(width, height, painting.layout);
    let chroma = match painting.layout {
        PaintLayout::Rgb => [false; 3],
        PaintLayout::Yuv { .. } => [false, true, true],
    };

    [0, 1, 2].map(|index| {
        let (plane_width, plane_height) = sizes[index];
        let (offset, scale) = scaling(painting.samples, painting.range, chroma[index]);
        let seeded = seeded_plane(plane_width, plane_height + index as u32);
        let count = (plane_width * plane_height) as usize;

        seeded[seeded.len() - count..]
            .iter()
            .map(|value| {
                let spread = 1.1 * value - 0.05;
                let code = offset as f32 + scale as f32 * spread;
                stored(code, painting.samples)
            })
            .collect()
    })
}

/// A map that peaks at twice the scale in the middle of the frame and fades to nothing.
fn blob_map(width: u32, height: u32) -> Vec<f32> {
    let centre_x = width as f32 / 2.0;
    let centre_y = height as f32 / 2.0;
    let spread = (width.max(height) as f32 / 5.0).max(1.0);

    (0..width * height)
        .map(|index| {
            let distance_x = (index % width) as f32 - centre_x;
            let distance_y = (index / width) as f32 - centre_y;
            let squared = distance_x * distance_x + distance_y * distance_y;
            2.0 * SCALE * (-squared / (2.0 * spread * spread)).exp()
        })
        .collect()
}

fn painting(style: Style, layout: PaintLayout, samples: MatrixSamples, range: SampleRange) -> Painting {
    Painting {
        style,
        scale: SCALE,
        floor: FLOOR,
        smooth: true,
        layout,
        samples,
        range,
        coefficients: Coefficients::Bt709,
    }
}

fn yuv(width_shift: u32, height_shift: u32) -> PaintLayout {
    PaintLayout::Yuv {
        width_shift,
        height_shift,
    }
}

fn to_bytes(values: &[f32], samples: MatrixSamples) -> Vec<u8> {
    let wide: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();

    encode_samples(&wide, samples)
}

fn painted_on_device(
    picture: &[Vec<f32>; 3],
    map: Vec<f32>,
    width: u32,
    height: u32,
    painting: Painting,
) -> [Vec<f32>; 3] {
    let picture_bytes = picture
        .each_ref()
        .map(|values| to_bytes(values, painting.samples));
    let task = Paint {
        painting,
        picture: picture_bytes,
        map,
        width,
        height,
    };
    let written = backend().build(task);

    written.map(|bytes| {
        let decoded = decode_samples(&bytes, painting.samples);
        decoded.iter().map(|value| *value as f32).collect()
    })
}

fn assert_matches(found: &[Vec<f32>; 3], expected: &[Vec<f32>; 3], samples: MatrixSamples) {
    let tolerance = match samples {
        MatrixSamples::F32 => 1e-4,
        _ => 1.0,
    };

    for channel in 0..3 {
        assert_eq!(found[channel].len(), expected[channel].len(), "plane {channel}");

        let pairs = found[channel].iter().zip(&expected[channel]);
        for (index, (got, want)) in pairs.enumerate() {
            let stored_want = stored(*want, samples);
            assert!(
                (got - stored_want).abs() <= tolerance,
                "plane {channel} sample {index} was {got}, expected {stored_want}",
            );
        }
    }
}

fn assert_matches_reference(width: u32, height: u32, painting: Painting) {
    let picture = patterned_picture(width, height, &painting);
    let map = blob_map(width, height);
    let picture_slices = picture.each_ref().map(Vec::as_slice);

    let expected = reference_heatmap(picture_slices, &map, width, height, &painting);
    let found = painted_on_device(&picture, map, width, height, painting);

    assert_matches(&found, &expected, painting.samples);
}

#[test]
fn blue_on_8_bit_limited_420_matches_the_reference() {
    let painting = painting(Style::Blue, yuv(1, 1), MatrixSamples::U8, SampleRange::Limited);

    assert_matches_reference(64, 36, painting);
}

#[test]
fn heat_on_10_bit_limited_420_matches_the_reference() {
    let samples = MatrixSamples::U16 { bits: 10 };
    let painting = painting(Style::Heat, yuv(1, 1), samples, SampleRange::Limited);

    assert_matches_reference(64, 36, painting);
}

#[test]
fn blue_on_8_bit_full_444_matches_the_reference() {
    let painting = painting(Style::Blue, yuv(0, 0), MatrixSamples::U8, SampleRange::Full);

    assert_matches_reference(64, 36, painting);
}

#[test]
fn heat_on_16_bit_rgb_matches_the_reference() {
    let samples = MatrixSamples::U16 { bits: 16 };
    let painting = painting(Style::Heat, PaintLayout::Rgb, samples, SampleRange::Full);

    assert_matches_reference(64, 36, painting);
}

#[test]
fn blue_on_f32_420_matches_the_reference() {
    let painting = painting(Style::Blue, yuv(1, 1), MatrixSamples::F32, SampleRange::Full);

    assert_matches_reference(64, 36, painting);
}

#[test]
fn an_odd_sized_420_frame_matches_the_reference() {
    let painting = painting(Style::Heat, yuv(1, 1), MatrixSamples::U8, SampleRange::Limited);

    assert_matches_reference(7, 5, painting);
}

#[test]
fn a_tiny_frame_matches_the_reference() {
    let painting = painting(Style::Blue, yuv(1, 1), MatrixSamples::U8, SampleRange::Limited);

    assert_matches_reference(8, 6, painting);
}

#[test]
fn a_nan_in_the_map_leaves_plain_grey() {
    let (width, height) = (64, 36);
    let painting = painting(Style::Blue, yuv(1, 1), MatrixSamples::U8, SampleRange::Limited);
    let picture = patterned_picture(width, height, &painting);
    let clean_map = vec![0.0; (width * height) as usize];
    let mut map = clean_map.clone();
    let centre = (height / 2 * width + width / 2) as usize;
    map[centre] = f32::NAN;
    let picture_slices = picture.each_ref().map(Vec::as_slice);

    let grey = reference_heatmap(picture_slices, &clean_map, width, height, &painting);
    let found = painted_on_device(&picture, map, width, height, painting);

    assert_matches(&found, &grey, painting.samples);
}

fn unsmoothed(painting: Painting) -> Painting {
    Painting {
        smooth: false,
        ..painting
    }
}

#[test]
fn blue_on_8_bit_limited_420_without_smoothing_matches_the_reference() {
    let painting = painting(Style::Blue, yuv(1, 1), MatrixSamples::U8, SampleRange::Limited);
    let painting = unsmoothed(painting);

    assert_matches_reference(64, 36, painting);
}

#[test]
fn heat_on_10_bit_444_with_no_floor_matches_the_reference() {
    let samples = MatrixSamples::U16 { bits: 10 };
    let painting = painting(Style::Heat, yuv(0, 0), samples, SampleRange::Limited);
    let painting = Painting {
        floor: 0.0,
        ..painting
    };

    assert_matches_reference(64, 36, painting);
}

#[test]
fn an_odd_sized_420_frame_without_smoothing_matches_the_reference() {
    let painting = painting(Style::Heat, yuv(1, 1), MatrixSamples::U8, SampleRange::Limited);
    let painting = unsmoothed(painting);

    assert_matches_reference(7, 5, painting);
}

#[test]
fn overlay_on_8_bit_limited_420_matches_the_reference() {
    let painting = painting(Style::Overlay, yuv(1, 1), MatrixSamples::U8, SampleRange::Limited);

    assert_matches_reference(64, 36, painting);
}

#[test]
fn overlay_without_smoothing_on_16_bit_rgb_matches_the_reference() {
    let samples = MatrixSamples::U16 { bits: 16 };
    let painting = painting(Style::Overlay, PaintLayout::Rgb, samples, SampleRange::Full);
    let painting = Painting {
        floor: 0.0,
        ..unsmoothed(painting)
    };

    assert_matches_reference(64, 36, painting);
}

#[test]
fn a_nan_without_smoothing_greys_only_its_pixel() {
    let (width, height) = (64, 36);
    let painting = painting(Style::Blue, yuv(1, 1), MatrixSamples::U8, SampleRange::Limited);
    let painting = Painting {
        scale: 1.0,
        floor: 0.0,
        smooth: false,
        ..painting
    };
    let picture = patterned_picture(width, height, &painting);
    let size = (width * height) as usize;
    let full_map = vec![1.0; size];
    let clean_map = vec![0.0; size];
    let mut map = full_map.clone();
    let centre = (height / 2 * width + width / 2) as usize;
    map[centre] = f32::NAN;
    let picture_slices = picture.each_ref().map(Vec::as_slice);

    let full = reference_heatmap(picture_slices, &full_map, width, height, &painting);
    let grey = reference_heatmap(picture_slices, &clean_map, width, height, &painting);
    let found = painted_on_device(&picture, map, width, height, painting);

    let centre_want = stored(grey[0][centre], painting.samples);
    assert!(
        (found[0][centre] - centre_want).abs() <= 1.0,
        "centre was {}, expected {centre_want}",
        found[0][centre],
    );

    let neighbours = [
        centre - width as usize,
        centre - 1,
        centre + 1,
        centre + width as usize,
    ];
    for index in neighbours {
        let want = stored(full[0][index], painting.samples);
        assert!(
            (found[0][index] - want).abs() <= 1.0,
            "neighbour {index} was {}, expected {want}",
            found[0][index],
        );
    }
}
