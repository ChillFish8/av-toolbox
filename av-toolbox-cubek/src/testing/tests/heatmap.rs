use crate::heatmap::{PaintLayout, Painting, Style};
use crate::matrix::{Coefficients, MatrixSamples, SampleRange};
use crate::testing::{reference_colour, reference_heatmap, reference_strength};

fn assert_colour(actual: [f32; 3], expected: [f32; 3]) {
    for channel in 0..3 {
        assert!(
            (actual[channel] - expected[channel]).abs() < 1e-6,
            "channel {channel}: {actual:?} against {expected:?}"
        );
    }
}

fn painting(layout: PaintLayout, range: SampleRange, scale: f32) -> Painting {
    Painting {
        style: Style::Blue,
        scale,
        layout,
        samples: MatrixSamples::U8,
        range,
        coefficients: Coefficients::Bt709,
    }
}

#[test]
fn strength_starts_above_the_floor_and_saturates_at_the_scale() {
    assert_eq!(reference_strength(0.0, 0.003), 0.0);
    assert_eq!(reference_strength(0.00075, 0.003), 0.0);
    assert!((reference_strength(0.001875, 0.003) - 0.5).abs() < 1e-6);
    assert_eq!(reference_strength(0.003, 0.003), 1.0);
    assert_eq!(reference_strength(1.0, 0.003), 1.0);
}

#[test]
fn a_nan_value_has_no_strength() {
    assert_eq!(reference_strength(f32::NAN, 1.0), 0.0);
}

#[test]
fn a_tiny_negative_value_has_no_strength() {
    assert_eq!(reference_strength(-1.5e-7, 0.003), 0.0);
}

#[test]
fn blue_runs_from_the_grey_picture_to_deep_blue() {
    assert_colour(reference_colour(0.5, 0.0, Style::Blue), [0.5, 0.5, 0.5]);
    assert_colour(
        reference_colour(1.0, 0.5, Style::Blue),
        [110.0 / 255.0, 160.0 / 255.0, 1.0],
    );
    assert_colour(
        reference_colour(1.0, 1.0, Style::Blue),
        [20.0 / 255.0, 40.0 / 255.0, 200.0 / 255.0],
    );
}

#[test]
fn heat_lifts_black_and_runs_cyan_to_red() {
    assert_colour(
        reference_colour(0.0, 0.0, Style::Heat),
        [0.25 * 60.0 / 255.0, 0.25 * 220.0 / 255.0, 0.25 * 230.0 / 255.0],
    );
    assert_colour(
        reference_colour(1.0, 0.5, Style::Heat),
        [240.0 / 255.0, 220.0 / 255.0, 40.0 / 255.0],
    );
    assert_colour(
        reference_colour(1.0, 1.0, Style::Heat),
        [230.0 / 255.0, 30.0 / 255.0, 30.0 / 255.0],
    );
}

#[test]
fn a_flat_map_stays_flat_through_smoothing() {
    let (width, height) = (16, 12);
    let layout = PaintLayout::Yuv {
        width_shift: 0,
        height_shift: 0,
    };
    let painting = painting(layout, SampleRange::Full, 0.003);
    let size = (width * height) as usize;
    let luma = vec![128.0; size];
    let chroma = vec![128.0; size];
    let map = vec![0.003; size];

    let output = reference_heatmap([&luma, &chroma, &chroma], &map, width, height, &painting);

    let colour = reference_colour(128.0 / 255.0, 1.0, Style::Blue);
    let expected = 255.0 * (0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]);
    for value in &output[0] {
        assert!((value - expected).abs() < 1e-3, "{value} against {expected}");
    }
}

#[test]
fn a_clean_frame_is_the_grey_picture() {
    let (width, height) = (8, 6);
    let layout = PaintLayout::Yuv {
        width_shift: 1,
        height_shift: 1,
    };
    let painting = painting(layout, SampleRange::Limited, 0.003);
    let luma: Vec<f32> = (0..width * height).map(|index| 16.0 + index as f32).collect();
    let chroma = vec![90.0; 12];
    let map = vec![0.0; (width * height) as usize];

    let output = reference_heatmap([&luma, &chroma, &chroma], &map, width, height, &painting);

    for (value, expected) in output[0].iter().zip(&luma) {
        assert!((value - expected).abs() < 1e-3, "{value} against {expected}");
    }
    for plane in &output[1..] {
        assert_eq!(plane.len(), 12);
        for value in plane {
            assert!((value - 128.0).abs() < 1e-3, "{value}");
        }
    }
}

#[test]
fn superblack_luma_reads_as_black() {
    let layout = PaintLayout::Yuv {
        width_shift: 0,
        height_shift: 0,
    };
    let painting = painting(layout, SampleRange::Limited, 0.003);
    let luma = vec![4.0; 16];
    let chroma = vec![128.0; 16];
    let map = vec![0.0; 16];

    let output = reference_heatmap([&luma, &chroma, &chroma], &map, 4, 4, &painting);

    for value in &output[0] {
        assert!((value - 16.0).abs() < 1e-3, "{value}");
    }
}
