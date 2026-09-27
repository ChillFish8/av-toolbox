use super::alpha::haloed_step;
use crate::dehalo::PADDING;
use crate::dehalo::yahr::YahrParams;
use crate::depth::Element;
use crate::testing::{
    reference_crop,
    reference_narrow,
    reference_pad,
    reference_padded_warp,
    reference_warp_sharp,
    reference_yahr,
};
use crate::warp::{BlurKind, WarpParams};

fn defaults(bits: u32) -> YahrParams {
    YahrParams {
        blur: 2,
        depth: 32,
        element: Element::U16,
        bits,
    }
}

/// The value at `column` on interior row 8.
fn at(values: &[f32], width: u32, column: u32) -> f32 {
    values[(8 * width + column) as usize]
}

#[test]
fn the_warp_params_follow_yahr() {
    let params = YahrParams {
        blur: 3,
        depth: -8,
        element: Element::U8,
        bits: 8,
    };

    let warp_params = params.warp_params();

    assert_eq!(warp_params.thresh, 128);
    assert_eq!(warp_params.blur, BlurKind::R6);
    assert_eq!(warp_params.passes, 3);
    assert_eq!(warp_params.depth, -8);
    assert_eq!(warp_params.element, Element::U8);
    assert_eq!(warp_params.bits, 8);
}

#[test]
fn a_flat_plane_comes_back_flat() {
    for bits in [8, 10] {
        let flat = vec![120.0f32; 64 * 32];
        let params = defaults(bits);

        let output = reference_yahr(&flat, 64, 32, &params);

        assert!(
            output.iter().all(|value| (value - 120.0).abs() < 1e-3),
            "bits {bits}"
        );
    }
}

#[test]
fn a_line_art_halo_loses_at_least_half_its_overshoot() {
    let width = 64;
    let height = 16;
    let row: Vec<f32> = (0..width)
        .map(|column| match column {
            30 => 200.0,
            31..33 => 30.0,
            33 => 200.0,
            _ => 128.0,
        })
        .collect();
    let input = row.repeat(height as usize);
    let params = defaults(8);

    let output = reference_yahr(&input, width, height, &params);

    // Half of the halo's 72 overshoot above the 128 background.
    let limit = 36.0;
    for column in [30, 33] {
        let overshoot = at(&output, width, column) - 128.0;
        assert!(overshoot <= limit, "column {column}: overshoot {overshoot}");
    }

    // The dark line itself must stay below the background, so this is dehaloing and not just
    // flattening the whole area to 128.
    for column in [31, 32] {
        let value = at(&output, width, column);
        assert!(value < 128.0, "column {column}: {value}");
    }
}

#[test]
fn a_soft_edge_overshoot_loses_at_least_half() {
    let width = 64;
    let height = 16;
    let row: Vec<f32> = (0..width)
        .map(|column| match column {
            0..32 => 60.0,
            32 => 95.0,
            33 => 130.0,
            34 => 165.0,
            35 => 200.0,
            36 => 240.0,
            _ => 200.0,
        })
        .collect();
    let input = row.repeat(height as usize);
    let params = defaults(8);

    let output = reference_yahr(&input, width, height, &params);

    // Half of the peak's 40 overshoot above the 200 it settles at either side.
    let overshoot = at(&output, width, 36) - 200.0;
    assert!(overshoot <= 20.0, "overshoot {overshoot}");
}

#[test]
fn a_wide_saturated_step_passes_through_nearly_unchanged() {
    let width = 64;
    let height = 16;
    let input = haloed_step(width, height);
    let params = defaults(8);

    let output = reference_yahr(&input, width, height, &params);

    // A wide, fully saturated ring gives the warp's edge mask no slope to act on. YAHR should
    // barely touch it.
    for row in 1..(height - 1) {
        for column in 1..(width - 1) {
            let index = (row * width + column) as usize;
            let moved = (output[index] - input[index]).abs();

            assert!(moved <= 2.0, "row {row} column {column}: moved {moved}");
        }
    }
}

#[test]
fn the_padded_warp_matches_warping_a_padded_plane() {
    let width = 20;
    let height = 12;
    let input: Vec<f32> = (0..width * height)
        .map(|index| ((index * 37 + 11) % 256) as f32)
        .collect();
    let params = WarpParams {
        element: Element::U8,
        bits: 8,
        thresh: 128,
        blur: BlurKind::R6,
        passes: 2,
        depth: 32,
    };

    let actual = reference_padded_warp(&input, width, height, &params);

    let padded = reference_pad(&input, width, height, PADDING);
    let padded_width = width + 2 * PADDING;
    let padded_height = height + 2 * PADDING;
    let peak = params.peak() as f32;
    let narrowed = reference_narrow(&padded, peak);
    let warped = reference_warp_sharp(&narrowed, padded_width, padded_height, &params);
    let widened: Vec<f32> = warped.iter().map(|value| f32::from(*value)).collect();
    let expected = reference_crop(&widened, padded_width, padded_height, PADDING);

    assert_eq!(actual, expected);
}
