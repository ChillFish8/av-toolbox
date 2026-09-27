use crate::depth::Element;
use crate::testing::{reference_blur_mask, reference_edge_mask, reference_warp, reference_warp_sharp};
use crate::warp::{BlurKind, WarpParams};

fn eight_bit(thresh: u32, depth: i32) -> WarpParams {
    WarpParams {
        element: Element::U8,
        bits: 8,
        thresh,
        blur: BlurKind::R6,
        passes: 2,
        depth,
    }
}

/// Three rows of `row`.
fn three_rows(row: &[u16]) -> Vec<u16> {
    row.repeat(3)
}

#[test]
fn the_peak_and_threshold_follow_the_depth() {
    let ten_bit = WarpParams {
        element: Element::U16,
        bits: 10,
        ..eight_bit(128, 16)
    };

    assert_eq!(ten_bit.peak(), 1023);
    assert_eq!(ten_bit.scaled_thresh(), 512);
}

#[test]
fn a_small_step_gives_an_even_mask() {
    // Every inner sample sees left 10 and right 14, which works out to 48, and the border copies it.
    let input = three_rows(&[10, 10, 14, 14]);
    let params = eight_bit(255, 16);

    let mask = reference_edge_mask(&input, 4, 3, &params);

    assert_eq!(mask, vec![48; 12]);
}

#[test]
fn the_threshold_caps_the_mask() {
    let input = three_rows(&[10, 10, 14, 14]);
    let params = eight_bit(40, 16);

    let mask = reference_edge_mask(&input, 4, 3, &params);

    assert_eq!(mask, vec![40; 12]);
}

#[test]
fn a_plane_too_small_for_a_neighbourhood_has_no_edges() {
    let input = vec![0, 255, 0, 255];
    let params = eight_bit(255, 16);

    let mask = reference_edge_mask(&input, 2, 2, &params);

    assert_eq!(mask, vec![0; 4]);
}

#[test]
fn the_r2_blur_matches_the_plugin_by_hand() {
    // A single row, so only the horizontal pass changes anything.
    let mask = vec![0, 0, 0, 0, 64];

    let blurred = reference_blur_mask(&mask, 5, 1, BlurKind::R2);

    assert_eq!(blurred[2], 4);
    assert_eq!(blurred[4], 44);
}

#[test]
fn the_r6_blur_weighs_the_centre_three_eighths() {
    let mut mask = vec![0u32; 13];
    mask[6] = 64;

    let blurred = reference_blur_mask(&mask, 13, 1, BlurKind::R6);

    assert_eq!(blurred[6], 24);
}

#[test]
fn a_flat_mask_stays_flat_through_either_blur() {
    let mask = vec![77u32; 20 * 15];

    let wide = reference_blur_mask(&mask, 20, 15, BlurKind::R6);
    let narrow = reference_blur_mask(&mask, 20, 15, BlurKind::R2);

    assert_eq!(wide, mask);
    assert_eq!(narrow, mask);
}

#[test]
fn a_shallow_depth_blends_toward_the_edge() {
    // The centre sees left 0 and right 64, so it moves a quarter sample left and blends 0 with 100.
    let input = three_rows(&[0, 100, 200]);
    let mask = vec![0, 0, 64, 0, 0, 64, 0, 0, 64];
    let params = eight_bit(128, 1);

    let warped = reference_warp(&input, &mask, 3, 3, &params);

    assert_eq!(warped[4], 75);
}

#[test]
fn a_deep_warp_clamps_at_the_plane_edge() {
    let input = three_rows(&[0, 100, 200]);
    let mask = vec![0, 0, 64, 0, 0, 64, 0, 0, 64];
    let params = eight_bit(128, 16);

    let warped = reference_warp(&input, &mask, 3, 3, &params);

    assert_eq!(warped[4], 0);
}

#[test]
fn a_flat_plane_is_unchanged_by_the_whole_warp() {
    let input = vec![90u16; 24 * 16];
    let params = eight_bit(128, 16);

    let warped = reference_warp_sharp(&input, 24, 16, &params);

    assert_eq!(warped, input);
}

#[test]
fn a_zero_threshold_moves_only_the_last_row() {
    // The plugin clamps the last row's vertical offset to -1/128, so it always takes 1/128 of the row above.
    let input: Vec<u16> = (0..24 * 16).map(|index| (index * 37 % 256) as u16).collect();
    let params = eight_bit(0, 16);

    let warped = reference_warp_sharp(&input, 24, 16, &params);

    let last_row_start = 24 * 15;
    assert_eq!(warped[..last_row_start], input[..last_row_start]);

    for column in 0..24 {
        let here = u32::from(input[last_row_start + column]);
        let above = u32::from(input[last_row_start - 24 + column]);
        let blended = (above + here * 127 + 64) >> 7;

        assert_eq!(
            u32::from(warped[last_row_start + column]),
            blended,
            "column {column}"
        );
    }
}
