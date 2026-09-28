use crate::dehalo::dering::{DeringParams, contra_limit_sample, limit_sample, ring_mask_sample};
use crate::rgvs::RepairMode;
use crate::testing::reference_hq_dering;

fn test_params(bits: u32) -> DeringParams {
    DeringParams {
        mrad: 1,
        msmooth: 1,
        incedge: false,
        mthr: 60.0,
        minp: 1,
        nrmode: 2,
        sharp: 1,
        drrep: None,
        thr: 12.0,
        elast: 2.0,
        darkthr: 4.0,
        bits,
    }
}

#[test]
fn contra_limit_takes_the_smaller_change() {
    // |3| > |5| is false, so it picks the repaired value, 3: 100 + 3 = 103.
    let picks_repaired = contra_limit_sample(100.0, 5.0, 3.0);
    // |-8| > |5| is true, so it picks the sharpened value, 5: 100 + 5 = 105.
    let picks_sharpened = contra_limit_sample(100.0, 5.0, -8.0);
    // |5| > |-5| is false, since it is not a strictly greater magnitude, so it picks the repaired
    // value, 5: 100 + 5 = 105.
    let ties_pick_repaired = contra_limit_sample(100.0, -5.0, 5.0);

    assert_eq!(picks_repaired, 103.0);
    assert_eq!(picks_sharpened, 105.0);
    assert_eq!(ties_pick_repaired, 105.0);
}

#[test]
fn limit_passes_small_changes_through() {
    // d = 110 - 100 = 10, which is > 0, so t = darkthr = 3. t * elast = 6, and 10 >= 6, so it falls
    // back to the plane, 100.
    let falls_back_to_the_plane = limit_sample(100.0, 110.0, 12.0, 3.0, 2.0);
    // d = 90 - 100 = -10, so t = thr = 12. 10 <= 12, so it passes the filtered value through, 90.
    let passes_through = limit_sample(100.0, 90.0, 12.0, 3.0, 2.0);

    assert_eq!(falls_back_to_the_plane, 100.0);
    assert_eq!(passes_through, 90.0);
}

#[test]
fn limit_blends_between_the_thresholds() {
    // d = 82 - 100 = -18, so t = thr = 12 and t * elast = 24. 18 sits between 12 and 24, so it
    // blends: 100 - 18 * (24 - 18) / (24 - 12) = 100 - 18 * 6 / 12 = 100 - 9 = 91.
    let blended = limit_sample(100.0, 82.0, 12.0, 3.0, 2.0);

    assert_eq!(blended, 91.0);
}

#[test]
fn limit_never_divides_by_zero() {
    // thr = 0, darkthr = 0, elast = 1. d = 5 is > 0, so t = darkthr = 0 and t * elast = 0. 5 >= 0,
    // so it falls back to the plane rather than dividing by a zero span.
    let falls_back = limit_sample(100.0, 105.0, 0.0, 0.0, 1.0);
    // d = 0, so t = thr = 0. 0 <= 0, so it passes the filtered value through.
    let passes_through = limit_sample(100.0, 100.0, 0.0, 0.0, 1.0);

    assert_eq!(falls_back, 100.0);
    assert_eq!(passes_through, 100.0);
    assert!(falls_back.is_finite());
    assert!(passes_through.is_finite());
}

#[test]
fn ring_mask_cuts_the_inner_mask_out() {
    // outer == inner == peak: 255 * (255 - 255) / 255 = 0.
    let fully_cut = ring_mask_sample(255.0, 255.0, false, 255.0);
    // inner 0: 255 * (255 - 0) / 255 = 255.
    let untouched = ring_mask_sample(255.0, 0.0, false, 255.0);
    // incedge always gives the outer mask back, regardless of inner.
    let incedge_high_inner = ring_mask_sample(255.0, 255.0, true, 255.0);
    let incedge_low_inner = ring_mask_sample(255.0, 0.0, true, 255.0);

    assert_eq!(fully_cut, 0.0);
    assert_eq!(untouched, 255.0);
    assert_eq!(incedge_high_inner, 255.0);
    assert_eq!(incedge_low_inner, 255.0);
}

#[test]
fn a_flat_plane_comes_back_flat() {
    let flat = vec![120.0f32; 64 * 32];
    let modes = [
        None,
        Some(RepairMode::One),
        Some(RepairMode::Two),
        Some(RepairMode::Thirteen),
        Some(RepairMode::Seventeen),
        Some(RepairMode::TwentyFour),
    ];

    for sharp in 0..=3 {
        for minp in 0..=4 {
            for incedge in [false, true] {
                for drrep in modes {
                    let params = DeringParams {
                        sharp,
                        minp,
                        incedge,
                        drrep,
                        ..test_params(8)
                    };

                    let output = reference_hq_dering(&flat, 64, 32, &params);

                    assert!(
                        output.iter().all(|value| (value - 120.0).abs() < 1e-3),
                        "sharp {sharp} minp {minp} incedge {incedge} drrep {drrep:?}"
                    );
                }
            }
        }
    }
}

/// One row of a sharp step at column 20, with small alternating ringing on both sides of it, and a
/// smaller alternating ripple on the far columns this test expects to stay untouched.
fn ringing_row(width: u32) -> Vec<f32> {
    let low = 60.0;
    let high = 200.0;
    let ripple = 15.0;
    let far_ripple = 3.0;
    let edge = 20u32;
    let ring_before = 15..20u32;
    let ring_after = 20..25u32;
    let far_before = 0..7u32;
    let far_after = 37..44u32;

    (0..width)
        .map(|column| {
            let sign = if column % 2 == 0 { 1.0 } else { -1.0 };

            if far_before.contains(&column) {
                low + sign * far_ripple
            } else if far_after.contains(&column) {
                high + sign * far_ripple
            } else if ring_before.contains(&column) {
                let ring_sign = if (column - 15) % 2 == 0 { 1.0 } else { -1.0 };
                low + ring_sign * ripple
            } else if ring_after.contains(&column) {
                let ring_sign = if (column - 20) % 2 == 0 { -1.0 } else { 1.0 };
                high + ring_sign * ripple
            } else if column < edge {
                low
            } else {
                high
            }
        })
        .collect()
}

#[test]
fn a_ringing_edge_is_smoothed_near_the_edge_only() {
    let width = 44;
    let height = 6;
    let row = ringing_row(width);
    let plane: Vec<f32> = (0..height).flat_map(|_| row.clone()).collect();

    let params = DeringParams {
        mrad: 2,
        msmooth: 1,
        incedge: true,
        // The horizontal Prewitt response is 3|c(x-1) - c(x+1)|, which is 0 inside the far ripple
        // and 9 at its edge. 20 sits above that 9 and far below the step edge's own response, so it
        // zeroes the far ripple out of the mask without touching the real edge.
        mthr: 20.0,
        minp: 1,
        nrmode: 1,
        sharp: 0,
        drrep: None,
        thr: 2.0,
        elast: 10.0,
        darkthr: 2.0,
        bits: 8,
    };

    let output = reference_hq_dering(&plane, width, height, &params);

    // Columns 19 and 20 are the true step edge itself, not ringing beside it, so they are left out
    // of both sets. MinBlur is expected to preserve a genuine edge rather than smooth it away.
    let ringing_columns: Vec<u32> = (15..19).chain(21..25).collect();
    let far_columns: Vec<u32> = (0..=6).chain(37..=43).collect();

    for row_index in 0..height {
        let row_start = (row_index * width) as usize;
        let row_end = row_start + width as usize;
        let input_row = &plane[row_start..row_end];
        let output_row = &output[row_start..row_end];

        for column in &ringing_columns {
            let index = (row_index * width + column) as usize;
            let changed = (output[index] - plane[index]).abs() > 1e-3;

            assert!(
                changed,
                "row {row_index} column {column} did not change\ninput row:  {input_row:?}\noutput row: {output_row:?}"
            );
        }

        for column in &far_columns {
            let index = (row_index * width + column) as usize;

            assert!(
                (output[index] - plane[index]).abs() < 1e-3,
                "row {row_index} column {column} changed\ninput row:  {input_row:?}\noutput row: {output_row:?}"
            );
        }
    }
}
