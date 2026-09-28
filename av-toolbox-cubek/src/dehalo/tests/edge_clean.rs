use crate::dehalo::edge_clean::{EdgeCleanParams, levels_sample, threshold_sample};
use crate::depth::Element;
use crate::rgvs::RepairMode;
use crate::testing::reference_edge_clean;

fn defaults(bits: u32) -> EdgeCleanParams {
    EdgeCleanParams {
        strength: 10,
        rep: false,
        rmode: RepairMode::Seventeen,
        smode: false,
        hot: false,
        element: Element::U8,
        bits,
    }
}

#[test]
fn the_warp_depth_rounds_half_the_strength_up() {
    // depth = (strength + 4 * smode).div_ceil(2).
    //
    // strength 10, smode false: (10 + 0).div_ceil(2) = 10.div_ceil(2) = 5.
    let rounds_evenly = EdgeCleanParams {
        strength: 10,
        ..defaults(8)
    };

    // strength 9, smode false: (9 + 0).div_ceil(2) = 9.div_ceil(2) = 5, so 4.5 rounds up to 5.
    let rounds_up_from_a_half = EdgeCleanParams {
        strength: 9,
        ..defaults(8)
    };

    // strength 250, smode true: (250 + 4).div_ceil(2) = 254.div_ceil(2) = 127.
    let smode_shifts_the_strength = EdgeCleanParams {
        strength: 250,
        smode: true,
        ..defaults(8)
    };

    assert_eq!(rounds_evenly.warp_params().depth, 5);
    assert_eq!(rounds_up_from_a_half.warp_params().depth, 5);
    assert_eq!(smode_shifts_the_strength.warp_params().depth, 127);
}

#[test]
fn the_threshold_zeroes_weak_edges_and_fills_strong_ones() {
    // At 8 bits: peak 255, low 4, high 32.
    //
    // 3 is below low, so it zeroes.
    // 4 sits at low, so it passes through unchanged.
    // 20 sits between low and high, so it passes through unchanged.
    // 32 sits at high, so it passes through unchanged.
    // 33 is above high, so it fills to peak, 255.
    let peak = 255.0;
    let low = 4.0;
    let high = 32.0;
    let values = [3.0, 4.0, 20.0, 32.0, 33.0];

    let plain: Vec<f32> = values
        .iter()
        .map(|value| threshold_sample(*value, low, high, peak, false))
        .collect();

    // Inverted flips every result around peak: peak - result.
    let inverted: Vec<f32> = values
        .iter()
        .map(|value| threshold_sample(*value, low, high, peak, true))
        .collect();

    assert_eq!(plain, vec![0.0, 4.0, 20.0, 32.0, 255.0]);
    assert_eq!(inverted, vec![255.0, 251.0, 235.0, 223.0, 0.0]);
}

#[test]
fn levels_stretch_and_curve() {
    // At 8 bits: peak 255, low 40, span 128, exponent 1 / 0.35 ~= 2.857143.
    //
    // 40 sits at low, so stretched = 0 / 128 = 0, then 0 ^ exponent * 255 = 0.
    // 168 sits at low + span, so stretched = 128 / 128 = 1, then 1 ^ exponent * 255 = 255.
    // 104 sits halfway, so stretched = 64 / 128 = 0.5.
    //   ln(0.5) = -0.693147, times 2.857143 = -1.980420.
    //   e^-1.980420 = e^-2 * e^0.019580 ~= 0.135335 * 1.019758 ~= 0.138013.
    //   0.138013 * 255 ~= 35.193.
    // 20 sits below low, so stretched clamps to 0, giving 0.
    // 200 sits above low + span, so stretched clamps to 1, giving 255.
    let peak = 255.0;
    let low = 40.0;
    let span = 128.0;
    let exponent = 1.0 / 0.35;

    let at_low = levels_sample(40.0, low, span, exponent, peak);
    let at_high = levels_sample(168.0, low, span, exponent, peak);
    let midpoint = levels_sample(104.0, low, span, exponent, peak);
    let below_the_window = levels_sample(20.0, low, span, exponent, peak);
    let above_the_window = levels_sample(200.0, low, span, exponent, peak);

    assert_eq!(at_low, 0.0);
    assert_eq!(at_high, 255.0);
    assert!((midpoint - 35.193).abs() < 0.01, "got {midpoint}");
    assert_eq!(below_the_window, 0.0);
    assert_eq!(above_the_window, 255.0);
}

#[test]
fn a_flat_plane_comes_back_flat() {
    let flat = vec![120.0f32; 64 * 32];

    for rep in [false, true] {
        for smode in [false, true] {
            for hot in [false, true] {
                let params = EdgeCleanParams {
                    rep,
                    smode,
                    hot,
                    ..defaults(8)
                };

                let output = reference_edge_clean(&flat, 64, 32, &params);

                assert!(
                    output.iter().all(|value| (value - 120.0).abs() < 1e-3),
                    "rep {rep} smode {smode} hot {hot}"
                );
            }
        }
    }
}
