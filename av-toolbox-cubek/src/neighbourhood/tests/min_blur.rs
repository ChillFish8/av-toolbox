use crate::neighbourhood::median_of_three_sample;
use crate::testing::reference_min_blur;

#[test]
fn the_pick_keeps_the_source_between_the_two() {
    // (5 - 3)(5 - 8) = 2 * -3 = -6 < 0, so the source is between the two.
    assert_eq!(median_of_three_sample(5.0, 3.0, 8.0), 5.0);
}

#[test]
fn the_pick_takes_the_nearer_of_the_two_otherwise() {
    // (5 - 6)(5 - 9) = -1 * -4 = 4, not between. |−1| < |−4|, so the nearer is the blur.
    assert_eq!(median_of_three_sample(5.0, 6.0, 9.0), 6.0);

    // (5 - 9)(5 - 6) = -4 * -1 = 4, not between. |−4| < |−1| is false, so it takes the median.
    assert_eq!(median_of_three_sample(5.0, 9.0, 6.0), 6.0);

    // A blur and median tied at 9: (5 - 9)(5 - 9) = 16, not between, and |−4| < |−4| is false, so
    // it takes the median.
    assert_eq!(median_of_three_sample(5.0, 9.0, 9.0), 9.0);
}

/// A 9x9 plane of 10 with a single 200 spike in the centre.
fn spiked_plane() -> Vec<f32> {
    let mut plane = vec![10.0; 81];
    plane[40] = 200.0;
    plane
}

#[test]
fn min_blur_dampens_a_lone_spike_more_at_larger_radii() {
    let plane = spiked_plane();

    // Every radius's median already lands on the background, since a lone spike is always the odd
    // one out in its window. The pick therefore always takes the blur, which comes closer to the
    // background as more blur passes run over it.
    let radius_1 = reference_min_blur(&plane, 9, 9, 1);
    let radius_2 = reference_min_blur(&plane, 9, 9, 2);
    let radius_3 = reference_min_blur(&plane, 9, 9, 3);

    // Radius 1 blurs once with Binomial: the centre keeps weight 4 of 16, its 4 edge neighbours
    // weight 2 each and its 4 corner neighbours weight 1 each, all at 10 except the centre at 200:
    // (200*4 + 10*2*4 + 10*1*4) / 16 = 920 / 16 = 57.5.
    assert!((radius_1[40] - 57.5).abs() < 1e-3, "got {}", radius_1[40]);

    // Radius 2 blurs again with Box (a plain average) over that same 3x3 of Binomial results
    // (57.5 at the centre, 33.75 on each edge neighbour, 21.875 on each corner neighbour):
    // (57.5 + 4*33.75 + 4*21.875) / 9 = 280 / 9 ≈ 31.111.
    assert!((radius_2[40] - 31.111_11).abs() < 1e-3, "got {}", radius_2[40]);

    // Radius 3 runs a second Box pass over the radius 2 blur. That window now sees the centre's own
    // radius 2 value (280 / 9), its 4 edge neighbours' (155 / 6, boxing the Binomial window around
    // each) and its 4 corner neighbours' (21.875, unchanged since boxing their own Binomial value
    // only pulls in more background):
    // (280 / 9 + 4 * 155 / 6 + 4 * 21.875) / 9 = 15980 / 648 = 3995 / 162 ≈ 24.660.
    assert!((radius_3[40] - 24.660_49).abs() < 1e-3, "got {}", radius_3[40]);

    // Each extra pass halves roughly how far the centre sits above the background, but a lone spike
    // is never fully removed by blurring alone.
    assert!(radius_1[40] > radius_2[40]);
    assert!(radius_2[40] > radius_3[40]);
    assert!(radius_3[40] > 10.0);
}

#[test]
fn min_blur_keeps_a_flat_plane_flat() {
    let plane = vec![42.0; 49];

    for radius in 1..=3 {
        let filtered = reference_min_blur(&plane, 7, 7, radius);

        assert_eq!(filtered, plane, "radius {radius}");
    }
}
