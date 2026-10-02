use av_toolbox_cubek::testing::{CloseDecisionTally, Eedi3Parameters, SplitMix, random_eedi3_parameters};

use super::{Bands, PlaneCase, assert_tally, column_band_mask, compare_case, diagonal_bands};

/// Mask runs of 1 to 9 columns, separated by gaps of 1 to 12 columns.
fn stuttering_bands(width: u32) -> Vec<(u32, u32)> {
    let mut bands = Vec::new();
    let mut left = 3;
    let mut length = 1;
    let mut gap = 1;
    while left + length < width {
        bands.push((left, left + length));
        left += length + gap;
        length = length % 9 + 1;
        gap = gap % 12 + 1;
    }

    bands
}

#[test]
fn band_sums_restart_after_every_gap() {
    let width = 301;
    let height = 24;
    let input_bands = Bands {
        slope: 3,
        period: 11,
        low: 30,
        high: 220,
        noise: 25,
    };
    let input = diagonal_bands(width, height, &input_bands);
    let bands = stuttering_bands(width);
    let mclip = column_band_mask(width, height, &bands);

    let mut tally = CloseDecisionTally::new();
    for nrad in 0..=3 {
        for mdis in [1, 4, 20] {
            let mut random = SplitMix(u64::from(nrad * 100 + mdis));
            let drawn = random_eedi3_parameters(&mut random, 8);
            let parameters = Eedi3Parameters { nrad, mdis, ..drawn };
            let label = format!("nrad {nrad}, mdis {mdis}");

            let mut case = PlaneCase::new(&label, width, height, parameters, input.clone());
            case.mclip = Some(mclip.clone());
            compare_case(&mut tally, &case);
        }
    }

    assert_tally(&tally, "band sums across gaps");
}

#[test]
fn band_sums_hold_at_both_sides_of_the_row() {
    let width = 90;
    let height = 20;
    let mdis = 12;
    let input_bands = Bands {
        slope: 1,
        period: 7,
        low: 10,
        high: 240,
        noise: 30,
    };
    let input = diagonal_bands(width, height, &input_bands);
    let sides = [(0, mdis + 2), (width - mdis - 2, width)];
    let mclip = column_band_mask(width, height, &sides);

    let mut tally = CloseDecisionTally::new();
    for nrad in 0..=3 {
        let mut random = SplitMix(u64::from(nrad) + 7);
        let drawn = random_eedi3_parameters(&mut random, 8);
        let parameters = Eedi3Parameters { nrad, mdis, ..drawn };

        let masked_label = format!("sides at nrad {nrad}");
        let mut masked = PlaneCase::new(&masked_label, width, height, parameters, input.clone());
        masked.mclip = Some(mclip.clone());
        compare_case(&mut tally, &masked);

        let unmasked_label = format!("unmasked at nrad {nrad}");
        let unmasked = PlaneCase::new(&unmasked_label, width, height, parameters, input.clone());
        compare_case(&mut tally, &unmasked);
    }

    assert_tally(&tally, "band sums at the sides");
}
