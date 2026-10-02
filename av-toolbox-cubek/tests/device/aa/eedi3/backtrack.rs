use av_toolbox_cubek::testing::{CloseDecisionTally, Eedi3Parameters, SplitMix, random_eedi3_parameters};

use super::{Bands, PlaneCase, assert_tally, compare_passes, diagonal_bands, on_device};

/// The columns of a mask set on single columns spaced `gap` apart.
fn island_columns(width: u32, gap: u32) -> Vec<u32> {
    (gap / 2..width).step_by(gap as usize).collect()
}

/// A mask set on `columns` on every row.
fn islands(width: u32, height: u32, columns: &[u32]) -> Vec<f32> {
    let mut mask = vec![0.0; width as usize * height as usize];
    for row in 0..height {
        for column in columns {
            mask[(row * width + column) as usize] = 255.0;
        }
    }

    mask
}

/// The columns further than `mdis` from every island.
fn inactive_columns(width: u32, columns: &[u32], mdis: u32) -> Vec<u32> {
    (0..width)
        .filter(|column| columns.iter().all(|island| column.abs_diff(*island) > mdis))
        .collect()
}

#[test]
fn inactive_columns_beside_active_ones_match() {
    let input_bands = Bands {
        slope: 2,
        period: 9,
        low: 20,
        high: 230,
        noise: 15,
    };
    let mdis = 3;
    let height = 16;

    let mut tally = CloseDecisionTally::new();
    for (width, gap) in [(31, 9), (32, 11), (33, 9), (97, 13), (256, 40)] {
        let mut random = SplitMix(u64::from(width + gap));
        let drawn = random_eedi3_parameters(&mut random, 8);
        let parameters = Eedi3Parameters { mdis, ..drawn };
        let input = diagonal_bands(width, height, &input_bands);
        let columns = island_columns(width, gap);
        let label = format!("{width} wide, islands {gap} apart");

        let mut case = PlaneCase::new(&label, width, height, parameters, input);
        case.mclip = Some(islands(width, height, &columns));
        let passes = on_device(&case);
        compare_passes(&mut tally, &case, &passes);

        let inactive = inactive_columns(width, &columns, mdis);
        assert!(!inactive.is_empty(), "{label}: no column is inactive");

        for pass in &passes {
            for row in pass.directions.chunks(width as usize) {
                for column in &inactive {
                    let direction = row[*column as usize];
                    assert_eq!(direction, 0, "{label}: inactive column {column} took a direction");
                }
            }
        }
    }

    assert_tally(&tally, "islands");
}
