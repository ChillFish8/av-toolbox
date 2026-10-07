use crate::cnn::{BodyPath, Precision};
use crate::dpir::{DPIR_BUDGET, OVERLAP, Padded, ScratchSizes, scratch_sizes};

/// The bytes of scratch a region of `sizes` takes, four maps a level and the columns.
fn total_bytes(sizes: ScratchSizes) -> usize {
    let maps: usize = sizes.level_bytes.iter().sum();

    4 * maps + sizes.column_bytes
}

#[test]
fn padded_of_matches_the_reference_rule() {
    let odd = Padded {
        width: 56,
        height: 40,
        left: 3,
        top: 1,
    };
    let aligned = Padded {
        width: 48,
        height: 40,
        left: 0,
        top: 0,
    };
    let narrow = Padded {
        width: 8,
        height: 8,
        left: 2,
        top: 1,
    };

    assert_eq!(Padded::of(50, 38), odd);
    assert_eq!(Padded::of(48, 40), aligned);
    assert_eq!(Padded::of(3, 5), narrow);
}

#[test]
fn a_1080p_f16_frame_runs_whole() {
    let sizes = scratch_sizes(Precision::F16, BodyPath::Convolution, 1920, 1080, 0);

    assert_eq!(sizes.tile, 1920);
    assert_eq!(sizes.column_bytes, 0);
}

#[test]
fn a_2160p_frame_tiles_at_a_multiple_of_8() {
    let sizes = scratch_sizes(Precision::F16, BodyPath::Convolution, 3840, 2160, 0);

    assert!(sizes.tile.is_multiple_of(8), "{sizes:?}");
    assert!(sizes.tile < 3840, "{sizes:?}");
    assert!(total_bytes(sizes) <= DPIR_BUDGET, "{sizes:?}");
}

#[test]
fn a_small_allocation_forces_tiles() {
    let max_allocation = 64 << 20;
    let sizes = scratch_sizes(Precision::F16, BodyPath::Convolution, 1920, 1080, max_allocation);

    assert!(sizes.tile < 1920, "{sizes:?}");
    assert!(sizes.tile.is_multiple_of(8), "{sizes:?}");
    assert!(sizes.level_bytes[0] <= max_allocation, "{sizes:?}");
}

#[test]
fn an_allocation_the_first_map_exactly_fills_runs_whole() {
    let whole = scratch_sizes(Precision::F16, BodyPath::Convolution, 1920, 1080, 0);
    let exact = scratch_sizes(
        Precision::F16,
        BodyPath::Convolution,
        1920,
        1080,
        whole.level_bytes[0],
    );

    assert_eq!(exact, whole);
}

#[test]
fn im2col_counts_the_columns() {
    let sizes = scratch_sizes(Precision::F32, BodyPath::Im2col, 1920, 1080, 0);

    assert!(sizes.tile < 1920, "{sizes:?}");
    assert!(sizes.column_bytes > 0, "{sizes:?}");
    assert!(total_bytes(sizes) <= DPIR_BUDGET, "{sizes:?}");
}

#[test]
fn a_tile_reports_its_region_clipped_to_the_padded_frame() {
    let sizes = scratch_sizes(Precision::F32, BodyPath::Im2col, 1920, 1080, 0);
    let region = sizes.tile + 2 * OVERLAP;
    let region_height = region.min(1080) as usize;
    let first_level = region as usize * region_height * 64 * 4;

    assert_eq!(sizes.level_bytes[0], first_level, "{sizes:?}");
    assert_eq!(sizes.column_bytes, 9 * first_level, "{sizes:?}");
}

#[test]
fn the_tile_is_the_largest_that_fits() {
    let max_allocation = 64 << 20;
    let sizes = scratch_sizes(Precision::F16, BodyPath::Convolution, 1920, 1080, max_allocation);
    let larger = sizes.tile + 8 + 2 * OVERLAP;
    let larger_height = larger.min(1080) as usize;
    let larger_first_level = larger as usize * larger_height * 64 * 2;

    assert!(larger_first_level > max_allocation, "{sizes:?}");
}
