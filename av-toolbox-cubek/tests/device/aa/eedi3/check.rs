use av_toolbox_cubek::aa::Eedi3Check;
use av_toolbox_cubek::testing::{CloseDecisionTally, Eedi3Fixture, Eedi3Parameters, assert_same_plane};

use super::{
    Bands,
    PlaneCase,
    assert_tally,
    compare_passes,
    diagonal_bands,
    larger_masked_case,
    on_device,
    on_device_together,
    on_device_with_capacity,
    random_based_case,
};

/// Asserts that `case` runs the persistent check at `persistent` and the per-row check at `per_row`, and
/// that both runs give the same output and directions exactly.
///
/// Without a `persistent` capacity the run takes the device's.
fn assert_checks_agree(case: &PlaneCase, persistent: Option<u32>, per_row: u32) {
    let persistent_passes = on_device_with_capacity(case, persistent);
    let per_row_passes = on_device_with_capacity(case, Some(per_row));

    let pairs = persistent_passes.iter().zip(&per_row_passes);
    for (offset, (persistent_pass, per_row_pass)) in pairs.enumerate() {
        let label = format!("{} at offset {offset}", case.label);
        assert_eq!(persistent_pass.check, Eedi3Check::Persistent, "{label}");
        assert_eq!(per_row_pass.check, Eedi3Check::PerRow, "{label}");

        let output_label = format!("{label}: the output");
        assert_same_plane(
            &persistent_pass.output,
            &per_row_pass.output,
            case.width,
            &output_label,
        );

        let directions_label = format!("{label}: the directions");
        assert_same_plane(
            &persistent_pass.directions,
            &per_row_pass.directions,
            case.width,
            &directions_label,
        );
    }
}

/// Noisy diagonal bands, `width` by `height`, at the default arguments.
fn noisy_bands_case(label: &str, width: u32, height: u32) -> PlaneCase {
    let bands = Bands {
        slope: 3,
        period: 23,
        low: 20,
        high: 230,
        noise: 40,
    };
    let input = diagonal_bands(width, height, &bands);

    PlaneCase::new(label, width, height, Eedi3Parameters::default(), input)
}

#[test]
fn the_persistent_check_matches_the_per_row_check() {
    for seed in 0..=30 {
        let case = random_based_case(seed);
        assert_checks_agree(&case, None, 1);
    }

    let masked = larger_masked_case(Eedi3Parameters::default());
    assert_checks_agree(&masked, None, 1);

    let wide = noisy_bands_case("1500 wide", 1500, 40);
    assert_checks_agree(&wide, None, 1);
}

#[test]
fn the_capacity_edge_picks_the_path() {
    let case = noisy_bands_case("64 wide", 64, 12);

    assert_checks_agree(&case, Some(64), 63);
}

#[test]
fn a_plane_with_no_checked_row_is_still_assembled() {
    let case = noisy_bands_case("9 by 4", 9, 4);

    assert_checks_agree(&case, None, 1);
}

#[test]
fn vcheck_zero_skips_the_check() {
    let mut case = noisy_bands_case("vcheck 0", 240, 48);
    case.parameters.vcheck = 0;

    let passes = on_device(&case);
    let forced_passes = on_device_with_capacity(&case, Some(1));
    let every_pass = passes.iter().chain(&forced_passes);
    for pass in every_pass {
        assert_eq!(pass.check, Eedi3Check::Skipped);
    }

    let mut tally = CloseDecisionTally::new();
    compare_passes(&mut tally, &case, &passes);
    assert_tally(&tally, "vcheck 0");
}

#[test]
fn a_tall_plane_checks_every_row_in_order() {
    let case = noisy_bands_case("200 by 300", 200, 300);

    assert_checks_agree(&case, None, 1);
}

/// Asserts that both passes of `case` through one call match two single passes exactly, with the
/// persistent check running only on planes at most `capacity` wide when there is one.
fn assert_together_matches_single(case: &PlaneCase, capacity: Option<u32>) {
    let together_passes = on_device_together(case, capacity);
    let single_passes = on_device_with_capacity(case, capacity);
    assert_eq!(together_passes.len(), single_passes.len(), "{}", case.label);

    let pairs = together_passes.iter().zip(&single_passes);
    for (offset, (together_pass, single_pass)) in pairs.enumerate() {
        let label = format!("{} at offset {offset}", case.label);
        assert_eq!(together_pass.check, single_pass.check, "{label}");

        let output_label = format!("{label}: the output");
        assert_same_plane(
            &together_pass.output,
            &single_pass.output,
            case.width,
            &output_label,
        );

        let directions_label = format!("{label}: the directions");
        assert_same_plane(
            &together_pass.directions,
            &single_pass.directions,
            case.width,
            &directions_label,
        );
    }
}

#[test]
fn both_parities_in_one_launch_match_two_single_passes() {
    let mut cases: Vec<PlaneCase> = (0..=30).map(random_based_case).collect();
    cases.push(larger_masked_case(Eedi3Parameters::default()));
    cases.push(noisy_bands_case("1500 wide", 1500, 40));

    let fixture = Eedi3Fixture::named("based");
    let mut based = PlaneCase::of_fixture(fixture);
    based.sclip = fixture.sclip_plane();
    based.mclip = fixture.mclip_plane();
    cases.push(based);

    let mut unchecked = noisy_bands_case("vcheck 0", 240, 48);
    unchecked.parameters.vcheck = 0;
    cases.push(unchecked);

    for case in &cases {
        assert_together_matches_single(case, None);
        assert_together_matches_single(case, Some(1));
    }
}
