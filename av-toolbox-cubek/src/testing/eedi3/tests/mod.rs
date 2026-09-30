use std::panic;

use super::path::{Grid, connection_cost};
use super::{
    CheckOrder,
    CloseDecisionTally,
    Contraction,
    EEDI3_FIXTURES,
    Eedi3Clips,
    Eedi3Fixture,
    Eedi3Parameters,
    Eedi3Pass,
    Eedi3Variant,
    Lending,
    Settings,
    TieOrder,
    reference_eedi3_antialias,
    reference_eedi3_field,
    reference_eedi3_field_variant,
};

const WORKED_WIDTH: usize = 12;
const WORKED_HEIGHT: usize = 8;

/// An edge between 40 and 200 that moves two columns per row.
fn worked_plane() -> Vec<f32> {
    let mut plane = Vec::with_capacity(WORKED_WIDTH * WORKED_HEIGHT);
    for row in 0..WORKED_HEIGHT {
        for column in 0..WORKED_WIDTH {
            let value = if column > 2 * row { 200.0 } else { 40.0 };
            plane.push(value);
        }
    }

    plane
}

fn worked_parameters() -> Eedi3Parameters {
    Eedi3Parameters {
        mdis: 3,
        nrad: 1,
        ..Eedi3Parameters::default()
    }
}

fn plane_of(rows: &[[u16; WORKED_WIDTH]; WORKED_HEIGHT]) -> Vec<f32> {
    rows.iter().flatten().map(|value| f32::from(*value)).collect()
}

#[test]
fn the_worked_example_matches_by_hand() {
    let plane = worked_plane();
    let parameters = worked_parameters();
    let plugin = Eedi3Variant::default();
    let settings = Settings::new(&parameters, plugin);
    let grid = Grid::new(&plane, 12, 8);

    // Costs of rebuilt row 3.
    let row_cost = |column, direction| connection_cost(&grid, 3, column, direction, &settings);
    assert_eq!(row_cost(0, 0), 64.0);
    assert_eq!(row_cost(5, -2), 88.5);
    assert_eq!(row_cost(3, -2), 0.5);
    assert_eq!(row_cost(9, -2), 0.5);
    assert_eq!(row_cost(6, 0), 184.0);
    // The table rounds these to 3 decimals.
    let right_edge = row_cost(11, 0);
    let steep = row_cost(2, -2);
    let widest = row_cost(3, 3);
    assert!((right_edge - 96.0).abs() < 0.001, "C(11, 0) is {right_edge}");
    assert!((steep - 11.167).abs() < 0.001, "C(2, -2) is {steep}");
    assert!((widest - 323.417).abs() < 0.001, "C(3, 3) is {widest}");

    let clips = Eedi3Clips::default();
    let pass0 = reference_eedi3_field(&plane, 12, 8, 0, &parameters, clips);
    let pass1 = reference_eedi3_field(&plane, 12, 8, 1, &parameters, clips);
    let merged = reference_eedi3_antialias(&plane, 12, 8, &parameters, clips);

    let row_3 = 3 * WORKED_WIDTH..4 * WORKED_WIDTH;
    let path = [0, -1, -2, -2, -2, -2, -2, -2, -2, -2, -1, 0];
    assert_eq!(pass0.directions[row_3], path);

    let expected_pass0 = plane_of(&[
        [40, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
        [40, 40, 40, 200, 200, 200, 200, 200, 200, 200, 200, 200],
        [40, 40, 40, 40, 40, 200, 200, 200, 200, 200, 200, 200],
        [40, 32, 35, 35, 35, 80, 80, 160, 150, 207, 207, 210],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 200, 200, 200],
        [40, 40, 40, 40, 40, 35, 35, 35, 30, 120, 120, 120],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20],
    ]);
    let expected_pass1 = plane_of(&[
        [40, 40, 40, 220, 210, 210, 210, 200, 200, 200, 200, 200],
        [40, 40, 40, 200, 200, 200, 200, 200, 200, 200, 200, 200],
        [40, 40, 40, 120, 120, 160, 160, 205, 205, 205, 207, 200],
        [40, 40, 40, 40, 40, 40, 40, 200, 200, 200, 200, 200],
        [40, 40, 40, 35, 35, 33, 33, 80, 80, 145, 120, 210],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 200],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 110],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
    ]);
    let expected_merged = plane_of(&[
        [40, 120, 120, 210, 205, 205, 205, 200, 200, 200, 200, 200],
        [40, 40, 40, 200, 200, 200, 200, 200, 200, 200, 200, 200],
        [40, 40, 40, 80, 80, 180, 180, 203, 203, 203, 204, 200],
        [40, 36, 38, 38, 38, 60, 60, 180, 175, 204, 204, 205],
        [40, 40, 40, 38, 38, 37, 37, 60, 60, 173, 160, 205],
        [40, 40, 40, 40, 40, 38, 38, 38, 35, 80, 80, 160],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 75],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 30],
    ]);
    assert_eq!(pass0.output, expected_pass0);
    assert_eq!(pass1.output, expected_pass1);
    assert_eq!(merged.output, expected_merged);
}

/// A plane with its size, arguments and optional clips.
struct PlaneInputs {
    width: u32,
    height: u32,
    parameters: Eedi3Parameters,
    plane: Vec<f32>,
    sclip: Option<Vec<f32>>,
    mclip: Option<Vec<f32>>,
}

impl PlaneInputs {
    fn of_fixture(fixture: &Eedi3Fixture) -> Self {
        Self {
            width: fixture.setting("width"),
            height: fixture.setting("height"),
            parameters: fixture.parameters(),
            plane: fixture.input_plane(),
            sclip: fixture.sclip_plane(),
            mclip: fixture.mclip_plane(),
        }
    }

    fn clips(&self) -> Eedi3Clips<'_> {
        Eedi3Clips {
            sclip: self.sclip.as_deref(),
            mclip: self.mclip.as_deref(),
        }
    }

    fn field(&self, offset: u32, variant: &Eedi3Variant) -> Eedi3Pass {
        let clips = self.clips();

        reference_eedi3_field_variant(
            &self.plane,
            self.width,
            self.height,
            offset,
            &self.parameters,
            clips,
            variant,
        )
    }
}

fn fixture_named(name: &str) -> &'static Eedi3Fixture {
    EEDI3_FIXTURES
        .iter()
        .find(|fixture| fixture.name == name)
        .expect("the fixture exists")
}

/// Runs pass `offset` of `fixture` under `variant`, with the fixture's clips.
fn run_fixture(fixture: &Eedi3Fixture, offset: u32, variant: &Eedi3Variant) -> Eedi3Pass {
    let inputs = PlaneInputs::of_fixture(fixture);

    inputs.field(offset, variant)
}

/// Whether either pass of `fixture` under `variant` differs from the plugin, in samples or directions.
fn variant_differs(fixture: &Eedi3Fixture, variant: &Eedi3Variant) -> bool {
    (0..2).any(|offset| {
        let pass = run_fixture(fixture, offset, variant);
        let expected_output = fixture.pass_plane(offset);
        let expected_directions = fixture.directions(offset);

        pass.output != expected_output || pass.directions != expected_directions
    })
}

#[test]
fn every_fixture_matches_the_plugin() {
    for fixture in &EEDI3_FIXTURES {
        let inputs = PlaneInputs::of_fixture(fixture);
        let PlaneInputs {
            width,
            height,
            parameters,
            plane,
            ..
        } = &inputs;
        let clips = inputs.clips();
        let samples = *width as usize * *height as usize;
        assert_eq!(plane.len(), samples, "{} input", fixture.name);

        for offset in 0..2 {
            let expected_output = fixture.pass_plane(offset);
            let expected_directions = fixture.directions(offset);
            assert_eq!(expected_output.len(), samples, "{} pass {offset}", fixture.name);
            assert_eq!(
                expected_directions.len(),
                samples,
                "{} directions {offset}",
                fixture.name
            );

            let pass = reference_eedi3_field(plane, *width, *height, offset, parameters, clips);

            assert_eq!(
                pass.output, expected_output,
                "{} at offset {offset}",
                fixture.name
            );
            assert_eq!(
                pass.directions, expected_directions,
                "{} directions at offset {offset}",
                fixture.name
            );
        }

        let expected_merged = fixture.merged_plane();
        let merged = reference_eedi3_antialias(plane, *width, *height, parameters, clips);
        assert_eq!(merged.output, expected_merged, "{} merged", fixture.name);
    }
}

#[test]
fn the_check_reads_the_checked_row_above() {
    let fixture = fixture_named("check_order");
    let plugin = Eedi3Variant::default();
    let unchecked_above = Eedi3Variant {
        check_order: CheckOrder::UncheckedAbove,
        ..Eedi3Variant::default()
    };

    assert!(!variant_differs(fixture, &plugin));
    assert!(
        variant_differs(fixture, &unchecked_above),
        "reading the unchecked row above left both passes unchanged"
    );
}

#[test]
fn lent_costs_matter_under_mclip() {
    let fixture = fixture_named("based");
    let plugin = Eedi3Variant::default();
    let without_lending = Eedi3Variant {
        lending: Lending::Off,
        ..Eedi3Variant::default()
    };

    assert!(fixture.mclip.is_some());
    assert!(!variant_differs(fixture, &plugin));
    assert!(
        variant_differs(fixture, &without_lending),
        "dropping the lent costs left both passes unchanged"
    );
}

#[test]
fn a_tie_goes_to_the_smallest_direction() {
    let fixture = fixture_named("flat");
    let smallest = Eedi3Variant::default();
    let largest = Eedi3Variant {
        ties: TieOrder::LargestDirection,
        ..Eedi3Variant::default()
    };

    let mut plugin_matches = true;
    let mut directions_differ = false;
    for offset in 0..2 {
        let expected_directions = fixture.directions(offset);
        let smallest_pass = run_fixture(fixture, offset, &smallest);
        let largest_pass = run_fixture(fixture, offset, &largest);

        plugin_matches &= smallest_pass.directions == expected_directions;
        directions_differ |= largest_pass.directions != expected_directions;
    }

    assert!(plugin_matches);
    assert!(
        directions_differ,
        "sending ties to the largest direction changed no direction"
    );
}

/// A small fixed-seed generator, so every run draws the same planes.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

        mixed ^ (mixed >> 31)
    }

    fn below(&mut self, bound: u32) -> u32 {
        let drawn = self.next() % u64::from(bound);
        drawn as u32
    }

    /// A value between 0 and 1, in steps of 2^-24.
    fn unit(&mut self) -> f32 {
        let drawn = self.next() >> 40;
        drawn as f32 / (1u32 << 24) as f32
    }
}

fn random_parameters(random: &mut SplitMix, bits: u32) -> Eedi3Parameters {
    let alpha = random.unit();
    let mut beta = random.unit() * (1.0 - alpha);
    while alpha + beta > 1.0 {
        beta *= 0.999;
    }

    Eedi3Parameters {
        alpha,
        beta,
        gamma: random.unit() * 100.0,
        nrad: random.below(4),
        mdis: 1 + random.below(40),
        vcheck: random.below(4),
        vthresh: [
            0.5 + random.unit() * 100.0,
            0.5 + random.unit() * 100.0,
            0.5 + random.unit() * 10.0,
        ],
        bits,
    }
}

/// Noise, a hard sloped edge, the edge with light noise, or low noise around one level.
fn random_plane(random: &mut SplitMix, width: u32, height: u32, bits: u32) -> Vec<f32> {
    let peak = (1u32 << bits) - 1;
    let kind = random.below(4);
    let low = random.below(peak + 1) as i32;
    let high = random.below(peak + 1) as i32;
    let slope = (random.unit() - 0.5) * 16.0;
    let shift = random.unit() * width as f32;
    let spread = match kind {
        2 => peak as i32 / 40,
        3 => 1 + random.below(3) as i32,
        _ => 0,
    };

    let mut plane = Vec::with_capacity(width as usize * height as usize);
    for row in 0..height {
        for column in 0..width {
            if kind == 0 {
                let value = random.below(peak + 1);
                plane.push(value as f32);
                continue;
            }

            let beyond_edge = column as f32 > slope * row as f32 + shift;
            let level = if beyond_edge && kind != 3 { high } else { low };
            let noise = random.below(2 * spread as u32 + 1) as i32 - spread;
            let value = (level + noise).clamp(0, peak as i32);
            plane.push(value as f32);
        }
    }

    plane
}

/// Sparse dots, a few blocks, or every sample set.
fn random_mask(random: &mut SplitMix, width: u32, height: u32) -> Vec<f32> {
    let samples = width as usize * height as usize;
    let kind = random.below(3);

    let mut mask = vec![0.0; samples];
    match kind {
        0 => {
            let odds = 20 + random.below(200);
            for value in &mut mask {
                if random.below(odds) == 0 {
                    *value = (1 + random.below(255)) as f32;
                }
            }
        },
        1 => {
            let blocks = 1 + random.below(4);
            for _ in 0..blocks {
                let left = random.below(width);
                let top = random.below(height);
                let right = (left + 1 + random.below(12)).min(width);
                let bottom = (top + 1 + random.below(12)).min(height);
                for row in top..bottom {
                    for column in left..right {
                        mask[(row * width + column) as usize] = 255.0;
                    }
                }
            }
        },
        _ => mask.fill(255.0),
    }

    mask
}

/// A plane of 8 to 16 bits and 8 to 96 columns, with a third of the seeds taking `sclip` and a third
/// taking `mclip`.
fn random_case(seed: u64) -> PlaneInputs {
    let mut random = SplitMix(seed);
    let bits = 8 + random.below(9);
    let width = 8 + random.below(89);
    let height = 4 + 2 * random.below(19);
    let parameters = random_parameters(&mut random, bits);
    let plane = random_plane(&mut random, width, height, bits);
    let sclip = random_plane(&mut random, width, height, bits);
    let mclip = random_mask(&mut random, width, height);

    PlaneInputs {
        width,
        height,
        parameters,
        plane,
        sclip: (seed % 3 == 1).then_some(sclip),
        mclip: (seed % 3 == 2).then_some(mclip),
    }
}

/// How many fused runs changed anything, per fused order, and how many changed a direction.
#[derive(Default)]
struct FusedChanges {
    runs: [usize; 2],
    directions: usize,
}

/// Compares both fused orders with the plain reference through `tally`, counting the runs they changed.
fn compare_fused(
    tally: &mut CloseDecisionTally,
    changes: &mut FusedChanges,
    label: &str,
    inputs: &PlaneInputs,
) {
    let plain_variant = Eedi3Variant::default();
    let orders = [Contraction::FuseLater, Contraction::FuseEarlier];

    for offset in 0..2 {
        let plain = inputs.field(offset, &plain_variant);

        for (order, contraction) in orders.into_iter().enumerate() {
            let variant = Eedi3Variant {
                contraction,
                ..Eedi3Variant::default()
            };
            let fused = inputs.field(offset, &variant);
            let run = format!("{label} at offset {offset} with {contraction:?}");

            tally.compare(&run, &fused.output, &plain.output, &plain.touched, inputs.width);
            tally.compare_directions(
                &run,
                &fused.directions,
                &plain.directions,
                &plain.touched_directions,
                inputs.width,
            );

            let directions_changed = fused.directions != plain.directions;
            if fused.output != plain.output || directions_changed {
                changes.runs[order] += 1;
            }

            if directions_changed {
                changes.directions += 1;
            }
        }
    }
}

#[test]
fn fused_multiplies_change_only_touched_samples() {
    let mut tally = CloseDecisionTally::new();
    let mut changes = FusedChanges::default();

    for fixture in &EEDI3_FIXTURES {
        let inputs = PlaneInputs::of_fixture(fixture);
        compare_fused(&mut tally, &mut changes, fixture.name, &inputs);
    }

    for seed in 0..200 {
        let inputs = random_case(seed);
        let label = format!("seed {seed}");
        compare_fused(&mut tally, &mut changes, &label, &inputs);
    }

    tally.assert_bounds("fused multiplies");
    assert!(changes.runs[0] > 0, "fusing the later products changed no run");
    assert!(changes.runs[1] > 0, "fusing the earlier products changed no run");
    assert!(
        changes.directions > 0,
        "fusing the multiplies changed no direction"
    );
}

#[test]
#[should_panic(expected = "tiny: row 1, column 0 differs at an untouched sample, found 7 and expected 3")]
fn tally_refuses_an_untouched_difference() {
    let expected = [1.0, 2.0, 3.0, 4.0];
    let found = [1.0, 5.0, 7.0, 4.0];
    let touched = [false, true, false, false];

    let mut tally = CloseDecisionTally::new();
    tally.compare("tiny", &found, &expected, &touched, 2);
}

/// The message `tally` panics with when it checks its bounds, if it panics.
fn bounds_failure(tally: &CloseDecisionTally) -> Option<String> {
    let outcome = panic::catch_unwind(|| tally.assert_bounds("bounds"));
    let payload = outcome.err()?;

    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .expect("the tally panics with a formatted message");
    Some(message)
}

/// A tally of `samples` samples, the first `touched` of them touched and the first `differing` of those
/// different.
fn tally_of(samples: usize, touched: usize, differing: usize) -> CloseDecisionTally {
    let expected = vec![0.0; samples];
    let mut found = vec![0.0; samples];
    found[..differing].fill(1.0);
    let mut marks = vec![false; samples];
    marks[..touched].fill(true);

    let mut tally = CloseDecisionTally::new();
    tally.compare("bounds", &found, &expected, &marks, 100);
    tally
}

#[test]
fn tally_enforces_its_bounds() {
    let at_the_bounds = tally_of(10_000, 200, 5);
    let too_many_touched = tally_of(10_000, 201, 0);
    let too_many_differing = tally_of(10_000, 200, 6);

    assert_eq!(bounds_failure(&at_the_bounds), None);
    let touched_failure = bounds_failure(&too_many_touched).expect("201 of 10000 touched fails");
    let differing_failure = bounds_failure(&too_many_differing).expect("6 of 10000 differing fails");
    assert!(touched_failure.contains("201 of 10000"), "{touched_failure}");
    assert!(differing_failure.contains("6 differ"), "{differing_failure}");

    // Directions count toward the same bounds as samples.
    let mut with_directions = tally_of(5_000, 100, 2);
    let expected_directions = vec![0; 5_000];
    let mut found_directions = vec![0; 5_000];
    found_directions[..2].fill(1);
    let mut direction_marks = vec![false; 5_000];
    direction_marks[..101].fill(true);
    with_directions.compare_directions(
        "bounds",
        &found_directions,
        &expected_directions,
        &direction_marks,
        100,
    );

    let directions_failure = bounds_failure(&with_directions).expect("201 of 10000 touched fails");
    assert!(
        directions_failure.contains("201 of 10000"),
        "{directions_failure}"
    );
}
