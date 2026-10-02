mod backtrack;
mod bands;
mod exact;

use av_toolbox_cubek::aa::{
    Eedi3Arguments,
    Eedi3DeviceClips,
    Eedi3Scratch,
    Eedi3Settings,
    eedi3_field,
    eedi3_scratch,
};
use av_toolbox_cubek::testing::{
    CloseDecisionTally,
    EEDI3_FIXTURES,
    Eedi3Clips,
    Eedi3Fixture,
    Eedi3Parameters,
    Eedi3Pass,
    Eedi3PlaneKind,
    Eedi3Variant,
    Lending,
    SplitMix,
    TieOrder,
    eedi3_checked_rows,
    merge_eedi3_fields,
    random_eedi3_mask,
    random_eedi3_parameters,
    random_eedi3_plane,
    reference_eedi3_field,
    reference_eedi3_field_variant,
    worked_edge_plane,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::aa::transposed_plane;
use crate::support::{Sizing, backend, empty, plane, read_f32, read_u32, upload_f32};

fn arguments_for(parameters: &Eedi3Parameters) -> Eedi3Arguments {
    Eedi3Arguments {
        alpha: parameters.alpha,
        beta: parameters.beta,
        gamma: parameters.gamma,
        nrad: parameters.nrad,
        mdis: parameters.mdis,
        vcheck: parameters.vcheck,
        vthresh: parameters.vthresh,
    }
}

fn settings_for(parameters: &Eedi3Parameters) -> Eedi3Settings {
    let arguments = arguments_for(parameters);

    Eedi3Settings::new(&arguments, parameters.bits)
}

/// A plane with its size, arguments and optional `sclip` and `mclip`.
struct PlaneCase {
    label: String,
    width: u32,
    height: u32,
    parameters: Eedi3Parameters,
    input: Vec<f32>,
    sclip: Option<Vec<f32>>,
    mclip: Option<Vec<f32>>,
}

impl PlaneCase {
    fn new(label: &str, width: u32, height: u32, parameters: Eedi3Parameters, input: Vec<f32>) -> Self {
        Self {
            label: label.to_string(),
            width,
            height,
            parameters,
            input,
            sclip: None,
            mclip: None,
        }
    }

    fn of_fixture(fixture: &Eedi3Fixture) -> Self {
        let width = fixture.setting("width");
        let height = fixture.setting("height");
        let parameters = fixture.parameters();
        let input = fixture.input_plane();

        Self::new(fixture.name, width, height, parameters, input)
    }

    fn clips(&self) -> Eedi3Clips<'_> {
        Eedi3Clips {
            sclip: self.sclip.as_deref(),
            mclip: self.mclip.as_deref(),
        }
    }

    fn reference(&self, offset: u32) -> Eedi3Pass {
        let clips = self.clips();

        reference_eedi3_field(
            &self.input,
            self.width,
            self.height,
            offset,
            &self.parameters,
            clips,
        )
    }

    fn reference_variant(&self, offset: u32, variant: &Eedi3Variant) -> Eedi3Pass {
        let clips = self.clips();

        reference_eedi3_field_variant(
            &self.input,
            self.width,
            self.height,
            offset,
            &self.parameters,
            clips,
            variant,
        )
    }
}

/// One field pass the device computed, its output and its direction map at the input's size.
struct DevicePass {
    output: Vec<f32>,
    directions: Vec<i32>,
}

/// Spreads the direction map of the rebuilt rows over the plane, with 0 on the kept rows.
fn spread_directions(rebuilt: &[u32], width: u32, height: u32, offset: u32) -> Vec<i32> {
    let columns = width as usize;
    let mut directions = vec![0; columns * height as usize];

    let rebuilt_rows = rebuilt.chunks(columns).take(height as usize / 2);
    for (field_row, values) in rebuilt_rows.enumerate() {
        let row = 2 * field_row + 1 - offset as usize;
        let start = row * columns;
        for (column, value) in values.iter().enumerate() {
            directions[start + column] = *value as i32;
        }
    }

    directions
}

/// Both field passes of one plane on the device, sharing one scratch.
struct FieldPasses {
    input: Vec<f32>,
    width: u32,
    height: u32,
    settings: Eedi3Settings,
    sclip: Option<Vec<f32>>,
    mclip: Option<Vec<f32>>,
}

impl BuildTask for FieldPasses {
    type Output = Vec<DevicePass>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();
        let mdis = self.settings.mdis();
        let sizes = eedi3_scratch(self.width, self.height, mdis);

        let input = upload_f32(client, &self.input);
        let sclip = self.sclip.as_deref().map(|values| upload_f32(client, values));
        let mclip = self.mclip.as_deref().map(|values| upload_f32(client, values));
        let rebuilt = empty(client, sizes.rebuilt);
        let directions = empty(client, sizes.directions);
        let backpointers = empty(client, sizes.backpointers);
        let lenders = empty(client, sizes.lenders);
        let scratch = Eedi3Scratch {
            rebuilt: &rebuilt,
            directions: &directions,
            backpointers: &backpointers,
            lenders: &lenders,
        };

        let mut passes = Vec::new();
        for offset in 0..2 {
            let output = empty(client, bytes);
            let input_plane = plane(&input, self.width, self.height);
            let output_plane = plane(&output, self.width, self.height);
            let clips = Eedi3DeviceClips {
                sclip: sclip
                    .as_ref()
                    .map(|handle| plane(handle, self.width, self.height)),
                mclip: mclip
                    .as_ref()
                    .map(|handle| plane(handle, self.width, self.height)),
            };

            eedi3_field(
                client,
                &sizing,
                input_plane,
                output_plane,
                scratch,
                offset,
                &self.settings,
                clips,
            );

            cubecl::future::block_on(client.sync()).expect("the device should finish");

            let output_values = read_f32(client, &output);
            let rebuilt_directions = read_u32(client, &directions);
            let spread = spread_directions(&rebuilt_directions, self.width, self.height, offset);
            passes.push(DevicePass {
                output: output_values,
                directions: spread,
            });
        }

        passes
    }
}

fn on_device(case: &PlaneCase) -> Vec<DevicePass> {
    let task = FieldPasses {
        input: case.input.clone(),
        width: case.width,
        height: case.height,
        settings: settings_for(&case.parameters),
        sclip: case.sclip.clone(),
        mclip: case.mclip.clone(),
    };

    backend().build(task)
}

/// Adds both field passes of `case` to `tally`, the device's against the reference's.
fn compare_case(tally: &mut CloseDecisionTally, case: &PlaneCase) {
    let passes = on_device(case);

    compare_passes(tally, case, &passes);
}

/// Adds both field passes the device computed for `case` to `tally`, against the reference's.
fn compare_passes(tally: &mut CloseDecisionTally, case: &PlaneCase, passes: &[DevicePass]) {
    for (offset, found) in passes.iter().enumerate() {
        let expected = case.reference(offset as u32);
        let label = format!("{} at offset {offset}", case.label);

        tally.compare(
            &label,
            &found.output,
            &expected.output,
            &expected.touched,
            case.width,
        );
        tally.compare_directions(
            &label,
            &found.directions,
            &expected.directions,
            &expected.touched_directions,
            case.width,
        );
    }
}

/// Prints the tally's counts, then asserts its bounds.
fn assert_tally(tally: &CloseDecisionTally, label: &str) {
    println!("{label}: {tally:?}");

    tally.assert_bounds(label);
}

/// A deterministic hash of a sample's position, for noise.
fn scramble(column: u32, row: u32) -> u32 {
    let mut state = column.wrapping_mul(0x9e37_79b1) ^ row.wrapping_mul(0x85eb_ca77);
    state ^= state >> 15;
    state = state.wrapping_mul(0x2c1b_3c6d);
    state ^= state >> 12;

    state
}

/// The shape of a plane of diagonal bands.
struct Bands {
    /// How many columns an edge moves per row.
    slope: u32,
    /// How many columns each band spans.
    period: u32,
    low: u32,
    high: u32,
    noise: u32,
}

/// Diagonal bands of `low` and `high` with some noise.
fn diagonal_bands(width: u32, height: u32, bands: &Bands) -> Vec<f32> {
    let mut input = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let band = (column + bands.slope * row) / bands.period % 2;
            let base = if band == 0 { bands.low } else { bands.high };
            let jitter = scramble(column, row) % (bands.noise + 1);
            let value = base + jitter;
            input.push(value as f32);
        }
    }

    input
}

#[test]
fn every_fixture_matches_on_the_device() {
    let mut tally = CloseDecisionTally::new();
    for fixture in &EEDI3_FIXTURES {
        if fixture.mclip.is_some() {
            continue;
        }

        let case = PlaneCase::of_fixture(fixture);
        compare_case(&mut tally, &case);
    }

    assert_tally(&tally, "every fixture");
}

/// The worked example's plane is too small for the tally's bounds on its own, so the same edge on a larger
/// plane counts toward them too.
#[test]
fn the_worked_example_matches_on_the_device() {
    let fixture = Eedi3Fixture::named("worked");
    let case = PlaneCase::of_fixture(fixture);
    let passes = on_device(&case);

    // Rebuilt row 3 of pass 0 follows the path worked out by hand.
    let path = [0, -1, -2, -2, -2, -2, -2, -2, -2, -2, -1, 0];
    assert_eq!(passes[0].directions[36..48], path);

    let larger_input = worked_edge_plane(480, 64);
    let larger = PlaneCase::new("the worked edge", 480, 64, case.parameters, larger_input);

    let mut tally = CloseDecisionTally::new();
    compare_case(&mut tally, &case);
    compare_case(&mut tally, &larger);
    assert_tally(&tally, "the worked example");
}

/// The widest direction either field pass of `case` takes on the reference.
fn widest_direction(case: &PlaneCase) -> i32 {
    (0..2)
        .flat_map(|offset| case.reference(offset).directions)
        .map(i32::abs)
        .max()
        .expect("the plane has samples")
}

#[test]
fn a_wide_plane_matches_the_reference() {
    let width = 1920;
    let height = 16;
    let bands = Bands {
        slope: 38,
        period: 1200,
        low: 40,
        high: 200,
        noise: 30,
    };
    let input = diagonal_bands(width, height, &bands);
    let parameters = Eedi3Parameters {
        mdis: 40,
        ..Eedi3Parameters::default()
    };
    let case = PlaneCase::new("1920 wide", width, height, parameters, input);

    let widest = widest_direction(&case);
    assert!(widest >= 36, "the widest direction is {widest}, not at least 36");

    let mut tally = CloseDecisionTally::new();
    compare_case(&mut tally, &case);
    assert_tally(&tally, "1920 wide");
}

#[test]
fn a_sixteen_bit_plane_matches() {
    let width = 320;
    let height = 64;
    let bands = Bands {
        slope: 2,
        period: 64,
        low: 3000,
        high: 62000,
        noise: 800,
    };
    let input = diagonal_bands(width, height, &bands);
    let parameters = Eedi3Parameters {
        nrad: 3,
        vcheck: 3,
        bits: 16,
        ..Eedi3Parameters::default()
    };
    let case = PlaneCase::new("16-bit", width, height, parameters, input);

    let mut tally = CloseDecisionTally::new();
    compare_case(&mut tally, &case);
    assert_tally(&tally, "16-bit");
}

/// Flat areas split by hard edges of several slopes, with a flat level on each side.
fn flat_cases() -> Vec<PlaneCase> {
    let shapes = [
        (1, 20, 40, 200),
        (2, 36, 16, 235),
        (5, 64, 90, 110),
        (9, 90, 0, 255),
    ];

    shapes
        .iter()
        .enumerate()
        .map(|(index, (slope, period, low, high))| {
            let bands = Bands {
                slope: *slope,
                period: *period,
                low: *low,
                high: *high,
                noise: 0,
            };
            let input = diagonal_bands(240, 48, &bands);
            let label = format!("flat plane {index}");
            PlaneCase::new(&label, 240, 48, Eedi3Parameters::default(), input)
        })
        .collect()
}

/// How many samples and directions of `cases` change when ties go to the largest direction.
fn largest_direction_changes(cases: &[PlaneCase]) -> (usize, usize) {
    let largest = Eedi3Variant {
        ties: TieOrder::LargestDirection,
        ..Eedi3Variant::default()
    };

    let mut compared = 0;
    let mut changed = 0;
    for case in cases {
        for offset in 0..2 {
            let plugin = case.reference(offset);
            let flipped = case.reference_variant(offset, &largest);
            let outputs = plugin.output.iter().zip(&flipped.output);
            let output_changes = outputs.filter(|(first, second)| first != second).count();
            let directions = plugin.directions.iter().zip(&flipped.directions);
            let direction_changes = directions.filter(|(first, second)| first != second).count();

            compared += 2 * plugin.output.len();
            changed += output_changes + direction_changes;
        }
    }

    (changed, compared)
}

#[test]
fn flat_planes_with_hard_edges_match() {
    let cases = flat_cases();

    // Sending ties to the largest direction changes more than the 0.05 % the tally allows.
    let (changed, compared) = largest_direction_changes(&cases);
    assert!(
        changed * 10_000 > compared * 5,
        "ties decide only {changed} of {compared} samples"
    );

    let mut tally = CloseDecisionTally::new();
    for case in &cases {
        compare_case(&mut tally, case);
    }

    assert_tally(&tally, "flat planes");
}

/// The `sclip` fixture is too small for the tally's bounds on its own, so a larger plane with `sclip` and
/// the same arguments counts toward them too.
#[test]
fn sclip_replaces_the_cubic_fallback() {
    let fixture = Eedi3Fixture::named("based");
    let mut case = PlaneCase::of_fixture(fixture);
    let without_sclip: Vec<Eedi3Pass> = (0..2).map(|offset| case.reference(offset)).collect();
    case.sclip = fixture.sclip_plane();
    assert!(case.sclip.is_some());

    let with_sclip: Vec<Eedi3Pass> = (0..2).map(|offset| case.reference(offset)).collect();

    let differs = without_sclip
        .iter()
        .zip(&with_sclip)
        .any(|(first, second)| first.output != second.output);
    assert!(differs, "sclip left both passes unchanged");

    let input_bands = Bands {
        slope: 2,
        period: 40,
        low: 40,
        high: 200,
        noise: 10,
    };
    let sclip_bands = Bands {
        slope: 1,
        period: 25,
        low: 60,
        high: 180,
        noise: 20,
    };
    let larger_input = diagonal_bands(480, 64, &input_bands);
    let larger_sclip = diagonal_bands(480, 64, &sclip_bands);
    let mut larger = PlaneCase::new("a larger sclip", 480, 64, case.parameters, larger_input);
    larger.sclip = Some(larger_sclip);

    let mut tally = CloseDecisionTally::new();
    compare_case(&mut tally, &case);
    compare_case(&mut tally, &larger);
    assert_tally(&tally, "sclip");
}

#[test]
fn settings_follow_the_plugin() {
    let defaults = Eedi3Arguments::default();
    let default_settings = Eedi3Settings::new(&defaults, 8);
    let default_weights = default_settings.weights();

    assert_eq!(default_weights.alpha3.to_bits(), 0x3D88_8889);
    // 0.55000001, the f32 nearest 0.55.
    assert_eq!(default_weights.vertical.to_bits(), 0x3F0C_CCCD);
    assert_eq!(default_weights.gamma, 20.0);
    assert_eq!(default_weights.reciprocal0, 1.0 / 32.0);
    assert_eq!(default_weights.reciprocal1, 1.0 / 64.0);
    assert_eq!(default_weights.reciprocal2, 0.25);

    let based = Eedi3Arguments {
        alpha: 0.125,
        beta: 0.25,
        gamma: 40.0,
        vthresh: [12.0, 24.0, 4.0],
        ..Eedi3Arguments::default()
    };

    // The plugin's weights for these arguments at 8 and 10 bits, as f32 bits.
    let expected = [
        (
            8,
            [
                0x3D2A_AAAB,
                0x3E80_0000,
                0x4220_0000,
                0x3F20_0000,
                0x3DAA_AAAB,
                0x3D2A_AAAB,
                0x3E80_0000,
            ],
        ),
        (
            10,
            [
                0x3D2A_AAAB,
                0x3F80_0000,
                0x4320_0000,
                0x3F20_0000,
                0x3CAA_AAAB,
                0x3C2A_AAAB,
                0x3E80_0000,
            ],
        ),
    ];
    for (bits, weight_bits) in expected {
        let settings = Eedi3Settings::new(&based, bits);
        let weights = settings.weights();
        let found = [
            weights.alpha3,
            weights.beta,
            weights.gamma,
            weights.vertical,
            weights.reciprocal0,
            weights.reciprocal1,
            weights.reciprocal2,
        ]
        .map(f32::to_bits);

        assert_eq!(found, weight_bits, "based_aa's weights at {bits} bits");
    }
}

/// A mask set over each band of columns, from `left` up to `right`, on every row.
fn column_band_mask(width: u32, height: u32, bands: &[(u32, u32)]) -> Vec<f32> {
    let mut mask = vec![0.0; width as usize * height as usize];
    for row in 0..height {
        for (left, right) in bands {
            for column in *left..*right {
                mask[(row * width + column) as usize] = 255.0;
            }
        }
    }

    mask
}

/// The larger plane an `mclip` fixture's tally adds, with its arguments and both clips.
fn larger_masked_case(parameters: Eedi3Parameters) -> PlaneCase {
    let input_bands = Bands {
        slope: 2,
        period: 40,
        low: 40,
        high: 200,
        noise: 10,
    };
    let sclip_bands = Bands {
        slope: 1,
        period: 25,
        low: 60,
        high: 180,
        noise: 20,
    };
    let input = diagonal_bands(480, 64, &input_bands);
    let sclip = diagonal_bands(480, 64, &sclip_bands);
    let mclip = column_band_mask(480, 64, &[(30, 90), (200, 210), (400, 470)]);

    let mut case = PlaneCase::new("a larger mclip", 480, 64, parameters, input);
    case.sclip = Some(sclip);
    case.mclip = Some(mclip);

    case
}

/// The `mclip` fixtures are too small for the tally's bounds on their own, so a larger plane with the same
/// arguments and both clips counts toward them too.
#[test]
fn every_mclip_fixture_matches_on_the_device() {
    let masked_fixtures: Vec<&Eedi3Fixture> = EEDI3_FIXTURES
        .iter()
        .filter(|fixture| fixture.mclip.is_some())
        .collect();
    assert!(!masked_fixtures.is_empty(), "no fixture has an mclip");

    let mut tally = CloseDecisionTally::new();
    for fixture in masked_fixtures {
        let mut case = PlaneCase::of_fixture(fixture);
        case.sclip = fixture.sclip_plane();
        case.mclip = fixture.mclip_plane();
        compare_case(&mut tally, &case);

        let larger = larger_masked_case(case.parameters);
        compare_case(&mut tally, &larger);
    }

    assert_tally(&tally, "every mclip fixture");
}

/// Both field passes of `case` on the reference, under `variant`.
fn reference_passes(case: &PlaneCase, variant: &Eedi3Variant) -> Vec<Eedi3Pass> {
    (0..2)
        .map(|offset| case.reference_variant(offset, variant))
        .collect()
}

/// How many output samples and directions differ between two runs of both field passes.
fn changed_samples(first: &[Eedi3Pass], second: &[Eedi3Pass]) -> usize {
    let mut changed = 0;
    for (first_pass, second_pass) in first.iter().zip(second) {
        let outputs = first_pass.output.iter().zip(&second_pass.output);
        let directions = first_pass.directions.iter().zip(&second_pass.directions);

        changed += outputs.filter(|(left, right)| left != right).count();
        changed += directions.filter(|(left, right)| left != right).count();
    }

    changed
}

/// A mask where the second rebuilt row of each pass lends, the third borrows for column 0 alone, and every
/// other row borrows for column 0 and column 1. The first rebuilt row borrows with no row above to lend.
fn lending_mask(width: u32, height: u32, mdis: u32) -> Vec<f32> {
    let mut mask = vec![0.0; width as usize * height as usize];
    for row in 0..height {
        let field_row = row / 2;
        let set_column = if field_row == 1 {
            0
        } else if field_row == 2 {
            mdis + 1
        } else {
            mdis + 6
        };

        let start = row * width;
        mask[(start + set_column) as usize] = 255.0;
        mask[(start + width / 2) as usize] = 255.0;
    }

    mask
}

/// A gentle ramp, with noise at the left of the top rows, so the rows that lend have costly connections
/// and the rows that borrow cheap ones.
fn noisy_top_left(width: u32, height: u32) -> Vec<f32> {
    let mut input = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let mut value = 100 + column / 8;
            if row < 12 && column < 24 {
                value = scramble(column, row) % 256;
            }

            input.push(value as f32);
        }
    }

    input
}

#[test]
fn lent_costs_reach_the_device() {
    let width = 160;
    let height = 48;
    let input = noisy_top_left(width, height);
    let parameters = Eedi3Parameters {
        gamma: 60.0,
        mdis: 8,
        ..Eedi3Parameters::default()
    };
    let mut case = PlaneCase::new("lent costs", width, height, parameters, input);
    let mask = lending_mask(width, height, parameters.mdis);
    case.mclip = Some(mask);

    let plugin_variant = Eedi3Variant::default();
    let plugin = reference_passes(&case, &plugin_variant);
    let without_lending = Eedi3Variant {
        lending: Lending::Off,
        ..Eedi3Variant::default()
    };
    let unlent = reference_passes(&case, &without_lending);
    let changed = changed_samples(&plugin, &unlent);
    assert!(changed > 0, "dropping the lent costs left both passes unchanged");

    let mut tally = CloseDecisionTally::new();
    compare_case(&mut tally, &case);
    assert_tally(&tally, "lent costs");
}

#[test]
fn a_mask_changes_the_path_of_active_columns() {
    let width = 240;
    let height = 48;
    let bands = Bands {
        slope: 2,
        period: 36,
        low: 20,
        high: 230,
        noise: 30,
    };
    let input = diagonal_bands(width, height, &bands);
    let parameters = Eedi3Parameters {
        mdis: 8,
        ..Eedi3Parameters::default()
    };
    let mask = column_band_mask(width, height, &[(20, 60), (100, 110), (170, 200)]);
    let unmasked = PlaneCase::new("unmasked", width, height, parameters, input.clone());
    let mut masked = PlaneCase::new("masked", width, height, parameters, input);
    masked.mclip = Some(mask.clone());

    let mut changed_under_mask = 0;
    for offset in 0..2 {
        let without_mclip = unmasked.reference(offset);
        let with_mclip = masked.reference(offset);
        let samples = without_mclip.output.iter().zip(&with_mclip.output);
        for (index, (first, second)) in samples.enumerate() {
            if mask[index] != 0.0 && first != second {
                changed_under_mask += 1;
            }
        }
    }

    assert!(
        changed_under_mask > 0,
        "the mask changed no sample where it is set"
    );

    let mut tally = CloseDecisionTally::new();
    compare_case(&mut tally, &masked);
    assert_tally(&tally, "a mask changes the path");
}

/// A plane of 8 to 16 bits, 8 to 200 columns and 4 to 40 rows, with random arguments and a random mask.
fn random_masked_case(seed: u64) -> PlaneCase {
    let mut random = SplitMix(seed);
    let bits = 8 + random.below(9);
    let width = 8 + random.below(193);
    let height = 4 + 2 * random.below(19);
    let parameters = random_eedi3_parameters(&mut random, bits);
    let input = random_eedi3_plane(&mut random, width, height, bits, &[Eedi3PlaneKind::NoisyEdge]);
    let mclip = random_eedi3_mask(&mut random, width, height);

    let label = format!("seed {seed}");
    let mut case = PlaneCase::new(&label, width, height, parameters, input);
    case.mclip = Some(mclip);

    case
}

#[test]
fn seeded_random_masks_match() {
    let mut tally = CloseDecisionTally::new();
    for seed in 0..50 {
        let case = random_masked_case(seed);
        compare_case(&mut tally, &case);
    }

    assert_tally(&tally, "seeded random masks");
}

/// Asserts that the device's checked rows of `case` equal the reference's at every untouched sample,
/// handing back how many samples it compared.
fn assert_checked_rows_match(case: &PlaneCase) -> usize {
    let passes = on_device(case);
    let columns = case.width as usize;

    let mut compared = 0;
    for (offset, found) in passes.iter().enumerate() {
        let expected = case.reference(offset as u32);
        let first_rebuilt = 1 - offset as u32;
        let rows = eedi3_checked_rows(first_rebuilt, case.height);
        for row in rows {
            let start = row as usize * columns;
            for index in start..start + columns {
                if expected.touched[index] {
                    continue;
                }

                let column = index - start;
                assert_eq!(
                    found.output[index], expected.output[index],
                    "{} at offset {offset}: row {row}, column {column} differs",
                    case.label
                );
                compared += 1;
            }
        }
    }

    compared
}

/// A random plane at `based_aa`'s `vthresh`, of 8 to 16 bits, 8 to 200 columns and 4 to 40 rows.
fn random_based_case(seed: u64) -> PlaneCase {
    let mut random = SplitMix(seed);
    let bits = 8 + random.below(9);
    let width = 8 + random.below(193);
    let height = 4 + 2 * random.below(19);
    let drawn = random_eedi3_parameters(&mut random, bits);
    let kinds = [
        Eedi3PlaneKind::Edge,
        Eedi3PlaneKind::NoisyEdge,
        Eedi3PlaneKind::Noise,
    ];
    let input = random_eedi3_plane(&mut random, width, height, bits, &kinds);
    let sclip = random_eedi3_plane(&mut random, width, height, bits, &kinds);
    let parameters = Eedi3Parameters {
        vcheck: 1 + (seed % 3) as u32,
        vthresh: [12.0, 24.0, 4.0],
        ..drawn
    };

    let label = format!("seed {seed}");
    let mut case = PlaneCase::new(&label, width, height, parameters, input);
    if seed % 2 == 1 {
        case.sclip = Some(sclip);
    }

    case
}

#[test]
fn the_blend_is_exact_on_the_device() {
    let fixture = Eedi3Fixture::named("based");
    let mut based = PlaneCase::of_fixture(fixture);
    based.sclip = fixture.sclip_plane();
    based.mclip = fixture.mclip_plane();
    assert_eq!(based.parameters.vthresh, [12.0, 24.0, 4.0]);

    let mut compared = assert_checked_rows_match(&based);
    for seed in 0..100 {
        let case = random_based_case(seed);
        compared += assert_checked_rows_match(&case);
    }

    assert!(
        compared > 100_000,
        "only {compared} checked samples were compared"
    );
}

/// `based_aa`'s EEDI3 arguments on a plane of `bits` per sample.
fn based_parameters(bits: u32) -> Eedi3Parameters {
    Eedi3Parameters {
        alpha: 0.125,
        gamma: 40.0,
        vthresh: [12.0, 24.0, 4.0],
        bits,
        ..Eedi3Parameters::default()
    }
}

/// Lines 3 samples wide at 235 on 16, on the 8-bit scale, repeating every 16 columns and moving `slope`
/// columns per row.
fn periodic_diagonals(width: u32, height: u32, slope: u32, bits: u32) -> Vec<f32> {
    let unit = 1 << (bits - 8);

    let mut plane = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let on_line = (column + slope * row) % 16 < 3;
            let level = if on_line { 235 } else { 16 };
            let value = level * unit;
            plane.push(value as f32);
        }
    }

    plane
}

/// Checks one direction of `case` through a tally of its own, handing back the device's merged output.
fn check_direction(case: &PlaneCase, tallied: bool) -> Vec<f32> {
    let passes = on_device(case);

    if tallied {
        let mut tally = CloseDecisionTally::new();
        compare_passes(&mut tally, case, &passes);
        assert_tally(&tally, &case.label);
    }

    merge_eedi3_fields(&passes[0].output, &passes[1].output)
}

/// Checks periodic diagonals of `slope` on `bits` per sample, vertically, horizontally and both, each
/// through a tally of its own.
fn check_diagonals(bits: u32, slope: u32) {
    let width = 640;
    let height = 360;
    let parameters = based_parameters(bits);
    let input = periodic_diagonals(width, height, slope, bits);
    let turned = transposed_plane(&input, width, height);

    // Slope 2 ties exactly along every vertical path, which touches 16 % of samples.
    let vertical_label = format!("{bits}-bit slope {slope} vertical");
    let vertical = PlaneCase::new(&vertical_label, width, height, parameters, input);
    let vertical_output = check_direction(&vertical, slope != 2);

    let horizontal_label = format!("{bits}-bit slope {slope} horizontal");
    let horizontal = PlaneCase::new(&horizontal_label, height, width, parameters, turned);
    check_direction(&horizontal, true);

    let both_input = transposed_plane(&vertical_output, width, height);
    let both_label = format!("{bits}-bit slope {slope} both");
    let both = PlaneCase::new(&both_label, height, width, parameters, both_input);
    check_direction(&both, true);
}

#[test]
fn periodic_diagonals_match_at_based_settings() {
    // Each plane takes about six seconds on the reference, so they run side by side.
    std::thread::scope(|scope| {
        for bits in [8, 10] {
            for slope in 2..=5 {
                scope.spawn(move || check_diagonals(bits, slope));
            }
        }
    });
}
