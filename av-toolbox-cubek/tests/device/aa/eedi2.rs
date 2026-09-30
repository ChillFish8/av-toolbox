use av_toolbox_cubek::aa::{
    Eedi2Arguments,
    Eedi2Scratch,
    Eedi2Settings,
    Eedi2Stage,
    eedi2_passes_through,
    eedi2_scratch,
};
use av_toolbox_cubek::testing::{EEDI2_FIXTURES, Eedi2Parameters, Eedi2Stages, reference_eedi2_field_stages};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// Every intermediate the device tests compare, in the order the passes write them.
const STAGES: [Eedi2Stage; 6] = [
    Eedi2Stage::RawEdges,
    Eedi2Stage::Edges,
    Eedi2Stage::RawDirections,
    Eedi2Stage::Directions,
    Eedi2Stage::GapDirections,
    Eedi2Stage::ExpandedGaps,
];

fn settings_for(parameters: &Eedi2Parameters) -> Eedi2Settings {
    let arguments = Eedi2Arguments {
        mthresh: parameters.mthresh,
        lthresh: parameters.lthresh,
        vthresh: parameters.vthresh,
        estr: parameters.estr,
        dstr: parameters.dstr,
        maxd: parameters.maxd,
        nt: parameters.nt,
        pp: parameters.pp,
    };

    Eedi2Settings::new(&arguments, parameters.bits, parameters.ssw)
}

fn is_gap_stage(stage: Eedi2Stage) -> bool {
    matches!(stage, Eedi2Stage::GapDirections | Eedi2Stage::ExpandedGaps)
}

/// Every stage of one field pass, each run from pass 1 into one shared scratch.
struct StagePasses {
    input: Vec<f32>,
    width: u32,
    height: u32,
    offset: u32,
    settings: Eedi2Settings,
}

impl BuildTask for StagePasses {
    type Output = Vec<Vec<f32>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let sizes = eedi2_scratch(self.width, self.height);
        let field_rows = self.height as usize / 2;
        let field_samples = field_rows * self.width as usize;
        let gap_samples = (field_rows - 1) * self.width as usize;

        let input = upload_f32(client, &self.input);
        let edges = empty(client, sizes.edges);
        let working = empty(client, sizes.working);
        let directions = empty(client, sizes.directions);
        let spare = empty(client, sizes.spare);
        let scratch = Eedi2Scratch {
            edges: &edges,
            working: &working,
            directions: &directions,
            spare: &spare,
        };

        let mut outputs = Vec::new();
        for stage in STAGES {
            let input_plane = plane(&input, self.width, self.height);
            let handle = eedi2_passes_through(
                client,
                &sizing,
                input_plane,
                scratch,
                self.offset,
                &self.settings,
                stage,
            );

            cubecl::future::block_on(client.sync()).expect("the device should finish");

            let mut values = read_f32(client, handle);
            let length = if is_gap_stage(stage) {
                gap_samples
            } else {
                field_samples
            };
            values.truncate(length);
            outputs.push(values);
        }

        outputs
    }
}

fn on_device(
    input: &[f32],
    width: u32,
    height: u32,
    offset: u32,
    parameters: &Eedi2Parameters,
) -> Vec<Vec<f32>> {
    let task = StagePasses {
        input: input.to_vec(),
        width,
        height,
        offset,
        settings: settings_for(parameters),
    };

    backend().build(task)
}

fn expected_stage(stages: &Eedi2Stages, stage: Eedi2Stage) -> &[f32] {
    match stage {
        Eedi2Stage::RawEdges => &stages.e0,
        Eedi2Stage::Edges => &stages.e,
        Eedi2Stage::RawDirections => &stages.d0,
        Eedi2Stage::Directions => &stages.d3,
        Eedi2Stage::GapDirections => &stages.x0,
        Eedi2Stage::ExpandedGaps => &stages.x2,
    }
}

/// Fails at the first sample that differs, naming its row and column.
fn assert_exact(actual: &[f32], expected: &[f32], width: u32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: the sizes differ");

    let width = width as usize;
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let row = index / width;
        let column = index % width;
        assert_eq!(got, want, "{label}: row {row}, column {column}");
    }
}

fn assert_matches_reference(
    input: &[f32],
    width: u32,
    height: u32,
    parameters: &Eedi2Parameters,
    label: &str,
) {
    for offset in 0..2 {
        let outputs = on_device(input, width, height, offset, parameters);
        let expected = reference_eedi2_field_stages(input, width, height, offset, parameters);

        for (stage, actual) in STAGES.iter().zip(&outputs) {
            let expected_values = expected_stage(&expected, *stage);
            let stage_label = format!("{label} at offset {offset}, {stage:?}");
            assert_exact(actual, expected_values, width, &stage_label);
        }
    }
}

/// A deterministic hash of a sample's position, for noise.
fn scramble(column: u32, row: u32) -> u32 {
    let mut state = column.wrapping_mul(0x9e37_79b1) ^ row.wrapping_mul(0x85eb_ca77);
    state ^= state >> 15;
    state = state.wrapping_mul(0x2c1b_3c6d);
    state ^= state >> 12;

    state
}

/// Diagonal bands of `low` and `high` with some noise, `slope` columns per row.
fn noisy_diagonals(width: u32, height: u32, slope: u32, low: u32, high: u32, noise: u32) -> Vec<f32> {
    let mut input = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let band = (column + slope * row) / 9 % 2;
            let base = if band == 0 { low } else { high };
            let jitter = scramble(column, row) % (noise + 1);
            let value = base + jitter;
            input.push(value as f32);
        }
    }

    input
}

#[test]
fn every_fixture_matches_through_pass_12() {
    for fixture in &EEDI2_FIXTURES {
        let width = fixture.setting("width");
        let height = fixture.setting("height");
        let parameters = fixture.parameters();
        let input = fixture.input_plane();

        assert_matches_reference(&input, width, height, &parameters, fixture.name);
    }
}

#[test]
fn the_fixtures_give_gap_directions() {
    let mut gap_directions = 0;
    for fixture in &EEDI2_FIXTURES {
        let width = fixture.setting("width");
        let height = fixture.setting("height");
        let parameters = fixture.parameters();
        let peak = ((1u32 << parameters.bits) - 1) as f32;
        let input = fixture.input_plane();
        let stages = reference_eedi2_field_stages(&input, width, height, 0, &parameters);

        gap_directions += stages.x2.iter().filter(|value| **value != peak).count();
    }

    assert!(gap_directions > 0, "the fixtures should give some gap directions");
}

/// The 8-bit directions other than 0 in `columns` of every row.
fn count_directions(rows: &[&[f32]], columns: std::ops::Range<usize>) -> usize {
    rows.iter()
        .flat_map(|row| &row[columns.clone()])
        .filter(|value| **value != 255.0 && **value != 128.0)
        .count()
}

#[test]
fn directions_at_the_side_edges_match() {
    let width = 48;
    let height = 40;
    let input = noisy_diagonals(width, height, 1, 30, 220, 6);
    let parameters = Eedi2Parameters::default();

    let offset_0 = reference_eedi2_field_stages(&input, width, height, 0, &parameters);
    let rows: Vec<&[f32]> = offset_0.d0.chunks(width as usize).collect();
    let left_directions = count_directions(&rows, 2..6);
    let right_directions = count_directions(&rows, width as usize - 6..width as usize - 2);

    assert!(
        left_directions > 0,
        "the plane should give directions beside the left edge"
    );
    assert!(
        right_directions > 0,
        "the plane should give directions beside the right edge"
    );
    assert_matches_reference(&input, width, height, &parameters, "side edges");
}

#[test]
fn a_ten_bit_noisy_plane_matches() {
    let width = 200;
    let height = 64;
    let input = noisy_diagonals(width, height, 3, 100, 900, 60);
    let parameters = Eedi2Parameters {
        bits: 10,
        ..Eedi2Parameters::default()
    };

    assert_matches_reference(&input, width, height, &parameters, "10-bit noise");
}

#[test]
fn a_sixteen_bit_chroma_plane_matches() {
    let width = 120;
    let height = 48;
    let input = noisy_diagonals(width, height, 2, 4000, 60000, 3000);
    let parameters = Eedi2Parameters {
        bits: 16,
        ssw: 1,
        ..Eedi2Parameters::default()
    };

    assert_matches_reference(&input, width, height, &parameters, "16-bit chroma");
}

/// Passes 1 to 12 given a `spare` scratch one value short.
struct UndersizedScratch;

impl BuildTask for UndersizedScratch {
    type Output = ();

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let width = 8;
        let height = 14;
        let samples = (width * height) as usize;
        let input_values = vec![0.0; samples];
        let sizes = eedi2_scratch(width, height);
        let short_bytes = sizes.spare - size_of::<f32>();
        let parameters = Eedi2Parameters::default();
        let settings = settings_for(&parameters);

        let input = upload_f32(client, &input_values);
        let edges = empty(client, sizes.edges);
        let working = empty(client, sizes.working);
        let directions = empty(client, sizes.directions);
        let spare = empty(client, short_bytes);
        let scratch = Eedi2Scratch {
            edges: &edges,
            working: &working,
            directions: &directions,
            spare: &spare,
        };

        let input_plane = plane(&input, width, height);
        eedi2_passes_through(
            client,
            &sizing,
            input_plane,
            scratch,
            0,
            &settings,
            Eedi2Stage::ExpandedGaps,
        );
    }
}

#[test]
#[should_panic(expected = "the spare scratch holds 188 bytes, expected 192")]
fn an_undersized_scratch_is_refused() {
    backend().build(UndersizedScratch);
}
