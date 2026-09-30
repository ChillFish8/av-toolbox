use av_toolbox_cubek::aa::{SangnomPlanes, sangnom_aaf, sangnom_field, sangnom_scratch};
use av_toolbox_cubek::testing::{SANGNOM_FIXTURES, reference_sangnom_field};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// Both field passes of one plane, sharing one scratch.
struct FieldPasses {
    input: Vec<f32>,
    width: u32,
    height: u32,
    aaf: f32,
    max: f32,
}

impl BuildTask for FieldPasses {
    type Output = [Vec<f32>; 2];

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();
        let smoothed_bytes = sangnom_scratch(self.width, self.height);

        let input = upload_f32(client, &self.input);
        let smoothed = empty(client, smoothed_bytes);
        let outputs = [empty(client, bytes), empty(client, bytes)];

        for (offset, output) in outputs.iter().enumerate() {
            let planes = SangnomPlanes {
                input: plane(&input, self.width, self.height),
                output: plane(output, self.width, self.height),
                smoothed: &smoothed,
            };
            sangnom_field(client, &sizing, planes, offset as u32, self.aaf, self.max);
        }

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        outputs.map(|output| read_f32(client, &output))
    }
}

fn on_device(input: &[f32], width: u32, height: u32, aaf: f32, max: f32) -> [Vec<f32>; 2] {
    let task = FieldPasses {
        input: input.to_vec(),
        width,
        height,
        aaf,
        max,
    };

    backend().build(task)
}

/// Fails at the first sample that differs, naming its row and column.
fn assert_exact(actual: &[f32], expected: &[f32], width: u32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: the plane sizes differ");

    let width = width as usize;
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let row = index / width;
        let column = index % width;
        assert_eq!(got, want, "{label}: row {row}, column {column}");
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

fn assert_matches_reference(input: &[f32], width: u32, height: u32, aaf: f32, max: f32, label: &str) {
    let [pass0, pass1] = on_device(input, width, height, aaf, max);
    let expected_pass0 = reference_sangnom_field(input, width, height, 0, aaf, max);
    let expected_pass1 = reference_sangnom_field(input, width, height, 1, aaf, max);

    let label_pass0 = format!("{label} at offset 0");
    let label_pass1 = format!("{label} at offset 1");
    assert_exact(&pass0, &expected_pass0, width, &label_pass0);
    assert_exact(&pass1, &expected_pass1, width, &label_pass1);
}

#[test]
fn every_fixture_matches_on_the_device() {
    for fixture in &SANGNOM_FIXTURES {
        let width = fixture.setting("width");
        let height = fixture.setting("height");
        let bits = fixture.setting("bits");
        let aa = fixture.setting("aa");
        let aaf = sangnom_aaf(aa, bits);
        let max = if bits == 8 { 255.0 } else { 65535.0 };
        let input = fixture.input_plane();
        let expected_pass0 = fixture.pass0_plane();
        let expected_pass1 = fixture.pass1_plane();

        let [pass0, pass1] = on_device(&input, width, height, aaf, max);

        let label_pass0 = format!("{} at offset 0", fixture.name);
        let label_pass1 = format!("{} at offset 1", fixture.name);
        assert_exact(&pass0, &expected_pass0, width, &label_pass0);
        assert_exact(&pass1, &expected_pass1, width, &label_pass1);
    }
}

#[test]
fn a_wide_plane_matches_the_reference() {
    let width = 3840;
    let height = 6;
    let mut input = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let band = (column + 3 * row) / 5 % 2;
            let base = if band == 0 { 40 } else { 200 };
            let noise = scramble(column, row) % 41;
            let value = (base + noise - 20).min(255);
            input.push(value as f32);
        }
    }

    let aaf = sangnom_aaf(48, 8);

    assert_matches_reference(&input, width, height, aaf, 255.0, "3840x6");
}

#[test]
fn a_ten_bit_pass_may_exceed_the_peak() {
    let width = 256;
    let height = 256;
    let mut input = Vec::new();
    for row in 0..height {
        for column in 0..width {
            // Edges that curve away from the vertical, cut between 0 and the 10-bit peak.
            let curve = (column * column + 3 * row) / 11;
            let value = if curve % 2 == 0 { 0.0 } else { 1023.0 };
            input.push(value);
        }
    }

    let aaf = sangnom_aaf(128, 10);
    let [pass0, pass1] = on_device(&input, width, height, aaf, 65535.0);
    let expected_pass0 = reference_sangnom_field(&input, width, height, 0, aaf, 65535.0);
    let expected_pass1 = reference_sangnom_field(&input, width, height, 1, aaf, 65535.0);

    let above_peak = pass0
        .iter()
        .chain(&pass1)
        .filter(|value| **value > 1023.0)
        .count();
    println!("{above_peak} of {} samples are above 1023", 2 * input.len());

    assert_exact(&pass0, &expected_pass0, width, "10-bit at offset 0");
    assert_exact(&pass1, &expected_pass1, width, "10-bit at offset 1");
    assert!(
        above_peak > 0,
        "the plane should push some samples above the peak"
    );
}
