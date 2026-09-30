use super::fixtures::decode_plane;
use super::{
    EEDI2_FIXTURES,
    Eedi2Parameters,
    LeftDirection,
    Typo,
    field_stages,
    reference_eedi2_antialias,
    reference_eedi2_field,
    reference_eedi2_field_stages,
};

const WORKED_WIDTH: usize = 16;
const PEAK: f32 = 255.0;

/// A map value for a direction of `pixels` at 8 bits.
fn direction(pixels: f32) -> f32 {
    128.0 + 4.0 * pixels
}

/// The 45 degree staircase of reference section 6, 40 left of the diagonal and 200 right of it.
fn staircase() -> Vec<f32> {
    let mut plane = Vec::with_capacity(WORKED_WIDTH * WORKED_WIDTH);
    for row in 0..WORKED_WIDTH as i64 {
        for column in 0..WORKED_WIDTH as i64 {
            let value = if row > column - 2 { 40.0 } else { 200.0 };
            plane.push(value);
        }
    }

    plane
}

/// A 16 sample row starting with `start` and filled out with `rest`.
fn row_of(start: &[f32], rest: f32) -> Vec<f32> {
    let mut row = start.to_vec();
    row.resize(WORKED_WIDTH, rest);
    row
}

/// A mask drawn with `#` for set and `.` for clear, one string per row.
fn mask_of(rows: &[&str]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| row.chars())
        .map(|symbol| if symbol == '#' { 1.0 } else { 0.0 })
        .collect()
}

fn row_slice(plane: &[f32], row: usize) -> &[f32] {
    let start = row * WORKED_WIDTH;
    &plane[start..start + WORKED_WIDTH]
}

#[test]
fn the_worked_example_matches_by_hand() {
    let plane = staircase();
    let without_pp = Eedi2Parameters {
        pp: 0,
        ..Eedi2Parameters::default()
    };
    let with_pp = Eedi2Parameters::default();

    let pass_a_without_pp = reference_eedi2_field(&plane, 16, 16, 0, &without_pp);
    let pass_a = reference_eedi2_field_stages(&plane, 16, 16, 0, &with_pp);
    let pass_b = reference_eedi2_field(&plane, 16, 16, 1, &with_pp);
    let merged = reference_eedi2_antialias(&plane, 16, 16, &with_pp);

    // The masks of reference section 6.1.
    let e0 = mask_of(&[
        "................",
        "..####..........",
        "....####........",
        "......####......",
        "........####....",
        "..........####..",
        "............###.",
        "................",
    ]);
    let e1 = mask_of(&[
        "................",
        "...###..........",
        "....####........",
        "......####......",
        "........####....",
        "..........####..",
        "............###.",
        "................",
    ]);
    let e2 = mask_of(&[
        "................",
        "...####.........",
        "....#####.......",
        ".....######.....",
        ".......######...",
        ".........#####..",
        "...........####.",
        "................",
    ]);
    assert_eq!(pass_a.e0, e0);
    assert_eq!(pass_a.e1, e1);
    assert_eq!(pass_a.e2, e2);
    assert_eq!(pass_a.e3, e2);
    assert_eq!(pass_a.e, e2);

    // The directions of reference sections 6.2 and 6.3.
    let two_left = direction(-2.0);
    assert_eq!(pass_a.d0[2 * WORKED_WIDTH + 5], two_left);
    assert_eq!(pass_a.d1[2 * WORKED_WIDTH + 5], two_left);
    assert_eq!(pass_a.x0[WORKED_WIDTH + 5], two_left);
    let d3_row_1 = row_of(&[PEAK, PEAK, PEAK, PEAK, two_left, two_left, two_left], PEAK);
    let d3_row_2 = row_of(
        &[
            PEAK, PEAK, PEAK, PEAK, two_left, two_left, two_left, two_left, two_left,
        ],
        PEAK,
    );
    let x4_row_1 = row_of(
        &[PEAK, PEAK, PEAK, PEAK, two_left, two_left, two_left, two_left],
        PEAK,
    );
    assert_eq!(row_slice(&pass_a.d3, 1), d3_row_1);
    assert_eq!(row_slice(&pass_a.d3, 2), d3_row_2);
    assert_eq!(row_slice(&pass_a.x4, 1), x4_row_1);

    // The lattice at gap row 1 of reference section 6.4.
    assert_eq!(pass_a.l[WORKED_WIDTH + 5], two_left);
    assert_eq!(pass_a.r[WORKED_WIDTH + 5], 160.0);
    assert_eq!(pass_a.l[WORKED_WIDTH + 4], direction(-3.0));
    assert_eq!(pass_a.r[WORKED_WIDTH + 4], 40.0);
    assert_eq!(pass_a.y2[WORKED_WIDTH + 4], PEAK);

    // Pass A of reference section 6.5, without and with post-processing.
    let row_1 = row_of(&[40.0, 40.0, 120.0, 120.0], 200.0);
    let row_3_without_pp = row_of(&[40.0, 40.0, 40.0, 40.0, 40.0, 160.0], 200.0);
    let row_3 = row_of(&[40.0, 40.0, 40.0, 40.0, 120.0, 160.0], 200.0);
    let row_5 = row_of(&[40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 80.0, 160.0], 200.0);
    let row_7 = row_of(
        &[40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 80.0, 160.0],
        200.0,
    );
    let row_13 = row_of(&[40.0; 14], 120.0);
    let row_15 = row_of(&[], 40.0);
    assert_eq!(row_slice(&pass_a_without_pp, 1), row_1);
    assert_eq!(row_slice(&pass_a_without_pp, 3), row_3_without_pp);
    assert_eq!(row_slice(&pass_a_without_pp, 5), row_5);
    assert_eq!(row_slice(&pass_a_without_pp, 7), row_7);
    assert_eq!(row_slice(&pass_a_without_pp, 13), row_13);
    assert_eq!(row_slice(&pass_a_without_pp, 15), row_15);
    assert_eq!(row_slice(&pass_a.output, 1), row_1);
    assert_eq!(row_slice(&pass_a.output, 3), row_3);
    assert_eq!(row_slice(&pass_a.output, 5), row_5);
    assert_eq!(row_slice(&pass_a.output, 7), row_7);
    assert_eq!(row_slice(&pass_a.output, 13), row_13);
    assert_eq!(row_slice(&pass_a.output, 15), row_15);

    // Pass B of reference section 6.5.
    let row_0 = row_of(&[40.0, 40.0, 40.0], 200.0);
    let row_2 = row_of(&[40.0, 40.0, 40.0, 120.0, 120.0], 200.0);
    let row_4 = row_of(&[40.0, 40.0, 40.0, 40.0, 40.0, 120.0, 160.0], 200.0);
    let row_6 = row_of(&[40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 80.0, 160.0], 200.0);
    let row_14 = row_of(&[40.0; 15], 120.0);
    assert_eq!(row_slice(&pass_b, 0), row_0);
    assert_eq!(row_slice(&pass_b, 2), row_2);
    assert_eq!(row_slice(&pass_b, 4), row_4);
    assert_eq!(row_slice(&pass_b, 6), row_6);
    assert_eq!(row_slice(&pass_b, 14), row_14);

    // The double-rate anti-alias of reference section 6.5.
    let rows: [[u16; 16]; 16] = [
        [
            40, 40, 120, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 40, 40, 80, 180, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 40, 40, 40, 80, 180, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 40, 40, 40, 40, 60, 180, 200, 200, 200, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 40, 40, 40, 40, 40, 60, 180, 200, 200, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 40, 40, 40, 40, 40, 40, 60, 180, 200, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 180, 200, 200, 200, 200, 200,
        ],
        [
            40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 180, 200, 200, 200, 200,
        ],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 180, 200, 200, 200],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 160, 200, 200],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 80, 160, 200],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 80, 160],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 80],
        [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
    ];
    let expected: Vec<f32> = rows.iter().flatten().map(|value| f32::from(*value)).collect();
    assert_eq!(merged, expected);
}

#[test]
fn every_fixture_matches_the_plugin() {
    for fixture in &EEDI2_FIXTURES {
        let width = fixture.setting("width");
        let height = fixture.setting("height");
        let parameters = fixture.parameters();
        let plane = fixture.input_plane();
        let expected_pass0 = fixture.pass0_plane();
        let expected_pass1 = fixture.pass1_plane();
        let expected_merged = fixture.merged_plane();
        let samples = width as usize * height as usize;
        assert_eq!(plane.len(), samples, "{} input", fixture.name);
        assert_eq!(expected_pass0.len(), samples, "{} pass 0", fixture.name);
        assert_eq!(expected_pass1.len(), samples, "{} pass 1", fixture.name);
        assert_eq!(expected_merged.len(), samples, "{} merged", fixture.name);

        let pass0 = reference_eedi2_field(&plane, width, height, 0, &parameters);
        let pass1 = reference_eedi2_field(&plane, width, height, 1, &parameters);
        let merged = reference_eedi2_antialias(&plane, width, height, &parameters);

        assert_eq!(pass0, expected_pass0, "{} at offset 0", fixture.name);
        assert_eq!(pass1, expected_pass1, "{} at offset 1", fixture.name);
        assert_eq!(merged, expected_merged, "{} merged", fixture.name);
    }
}

#[test]
fn the_worked_fixture_is_the_staircase() {
    let fixture = &EEDI2_FIXTURES[0];
    let plane = decode_plane(fixture.input);

    assert_eq!(fixture.name, "worked");
    assert_eq!(plane, staircase());
}

#[test]
fn the_lattice_left_dependency_matters() {
    let fixture = EEDI2_FIXTURES
        .iter()
        .find(|fixture| fixture.name == "left_dependency")
        .expect("the left dependency fixture exists");
    let width = fixture.setting("width");
    let height = fixture.setting("height");
    let parameters = fixture.parameters();
    let plane = fixture.input_plane();
    let expected_pass0 = fixture.pass0_plane();
    let expected_pass1 = fixture.pass1_plane();

    let updated_pass0 = field_stages(
        &plane,
        width,
        height,
        0,
        &parameters,
        LeftDirection::Updated,
        Typo::Kept,
    );
    let updated_pass1 = field_stages(
        &plane,
        width,
        height,
        1,
        &parameters,
        LeftDirection::Updated,
        Typo::Kept,
    );
    let original_pass0 = field_stages(
        &plane,
        width,
        height,
        0,
        &parameters,
        LeftDirection::Original,
        Typo::Kept,
    );
    let original_pass1 = field_stages(
        &plane,
        width,
        height,
        1,
        &parameters,
        LeftDirection::Original,
        Typo::Kept,
    );

    assert_eq!(updated_pass0.output, expected_pass0);
    assert_eq!(updated_pass1.output, expected_pass1);
    let pass0_differs = original_pass0.output != expected_pass0;
    let pass1_differs = original_pass1.output != expected_pass1;
    assert!(
        pass0_differs || pass1_differs,
        "reading the original left direction left both passes unchanged"
    );
}

#[test]
fn the_b1_typo_matters() {
    let fixture = EEDI2_FIXTURES
        .iter()
        .find(|fixture| fixture.name == "diagonal_pp1")
        .expect("the diagonal fixture exists");
    let width = fixture.setting("width");
    let height = fixture.setting("height");
    let parameters = fixture.parameters();
    let plane = fixture.input_plane();
    let expected_pass0 = fixture.pass0_plane();
    let expected_pass1 = fixture.pass1_plane();

    let kept_pass0 = field_stages(
        &plane,
        width,
        height,
        0,
        &parameters,
        LeftDirection::Updated,
        Typo::Kept,
    );
    let kept_pass1 = field_stages(
        &plane,
        width,
        height,
        1,
        &parameters,
        LeftDirection::Updated,
        Typo::Kept,
    );
    let fixed_pass0 = field_stages(
        &plane,
        width,
        height,
        0,
        &parameters,
        LeftDirection::Updated,
        Typo::Fixed,
    );
    let fixed_pass1 = field_stages(
        &plane,
        width,
        height,
        1,
        &parameters,
        LeftDirection::Updated,
        Typo::Fixed,
    );

    assert_eq!(kept_pass0.output, expected_pass0);
    assert_eq!(kept_pass1.output, expected_pass1);
    assert!(
        kept_pass0.x0 != fixed_pass0.x0 || kept_pass1.x0 != fixed_pass1.x0,
        "fixing the typo left the gap directions unchanged"
    );
    let pass0_differs = fixed_pass0.output != expected_pass0;
    let pass1_differs = fixed_pass1.output != expected_pass1;
    assert!(
        pass0_differs || pass1_differs,
        "fixing the typo left both passes unchanged"
    );
}

const EDGE_WIDTH: usize = 40;
const EDGE_HEIGHT: usize = 16;

/// Hard stripes that cross both side edges at a shallow slope, so directions reach the edge columns.
fn edge_heavy_plane() -> Vec<f32> {
    let mut plane = Vec::with_capacity(EDGE_WIDTH * EDGE_HEIGHT);
    for row in 0..EDGE_HEIGHT {
        for column in 0..EDGE_WIDTH {
            let band = (column + 6 * row) / 8;
            let value = if band % 2 == 0 { 20.0 } else { 235.0 };
            plane.push(value);
        }
    }

    plane
}

/// Whether `map` holds a direction other than 0 within 3 columns of a side edge.
fn has_direction_near_an_edge(map: &[f32]) -> bool {
    let neutral = 128.0;
    map.iter().enumerate().any(|(index, value)| {
        let column = index % EDGE_WIDTH;
        let near_edge = column <= 3 || column >= EDGE_WIDTH - 4;
        near_edge && *value != PEAK && *value != neutral
    })
}

#[test]
fn the_clamp_keeps_every_read_in_the_row() {
    // Every read goes through a helper that asserts, in debug builds, that it stays inside its row.
    let plane = edge_heavy_plane();
    let parameters = Eedi2Parameters {
        maxd: 29,
        ..Eedi2Parameters::default()
    };

    let pass0 = reference_eedi2_field_stages(&plane, 40, 16, 0, &parameters);
    let pass1 = reference_eedi2_field_stages(&plane, 40, 16, 1, &parameters);

    assert!(has_direction_near_an_edge(&pass0.d0) || has_direction_near_an_edge(&pass1.d0));
    assert!(has_direction_near_an_edge(&pass0.l) || has_direction_near_an_edge(&pass1.l));
}

#[test]
fn the_fixtures_fill_gaps_and_post_process() {
    let mut filled = 0;
    let mut reset = 0;

    for fixture in &EEDI2_FIXTURES {
        let width = fixture.setting("width");
        let height = fixture.setting("height");
        let parameters = fixture.parameters();
        let input = fixture.input_plane();

        for offset in 0..2 {
            let stages = reference_eedi2_field_stages(&input, width, height, offset, &parameters);
            filled += stages
                .x2
                .iter()
                .zip(&stages.x4)
                .filter(|(x2, x4)| x2 != x4)
                .count();

            let rows = stages.r.chunks(width as usize);
            let rebuilt_rows = (0..).map(|gap| 2 * gap + 1 + offset as usize);
            for (row, rebuilt) in rebuilt_rows.zip(rows) {
                let start = row * width as usize;
                let output_row = &stages.output[start..start + width as usize];
                reset += output_row
                    .iter()
                    .zip(rebuilt)
                    .filter(|(output_value, rebuilt_value)| output_value != rebuilt_value)
                    .count();
            }
        }
    }

    assert!(filled > 0, "the fixtures should fill some gap directions");
    assert!(reset > 0, "the fixtures should post-process some rebuilt samples");
}
