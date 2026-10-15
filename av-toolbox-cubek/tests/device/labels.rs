use av_toolbox_cubek::labels::{CountPlanes, KeepPlanes, LabelPlanes, Settled, UNSET, count, keep, label};
use av_toolbox_cubek::testing::{SplitMix, reference_labels};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_u8, read_u32, upload_u8, upload_u32};

/// What labelling a mark plane gave.
struct Labelled {
    labels: Vec<u32>,
    counts: Vec<u32>,
    kept: Vec<u8>,
    settled: Settled,
}

/// Labels `marks`, counts each label's `counted` pixels, and keeps labels with `minimum` of them.
struct Label {
    marks: Vec<u8>,
    counted: Vec<u8>,
    width: u32,
    height: u32,
    minimum: u32,
    /// A value every label starts as before labelling, in place of a zeroed plane.
    stale: Option<u32>,
}

impl BuildTask for Label {
    type Output = Labelled;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let samples = self.marks.len();

        let marks = upload_u8(client, &self.marks);
        let counted = upload_u8(client, &self.counted);
        let labels = match self.stale {
            Some(value) => {
                let filled = vec![value; samples];
                upload_u32(client, &filled)
            },
            None => empty(client, samples * size_of::<u32>()),
        };
        let counts = empty(client, samples * size_of::<u32>());
        let changed = empty(client, size_of::<u32>());
        let kept = empty(client, samples);

        let marks_plane = plane(&marks, self.width, self.height);
        let labels_plane = plane(&labels, self.width, self.height);
        let changed_plane = plane(&changed, 1, 1);
        let label_planes = LabelPlanes {
            marks: marks_plane,
            labels: labels_plane,
            changed: changed_plane,
        };
        let settled =
            label(client, &sizing, label_planes).expect("the device should hand back the changed flag");

        let labels_plane = plane(&labels, self.width, self.height);
        let counted_plane = plane(&counted, self.width, self.height);
        let counts_plane = plane(&counts, self.width, self.height);
        let count_planes = CountPlanes {
            labels: labels_plane,
            marks: counted_plane,
            counts: counts_plane,
        };
        count(client, &sizing, count_planes);

        let labels_plane = plane(&labels, self.width, self.height);
        let counts_plane = plane(&counts, self.width, self.height);
        let kept_plane = plane(&kept, self.width, self.height);
        let keep_planes = KeepPlanes {
            labels: labels_plane,
            counts: counts_plane,
            output: kept_plane,
        };
        keep(client, &sizing, keep_planes, self.minimum);

        let labels = read_u32(client, &labels);
        let counts = read_u32(client, &counts);
        let kept = read_u8(client, &kept);

        Labelled {
            labels,
            counts,
            kept,
            settled,
        }
    }
}

/// Marks from a picture, where `#` is set and `.` is not.
fn picture(rows: &[&str]) -> (Vec<u8>, u32, u32) {
    let width = rows[0].len() as u32;
    let height = rows.len() as u32;
    let marks = rows
        .iter()
        .flat_map(|row| row.chars())
        .map(|cell| if cell == '#' { 255 } else { 0 })
        .collect();

    (marks, width, height)
}

fn run(rows: &[&str], minimum: u32) -> Labelled {
    let (marks, width, height) = picture(rows);

    backend().build(Label {
        counted: marks.clone(),
        marks,
        width,
        height,
        minimum,
        stale: None,
    })
}

#[test]
fn two_separate_blobs_get_two_labels() {
    let labelled = run(&["##...", "##...", ".....", "...##"], 1);

    let first = labelled.labels[0];
    let second = labelled.labels[18];
    assert_ne!(first, second);
    assert_eq!(labelled.labels[1], first);
    assert_eq!(labelled.labels[6], first);
    assert_eq!(labelled.labels[19], second);
    assert_eq!(labelled.labels[2], UNSET);
    assert!(matches!(labelled.settled, Settled::Converged { .. }));
}

#[test]
fn diagonal_neighbours_join() {
    let labelled = run(&["#..", ".#.", "..#"], 1);

    assert_eq!(labelled.labels[0], labelled.labels[4]);
    assert_eq!(labelled.labels[4], labelled.labels[8]);
}

#[test]
fn a_spiral_settles_into_one_label() {
    let rows = [
        "###########",
        "..........#",
        "#########.#",
        "#.......#.#",
        "#.#####.#.#",
        "#.#...#.#.#",
        "#.#.###.#.#",
        "#.#.....#.#",
        "#.#######.#",
        "#.........#",
        "###########",
    ];

    let labelled = run(&rows, 1);

    let spiral: Vec<u32> = labelled
        .labels
        .iter()
        .copied()
        .filter(|label| *label != UNSET)
        .collect();
    assert!(spiral.iter().all(|label| *label == spiral[0]));
    assert!(matches!(labelled.settled, Settled::Converged { .. }));
}

#[test]
fn counts_and_keep_drop_the_small_blob() {
    let labelled = run(&["##...", "##...", ".....", "....#"], 3);

    let big = labelled.labels[0] as usize;
    let small = labelled.labels[19] as usize;
    assert_eq!(labelled.counts[big], 4);
    assert_eq!(labelled.counts[small], 1);
    assert_eq!(labelled.kept[0], 255);
    assert_eq!(labelled.kept[19], 0);
    assert_eq!(labelled.kept[2], 0);
}

#[test]
fn counting_other_marks_flags_the_blobs_they_touch() {
    let (blob_marks, width, height) = picture(&["##...", "##...", ".....", "...##"]);
    let (strong_marks, _, _) = picture(&["#....", ".....", ".....", "....."]);

    let labelled = backend().build(Label {
        marks: blob_marks,
        counted: strong_marks,
        width,
        height,
        minimum: 1,
        stale: None,
    });

    assert_eq!(labelled.kept[0], 255);
    assert_eq!(labelled.kept[6], 255);
    assert_eq!(labelled.kept[18], 0);
}

fn label_marks(marks: Vec<u8>, width: u32, height: u32, stale: Option<u32>) -> Labelled {
    backend().build(Label {
        counted: marks.clone(),
        marks,
        width,
        height,
        minimum: 1,
        stale,
    })
}

/// Fails on the first label that differs, naming how many differ in all.
fn assert_labels(actual: &[u32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());

    let mismatches: Vec<usize> = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (found, wanted))| found != wanted)
        .map(|(index, _)| index)
        .collect();

    let Some(first) = mismatches.first() else {
        return;
    };

    panic!(
        "{} labels differ, the first at {first}: got {}, expected {}",
        mismatches.len(),
        actual[*first],
        expected[*first],
    );
}

/// A path that runs the full width along every even row, joined at alternating ends.
fn serpentine(width: u32, height: u32) -> Vec<u8> {
    let mut marks = vec![0u8; (width * height) as usize];

    for row in 0..height {
        for column in 0..width {
            let along = row.is_multiple_of(2);
            let right_turn = row % 4 == 1 && column == width - 1;
            let left_turn = row % 4 == 3 && column == 0;
            if along || right_turn || left_turn {
                marks[(row * width + column) as usize] = 255;
            }
        }
    }

    marks
}

fn checkerboard(width: u32, height: u32) -> Vec<u8> {
    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            if (column + row).is_multiple_of(2) { 255 } else { 0 }
        })
        .collect()
}

fn check_against_reference(marks: Vec<u8>, width: u32, height: u32) {
    let expected = reference_labels(&marks, width, height);

    let labelled = label_marks(marks, width, height, None);

    assert_labels(&labelled.labels, &expected);
}

#[test]
fn a_serpentine_matches_the_reference() {
    let marks = serpentine(257, 255);

    check_against_reference(marks, 257, 255);
}

#[test]
fn a_checkerboard_joins_through_diagonals() {
    let marks = checkerboard(64, 48);
    let expected = reference_labels(&marks, 64, 48);

    let labelled = label_marks(marks, 64, 48, None);

    assert_labels(&labelled.labels, &expected);
    assert_eq!(labelled.labels[0], 0);
    assert_eq!(labelled.labels[64 * 48 - 1], 0);
}

#[test]
fn a_full_plane_is_one_component() {
    let marks = vec![255u8; 300 * 200];

    let labelled = label_marks(marks, 300, 200, None);

    assert!(labelled.labels.iter().all(|label| *label == 0));
}

#[test]
fn an_empty_plane_is_all_unset() {
    let marks = vec![0u8; 300 * 200];

    let labelled = label_marks(marks, 300, 200, None);

    assert!(labelled.labels.iter().all(|label| *label == UNSET));
}

#[test]
fn a_single_row_matches_the_reference() {
    let marks = vec![255, 255, 0, 255, 0, 0, 255, 255, 255];

    check_against_reference(marks, 9, 1);
}

#[test]
fn a_single_column_matches_the_reference() {
    let marks = vec![255, 255, 0, 255, 0, 0, 255, 255, 255];

    check_against_reference(marks, 1, 9);
}

#[test]
fn stale_labels_are_overwritten() {
    let mut random = SplitMix(0x57a1e);
    let marks: Vec<u8> = (0..65 * 33)
        .map(|_| if random.below(100) < 45 { 255 } else { 0 })
        .collect();
    let expected = reference_labels(&marks, 65, 33);

    let from_zero = label_marks(marks.clone(), 65, 33, Some(0));
    let from_unset = label_marks(marks, 65, 33, Some(UNSET));

    assert_labels(&from_zero.labels, &expected);
    assert_labels(&from_unset.labels, &expected);
}

#[test]
fn a_random_mask_matches_the_reference() {
    let width = 1921;
    let height = 1081;
    let mut random = SplitMix(0x1abe_11ed);
    let marks: Vec<u8> = (0..width * height)
        .map(|_| if random.below(100) < 45 { 255 } else { 0 })
        .collect();

    check_against_reference(marks, width, height);
}
