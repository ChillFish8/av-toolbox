use av_toolbox_cubek::labels::{CountPlanes, KeepPlanes, LabelPlanes, Settled, UNSET, count, keep, label};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_u8, read_u32, upload_u8};

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
        let labels = empty(client, samples * size_of::<u32>());
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
        let settled = label(client, &sizing, label_planes);

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
    });

    assert_eq!(labelled.kept[0], 255);
    assert_eq!(labelled.kept[6], 255);
    assert_eq!(labelled.kept[18], 0);
}
