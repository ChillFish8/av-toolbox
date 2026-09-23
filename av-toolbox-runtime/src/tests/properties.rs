use super::engine::{
    build_engine,
    clear_recorded,
    filter_spec,
    multiplying_spec,
    recorded,
    recorded_indices,
    run_with_watchdog,
    sink_spec,
    source_spec,
};
use crate::Settings;

/// The stream that goes in is the stream that comes out, however many workers ran it.
#[test]
fn output_is_identical_at_every_width() {
    let widths = [1, 2, 3, 4, 8];
    let mut outputs = Vec::new();

    for width in widths {
        clear_recorded();
        let specs = vec![
            source_spec("input", &[0, 0, 1, 1, 2, 2, 3, 3], 0),
            filter_spec("wide", "input", Some(width)),
            sink_spec("out", "wide"),
        ];
        let engine = build_engine(&specs, Settings::default());
        run_with_watchdog(engine, "output is identical at every width");
        outputs.push(recorded());
    }

    for (width, output) in widths.iter().zip(outputs.iter()) {
        assert_eq!(output, &outputs[0], "width {width} disagreed with width one");
    }
    assert_eq!(outputs[0].len(), 8);
}

/// A stage that emits three frames for one input, none for the next, and one for the third
/// still produces the same output at every width, since the fan-out follows each frame's own
/// identity rather than the order calls happen to arrive in.
#[test]
fn output_is_identical_at_every_width_when_a_stage_changes_the_frame_count() {
    let widths = [1, 2, 3, 4, 8];
    let mut outputs = Vec::new();

    for width in widths {
        clear_recorded();
        let specs = vec![
            source_spec("input", &[0, 0, 1, 1, 2, 2, 3, 3], 0),
            multiplying_spec("multiply", "input", Some(width)),
            sink_spec("out", "multiply"),
        ];
        let engine = build_engine(&specs, Settings::default());
        run_with_watchdog(
            engine,
            "output is identical at every width when a stage changes the frame count",
        );
        outputs.push(recorded());
    }

    for (width, output) in widths.iter().zip(outputs.iter()) {
        assert_eq!(output, &outputs[0], "width {width} disagreed with width one");
    }

    // `MultiplyingInstance` cycles 3, 0, 1 copies by each frame's timestamp modulo three. The
    // eight input frames carry timestamps 0..=7, giving copy counts 3, 0, 1, 3, 0, 1, 3, 0, which
    // sum to 11.
    assert_eq!(outputs[0].len(), 11);
}

/// The same eight frames, chunked three different ways, still come out the same: every frame
/// its own chunk, one chunk for everything, and uneven runs.
#[test]
fn output_is_identical_however_the_stream_is_chunked() {
    let chunkings: [&[u64]; 3] = [
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[0, 0, 0, 0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0, 1, 2, 2],
    ];
    let mut outputs = Vec::new();

    for chunks in chunkings {
        clear_recorded();
        let specs = vec![
            source_spec("input", chunks, 0),
            filter_spec("wide", "input", Some(4)),
            sink_spec("out", "wide"),
        ];
        let engine = build_engine(&specs, Settings::default());
        run_with_watchdog(engine, "output is identical however the stream is chunked");
        outputs.push(recorded());
    }

    for output in &outputs {
        assert_eq!(output, &outputs[0]);
    }
    assert_eq!(outputs[0], (0..8).collect::<Vec<u64>>());
}

/// Two chunks and eight workers, so six of them never get anything assigned at all.
#[test]
fn a_width_above_the_chunk_count_still_runs_every_frame_once() {
    clear_recorded();

    let chunks = [0, 0, 0, 0, 1, 1, 1, 1];
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("wide", "input", Some(8)),
        sink_spec("out", "wide"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(
        engine,
        "a width above the chunk count still runs every frame once",
    );

    let mut seen = recorded();
    seen.sort_unstable();

    assert_eq!(seen, (0..8).collect::<Vec<u64>>());
}

/// Dense, zero based indices are only proof of correct renumbering when they are paired with
/// timestamps that prove the frames themselves came out in input order. Indices alone would
/// also pass on a stream that came back dense but scrambled.
#[test]
fn renumbering_is_dense_from_zero_whatever_the_width() {
    let widths = [1, 2, 3, 4, 8];

    for width in widths {
        clear_recorded();
        let specs = vec![
            source_spec("input", &[0, 0, 1, 1, 2, 2, 3, 3], 0),
            filter_spec("wide", "input", Some(width)),
            sink_spec("out", "wide"),
        ];
        let engine = build_engine(&specs, Settings::default());
        run_with_watchdog(engine, "renumbering is dense from zero whatever the width");

        let indices = recorded_indices();
        let dense: Vec<u64> = (0..indices.len() as u64).collect();
        assert_eq!(indices, dense, "width {width} did not renumber densely from zero");

        let timestamps = recorded();
        assert_eq!(
            timestamps,
            (0..8).collect::<Vec<u64>>(),
            "width {width}: dense indices are not proof the frames arrived in order",
        );
    }
}

/// No duplicates and none missing, at every width, checked against the multiset of timestamps
/// rather than the renumbered indices.
#[test]
fn every_frame_arrives_exactly_once_at_every_width() {
    let widths = [1, 2, 3, 4, 8];
    let chunks = [0, 0, 1, 1, 2, 2, 3, 3];
    let expected: Vec<u64> = (0..chunks.len() as u64).collect();

    for width in widths {
        clear_recorded();
        let specs = vec![
            source_spec("input", &chunks, 0),
            filter_spec("wide", "input", Some(width)),
            sink_spec("out", "wide"),
        ];
        let engine = build_engine(&specs, Settings::default());
        run_with_watchdog(engine, "every frame arrives exactly once at every width");

        let mut seen = recorded();
        seen.sort_unstable();

        assert_eq!(seen, expected, "width {width} lost or duplicated a frame");
    }
}
