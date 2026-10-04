use av_toolbox_cubek::ivtc::DecimateMetric;
use av_toolbox_cubek::testing::{
    SyntheticClip,
    decimate_drop_fixtures,
    reference_decimate_metric,
    synthetic_clips,
    vfm_matched_frames,
};

use crate::ivtc::decimation::{Thresholds, find_drop, first_cycle_metrics};

const CYCLE: usize = 5;
const WIDTH: usize = 64;
const HEIGHT: usize = 32;

fn metric(max_block: i64, total: i64) -> DecimateMetric {
    DecimateMetric { max_block, total }
}

fn thresholds(duplicate: i64, scene: i64) -> Thresholds {
    Thresholds { duplicate, scene }
}

fn planes_of(frames: &[[Vec<i32>; 3]], frame: usize) -> [&[i32]; 3] {
    let planes = &frames[frame];

    [&planes[0], &planes[1], &planes[2]]
}

fn frame_metric(
    frames: &[[Vec<i32>; 3]],
    frame: usize,
    block: (usize, usize),
    chroma: bool,
) -> DecimateMetric {
    let dims = [(WIDTH, HEIGHT), (WIDTH / 2, HEIGHT / 2), (WIDTH / 2, HEIGHT / 2)];
    let previous = planes_of(frames, frame.saturating_sub(1));
    let current = planes_of(frames, frame);
    let planes = if chroma { 3 } else { 1 };

    reference_decimate_metric(previous, current, dims, planes, block.0, block.1, (1, 1))
}

fn clip_named(clips: &[SyntheticClip], name: &str) -> usize {
    clips
        .iter()
        .position(|clip| clip.name == name)
        .unwrap_or_else(|| panic!("there is no synthetic clip {name}"))
}

#[test]
fn every_drop_matches_the_harness() {
    let fixtures = decimate_drop_fixtures();
    let clips = synthetic_clips();
    let matched: Vec<_> = clips.iter().map(vfm_matched_frames).collect();

    for fixture in &fixtures {
        let index = clip_named(&clips, &fixture.clip);
        let clip = &clips[index];
        let frames = if fixture.matched {
            &matched[index]
        } else {
            &clip.frames
        };
        let block = (fixture.block_x, fixture.block_y);
        let start = fixture.cycle * CYCLE;
        let end = start + fixture.length;
        let mut metrics: Vec<_> = (start..end)
            .map(|frame| frame_metric(frames, frame, block, fixture.chroma))
            .collect();

        let thresholds = Thresholds::new(
            fixture.dupthresh,
            fixture.scthresh,
            clip.bits,
            WIDTH as u32,
            HEIGHT as u32,
            fixture.block_x as u32,
            fixture.block_y as u32,
        );
        if fixture.cycle == 0 {
            first_cycle_metrics(&mut metrics, &thresholds);
        }

        let drop = find_drop(&metrics, &thresholds);
        let label = format!(
            "{} matched {} blocks {}x{} chroma {} thresholds {} {} cycle {}",
            fixture.clip,
            fixture.matched,
            fixture.block_x,
            fixture.block_y,
            fixture.chroma,
            fixture.dupthresh,
            fixture.scthresh,
            fixture.cycle,
        );
        assert_eq!(drop, fixture.drop, "{label}");

        let outputs: Vec<usize> = (start..end).filter(|frame| frame - start != drop).collect();
        assert_eq!(outputs, fixture.outputs, "{label}");
    }

    assert_eq!(fixtures.len(), 432);
}

// vivtc.cpp 1307-1308 sets `duppos` to `lowest` when its `maxbdiff` is below `dupthresh`, and 1314 drops
// `lowest`, so frame 2 goes even though frame 3 is a scene change (1301-1302).
#[test]
fn the_lowest_is_dropped_when_it_is_below_the_duplicate_threshold() {
    let metrics = [
        metric(50, 0),
        metric(40, 0),
        metric(9, 0),
        metric(30, 500),
        metric(60, 0),
    ];

    assert_eq!(find_drop(&metrics, &thresholds(10, 100)), 2);
}

// vivtc.cpp 1311-1312: a scene change with `duppos` still `DropUnknown` drops `scpos`, the last `totdiff`
// above `scthresh` (1301-1302), here frame 3 rather than frame 1.
#[test]
fn with_no_duplicate_the_frame_at_the_last_scene_change_is_dropped() {
    let metrics = [
        metric(50, 0),
        metric(40, 500),
        metric(20, 0),
        metric(30, 500),
        metric(60, 0),
    ];

    assert_eq!(find_drop(&metrics, &thresholds(10, 100)), 3);
}

// vivtc.cpp 1314: with `scpos` unknown and no duplicate under `dupthresh` (1307), `drop = lowest`.
#[test]
fn with_neither_the_lowest_is_dropped() {
    let metrics = [
        metric(50, 100),
        metric(40, 0),
        metric(20, 0),
        metric(30, 0),
        metric(60, 0),
    ];

    assert_eq!(find_drop(&metrics, &thresholds(10, 100)), 2);
}

// vivtc.cpp 1303 updates `lowest` only on a strictly lower `maxbdiff`, so the earliest of equals stays.
#[test]
fn ties_go_to_the_earliest() {
    let metrics = [
        metric(50, 0),
        metric(20, 0),
        metric(30, 0),
        metric(20, 0),
        metric(20, 0),
    ];

    assert_eq!(find_drop(&metrics, &thresholds(10, 100)), 1);
}

// vivtc.cpp 1301 compares `totdiff` with `>`, so a total equal to `scthresh` is no scene change and the
// lowest frame, 2, is dropped under 1314.
#[test]
fn a_total_equal_to_the_scene_threshold_is_no_scene_change() {
    let metrics = [metric(50, 100), metric(40, 0), metric(20, 0)];

    assert_eq!(find_drop(&metrics, &thresholds(10, 100)), 2);
}

// vivtc.cpp 1307 compares `maxbdiff` with `<`, so a lowest equal to `dupthresh` is no duplicate and the scene
// change, frame 0, is dropped under 1311-1312.
#[test]
fn a_lowest_equal_to_the_duplicate_threshold_is_no_duplicate() {
    let metrics = [metric(50, 101), metric(40, 0), metric(10, 0)];

    assert_eq!(find_drop(&metrics, &thresholds(10, 100)), 0);
}

// vivtc.cpp 1428-1429 sets frame 0's `maxbdiff` to frame 1's and its `totdiff` to `scthresh + 1`.
#[test]
fn the_first_frame_borrows_the_second_frames_block_difference() {
    let mut metrics = vec![metric(0, 0), metric(70, 12), metric(80, 3)];
    let limits = thresholds(10, 100);

    first_cycle_metrics(&mut metrics, &limits);

    assert_eq!(metrics, [metric(70, 101), metric(70, 12), metric(80, 3)]);
}

// vivtc.cpp 1428-1429 then 1301-1302 and 1311-1312: with no duplicate, frame 0's `scthresh + 1` is a scene
// change, so frame 0 is dropped even though frame 2 has the lowest difference.
#[test]
fn the_first_frame_is_dropped_as_a_scene_change_without_a_duplicate() {
    let mut metrics = vec![metric(0, 0), metric(70, 12), metric(40, 3)];
    let limits = thresholds(10, 100);

    first_cycle_metrics(&mut metrics, &limits);

    assert_eq!(find_drop(&metrics, &limits), 0);
}

// vivtc.cpp 1428-1429 copies frame 1's `maxbdiff` into frame 0, which then ties with it, and 1303 keeps the
// earliest.
#[test]
fn the_first_frame_wins_a_tie_with_the_second() {
    let mut metrics = vec![metric(0, 0), metric(5, 12), metric(40, 3)];
    let limits = thresholds(1, 100);

    first_cycle_metrics(&mut metrics, &limits);

    assert_eq!(find_drop(&metrics, &limits), 0);
}

// vivtc.cpp 1428-1429 would read `metrics[1]`, so a one-frame cycle is left as it is. 1301-1307 then find
// frame 0 the lowest.
#[test]
fn a_one_frame_cycle_is_left_alone() {
    let mut metrics = vec![metric(0, 0)];
    let limits = thresholds(10, 100);

    first_cycle_metrics(&mut metrics, &limits);

    assert_eq!(metrics, [metric(0, 0)]);
    assert_eq!(find_drop(&metrics, &limits), 0);
}

// vivtc.cpp 1572-1573 take `(int64_t)(((int64_t)max_value * ...) * threshold) / 100)`, truncating the double.
#[test]
fn thresholds_scale_as_create_vdecimate_scales_them() {
    // 8 bit, 64x32: 255 * 2048 * 12.3 / 100 = 64235.52, 255 * 1024 * 1.1 / 100 = 2872.32.
    let eight = Thresholds::new(1.1, 12.3, 8, 64, 32, 32, 32);
    // 10 bit: 1023 * 2048 * 12.3 / 100 = 257697.792, 1023 * 1024 * 1.1 / 100 = 11523.072.
    let ten = Thresholds::new(1.1, 12.3, 10, 64, 32, 32, 32);

    assert_eq!((eight.scene, eight.duplicate), (64235, 2872));
    assert_eq!((ten.scene, ten.duplicate), (257697, 11523));
}
