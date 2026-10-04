use av_toolbox_cubek::ivtc::{FieldSums, MatchSums};
use av_toolbox_cubek::testing::{
    SyntheticClip,
    VfmMetrics,
    VfmSettings,
    excluded_map_fixtures,
    map_fixtures,
    reference_field_sums,
    reference_vfm_metrics,
    synthetic_clips,
    vfm_decisions,
};

use crate::ivtc::decision::{
    B,
    C,
    Decision,
    FrameMetrics,
    N,
    P,
    PairSums,
    U,
    check_mm,
    compare_fields,
    is_scene_change,
};

const MI: u32 = 80;
const CTHRESH: i32 = 9;
const SCTHRESH: f64 = 12.0 / 100.0;

fn clip_named<'a>(clips: &'a [SyntheticClip], name: &str) -> &'a SyntheticClip {
    clips
        .iter()
        .find(|clip| clip.name == name)
        .unwrap_or_else(|| panic!("there is no synthetic clip {name}"))
}

fn decision_for(clip: &SyntheticClip, mode: u32, micmatch: u32, opposite: bool) -> Decision {
    Decision {
        mode,
        micmatch,
        mi: MI,
        shift: clip.bits - 8,
        opposite,
    }
}

/// VFM's resolved field for `clip`, the clip's order unless `opposite`.
fn resolved_field(clip: &SyntheticClip, opposite: bool) -> usize {
    let order = clip.order as usize;
    if opposite { 1 - order } else { order }
}

fn reference_metrics(clip: &SyntheticClip, frame: usize, decision: &Decision) -> VfmMetrics {
    let settings = VfmSettings {
        field: resolved_field(clip, decision.opposite),
        pairs: decision.pairs(),
        exclude_from: 0,
        exclude_to: 0,
        cthresh: CTHRESH,
        chroma: true,
        mchroma: true,
    };

    reference_vfm_metrics(clip, frame, &settings)
}

fn frame_metrics(reference: &VfmMetrics, micmatch: u32) -> FrameMetrics {
    let [current_previous, current_next, previous_next] = reference.sums;
    let scene = is_scene_change(reference.current_scene, reference.next_scene, SCTHRESH);

    FrameMetrics {
        pairs: PairSums {
            current_previous,
            current_next,
            previous_next,
        },
        mics: reference.mics,
        scene_change: micmatch == 1 && scene,
    }
}

#[test]
fn the_plugin_and_the_port_agree_on_every_frame() {
    let clips = synthetic_clips();
    let fixtures = vfm_decisions();
    for fixture in &fixtures {
        let clip = clip_named(&clips, &fixture.clip);
        let decision = decision_for(clip, fixture.mode, fixture.micmatch, fixture.opposite);
        let reference = reference_metrics(clip, fixture.frame, &decision);
        let metrics = frame_metrics(&reference, fixture.micmatch);
        let name = format!(
            "{} frame {} mode {} micmatch {} opposite {}",
            fixture.clip, fixture.frame, fixture.mode, fixture.micmatch, fixture.opposite,
        );

        let matched = decision.choose(&metrics);

        assert_eq!(metrics.mics, fixture.mics, "{name} mics");
        assert_eq!(metrics.scene_change, fixture.scene_change, "{name} scene change");
        assert_eq!(matched, fixture.matched, "{name} match");
        assert_eq!(metrics.mics[matched] >= MI, fixture.combed, "{name} combed");
    }

    assert_eq!(fixtures.len(), 3456);
}

#[test]
fn every_compare_choice_matches_the_harness() {
    let clips = synthetic_clips();
    let fixtures = map_fixtures().into_iter().chain(excluded_map_fixtures());
    for fixture in fixtures {
        let clip = clip_named(&clips, &fixture.clip);
        let shift = clip.bits - 8;
        let first = fixture.first as usize;
        let second = fixture.second as usize;
        let settings = VfmSettings {
            field: fixture.field as usize,
            pairs: [(first, second); 3],
            exclude_from: fixture.exclude_from,
            exclude_to: fixture.exclude_to,
            cthresh: CTHRESH,
            chroma: true,
            mchroma: true,
        };
        let sums = reference_field_sums(clip, fixture.frame, &settings, first, second);

        let choice = compare_fields(sums, first, second, shift);

        assert_eq!(
            choice,
            fixture.choice as usize,
            "{} frame {} field {} matches {} and {} excluding {} to {}",
            fixture.clip,
            fixture.frame,
            fixture.field,
            fixture.first,
            fixture.second,
            fixture.exclude_from,
            fixture.exclude_to,
        );
    }
}

#[test]
fn the_clamped_next_frame_shares_the_current_scene_measure() {
    let clips = synthetic_clips();
    let clip = clip_named(&clips, "cut_8");
    let last = clip.frames.len() - 1;
    let decision = decision_for(clip, 1, 1, false);

    let reference = reference_metrics(clip, last, &decision);
    let metrics = frame_metrics(&reference, 1);

    assert_eq!(reference.next_scene.to_bits(), reference.current_scene.to_bits());

    let fixtures = vfm_decisions();
    let last_frames = fixtures
        .iter()
        .filter(|fixture| fixture.clip == "cut_8" && fixture.frame == last && fixture.micmatch == 1);
    let mut checked = 0;
    for fixture in last_frames {
        assert_eq!(
            metrics.scene_change, fixture.scene_change,
            "mode {}",
            fixture.mode
        );
        checked += 1;
    }

    assert!(checked > 0, "decisions.txt has the last frame of cut_8");
}

#[test]
fn check_mm_needs_a_margin_of_thirty() {
    let close = [10, 39, 0, 0, 0];
    let just_apart = [10, 40, 0, 0, 0];
    let apart = [10, 50, 0, 0, 0];

    // vivtc.cpp:615, `abs(*m2mic-*m1mic) >= 30`.
    assert_eq!(check_mm(1, 0, &close, MI), 1);
    assert_eq!(check_mm(1, 0, &just_apart, MI), 0);
    assert_eq!(check_mm(1, 0, &apart, MI), 0);
}

#[test]
fn check_mm_keeps_the_first_when_the_second_reaches_mi() {
    let mics = [80, 255, 0, 0, 0];

    // vivtc.cpp:615, `*m2mic < MI`.
    assert_eq!(check_mm(1, 0, &mics, MI), 1);
}

fn match_sums(combing: u64, motion: u64, isolated: u64) -> MatchSums {
    MatchSums {
        combing,
        motion,
        isolated,
    }
}

/// Field sums whose normalised figures, each a sum divided by 6, are the given values.
fn sixths(first: (u64, u64, u64), second: (u64, u64, u64)) -> FieldSums {
    let (first_combing, first_motion, first_isolated) = first;
    let (second_combing, second_motion, second_isolated) = second;

    FieldSums {
        first: match_sums(6 * first_combing, 6 * first_motion, 6 * first_isolated),
        second: match_sums(6 * second_combing, 6 * second_motion, 6 * second_isolated),
    }
}

#[test]
fn compare_fields_takes_the_less_moving_match_past_five_hundred() {
    // The combing is so high that `mr` stays below 0.005, and the norms alone would keep `first`.
    let sums = sixths((0, 600, 0), (1_000_000, 200, 0));

    // vivtc.cpp:549 and 554, `mtn1 >= 500` with `mtn2 * 2 < mtn1` returns `match2`.
    assert_eq!(compare_fields(sums, C, P, 0), P);
}

#[test]
fn compare_fields_takes_the_less_moving_match_past_a_thousand() {
    let sums = sixths((0, 1200, 0), (0, 700, 0));

    // vivtc.cpp:550, `mtn1 >= 1000` with `mtn2 * 3 < mtn1 * 2`, where the 2:1 tests fail.
    assert_eq!(compare_fields(sums, C, P, 0), P);
}

#[test]
fn compare_fields_takes_the_less_moving_match_past_two_thousand() {
    let sums = sixths((0, 2400, 0), (0, 1800, 0));

    // vivtc.cpp:551, `mtn1 >= 2000` with `mtn2 * 5 < mtn1 * 4`, where the 3:2 tests fail.
    assert_eq!(compare_fields(sums, C, P, 0), P);
}

#[test]
fn compare_fields_takes_the_less_moving_match_past_four_thousand() {
    // `c2` is 1.1 and `c1` is 1, while the 5:4 tests fail.
    let sums = sixths((100, 4400, 0), (100, 4000, 0));

    // vivtc.cpp:552, `mtn1 >= 4000` with `c2 > c1`.
    assert_eq!(compare_fields(sums, C, P, 0), P);
}

#[test]
fn compare_fields_scales_its_thresholds_by_the_bit_depth() {
    let sums = sixths((0, 600, 0), (1_000_000, 200, 0));

    // vivtc.cpp:549, `500 << shift` is 2000 at 10 bit, so the norms decide.
    assert_eq!(compare_fields(sums, C, P, 2), C);
}

#[test]
fn compare_fields_takes_the_less_moving_match_on_some_motion() {
    let sums = sixths((0, 300, 0), (0, 100, 0));

    // vivtc.cpp:558, `mr > 0.005` with `max(mtn1, mtn2) > 150` and a 2:1 motion.
    assert_eq!(compare_fields(sums, C, P, 0), P);
}

#[test]
fn compare_fields_ignores_motion_of_150_or_less() {
    let sums = sixths((0, 150, 0), (0, 50, 0));

    // vivtc.cpp:558 fails on `max(mtn1, mtn2) > 150`, so vivtc.cpp:564 compares the equal norms.
    assert_eq!(compare_fields(sums, C, P, 0), C);
}

#[test]
fn compare_fields_falls_back_to_the_less_combed_match() {
    let more_combed = sixths((500, 0, 0), (100, 0, 0));
    let level = sixths((100, 0, 0), (100, 0, 0));

    // vivtc.cpp:564, `norm1 > norm2` returns `match2`, and a tie keeps `match1`.
    assert_eq!(compare_fields(more_combed, C, P, 0), P);
    assert_eq!(compare_fields(level, C, P, 0), C);
}

#[test]
fn compare_fields_rounds_its_norms_in_f32() {
    // In f32, 2^25 / 6 rounds up to 5592405.5 and its norm to 5592406, while 2^25 - 2 gives 5592405.
    let first = match_sums(33_554_432, 0, 0);
    let second = match_sums(33_554_430, 0, 0);
    let sums = FieldSums { first, second };

    // vivtc.cpp:542 and 543 normalise with `/ 6.0f`, then vivtc.cpp:564 compares the norms.
    assert_eq!(compare_fields(sums, C, P, 0), P);
}

#[test]
fn compare_fields_uses_the_isolated_sums_when_motion_is_small() {
    let sums = sixths((0, 0, 600), (0, 0, 100));

    // vivtc.cpp:536 swaps in the isolated sums, then vivtc.cpp:549 decides on them.
    assert_eq!(compare_fields(sums, C, P, 0), P);
}

#[test]
fn compare_fields_keeps_the_motion_sums_when_the_isolated_sums_are_close() {
    let sums = sixths((0, 0, 600), (0, 0, 300));

    // vivtc.cpp:536 needs `max(accumPml, accumNml) > 3 * min(accumPml, accumNml)`.
    assert_eq!(compare_fields(sums, C, P, 0), C);
}

/// Frame metrics with zero sums, so every field compare keeps its `first` match.
fn still_metrics(mics: [u32; 5], scene_change: bool) -> FrameMetrics {
    FrameMetrics {
        pairs: PairSums {
            current_previous: FieldSums::default(),
            current_next: FieldSums::default(),
            previous_next: FieldSums::default(),
        },
        mics,
        scene_change,
    }
}

fn decision(mode: u32, micmatch: u32, opposite: bool) -> Decision {
    Decision {
        mode,
        micmatch,
        mi: MI,
        shift: 0,
        opposite,
    }
}

#[test]
fn mode_zero_checks_the_discarded_match() {
    let metrics = still_metrics([10, 100, 100, 100, 100], false);

    // vivtc.cpp:750, the discarded match of `c` is `p`.
    assert_eq!(decision(0, 2, false).choose(&metrics), P);
}

#[test]
fn mode_one_checks_n() {
    let metrics = still_metrics([100, 100, 10, 100, 100], false);

    // vivtc.cpp:753.
    assert_eq!(decision(1, 2, false).choose(&metrics), N);
}

#[test]
fn mode_two_checks_u() {
    let metrics = still_metrics([100, 100, 100, 100, 10], false);

    // vivtc.cpp:755.
    assert_eq!(decision(2, 2, false).choose(&metrics), U);
}

#[test]
fn mode_three_checks_b_last() {
    let metrics = still_metrics([100, 100, 100, 10, 100], false);

    // vivtc.cpp:757 to 760, `n` and `u` keep `c`, then `b` wins.
    assert_eq!(decision(3, 2, false).choose(&metrics), B);
}

#[test]
fn mode_three_checks_each_step_against_the_match_so_far() {
    let metrics = still_metrics([100, 100, 40, 100, 0], false);

    // vivtc.cpp:758 takes `n` over `c`, then vivtc.cpp:759 takes `u` over `n`.
    assert_eq!(decision(3, 2, false).choose(&metrics), U);
}

#[test]
fn mode_four_checks_c_after_a_p_match() {
    let mut metrics = still_metrics([100, 10, 100, 100, 100], false);
    metrics.pairs.current_previous = sixths((100, 0, 0), (0, 0, 0));

    // vivtc.cpp:731 picks `p` on its norm, vivtc.cpp:733 keeps it against `n`, and vivtc.cpp:761 checks `c`.
    assert_eq!(decision(4, 2, false).choose(&metrics), C);
}

#[test]
fn mode_five_checks_u_then_b() {
    let metrics = still_metrics([100, 100, 100, 10, 100], false);

    // vivtc.cpp:764 to 766, `u` keeps `c`, then `b` wins.
    assert_eq!(decision(5, 2, false).choose(&metrics), B);
}

#[test]
fn the_mode_table_keeps_a_match_that_is_not_clearly_worse() {
    let metrics = still_metrics([100, 100, 100, 100, 100], false);

    // vivtc.cpp:615 never switches between equal mics.
    for mode in 0..6 {
        assert_eq!(decision(mode, 2, false).choose(&metrics), C, "mode {mode}");
    }
}

#[test]
fn micmatch_one_checks_mics_only_on_a_scene_change() {
    let quiet = still_metrics([100, 100, 10, 100, 100], false);
    let cut = still_metrics([100, 100, 10, 100, 100], true);

    // vivtc.cpp:748, `micmatch == 2 || (sc && micmatch == 1)`.
    assert_eq!(decision(1, 1, false).choose(&quiet), C);
    assert_eq!(decision(1, 1, false).choose(&cut), N);
    assert_eq!(decision(1, 0, false).choose(&cut), C);
}

#[test]
fn mode_four_compares_c_against_n() {
    let mut metrics = still_metrics([0; 5], false);
    metrics.pairs.current_next = sixths((100, 0, 0), (0, 0, 0));

    // vivtc.cpp:733 and 734 compare `c` against `n` with the current and next sums.
    assert_eq!(decision(4, 0, false).choose(&metrics), N);
    assert_eq!(decision(3, 0, false).choose(&metrics), C);
}

#[test]
fn mode_five_compares_p_against_n() {
    let mut metrics = still_metrics([0; 5], false);
    metrics.pairs.current_previous = sixths((100, 0, 0), (0, 0, 0));
    metrics.pairs.previous_next = sixths((100, 0, 0), (0, 0, 0));

    // vivtc.cpp:731 picks `p`, then vivtc.cpp:734 compares it against `n` with the previous and next sums.
    assert_eq!(decision(5, 0, false).choose(&metrics), N);
}

#[test]
fn the_opposite_field_swaps_p_and_n() {
    let opposite = decision(1, 2, true);
    let metrics = still_metrics([10, 100, 100, 100, 100], false);

    // vivtc.cpp:694 and 695, `fxo1m` makes `n` raw match 0 and `p` raw match 2.
    assert_eq!(opposite.order(), [2, 1, 0, 4, 3]);
    assert_eq!(opposite.pairs(), [(1, 2), (1, 0), (2, 0)]);
    assert_eq!(opposite.choose(&metrics), 0);
}

#[test]
fn the_same_field_keeps_the_match_numbers() {
    let same = decision(1, 2, false);

    // vivtc.cpp:693.
    assert_eq!(same.order(), [0, 1, 2, 3, 4]);
    assert_eq!(same.pairs(), [(1, 0), (1, 2), (0, 2)]);
}

#[test]
fn a_scene_change_is_either_measure_above_the_threshold() {
    // vivtc.cpp:722 and 726.
    assert!(!is_scene_change(0.12, 0.12, SCTHRESH));
    assert!(is_scene_change(0.13, 0.0, SCTHRESH));
    assert!(is_scene_change(0.0, 0.13, SCTHRESH));
}
