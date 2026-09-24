use std::sync::atomic::Ordering;

use av_toolbox_core::StageKind;
use av_toolbox_gpu::{
    Accelerator,
    ColourInfo,
    Depth,
    ElementType,
    FrameFormat,
    FrameLayout,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use super::*;
use crate::{Diagnostic, Span, Spanned, StageSpec, build, testing};

/// Whichever backend this build enables, so these tests compile under any single one.
///
/// The `compile_error!` guard in `av-toolbox-gpu` means the list is never empty.
fn any_accelerator() -> Accelerator {
    default_accelerators()[0]
}

/// A device that can do anything, so only the test that varies it sees a capability decide.
fn capabilities() -> Capabilities {
    let every = [
        ElementType::F32,
        ElementType::F16,
        ElementType::BF16,
        ElementType::U8,
        ElementType::U16,
        ElementType::U32,
        ElementType::I8,
        ElementType::I16,
        ElementType::I32,
    ];

    Capabilities::builder(any_accelerator())
        .storable(every)
        .computable(every)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn frames(sample: SampleFormat) -> StreamFormat {
    StreamFormat::Frames(FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
    })
}

fn source(name: &str, stage_type: &str) -> StageSpec {
    testing::spec(StageKind::Source, name, stage_type)
}

fn filter(name: &str, stage_type: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, stage_type);
    spec.inputs = vec![testing::positional(from)];

    spec
}

fn mask(name: &str, stage_type: &str, base: &str, overlay: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Mask, name, stage_type);
    spec.inputs = vec![testing::positional(base), testing::positional(overlay)];

    spec
}

fn sink(name: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Sink, name, "test/stdout");
    spec.inputs = vec![testing::positional(from)];

    spec
}

fn rated(name: &str, numerator: u32, denominator: u32) -> StageSpec {
    let mut spec = source(name, "test/rated");
    spec.args = Spanned::unknown(json!({ "numerator": numerator, "denominator": denominator }));

    spec
}

fn retimed(name: &str, from: &str, numerator: u32, denominator: u32) -> StageSpec {
    let mut spec = filter(name, "test/retime", from);
    spec.args = Spanned::unknown(json!({ "numerator": numerator, "denominator": denominator }));

    spec
}

fn rate_mismatches(specs: &[StageSpec]) -> Vec<Diagnostic> {
    let registry = testing::registry();
    let graph = build(&registry, specs).expect("should build").into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");

    diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code() == DiagnosticCode::FrameRateMismatch)
        .cloned()
        .collect()
}

fn solves(specs: &[StageSpec]) -> bool {
    let registry = testing::registry();
    let graph = build(&registry, specs).expect("should build").into_graph();

    solve(&graph, &registry, &capabilities()).is_ok()
}

fn both_accept_either() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("pick", "test/either", "input"),
        sink("out", "pick"),
    ]
}

fn consumer_needs_f32() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("exact", "test/precise", "input"),
        sink("out", "exact"),
    ]
}

fn nothing_in_common() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("small", "test/narrow", "input"),
        filter("exact", "test/precise", "small"),
        sink("out", "exact"),
    ]
}

fn one_source_two_filters() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("exact", "test/precise", "input"),
        filter("pick", "test/either", "input"),
        sink("out_exact", "exact"),
        sink("out_pick", "pick"),
    ]
}

fn prefers_u16() -> Vec<StageSpec> {
    vec![
        source("input", "test/deep"),
        filter("choosy", "test/prefers16", "input"),
        sink("out", "choosy"),
    ]
}

fn diamond_into_matched() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("pick", "test/either", "input"),
        filter("exact", "test/precise", "input"),
        mask("both", "test/matched", "pick", "exact"),
        sink("out", "both"),
    ]
}

fn matched_cannot_agree() -> Vec<StageSpec> {
    vec![
        source("first", "test/wide"),
        filter("small", "test/narrow", "first"),
        source("second", "test/wide"),
        filter("exact", "test/precise", "second"),
        mask("both", "test/matched", "small", "exact"),
        sink("out", "both"),
    ]
}

/// `test/wide` read by a pass-through filter and by `test/precise`, where F32 suits both.
fn pass_through_and_precise() -> Vec<StageSpec> {
    let mut plain = filter("plain", "test/blur", "input");
    plain.args = Spanned::unknown(json!({ "sigma": 0.8 }));

    vec![
        source("input", "test/wide"),
        plain,
        filter("exact", "test/precise", "input"),
        sink("out_plain", "plain"),
        sink("out_exact", "exact"),
    ]
}

fn anchored_that_agrees() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("pick", "test/either", "input"),
        filter("exact", "test/precise", "input"),
        mask("both", "test/anchored", "pick", "exact"),
        sink("out", "both"),
    ]
}

/// The same pipeline with the mask's inputs named and written in the other order.
fn anchored_named_in_reverse() -> Vec<StageSpec> {
    let mut specs = anchored_that_agrees();
    specs[3].inputs = vec![testing::named("overlay", "exact"), testing::named("base", "pick")];

    specs
}

/// `test/wide` read by one stage taking only U8 and another taking only F32.
fn fan_out_that_suits_nobody() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("small", "test/narrow", "input"),
        filter("exact", "test/precise", "input"),
        sink("out_small", "small"),
        sink("out_exact", "exact"),
    ]
}

/// The diamond with its two middle filters declared the other way round.
fn diamond_declared_backwards() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("exact", "test/precise", "input"),
        filter("pick", "test/either", "input"),
        mask("both", "test/matched", "pick", "exact"),
        sink("out", "both"),
    ]
}

/// One producer filling both slots of the same mask.
fn one_producer_in_both_slots() -> Vec<StageSpec> {
    vec![
        source("input", "test/floatfirst"),
        mask("both", "test/matched", "input", "input"),
        sink("out", "both"),
    ]
}

/// `test/sparse` with only its second, required input bound.
fn unbound_optional() -> Vec<StageSpec> {
    let mut sparse = testing::spec(StageKind::Filter, "sparse", "test/sparse");
    sparse.inputs = vec![testing::named("second", "input")];

    vec![source("input", "test/wide"), sparse, sink("out", "sparse")]
}

fn mask_that_emits_frames() -> Vec<StageSpec> {
    let mut wrong = testing::spec(StageKind::Mask, "wrong", "test/mislabelled");
    wrong.inputs = vec![testing::positional("input")];

    vec![source("input", "test/wide"), wrong, sink("out", "wrong")]
}

/// Two chains that never meet, each with an edge nothing can carry.
fn two_impossible_edges() -> Vec<StageSpec> {
    vec![
        source("first", "test/wide"),
        filter("small_a", "test/narrow", "first"),
        filter("exact_a", "test/precise", "small_a"),
        sink("out_a", "exact_a"),
        source("second", "test/wide"),
        filter("small_b", "test/narrow", "second"),
        filter("exact_b", "test/precise", "small_b"),
        sink("out_b", "exact_b"),
    ]
}

fn anchored_cannot_agree() -> Vec<StageSpec> {
    vec![
        source("first", "test/wide"),
        filter("pick", "test/either", "first"),
        source("second", "test/wide"),
        filter("exact", "test/precise", "second"),
        mask("both", "test/anchored", "pick", "exact"),
        sink("out", "both"),
    ]
}

#[test]
fn an_edge_carries_the_narrowest_format_both_ends_accept() {
    let registry = testing::registry();
    let graph = build(&registry, &both_accept_either())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(solved.output(NodeId::new(0)), Some(frames(SampleFormat::U8)));
}

#[test]
fn a_consumer_that_needs_precision_pushes_the_edge_wider() {
    let registry = testing::registry();
    let graph = build(&registry, &consumer_needs_f32())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(solved.output(NodeId::new(0)), Some(frames(SampleFormat::F32)));
}

#[test]
fn every_node_is_covered_and_a_sink_emits_nothing() {
    let registry = testing::registry();
    let graph = build(&registry, &both_accept_either())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(solved.len(), 3);
    assert!(!solved.is_empty());
    assert_eq!(solved.output(NodeId::new(2)), None);
    assert_eq!(
        solved.get(NodeId::new(2)).input(0),
        Some(frames(SampleFormat::U8))
    );
}

#[test]
fn an_edge_whose_ends_share_nothing_names_the_edge() {
    let registry = testing::registry();
    let graph = build(&registry, &nothing_in_common())
        .expect("should build")
        .into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert!(reported[0].message().contains("convert"));
    assert_eq!(reported[0].path(), "filter.exact.inputs[0]");
}

#[test]
fn a_fan_out_edge_picks_one_format_every_consumer_accepts() {
    let registry = testing::registry();
    let graph = build(&registry, &one_source_two_filters())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(solved.output(NodeId::new(0)), Some(frames(SampleFormat::F32)));
}

/// The case the whole shape of `accepts_input` exists for. A mask needing both inputs to match
/// is asked about the first while the second is only proposed, and steering on that proposal is
/// what lets the earlier producer commit to a format the later one can also reach.
#[test]
fn a_coupled_mask_steers_a_producer_before_it_commits() {
    let registry = testing::registry();
    let graph = build(&registry, &diamond_into_matched())
        .expect("should build")
        .into_graph();

    let order: Vec<usize> = graph.order().iter().map(|id| id.index()).collect();
    assert_eq!(order, vec![0, 1, 2, 3, 4], "node 1 has to settle before node 2");

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    let wide = solved.output(NodeId::new(1)).expect("a filter emits");
    let precise = solved.output(NodeId::new(2)).expect("a filter emits");
    assert_eq!(wide, precise);
    assert_eq!(wide, frames(SampleFormat::F32));
}

/// A mask whose two producers can never carry the same sample is caught while the second of
/// them is being decided, because by then the first is settled and the mask answers exactly.
#[test]
fn a_coupled_mask_whose_producers_cannot_agree_names_the_second_edge() {
    let registry = testing::registry();
    let graph = build(&registry, &matched_cannot_agree())
        .expect("should build")
        .into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert_eq!(reported[0].path(), "mask.both.inputs[1]");
}

/// The forward pass asks a coupled stage about a slot whose sibling is still unknown, and a
/// stage that only couples one way answers permissively and is never asked again. Holding it to
/// what it finally got is the validation pass, and nothing else here reaches it.
#[test]
fn a_stage_coupled_one_way_is_held_to_what_it_finally_got() {
    let registry = testing::registry();
    let graph = build(&registry, &anchored_cannot_agree())
        .expect("should build")
        .into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert_eq!(reported[0].path(), "mask.both.inputs[0]");
}

/// Every sample a producer offers is put to every reader on its own. Asking with one sample
/// and narrowing from there would lose F32 here, because the pass-through filter answers with
/// whatever it was shown and `test/precise` refuses the U8 it was shown first.
#[test]
fn a_reader_that_mirrors_the_offer_does_not_decide_the_edge_on_its_own() {
    let registry = testing::registry();
    let graph = build(&registry, &pass_through_and_precise())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(solved.output(NodeId::new(0)), Some(frames(SampleFormat::F32)));
}

/// A stage is told its inputs in signature order, not in the order the definition wrote them,
/// so naming them the other way round has to reach the same formats. `test/anchored` couples
/// only its first slot, so a solver passing the written position as the slot steers the wrong
/// input and never settles this graph at all.
#[test]
fn named_inputs_written_out_of_order_solve_the_same_as_positional_ones() {
    let registry = testing::registry();

    let positional = build(&registry, &anchored_that_agrees())
        .expect("should build")
        .into_graph();
    let named = build(&registry, &anchored_named_in_reverse())
        .expect("should build")
        .into_graph();

    let from_positional = solve(&positional, &registry, &capabilities()).expect("should solve");
    let from_named = solve(&named, &registry, &capabilities()).expect("should solve");

    for index in 0..from_positional.len() {
        let node = NodeId::new(index);
        assert_eq!(
            from_named.output(node),
            from_positional.output(node),
            "node {index}"
        );
    }

    let base = from_positional.output(NodeId::new(1));
    assert_eq!(base, Some(frames(SampleFormat::F32)));
    assert_eq!(
        from_named.get(NodeId::new(3)).inputs(),
        from_positional.get(NodeId::new(3)).inputs(),
    );
}

/// Two stages reading one producer, each needing a sample the other refuses, is a failure with
/// no "both" to name. Naming one of them alone would show two sets that overlap, so every
/// reader that refused is named with what it does take.
#[test]
fn a_fan_out_that_suits_no_reader_names_every_one_that_refused() {
    let registry = testing::registry();
    let graph = build(&registry, &fan_out_that_suits_nobody())
        .expect("should build")
        .into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert!(reported[0].message().contains("input emits u8, f32"));
    assert!(reported[0].message().contains("small takes u8"));

    let notes: Vec<&str> = reported[0]
        .notes()
        .iter()
        .map(|note| note.message.as_str())
        .collect();
    assert!(notes.contains(&"exact takes f32"), "{notes:?}");
}

/// The same diamond as above with the two filters declared the other way round, so the picky
/// one is decided first. The mask turns F32 down there while steering on a proposal that has
/// not settled, and a turndown made on a proposal only sets a sample aside. Declaration order
/// must not decide whether a coupled pipeline solves.
#[test]
fn a_coupled_mask_still_agrees_when_its_producers_are_declared_the_other_way_round() {
    let registry = testing::registry();
    let graph = build(&registry, &diamond_declared_backwards())
        .expect("should build")
        .into_graph();

    let order: Vec<usize> = graph.order().iter().map(|id| id.index()).collect();
    assert_eq!(order, vec![0, 1, 2, 3, 4], "the picky filter has to settle first");

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    let precise = solved.output(NodeId::new(1)).expect("a filter emits");
    let wide = solved.output(NodeId::new(2)).expect("a filter emits");
    assert_eq!(precise, wide);
    assert_eq!(precise, frames(SampleFormat::F32));
}

/// A producer feeding two slots of one consumer proposes the sample under test to both. Telling
/// the second slot the producer's preference instead would pin the edge to that preference,
/// which here is the wider of the two samples the source offers.
#[test]
fn a_producer_filling_two_slots_proposes_the_same_sample_to_both() {
    let registry = testing::registry();
    let graph = build(&registry, &one_producer_in_both_slots())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(solved.output(NodeId::new(0)), Some(frames(SampleFormat::U8)));
}

/// A mask emits a mask and everything else emits frames, and the graph relies on that rather
/// than on looking inside a buffer, so a stage contradicting its own kind is an error.
#[test]
fn a_stage_producing_the_wrong_kind_of_stream_is_reported() {
    let registry = testing::registry();
    let graph = build(&registry, &mask_that_emits_frames())
        .expect("should build")
        .into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::StreamTypeMismatch);
    assert_eq!(reported[0].path(), "mask.wrong");
    assert!(
        reported[0]
            .message()
            .contains("emits mask but this stage produced frames")
    );
}

/// An optional parameter nobody bound leaves a hole, and every place a stage hears about its
/// inputs has to put the bound one at the same index. The stage is asked about slot 1, is given
/// a list to produce from whose slot 1 is the filled one, and resolves with a hole at slot 0.
#[test]
fn an_unbound_optional_keeps_one_index_everywhere() {
    testing::SPARSE_ASKED_SLOT.store(usize::MAX, Ordering::SeqCst);
    testing::SPARSE_OFFERED_SLOTS.store(0, Ordering::SeqCst);
    testing::SPARSE_CHOSEN_SLOTS.store(0, Ordering::SeqCst);
    testing::SPARSE_FILLED_SLOTS.store(0, Ordering::SeqCst);

    let registry = testing::registry();
    let graph = build(&registry, &unbound_optional())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(testing::SPARSE_ASKED_SLOT.load(Ordering::SeqCst), 1);
    assert_eq!(testing::SPARSE_OFFERED_SLOTS.load(Ordering::SeqCst), 2);
    assert_eq!(testing::SPARSE_CHOSEN_SLOTS.load(Ordering::SeqCst), 2);
    assert_eq!(testing::SPARSE_FILLED_SLOTS.load(Ordering::SeqCst), 0b10);

    let formats = solved.get(NodeId::new(1));
    assert_eq!(formats.inputs().len(), 2);
    assert_eq!(formats.input(0), None);
    assert_eq!(formats.input(1), Some(frames(SampleFormat::U8)));
}

/// One solve reports every edge it could not settle, the way building reports every stage it
/// could not build.
#[test]
fn every_impossible_edge_is_reported_not_just_the_first() {
    let registry = testing::registry();
    let graph = build(&registry, &two_impossible_edges())
        .expect("should build")
        .into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 2);
    let paths: Vec<&str> = reported.iter().map(|entry| entry.path()).collect();
    assert_eq!(
        paths,
        vec!["filter.exact_a.inputs[0]", "filter.exact_b.inputs[0]"]
    );
}

#[test]
fn a_stage_takes_its_preference_when_the_device_can_compute_it() {
    let registry = testing::registry();
    let graph = build(&registry, &prefers_u16())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    let chosen = frames(SampleFormat::U16 { depth: Depth::TEN });
    assert_eq!(solved.output(NodeId::new(0)), Some(chosen));
}

/// A device without 16-bit arithmetic is what a stage's fallback exists for, and no accelerator
/// on this machine reports one. This is the pair of the test above, so a solver that ignored
/// capabilities entirely would fail one of them.
#[test]
fn a_stage_falls_back_when_the_device_cannot_compute_its_preference() {
    let capabilities = Capabilities::builder(any_accelerator())
        .storable([ElementType::F32, ElementType::U16])
        .computable([ElementType::F32])
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build();

    let registry = testing::registry();
    let graph = build(&registry, &prefers_u16())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities).expect("should solve");

    assert_eq!(solved.output(NodeId::new(0)), Some(frames(SampleFormat::F32)));
}

/// Solving prepares every stage itself, so arguments the registry rejects are reported here
/// rather than leaving a node without a format.
#[test]
fn a_stage_whose_arguments_are_invalid_is_reported_by_solving() {
    let mut specs = both_accept_either();
    specs[1] = filter("pick", "test/blur", "input");
    specs[1].args = Spanned::unknown(json!({ "sigma": "wide" }));

    let registry = testing::registry();
    let graph = build(&registry, &specs).expect("should build").into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported[0].code(), DiagnosticCode::InvalidArgs);
}

#[test]
fn frames_inputs_at_the_same_rate_settle() {
    let specs = vec![
        rated("left", 25, 1),
        rated("right", 50, 2),
        mask("join", "test/merge", "left", "right"),
        sink("out", "join"),
    ];

    assert!(solves(&specs));
}

#[test]
fn frames_inputs_at_different_rates_report_a_mismatch_on_the_join() {
    let specs = vec![
        rated("left", 25, 1),
        rated("right", 30, 1),
        mask("join", "test/merge", "left", "right"),
        sink("out", "join"),
    ];

    let reported = rate_mismatches(&specs);

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].path(), "mask.join");
    assert!(reported[0].message().contains("25/1"));
    assert!(reported[0].message().contains("30/1"));
}

#[test]
fn a_rate_mismatch_notes_each_input_where_it_is_read() {
    let mut join = mask("join", "test/merge", "left", "right");
    join.inputs[0].span = Span::new(10, 16);
    join.inputs[1].span = Span::new(18, 25);
    let specs = vec![
        rated("left", 25, 1),
        rated("right", 30, 1),
        join,
        sink("out", "join"),
    ];

    let reported = rate_mismatches(&specs);
    let notes: Vec<(&str, Span)> = reported[0]
        .notes()
        .iter()
        .map(|note| (note.message.as_str(), note.span))
        .collect();

    assert_eq!(notes.len(), 2);
    assert_eq!(notes[0].1, Span::new(10, 16));
    assert!(notes[0].0.contains("left"));
    assert!(notes[0].0.contains("25/1"));
    assert_eq!(notes[1].1, Span::new(18, 25));
    assert!(notes[1].0.contains("right"));
    assert!(notes[1].0.contains("30/1"));
}

#[test]
fn a_known_rate_against_an_unknown_one_is_a_mismatch() {
    let mut unknown = source("right", "test/file");
    unknown.args = Spanned::unknown(json!({ "path": "a.mkv" }));
    let specs = vec![
        rated("left", 25, 1),
        unknown,
        mask("join", "test/merge", "left", "right"),
        sink("out", "join"),
    ];

    let reported = rate_mismatches(&specs);

    assert_eq!(reported.len(), 1);
    assert!(reported[0].message().contains("unknown"));
}

#[test]
fn a_mask_input_takes_no_part_in_the_rate_check() {
    let mut clean = testing::spec(StageKind::Filter, "clean", "test/deband");
    clean.inputs = vec![
        testing::named("reference", "right"),
        testing::named("overlay", "veil"),
    ];
    let specs = vec![
        rated("left", 25, 1),
        rated("right", 30, 1),
        mask("veil", "test/merge", "left", "left"),
        clean,
        sink("out", "clean"),
    ];

    assert!(solves(&specs));
}

#[test]
fn a_retimed_stream_matches_a_sibling_at_its_new_rate() {
    let specs = vec![
        rated("input", 25, 1),
        retimed("slowed", "input", 30, 1),
        rated("other", 30, 1),
        mask("join", "test/merge", "slowed", "other"),
        sink("out", "join"),
    ];

    assert!(solves(&specs));
}
