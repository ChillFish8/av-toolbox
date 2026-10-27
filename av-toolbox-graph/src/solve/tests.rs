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
        field_order: None,
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
fn fan_out_into_narrow_and_precise() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("small", "test/narrow", "input"),
        filter("exact", "test/precise", "input"),
        sink("out_small", "small"),
        sink("out_exact", "exact"),
    ]
}

/// `test/float` read by two stages taking only U8.
fn fan_out_that_suits_nobody() -> Vec<StageSpec> {
    vec![
        source("input", "test/float"),
        filter("small", "test/narrow", "input"),
        filter("tiny", "test/narrow", "input"),
        sink("out_small", "small"),
        sink("out_tiny", "tiny"),
    ]
}

fn f32_into_narrow() -> Vec<StageSpec> {
    vec![
        source("input", "test/wide"),
        filter("exact", "test/precise", "input"),
        filter("small", "test/narrow", "exact"),
        sink("out", "small"),
    ]
}

/// A mask needing matched inputs, given F32 in its first slot and U8 in its second.
fn matched_beside_float() -> Vec<StageSpec> {
    vec![
        source("input", "test/float"),
        source("second", "test/wide"),
        filter("small", "test/narrow", "second"),
        mask("both", "test/matched", "input", "small"),
        sink("out", "both"),
    ]
}

/// A mask whose base must match its overlay, with the U8 overlay settled before the F32 base.
///
/// The mask takes anything as its overlay, so the only edge it refuses is the F32 base.
fn anchored_over_settled_u8() -> Vec<StageSpec> {
    vec![
        source("second", "test/wide"),
        filter("small", "test/narrow", "second"),
        source("first", "test/float"),
        mask("both", "test/anchored", "first", "small"),
        sink("out", "both"),
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

/// One producer filling both slots of the same mask, with a second reader that takes only U8.
fn one_producer_in_both_slots() -> Vec<StageSpec> {
    vec![
        source("input", "test/floatfirst"),
        mask("both", "test/matched", "input", "input"),
        filter("small", "test/narrow", "input"),
        sink("out", "both"),
        sink("small_out", "small"),
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
        filter("exact_a", "test/precise", "first"),
        filter("small_a", "test/narrow", "exact_a"),
        sink("out_a", "small_a"),
        source("second", "test/wide"),
        filter("exact_b", "test/precise", "second"),
        filter("small_b", "test/narrow", "exact_b"),
        sink("out_b", "small_b"),
    ]
}

/// A mask whose U8 base must match an F32 overlay.
fn anchored_u8_base_under_f32() -> Vec<StageSpec> {
    vec![
        source("first", "test/wide"),
        filter("pick", "test/either", "first"),
        source("second", "test/wide"),
        filter("exact", "test/precise", "second"),
        mask("both", "test/anchored", "pick", "exact"),
        sink("out", "both"),
    ]
}

/// A mask whose F32 base must match a U8 overlay that settles after it.
///
/// The mask takes anything as its overlay, so the only edge it refuses is the F32 base.
fn anchored_cannot_agree() -> Vec<StageSpec> {
    vec![
        source("first", "test/float"),
        source("second", "test/wide"),
        filter("small", "test/narrow", "second"),
        mask("both", "test/anchored", "first", "small"),
        sink("out", "both"),
    ]
}

fn note_messages(diagnostic: &Diagnostic) -> Vec<&str> {
    diagnostic
        .notes()
        .iter()
        .map(|note| note.message.as_str())
        .collect()
}

#[test]
fn an_edge_carries_the_producers_first_choice_both_ends_accept() {
    let registry = testing::registry();
    let specs = vec![
        source("input", "test/floatfirst"),
        filter("pick", "test/either", "input"),
        sink("out", "pick"),
    ];
    let graph = build(&registry, &specs).expect("should build").into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(solved.output(NodeId::new(0)), Some(frames(SampleFormat::F32)));
}

#[test]
fn an_edge_carries_the_first_format_both_ends_accept() {
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
fn an_f32_edge_into_a_narrow_reader_names_the_edge() {
    let registry = testing::registry();
    let graph = build(&registry, &f32_into_narrow())
        .expect("should build")
        .into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert!(reported[0].message().contains("resample/depth"));
    assert_eq!(reported[0].path(), "filter.small.inputs[0]");

    let notes = note_messages(reported[0]);
    assert_eq!(notes, vec!["exact is defined here"]);
}

#[test]
fn a_narrow_producer_widens_into_a_precise_reader() {
    let registry = testing::registry();
    let graph = build(&registry, &nothing_in_common())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    let expected = Widen {
        consumer: NodeId::new(2),
        edge: 0,
        from: frames(SampleFormat::U8),
        to: frames(SampleFormat::F32),
    };
    assert_eq!(solved.widens(), &[expected]);
    assert_eq!(solved.output(NodeId::new(1)), Some(frames(SampleFormat::U8)));
    assert_eq!(
        solved.get(NodeId::new(2)).input(0),
        Some(frames(SampleFormat::F32))
    );
}

#[test]
fn a_reader_that_needs_narrowing_is_still_an_error() {
    let registry = testing::registry();
    let specs = vec![
        source("input", "test/float"),
        filter("small", "test/narrow", "input"),
        sink("out", "small"),
    ];
    let graph = build(&registry, &specs).expect("should build").into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert!(reported[0].message().contains("resample/depth"));
}

#[test]
fn a_widen_on_one_input_lets_its_sibling_through() {
    let registry = testing::registry();
    let graph = build(&registry, &matched_beside_float())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    let expected = Widen {
        consumer: NodeId::new(3),
        edge: 1,
        from: frames(SampleFormat::U8),
        to: frames(SampleFormat::F32),
    };
    assert_eq!(solved.widens(), &[expected]);
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

/// The F32 producer is decided after the U8 one has settled, so the mask refuses it firmly and
/// only widening the earlier edge lets the two meet.
#[test]
fn a_coupled_mask_widens_the_narrower_producer() {
    let registry = testing::registry();
    let graph = build(&registry, &matched_cannot_agree())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    let expected = Widen {
        consumer: NodeId::new(4),
        edge: 0,
        from: frames(SampleFormat::U8),
        to: frames(SampleFormat::F32),
    };
    assert_eq!(solved.widens(), &[expected]);
    assert_eq!(
        solved.get(NodeId::new(4)).input(0),
        Some(frames(SampleFormat::F32))
    );

    let emitted = solved.output(NodeId::new(4)).expect("a mask emits");
    assert_eq!(emitted.sample(), SampleFormat::F32);
}

/// A frame source settles before the mask, so the mask is asked for a proposal while one of its
/// inputs still needs widening.
#[test]
fn a_mask_proposed_before_its_widen_still_emits_what_it_receives() {
    let specs = vec![
        source("first", "test/wide"),
        filter("small", "test/narrow", "first"),
        source("second", "test/wide"),
        filter("exact", "test/precise", "second"),
        source("third", "test/wide"),
        mask("both", "test/matched", "small", "exact"),
        deband_with_mask("third", "both"),
        sink("out", "clean"),
    ];

    let registry = testing::registry();
    let graph = build(&registry, &specs).expect("should build").into_graph();

    let order: Vec<usize> = graph.order().iter().map(|id| id.index()).collect();
    assert_eq!(
        order,
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        "third has to settle before both"
    );

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    assert_eq!(
        solved.get(NodeId::new(5)).input(0),
        Some(frames(SampleFormat::F32))
    );

    let emitted = solved.output(NodeId::new(5)).expect("a mask emits");
    assert_eq!(emitted.sample(), SampleFormat::F32);
}

#[test]
fn a_one_way_coupled_mask_widens_its_base_to_the_overlay() {
    let registry = testing::registry();
    let graph = build(&registry, &anchored_u8_base_under_f32())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    let expected = Widen {
        consumer: NodeId::new(4),
        edge: 0,
        from: frames(SampleFormat::U8),
        to: frames(SampleFormat::F32),
    };
    assert_eq!(solved.widens(), &[expected]);

    let emitted = solved.output(NodeId::new(4)).expect("a mask emits");
    assert_eq!(emitted.sample(), SampleFormat::F32);
}

/// The F32 base is decided after its U8 overlay has settled. Only an edge the mask refuses is
/// widened, and F32 widens to nothing.
#[test]
fn a_refused_f32_base_is_reported_even_though_its_overlay_could_widen() {
    let registry = testing::registry();
    let graph = build(&registry, &anchored_over_settled_u8())
        .expect("should build")
        .into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert_eq!(reported[0].path(), "mask.both.inputs[0]");
    assert!(reported[0].message().contains("resample/depth"));

    let notes = note_messages(reported[0]);
    assert_eq!(notes, vec!["first is defined here"]);
}

/// The forward pass asks a coupled stage about a slot whose sibling is still unknown, and a
/// stage that only couples one way answers permissively. Holding it to what it finally got is
/// left to the stage's own decision and the validation pass.
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

#[test]
fn a_fan_out_widens_only_the_reader_that_needs_it() {
    let registry = testing::registry();
    let graph = build(&registry, &fan_out_into_narrow_and_precise())
        .expect("should build")
        .into_graph();

    let solved = solve(&graph, &registry, &capabilities()).expect("should solve");

    let expected = Widen {
        consumer: NodeId::new(2),
        edge: 0,
        from: frames(SampleFormat::U8),
        to: frames(SampleFormat::F32),
    };
    assert_eq!(solved.output(NodeId::new(0)), Some(frames(SampleFormat::U8)));
    assert_eq!(solved.widens(), &[expected]);
}

/// Several readers refusing one producer is a failure with no "both" to name, so every reader
/// that refused is named with what it does take.
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
    assert!(reported[0].message().contains("input emits f32"));
    assert!(reported[0].message().contains("small takes u8"));
    assert!(reported[0].message().contains("resample/depth"));

    let notes = note_messages(reported[0]);
    assert!(notes.contains(&"tiny takes u8"), "{notes:?}");
}

/// A diamond of two filters reading `test/wide` into `test/matched`, with the filters declared the
/// other way round, so the picky one is decided first. The mask turns F32 down there while
/// steering on a proposal that has not settled, and a turndown made on a proposal only sets a
/// sample aside. Declaration order must not decide whether a coupled pipeline solves.
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

/// The other reader refuses the producer's first preference, so only U8 is left.
#[test]
fn a_producer_filling_two_slots_settles_on_what_every_reader_agrees() {
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
        vec!["filter.small_a.inputs[0]", "filter.small_b.inputs[0]"]
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

fn deband_with_mask(frames: &str, overlay: &str) -> StageSpec {
    let mut clean = testing::spec(StageKind::Filter, "clean", "test/deband");
    clean.inputs = vec![
        testing::named("reference", frames),
        testing::named("overlay", overlay),
    ];

    clean
}

fn unrated_mask(name: &str) -> Vec<StageSpec> {
    let mut file = source("plain", "test/file");
    file.args = Spanned::unknown(json!({ "path": "a.mkv" }));

    vec![file, mask(name, "test/merge", "plain", "plain")]
}

#[test]
fn a_mask_from_another_rate_is_refused() {
    let specs = vec![
        rated("left", 24, 1),
        rated("right", 30, 1),
        mask("veil", "test/merge", "left", "left"),
        deband_with_mask("right", "veil"),
        sink("out", "clean"),
    ];

    let reported = rate_mismatches(&specs);

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].path(), "filter.clean");
    assert!(reported[0].message().contains("every frames and mask input"));

    let notes = reported[0].notes();

    assert_eq!(notes.len(), 2);
    assert!(notes[1].message.contains("overlay reads veil at 24/1"));
}

#[test]
fn a_mask_at_the_clips_rate_is_accepted() {
    let specs = vec![
        rated("left", 24, 1),
        mask("veil", "test/merge", "left", "left"),
        deband_with_mask("left", "veil"),
        sink("out", "clean"),
    ];

    assert!(solves(&specs));
}

#[test]
fn a_mask_from_a_rateless_clip_is_refused_beside_a_rated_one() {
    let mut specs = vec![rated("left", 24, 1)];
    specs.extend(unrated_mask("veil"));
    specs.push(deband_with_mask("left", "veil"));
    specs.push(sink("out", "clean"));

    let reported = rate_mismatches(&specs);

    assert_eq!(reported.len(), 1);
    assert!(reported[0].message().contains("unknown"));
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

#[test]
fn a_mask_edge_is_never_widened() {
    let registry = testing::registry();
    let mut edges = testing::spec(StageKind::Mask, "edges", "test/merge");
    edges.inputs = vec![testing::positional("input")];
    let mut out = testing::spec(StageKind::Sink, "out", "test/float-mask");
    out.inputs = vec![testing::positional("edges")];
    let specs = vec![source("input", "test/eight"), edges, out];
    let graph = build(&registry, &specs).expect("should build").into_graph();

    let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
    let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert_eq!(reported[0].path(), "sink.out.inputs[0]");
}
