use std::collections::HashSet;

use av_toolbox_core::{InputSignature, PipelineRegistry, PreparedStage, StageError, StageKind, StreamType};
use av_toolbox_gpu::{
    Capabilities,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};

use crate::diagnostic::{input_path, stage_path};
use crate::graph::{CompiledGraph, Node, NodeId};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// One set of formats per node, produced by the solver and consumed by instantiation.
///
/// It is built only by solving, so it always covers every node of the graph it came from.
#[derive(Debug, Clone)]
pub struct SolvedFormats {
    per_node: Vec<ResolvedFormats>,
}

impl SolvedFormats {
    pub(crate) fn new(per_node: Vec<ResolvedFormats>) -> Self {
        Self { per_node }
    }

    pub fn get(&self, node: NodeId) -> &ResolvedFormats {
        &self.per_node[node.index()]
    }

    /// What `node` emits, which is nothing for a sink.
    pub fn output(&self, node: NodeId) -> Option<StreamFormat> {
        self.get(node).output()
    }

    pub fn len(&self) -> usize {
        self.per_node.len()
    }

    pub fn is_empty(&self) -> bool {
        self.per_node.is_empty()
    }

    /// Adds formats for a node appended to the graph after solving.
    pub(crate) fn push(&mut self, formats: ResolvedFormats) {
        self.per_node.push(formats);
    }
}

/// Chooses one sample format per edge, in the graph's topological order.
///
/// Every stage states what it could emit and what it would take, and an edge carries the narrowest
/// format both ends allow. No conversion is inserted, so an edge whose ends share nothing is an
/// error naming both sides.
pub fn solve(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    capabilities: &Capabilities,
) -> Result<SolvedFormats, Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);

    let slots = graph
        .nodes()
        .iter()
        .map(|node| signature_slots(registry, node))
        .collect();

    let mut solver = Solver {
        graph,
        capabilities,
        prepared,
        slots,
        resolved: (0..graph.len()).map(|_| None).collect(),
        produced: (0..graph.len()).map(|_| None).collect(),
        reported: HashSet::new(),
        diagnostics,
    };

    for id in graph.order() {
        solver.decide(*id);
    }

    solver.validate();

    solver.finish()
}

/// The state one solve carries from node to node.
struct Solver<'a> {
    graph: &'a CompiledGraph,
    capabilities: &'a Capabilities,
    prepared: Vec<Option<PreparedStage>>,
    slots: Vec<SignatureSlots>,
    resolved: Vec<Option<ResolvedFormats>>,
    produced: Vec<Option<OutputFormats>>,
    reported: HashSet<NodeId>,
    diagnostics: Diagnostics,
}

impl Solver<'_> {
    /// Settles one node's inputs and the format it emits.
    ///
    /// A node whose stage never prepared, and one reading a node that never settled, is left
    /// undecided rather than guessed at.
    fn decide(&mut self, id: NodeId) {
        if self.prepared[id.index()].is_none() {
            return;
        }

        let Some(inputs) = self.inputs_of(id) else {
            return;
        };

        if self.graph.node(id).kind() == StageKind::Sink {
            self.resolved[id.index()] = Some(ResolvedFormats::new(inputs, None));
            return;
        }

        let Some(output) = self.output_of(id, &inputs) else {
            return;
        };

        if !self.emits_its_own_kind(id, &output) {
            return;
        }

        let Some(sample) = self.choose(id, &output) else {
            return;
        };

        let emitted = output.template.with_sample(sample);
        self.resolved[id.index()] = Some(ResolvedFormats::new(inputs, Some(emitted)));
    }

    /// Rejects a stage whose template is not the kind of stream its kind emits.
    ///
    /// A mask carries a mask and everything else carries frames, and the two are told apart
    /// downstream by what the graph says a stage emits rather than by looking inside a buffer.
    fn emits_its_own_kind(&mut self, id: NodeId, output: &OutputFormats) -> bool {
        let node = self.graph.node(id);
        let produced = match output.template {
            StreamTemplate::Frames(_) => StreamType::Frames,
            StreamTemplate::Mask(_) => StreamType::Mask,
        };

        let declared = node
            .kind()
            .output()
            .expect("a sink is never asked what it produces");
        if produced == declared {
            return true;
        }

        let message = format!(
            "a {} emits {declared} but this stage produced {produced}",
            node.kind(),
        );

        self.diagnostics.push(Diagnostic::error(
            DiagnosticCode::StreamTypeMismatch,
            stage_path(node.kind(), node.name()),
            node.span(),
            message,
        ));

        false
    }

    /// The narrowest sample this node can emit that every reader of it will take.
    ///
    /// Each sample the stage offered is put to every reader on its own, so the answer does not
    /// depend on which one happened to be asked about first. A reader that turned a sample down
    /// while one of its other inputs was still unsettled may have been steering on a proposal that
    /// moves later, so that sample is only set aside rather than ruled out. It is taken up again
    /// when nothing else survives, and the validation pass holds the reader to it once everything
    /// has settled.
    fn choose(&mut self, id: NodeId, output: &OutputFormats) -> Option<SampleFormat> {
        if output.samples.is_empty() {
            let error = StageError::msg("the stage offered no sample format to emit");
            self.report_stage(id, &error);

            return None;
        }

        let readers = self.readers_of(id);
        let candidates = output.samples.preferences().to_vec();

        let mut agreed = FormatSet::new();
        let mut tolerated = FormatSet::new();
        let mut refused: Vec<Option<Refusal>> = vec![None; readers.len()];

        for candidate in candidates {
            let offered = output.template.with_sample(candidate);
            let turndowns = self.probe(id, &readers, candidate, offered)?;

            if turndowns.is_empty() {
                agreed.push(candidate);
                tolerated.push(candidate);
                continue;
            }

            let mut ruled_out = false;
            for turndown in turndowns {
                if !turndown.firm {
                    continue;
                }

                ruled_out = true;
                keep_refusal(&mut refused, &readers, turndown);
            }

            if !ruled_out {
                tolerated.push(candidate);
            }
        }

        if !agreed.is_empty() {
            return agreed.narrowest();
        }

        if !tolerated.is_empty() {
            return tolerated.narrowest();
        }

        let mut refusals = refused.into_iter().flatten();
        let first = refusals
            .next()
            .expect("a sample either survives every reader or one of them ruled it out");
        let rest: Vec<Refusal> = refusals.collect();

        self.report_no_common_format(&output.samples, None, &first, &rest);

        None
    }

    /// Puts one sample to every reader, collecting each one that will not take it.
    fn probe(
        &mut self,
        id: NodeId,
        readers: &[Reader],
        candidate: SampleFormat,
        offered: StreamFormat,
    ) -> Option<Vec<Turndown>> {
        let mut turndowns = Vec::new();

        for (index, reader) in readers.iter().enumerate() {
            let asking = Proposals::Forward(id);
            let answer = self.ask(reader.consumer, reader.slot, offered, asking)?;

            if answer.accepted.accepts(candidate) {
                continue;
            }

            turndowns.push(Turndown {
                reader: index,
                accepted: answer.accepted,
                firm: answer.firm,
            });
        }

        Some(turndowns)
    }

    /// Re-asks every consumer, now that none of its inputs is unknown.
    ///
    /// A stage whose acceptance depends on more than one input answers on incomplete information
    /// during the forward pass. This is where that answer is held to what the stage actually got.
    fn validate(&mut self) {
        let graph = self.graph;

        for id in graph.order() {
            if self.resolved[id.index()].is_none() {
                continue;
            }

            for (edge, source) in edges_of(graph, *id) {
                let Some(given) = self.emitted_by(source) else {
                    continue;
                };

                let slot = self.slot_of(*id, edge);
                let Some(answer) = self.ask(*id, slot, given, Proposals::Settled) else {
                    continue;
                };

                if answer.accepted.accepts(given.sample()) {
                    continue;
                }

                let emitted = self.emitted_samples(source, given.sample());
                let refusal = Refusal {
                    consumer: *id,
                    edge,
                    accepted: answer.accepted,
                };

                self.report_no_common_format(&emitted, Some(given.sample()), &refusal, &[]);
            }
        }
    }

    fn finish(self) -> Result<SolvedFormats, Diagnostics> {
        if self.diagnostics.has_errors() {
            return Err(self.diagnostics);
        }

        let per_node = self
            .resolved
            .into_iter()
            .map(|formats| formats.expect("every node settled without a diagnostic"))
            .collect();

        Ok(SolvedFormats::new(per_node))
    }

    /// Asks one consumer what it would take in `slot`, given what is known about its other inputs.
    ///
    /// The answer is firm when every other slot is settled. A slot nobody bound never becomes
    /// known, so it does not hold the answer back.
    fn ask(
        &mut self,
        consumer: NodeId,
        slot: usize,
        offered: StreamFormat,
        proposals: Proposals,
    ) -> Option<Answer> {
        let inputs = match proposals {
            Proposals::Forward(deciding) => self.offers_to(consumer, slot, offered, deciding),
            Proposals::Settled => self.settled_offers_to(consumer),
        };

        let slots = &self.slots[consumer.index()];
        let firm = inputs
            .iter()
            .enumerate()
            .all(|(index, offer)| index == slot || offer.is_resolved() || slots[index].is_none());

        let answer = {
            let stage = self.prepared[consumer.index()].as_ref()?;
            stage.accepts_input(self.capabilities, slot, offered, &inputs)
        };

        match answer {
            Ok(accepted) => Some(Answer { accepted, firm }),
            Err(error) => {
                self.report_stage(consumer, &error);
                None
            },
        }
    }

    /// What the solver can tell `consumer` about each of its inputs while `slot` is being decided.
    ///
    /// Every slot fed by the node being decided carries the sample under test, so a producer
    /// filling two slots of one consumer proposes the same thing to both.
    fn offers_to(
        &mut self,
        consumer: NodeId,
        slot: usize,
        offered: StreamFormat,
        deciding: NodeId,
    ) -> Vec<Offered> {
        let sources = self.sources_of(consumer);
        let mut inputs = Vec::with_capacity(sources.len());

        for (index, source) in sources.into_iter().enumerate() {
            if index == slot {
                inputs.push(Offered::Proposed(offered));
                continue;
            }

            let Some(source) = source else {
                inputs.push(Offered::Unknown);
                continue;
            };

            if source == deciding {
                inputs.push(Offered::Proposed(offered));
                continue;
            }

            if let Some(settled) = self.emitted_by(source) {
                inputs.push(Offered::Resolved(settled));
                continue;
            }

            match self.proposed_by(source) {
                Some(format) => inputs.push(Offered::Proposed(format)),
                None => inputs.push(Offered::Unknown),
            }
        }

        inputs
    }

    /// Every input of `consumer` as it finally settled.
    fn settled_offers_to(&mut self, consumer: NodeId) -> Vec<Offered> {
        let sources = self.sources_of(consumer);
        let mut inputs = Vec::with_capacity(sources.len());

        for source in sources {
            let settled = source.and_then(|source| self.emitted_by(source));

            match settled {
                Some(format) => inputs.push(Offered::Resolved(format)),
                None => inputs.push(Offered::Unknown),
            }
        }

        inputs
    }

    /// The format a node would prefer to emit, once everything it reads has settled.
    ///
    /// A node still waiting on one of its own inputs has nothing to propose yet, and a sink emits
    /// nothing at all.
    fn proposed_by(&mut self, id: NodeId) -> Option<StreamFormat> {
        if self.graph.node(id).kind() == StageKind::Sink {
            return None;
        }

        let inputs = self.inputs_of(id)?;
        let output = self.output_of(id, &inputs)?;
        let preferred = output.samples.preferences().first().copied()?;

        Some(output.template.with_sample(preferred))
    }

    /// What a node would emit, computed once and kept for whoever asks again.
    ///
    /// Every input is settled by the time this runs, so the answer cannot change between one
    /// reader asking and the next.
    fn output_of(&mut self, id: NodeId, inputs: &[Option<StreamFormat>]) -> Option<OutputFormats> {
        if let Some(cached) = &self.produced[id.index()] {
            return Some(cached.clone());
        }

        let answer = {
            let stage = self.prepared[id.index()].as_ref()?;
            stage.produces(self.capabilities, inputs)
        };

        match answer {
            Ok(output) => {
                self.produced[id.index()] = Some(output.clone());
                Some(output)
            },
            Err(error) => {
                self.report_stage(id, &error);
                None
            },
        }
    }

    /// The formats this node's producers settled on, one entry per signature slot.
    ///
    /// A slot nobody bound is a hole, so everything after it keeps its own index. A producer that
    /// never settled gives `None` for the node as a whole, because it cannot be decided at all.
    fn inputs_of(&self, id: NodeId) -> Option<Vec<Option<StreamFormat>>> {
        let sources = self.sources_of(id);
        let mut inputs = Vec::with_capacity(sources.len());

        for source in sources {
            let Some(source) = source else {
                inputs.push(None);
                continue;
            };

            let format = self.emitted_by(source)?;
            inputs.push(Some(format));
        }

        Some(inputs)
    }

    fn emitted_by(&self, id: NodeId) -> Option<StreamFormat> {
        let formats = self.resolved[id.index()].as_ref()?;

        formats.output()
    }

    /// Everything a node could have emitted, which is the set its format was settled from.
    fn emitted_samples(&self, id: NodeId, settled: SampleFormat) -> FormatSet {
        match &self.produced[id.index()] {
            Some(output) => output.samples.clone(),
            None => FormatSet::of([settled]),
        }
    }

    /// The node feeding each of this node's signature slots, empty where nothing was bound.
    fn sources_of(&self, id: NodeId) -> Vec<Option<NodeId>> {
        let node = self.graph.node(id);

        self.slots[id.index()]
            .iter()
            .map(|filled| filled.map(|edge| node.inputs()[edge].source()))
            .collect()
    }

    /// Where the edge at `edge` sits in its node's signature.
    fn slot_of(&self, id: NodeId, edge: usize) -> usize {
        let found = self.slots[id.index()]
            .iter()
            .position(|filled| *filled == Some(edge));

        found.unwrap_or(edge)
    }

    /// Every input that reads this node, named by the consumer, the edge and its signature slot.
    ///
    /// A consumer reading the same node twice is listed once per input.
    fn readers_of(&self, id: NodeId) -> Vec<Reader> {
        let mut seen: Vec<NodeId> = Vec::new();
        let mut readers = Vec::new();

        for consumer in self.graph.node(id).consumers() {
            if seen.contains(consumer) {
                continue;
            }

            seen.push(*consumer);

            for (edge, source) in edges_of(self.graph, *consumer) {
                if source != id {
                    continue;
                }

                readers.push(Reader {
                    consumer: *consumer,
                    edge,
                    slot: self.slot_of(*consumer, edge),
                });
            }
        }

        readers
    }

    /// Puts a failure raised by a stage itself on that stage.
    ///
    /// A stage is asked more than once in a solve, so only its first failure is kept.
    fn report_stage(&mut self, id: NodeId, error: &StageError) {
        if !self.reported.insert(id) {
            return;
        }

        let node = self.graph.node(id);
        self.diagnostics.push(Diagnostic::error(
            DiagnosticCode::StageBuildFailed,
            stage_path(node.kind(), node.name()),
            node.span(),
            error.to_string(),
        ));
    }

    /// Names an edge that has no format to carry, together with every reader that refused.
    ///
    /// One reader is a disagreement between two stages and reads as one. Several readers have no
    /// one format between them, so each is named with what it does take and the ones after the
    /// first become notes.
    fn report_no_common_format(
        &mut self,
        emitted: &FormatSet,
        settled: Option<SampleFormat>,
        first: &Refusal,
        rest: &[Refusal],
    ) {
        let consuming = self.graph.node(first.consumer);
        let reading = &consuming.inputs()[first.edge];
        let producing = self.graph.node(reading.source());

        let message = match (settled, rest.is_empty()) {
            (Some(sample), _) => format!(
                "{} emits {} and was settled on {sample}, but {} takes {}. \
                 Put a convert filter between them to pick one.",
                producing.name(),
                describe(emitted),
                consuming.name(),
                describe(&first.accepted),
            ),
            (None, true) => format!(
                "{} emits {} and {} takes {}, so this edge has no format to carry. \
                 Put a convert filter between them to pick one.",
                producing.name(),
                describe(emitted),
                consuming.name(),
                describe(&first.accepted),
            ),
            (None, false) => format!(
                "{} emits {} and no single one of those suits every stage reading it. {} takes {}. \
                 Put a convert filter before whichever reader needs a different one.",
                producing.name(),
                describe(emitted),
                consuming.name(),
                describe(&first.accepted),
            ),
        };

        let mut diagnostic = Diagnostic::error(
            DiagnosticCode::NoCommonFormat,
            input_path(consuming.kind(), consuming.name(), first.edge),
            reading.span(),
            message,
        );

        for refusal in rest {
            let reader = self.graph.node(refusal.consumer);
            let edge = &reader.inputs()[refusal.edge];
            let takes = format!("{} takes {}", reader.name(), describe(&refusal.accepted));

            diagnostic = diagnostic.with_note(takes, edge.span());
        }

        let defined = format!("{} is defined here", producing.name());
        self.diagnostics
            .push(diagnostic.with_note(defined, producing.span()));
    }
}

/// What a reader said about one sample, and how much it had to go on.
struct Answer {
    accepted: FormatSet,
    /// Whether every other slot was already settled, so the reader will not change its mind.
    firm: bool,
}

/// A reader turning down a sample it was offered.
///
/// A stage steers by leaving samples out of the set it returns, because a set cannot say "I prefer
/// this because of a sibling" apart from "I require this". So a turndown made while one of the
/// reader's other inputs was still a proposal only sets a sample aside, and one made with every
/// other input settled rules it out.
struct Turndown {
    reader: usize,
    accepted: FormatSet,
    firm: bool,
}

/// Records what a reader ruled out, keeping the first where it ruled out more than one sample.
fn keep_refusal(refused: &mut [Option<Refusal>], readers: &[Reader], turndown: Turndown) {
    if refused[turndown.reader].is_some() {
        return;
    }

    refused[turndown.reader] = Some(Refusal {
        consumer: readers[turndown.reader].consumer,
        edge: readers[turndown.reader].edge,
        accepted: turndown.accepted,
    });
}

/// One input of one consumer that reads a node whose format is being decided.
#[derive(Debug, Copy, Clone)]
struct Reader {
    consumer: NodeId,
    /// Where the input sits in the definition, which is what a diagnostic points at.
    edge: usize,
    /// Where the input sits in the stage's signature, which is what the stage is told.
    slot: usize,
}

/// A consumer turning down a sample, kept so the failure can name who refused what.
#[derive(Debug, Clone)]
struct Refusal {
    consumer: NodeId,
    edge: usize,
    accepted: FormatSet,
}

/// Which of a node's inputs fills each slot of its stage's signature.
///
/// An optional parameter nobody bound leaves its slot empty, and the variadic tail follows the
/// named parameters in the order the definition wrote it.
type SignatureSlots = Vec<Option<usize>>;

/// Lines a node's inputs up with the slots of its stage's signature.
///
/// A stage the registry does not know has no signature to line up against, so its inputs keep the
/// order they were written in.
fn signature_slots(registry: &PipelineRegistry, node: &Node) -> SignatureSlots {
    let Some(signature) = signature_of(registry, node) else {
        return (0..node.inputs().len()).map(Some).collect();
    };

    let params = signature.params();
    let mut slots: SignatureSlots = vec![None; params.len()];

    for (edge, input) in node.inputs().iter().enumerate() {
        let Some(name) = input.param() else {
            continue;
        };

        let Some(slot) = params.iter().position(|param| param.name() == name) else {
            continue;
        };

        slots[slot] = Some(edge);
    }

    for (edge, input) in node.inputs().iter().enumerate() {
        if input.param().is_none() {
            slots.push(Some(edge));
        }
    }

    slots
}

fn signature_of(registry: &PipelineRegistry, node: &Node) -> Option<&'static InputSignature> {
    let stage = registry
        .stages()
        .find(|info| info.kind() == node.kind() && info.full_identifier() == node.identifier());

    stage.map(|info| info.inputs())
}

/// Every input of a node, named by its position in the definition and the node it reads.
fn edges_of(graph: &CompiledGraph, id: NodeId) -> Vec<(usize, NodeId)> {
    graph
        .node(id)
        .inputs()
        .iter()
        .enumerate()
        .map(|(edge, input)| (edge, input.source()))
        .collect()
}

/// How much the solver knows about a consumer's other inputs when it asks about one of them.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Proposals {
    /// Whatever is known while the named node is being decided, which is partly only proposed.
    Forward(NodeId),
    /// Every input as it finally settled.
    Settled,
}

/// Spells out a set for a reader, so a diagnostic can name what each end of an edge allows.
fn describe(set: &FormatSet) -> String {
    if set.is_empty() {
        return String::from("nothing");
    }

    let names: Vec<String> = set
        .preferences()
        .iter()
        .map(|sample| sample.to_string())
        .collect();

    names.join(", ")
}

#[cfg(test)]
mod tests {
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

    use super::*;
    use crate::{Diagnostic, StageSpec, build, testing};

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
        plain.args = crate::Spanned::unknown(serde_json::json!({ "sigma": 0.8 }));

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
        specs[1].args = crate::Spanned::unknown(serde_json::json!({ "sigma": "wide" }));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();

        let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("should fail");
        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

        assert_eq!(reported[0].code(), DiagnosticCode::InvalidArgs);
    }
}
