mod rate;
#[cfg(test)]
mod tests;

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

use crate::demands::note_demanders;
use crate::diagnostic::{input_path, stage_path};
use crate::fill::note_templates;
use crate::graph::{CompiledGraph, Node, NodeId};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// One set of formats per node, produced by the solver and consumed by instantiation.
///
/// It is built only by solving, so it always covers every node of the graph it came from.
#[derive(Debug, Clone)]
pub struct SolvedFormats {
    per_node: Vec<ResolvedFormats>,
    widens: Vec<Widen>,
}

impl SolvedFormats {
    pub(crate) fn new(per_node: Vec<ResolvedFormats>, widens: Vec<Widen>) -> Self {
        Self { per_node, widens }
    }

    /// Every edge the solver widened, where the consumer's input holds the wider format.
    pub fn widens(&self) -> &[Widen] {
        &self.widens
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

    pub(crate) fn set_output(&mut self, node: NodeId, format: StreamFormat) {
        self.per_node[node.index()].set_output(format);
    }

    pub(crate) fn set_input(&mut self, node: NodeId, slot: usize, format: StreamFormat) {
        self.per_node[node.index()].set_input(slot, format);
    }
}

/// One edge the solver widened so its reader could take it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widen {
    pub consumer: NodeId,
    /// The input's position in the consumer's definition.
    pub edge: usize,
    pub from: StreamFormat,
    pub to: StreamFormat,
}

/// Chooses one sample format per edge, in the graph's topological order.
///
/// Every stage states what it could emit and what it would take, and an edge carries the producer's
/// most preferred format that both ends allow. An edge whose reader only takes a wider sample is
/// widened to the narrowest one holding every value exactly. An edge that would have to narrow is
/// an error naming both sides.
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
        widens: Vec::new(),
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
    widens: Vec<Widen>,
    diagnostics: Diagnostics,
}

impl Solver<'_> {
    /// Settles one node's inputs and the format it emits.
    ///
    /// A node whose stage never prepared, and one reading a node that never settled, is left
    /// undecided rather than guessed at. Inputs the node refuses are widened before its output is
    /// worked out, so the output follows from what it actually receives.
    fn decide(&mut self, id: NodeId) {
        if self.prepared[id.index()].is_none() {
            return;
        }

        if self.inputs_of(id).is_none() {
            return;
        }

        while self.widen_refused_inputs(id) {}

        let Some(inputs) = self.inputs_of(id) else {
            return;
        };

        if !self.rates_agree(id, &inputs) {
            return;
        }

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

    /// The producer's most preferred sample that every reader of this node will take.
    ///
    /// Each sample the stage offered is put to every reader on its own, so the answer does not
    /// depend on which one happened to be asked about first. A reader that turned a sample down
    /// while one of its other inputs was still unsettled may have been steering on a proposal that
    /// moves later, so that sample is only set aside rather than ruled out. It is taken up again
    /// when nothing else survives, and the validation pass holds the reader to it once everything
    /// has settled. When every sample was ruled out, the producer's first one is widened for each
    /// reader that refused it.
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

        for candidate in candidates {
            let offered = output.template.with_sample(candidate);
            let turndowns = self.probe(id, &readers, candidate, offered)?;

            if turndowns.is_empty() {
                agreed.push(candidate);
                tolerated.push(candidate);
                continue;
            }

            let ruled_out = turndowns.iter().any(|turndown| turndown.firm);
            if !ruled_out {
                tolerated.push(candidate);
            }
        }

        if !agreed.is_empty() {
            return agreed.first();
        }

        if !tolerated.is_empty() {
            return tolerated.first();
        }

        let first = output.samples.first()?;
        self.widen_for_readers(id, output, &readers, first)
    }

    /// Settles on `sample`, widening the edge into each reader that firmly refused it.
    ///
    /// A reader with no lossless widening it takes may have refused only because a sibling input
    /// settled narrower. That reader is left to widen the sibling when it is decided itself.
    /// Any other such reader makes the edge an error naming each one.
    fn widen_for_readers(
        &mut self,
        id: NodeId,
        output: &OutputFormats,
        readers: &[Reader],
        sample: SampleFormat,
    ) -> Option<SampleFormat> {
        let offered = output.template.with_sample(sample);
        let turndowns = self.probe(id, readers, sample, offered)?;

        let mut widens = Vec::new();
        let mut refusals = Vec::new();

        for turndown in turndowns {
            if !turndown.firm {
                continue;
            }

            let reader = readers[turndown.reader];
            let proposals = Proposals::Forward(id);
            let widened = self.widening(reader, offered, &turndown.accepted, proposals);

            if let Some(to) = widened {
                widens.push(Widen {
                    consumer: reader.consumer,
                    edge: reader.edge,
                    from: offered,
                    to,
                });
                continue;
            }

            if self.a_sibling_could_widen_to(reader, sample) {
                continue;
            }

            refusals.push(Refusal {
                consumer: reader.consumer,
                edge: reader.edge,
                accepted: turndown.accepted,
            });
        }

        let mut refusals = refusals.into_iter();
        let Some(first) = refusals.next() else {
            self.widens.extend(widens);
            return Some(sample);
        };

        let rest: Vec<Refusal> = refusals.collect();
        self.report_no_common_format(&output.samples, None, &first, &rest);

        None
    }

    /// The narrowest lossless widening of `offered` that `reader` takes, confirmed by asking again.
    ///
    /// Only frames widen, since a widening is carried out by a `resample/depth`.
    fn widening(
        &mut self,
        reader: Reader,
        offered: StreamFormat,
        accepted: &FormatSet,
        proposals: Proposals,
    ) -> Option<StreamFormat> {
        offered.as_frames()?;

        let widenings = offered.sample().lossless_widenings();
        let sample = widenings.into_iter().find(|wider| accepted.accepts(*wider))?;

        let template = StreamTemplate::from_format(&offered);
        let widened = template.with_sample(sample);
        let answer = self.ask(reader.consumer, reader.slot, widened, proposals)?;

        if !answer.accepted.accepts(sample) {
            return None;
        }

        Some(widened)
    }

    /// Whether another settled input of the reader could widen losslessly to `sample`.
    fn a_sibling_could_widen_to(&self, reader: Reader, sample: SampleFormat) -> bool {
        let slots = self.slots[reader.consumer.index()].len();

        (0..slots)
            .filter(|slot| *slot != reader.slot)
            .filter_map(|slot| self.received(reader.consumer, slot))
            .any(|format| format.sample().lossless_widenings().contains(&sample))
    }

    /// Records that the edge into `reader` now carries `to`, keeping the producer's format as `from`.
    ///
    /// The reader is still being decided, so its own formats are worked out from `to` afterwards.
    /// Any output kept for it from a proposal was worked out from the narrower input, so it is
    /// dropped.
    fn record_widen(&mut self, reader: Reader, given: StreamFormat, to: StreamFormat) {
        self.produced[reader.consumer.index()] = None;

        let existing = self
            .widens
            .iter_mut()
            .find(|widen| widen.consumer == reader.consumer && widen.edge == reader.edge);

        if let Some(widen) = existing {
            widen.to = to;
            return;
        }

        self.widens.push(Widen {
            consumer: reader.consumer,
            edge: reader.edge,
            from: given,
            to,
        });
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

    /// Widens each input of `id` that it refuses but takes a lossless widening of.
    ///
    /// Every producer has settled by now, so the node answers on what it actually gets. Returns
    /// whether any input was widened, since a widen can change what a sibling input is allowed.
    /// Each widen moves an edge strictly wider, so repeating this always comes to an end.
    fn widen_refused_inputs(&mut self, id: NodeId) -> bool {
        let mut widened = false;

        for (edge, _source) in edges_of(self.graph, id) {
            let slot = self.slot_of(id, edge);
            let Some(given) = self.received(id, slot) else {
                continue;
            };

            let Some(answer) = self.ask(id, slot, given, Proposals::Settled) else {
                continue;
            };

            if answer.accepted.accepts(given.sample()) {
                continue;
            }

            let reader = Reader {
                consumer: id,
                edge,
                slot,
            };
            let wider = self.widening(reader, given, &answer.accepted, Proposals::Settled);

            if let Some(to) = wider {
                self.record_widen(reader, given, to);
                widened = true;
            }
        }

        widened
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
                let slot = self.slot_of(*id, edge);
                let Some(given) = self.received(*id, slot) else {
                    continue;
                };

                let Some(answer) = self.ask(*id, slot, given, Proposals::Settled) else {
                    continue;
                };

                if answer.accepted.accepts(given.sample()) {
                    continue;
                }

                let Some(settled) = self.emitted_by(source) else {
                    continue;
                };

                let emitted = self.emitted_samples(source, settled.sample());
                let refusal = Refusal {
                    consumer: *id,
                    edge,
                    accepted: answer.accepted,
                };

                self.report_no_common_format(&emitted, Some(settled.sample()), &refusal, &[]);
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

        Ok(SolvedFormats::new(per_node, self.widens))
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
            Proposals::Settled => self.settled_offers_to(consumer, slot, offered),
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

            if let Some(settled) = self.received(consumer, index) {
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

    /// Every input of `consumer` as it finally settled, with `offered` in `slot`.
    fn settled_offers_to(&self, consumer: NodeId, slot: usize, offered: StreamFormat) -> Vec<Offered> {
        let slots = self.slots[consumer.index()].len();
        let mut inputs = Vec::with_capacity(slots);

        for index in 0..slots {
            if index == slot {
                inputs.push(Offered::Resolved(offered));
                continue;
            }

            match self.received(consumer, index) {
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
    /// Every input is settled by the time this runs. Widening one of the node's inputs drops what
    /// was kept, so the answer is worked out again from what the node receives.
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
    /// A producer that never settled gives `None` for the node as a whole, because it cannot be
    /// decided at all.
    fn inputs_of(&self, id: NodeId) -> Option<Vec<Option<StreamFormat>>> {
        let slots = &self.slots[id.index()];

        slot_inputs(slots, |edge| self.received_on(id, edge))
    }

    /// The format `consumer` gets in `slot`, which is the widened one where the edge was widened.
    ///
    /// It is `None` while the producer is unsettled and for a slot nobody bound.
    fn received(&self, consumer: NodeId, slot: usize) -> Option<StreamFormat> {
        let edge = self.slots[consumer.index()][slot]?;

        self.received_on(consumer, edge)
    }

    /// The format `consumer` gets on the input at `edge`, which is `None` while its producer is
    /// unsettled.
    fn received_on(&self, consumer: NodeId, edge: usize) -> Option<StreamFormat> {
        let widened = self
            .widens
            .iter()
            .find(|widen| widen.consumer == consumer && widen.edge == edge);
        if let Some(widen) = widened {
            return Some(widen.to);
        }

        let source = self.graph.node(consumer).inputs()[edge].source();
        self.emitted_by(source)
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
        let diagnostic = Diagnostic::error(
            DiagnosticCode::StageBuildFailed,
            stage_path(node.kind(), node.name()),
            node.span(),
            error.to_string(),
        );
        let demanders_noted = note_demanders(self.graph, node, diagnostic);
        let noted = note_templates(node, demanders_noted);

        self.diagnostics.push(noted);
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
                 Put a resample/depth filter between them to pick one.",
                producing.name(),
                describe(emitted),
                consuming.name(),
                describe(&first.accepted),
            ),
            (None, true) => format!(
                "{} emits {} and {} takes {}, so this edge has no format to carry. \
                 Put a resample/depth filter between them to pick one.",
                producing.name(),
                describe(emitted),
                consuming.name(),
                describe(&first.accepted),
            ),
            (None, false) => format!(
                "{} emits {} and no single one of those suits every stage reading it. {} takes {}. \
                 Put a resample/depth filter before whichever reader needs a different one.",
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
pub(crate) type SignatureSlots = Vec<Option<usize>>;

/// Lines a node's inputs up with the slots of its stage's signature.
///
/// A stage the registry does not know has no signature to line up against, so its inputs keep the
/// order they were written in.
pub(crate) fn signature_slots(registry: &PipelineRegistry, node: &Node) -> SignatureSlots {
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

/// Lays a node's input formats out by signature slot, asking `format_on` for each bound edge's.
///
/// A slot nobody bound is a hole, so everything after it keeps its own index. An edge with no
/// format gives `None` for the node as a whole.
pub(crate) fn slot_inputs(
    slots: &SignatureSlots,
    mut format_on: impl FnMut(usize) -> Option<StreamFormat>,
) -> Option<Vec<Option<StreamFormat>>> {
    let mut inputs = Vec::with_capacity(slots.len());

    for filled in slots {
        let Some(edge) = filled else {
            inputs.push(None);
            continue;
        };

        let format = format_on(*edge)?;
        inputs.push(Some(format));
    }

    Some(inputs)
}

pub(crate) fn signature_of(registry: &PipelineRegistry, node: &Node) -> Option<&'static InputSignature> {
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
