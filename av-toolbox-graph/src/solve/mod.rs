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
