use std::collections::{BTreeMap, HashSet};
use std::panic;
use std::thread::{self, ScopedJoinHandle};

use av_toolbox_core::{
    AnyProbe,
    PipelineRegistry,
    PreparedStage,
    ProbeKind,
    Probed,
    ProbedValue,
    RunProbeFn,
    Seeker,
    StageError,
};

use super::reach::{Reached, reached};
use super::{NodeProbe, unknown_probe};
use crate::demands::bound_edge;
use crate::graph::{CompiledGraph, Node, NodeId};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// One probe a stage made, with the sources it reaches.
struct Asked {
    node: NodeId,
    /// Where the probe sits in its node's probes.
    index: usize,
    probe: NodeProbe,
    reached: Reached,
}

/// The probes one source answers, grouped by kind.
struct Job {
    source: NodeId,
    seeker: Box<dyn Seeker>,
    batches: Vec<Batch>,
}

struct Batch {
    kind: ProbeKind,
    run: RunProbeFn,
    probes: Vec<AnyProbe>,
}

/// What opening a source's seeker gave.
enum Opening {
    Ready(Job),
    Unseekable,
    Failed,
}

/// What one source answered for one probe.
struct Answer {
    source: NodeId,
    probe: AnyProbe,
    value: ProbedValue,
}

type JobResult = (NodeId, Result<Vec<Answer>, StageError>);

fn has_probes(graph: &CompiledGraph) -> bool {
    graph.nodes().iter().any(|node| !node.probes().is_empty())
}

/// Runs every recorded probe against the sources it reaches, and stores what each stage's probes found.
pub fn run_probes(graph: &mut CompiledGraph, registry: &PipelineRegistry) -> Result<(), Diagnostics> {
    if !has_probes(graph) {
        return Ok(());
    }

    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let mut asked = asked_probes(graph);
    let jobs = jobs_for(graph, registry, &prepared, &asked, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let answers = run_jobs(graph, jobs, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let settled = settled_probes(graph, registry, &mut asked, &answers, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    for (node, probed) in settled {
        graph.node_mut(node).set_probed(probed);
    }

    for one in asked {
        let Some(found) = one.probe.found else {
            continue;
        };

        graph.node_mut(one.node).set_found(one.index, found);
    }

    Ok(())
}

fn asked_probes(graph: &CompiledGraph) -> Vec<Asked> {
    let mut asked = Vec::new();

    for id in graph.order() {
        let node = graph.node(*id);
        for (index, probe) in node.probes().iter().enumerate() {
            let Some(edge) = bound_edge(node, probe.input) else {
                continue;
            };

            let start = node.inputs()[edge].source();
            let reached = reached(graph, start);
            asked.push(Asked {
                node: *id,
                index,
                probe: probe.clone(),
                reached,
            });
        }
    }

    asked
}

/// Opens a seeker on every source a probe reaches, and gives each the probes it answers.
fn jobs_for(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    prepared: &[Option<PreparedStage>],
    asked: &[Asked],
    diagnostics: &mut Diagnostics,
) -> Vec<Job> {
    let mut openings = BTreeMap::new();
    let mut refused = HashSet::new();

    for one in asked {
        for source in &one.reached.sources {
            let opening = openings
                .entry(*source)
                .or_insert_with(|| open_source(graph, prepared, *source, diagnostics));

            match opening {
                Opening::Ready(job) => add_probe(graph, registry, job, one, diagnostics),
                Opening::Unseekable => {
                    if refused.insert((one.node, *source)) {
                        let probing = graph.node(one.node);
                        let source = graph.node(*source);
                        let kind = one.probe.probe.kind();
                        let diagnostic = unseekable(probing, source, kind);
                        diagnostics.push(diagnostic);
                    }
                },
                Opening::Failed => {},
            }
        }
    }

    openings.into_values().filter_map(ready_job).collect()
}

fn open_source(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    source: NodeId,
    diagnostics: &mut Diagnostics,
) -> Opening {
    let stage = prepared[source.index()]
        .as_ref()
        .expect("every stage is prepared once preparing reports no errors");

    match stage.seeker() {
        Ok(Some(seeker)) => Opening::Ready(Job {
            source,
            seeker,
            batches: Vec::new(),
        }),
        Ok(None) => Opening::Unseekable,
        Err(error) => {
            let node = graph.node(source);
            let diagnostic = source_failed(node, &error);
            diagnostics.push(diagnostic);
            Opening::Failed
        },
    }
}

/// Adds the probe `asked` makes to the batch of its kind, keeping each distinct probe once.
fn add_probe(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    job: &mut Job,
    asked: &Asked,
    diagnostics: &mut Diagnostics,
) {
    let probe = &asked.probe.probe;
    let kind = probe.kind();

    let position = job.batches.iter().position(|batch| batch.kind == kind);
    let batch = match position {
        Some(position) => &mut job.batches[position],
        None => {
            let Some(run) = registry.probe_runner(kind) else {
                let probing = graph.node(asked.node);
                let diagnostic = unknown_probe(probing, probe);
                diagnostics.push(diagnostic);
                return;
            };

            let batch = Batch {
                kind,
                run,
                probes: Vec::new(),
            };
            job.batches.push(batch);
            job.batches.last_mut().expect("a batch was just pushed")
        },
    };

    if !batch.probes.contains(probe) {
        batch.probes.push(probe.clone());
    }
}

fn ready_job(opening: Opening) -> Option<Job> {
    match opening {
        Opening::Ready(job) => Some(job),
        Opening::Unseekable | Opening::Failed => None,
    }
}

/// Runs each source's batches on its own thread, and gathers the answers in source order.
fn run_jobs(graph: &CompiledGraph, jobs: Vec<Job>, diagnostics: &mut Diagnostics) -> Vec<Answer> {
    let results: Vec<JobResult> = thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .into_iter()
            .map(|job| scope.spawn(move || run_job(job)))
            .collect();

        handles.into_iter().map(joined).collect()
    });

    let mut answers = Vec::new();
    for (source, result) in results {
        match result {
            Ok(found) => answers.extend(found),
            Err(error) => {
                let node = graph.node(source);
                let diagnostic = source_failed(node, &error);
                diagnostics.push(diagnostic);
            },
        }
    }

    answers
}

fn joined(handle: ScopedJoinHandle<'_, JobResult>) -> JobResult {
    match handle.join() {
        Ok(result) => result,
        Err(payload) => panic::resume_unwind(payload),
    }
}

fn run_job(job: Job) -> JobResult {
    let Job {
        source,
        mut seeker,
        batches,
    } = job;

    let answers = run_batches(source, seeker.as_mut(), &batches);
    (source, answers)
}

fn run_batches(
    source: NodeId,
    seeker: &mut dyn Seeker,
    batches: &[Batch],
) -> Result<Vec<Answer>, StageError> {
    let mut answers = Vec::new();

    for batch in batches {
        let values = (batch.run)(seeker, &batch.probes)?;
        for (probe, value) in batch.probes.iter().zip(values) {
            answers.push(Answer {
                source,
                probe: probe.clone(),
                value,
            });
        }
    }

    Ok(answers)
}

/// Hands each probing stage one value per probe, combining the answers of several sources.
///
/// Each settled probe in `asked` notes what it found, as it is shown to the user.
fn settled_probes(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    asked: &mut [Asked],
    answers: &[Answer],
    diagnostics: &mut Diagnostics,
) -> BTreeMap<NodeId, Probed> {
    let mut settled: BTreeMap<NodeId, Probed> = BTreeMap::new();

    for one in asked.iter_mut() {
        let node = graph.node(one.node);
        let kind = one.probe.probe.kind();
        let values = values_for(one, answers);
        if values.is_empty() {
            continue;
        }

        let value = match settled_value(registry, kind, values) {
            Ok(value) => value,
            Err(error) => {
                let meeting = one.reached.meeting.map(|meeting| graph.node(meeting));
                let diagnostic = not_combined(node, meeting, &error);
                diagnostics.push(diagnostic);
                continue;
            },
        };

        let sources = source_names(graph, &one.reached);
        let shown = registry.show_probed(kind, &value);
        tracing::info!(
            "`{}` probed {} from {}: {}",
            node.name(),
            kind.name(),
            sources,
            shown
        );

        let probed = settled.entry(one.node).or_default();
        probed.insert(one.probe.input, kind, value);
        one.probe.found = Some(shown);
    }

    settled
}

/// What each source `asked` reaches answered for its probe, in source order.
fn values_for(asked: &Asked, answers: &[Answer]) -> Vec<ProbedValue> {
    answers
        .iter()
        .filter(|answer| asked.reached.sources.contains(&answer.source))
        .filter(|answer| answer.probe == asked.probe.probe)
        .map(|answer| answer.value.clone())
        .collect()
}

fn settled_value(
    registry: &PipelineRegistry,
    kind: ProbeKind,
    mut values: Vec<ProbedValue>,
) -> Result<ProbedValue, StageError> {
    if values.len() == 1 {
        let value = values.pop().expect("one value");
        return Ok(value);
    }

    let Some(combine) = registry.probe_combiner(kind) else {
        return Err(no_combiner(kind));
    };

    combine(values)
}

fn source_names(graph: &CompiledGraph, reached: &Reached) -> String {
    let names: Vec<String> = reached
        .sources
        .iter()
        .map(|source| format!("`{}`", graph.node(*source).name()))
        .collect();

    names.join(", ")
}

fn no_combiner(kind: ProbeKind) -> StageError {
    let message = format!("{} is no registered probe kind", kind.name());

    StageError::msg(message)
}

fn unseekable(probing: &Node, source: &Node, kind: ProbeKind) -> Diagnostic {
    let message = format!(
        "`{}` probes {} of `{}` ({}), which cannot seek. Read it with a file source such as ffms2/file.",
        probing.name(),
        kind.name(),
        source.name(),
        source.identifier()
    );

    Diagnostic::error(
        DiagnosticCode::UnseekableSource,
        probing.path(),
        probing.span(),
        message,
    )
}

fn source_failed(source: &Node, error: &StageError) -> Diagnostic {
    let message = format!("`{}` could not be read ahead of the run: {error}", source.name());

    Diagnostic::error(DiagnosticCode::ProbeFailed, source.path(), source.span(), message)
}

fn not_combined(probing: &Node, meeting: Option<&Node>, error: &StageError) -> Diagnostic {
    let place = match meeting {
        Some(meeting) => format!("its sources meet at `{}`", meeting.name()),
        None => String::from("it reaches more than one source"),
    };
    let message = format!("`{}` cannot probe here, as {place}: {error}", probing.name());

    Diagnostic::error(
        DiagnosticCode::ProbeFailed,
        probing.path(),
        probing.span(),
        message,
    )
}
