use std::collections::HashMap;
use std::io;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle};

use av_toolbox_core::{
    CHANNEL_DEPTH,
    ChannelHub,
    ChannelStop,
    ChannelToken,
    InputFrame,
    OutputFrame,
    PipelineRegistry,
    StageKind,
};
use av_toolbox_gpu::{Backend, FrameFormat, Tuning};
use av_toolbox_graph::{Diagnostic, DiagnosticCode, NodeId, Span, StageSpec};
use av_toolbox_pipeline::{CompiledPipeline, StageRef};
use av_toolbox_runtime::{Cancel, Engine, Progress, RunError, RunReport, Settings};
use codespan_reporting::term::termcolor::WriteColor;
use serde_json::{Map, Value};

use crate::error::NamedStage;
use crate::input::PipelineInput;
use crate::plan::{self, PlannedNode};
use crate::report::{self, PipelineFiles};
use crate::source::PipelineSource;
use crate::{SessionError, steps};

/// The stage type of the caller's channel source and sink.
const CHANNEL_STAGE: &str = "channel/frames";

/// What a stream's thread hands back once its run ends.
type Finish = Result<RunReport, RunError>;

/// A loaded pipeline and an opened backend, ready to run as one stream after another.
///
/// Each stream rebuilds and compiles the pipeline against the same backend, so a new stream skips
/// reading the TOML and opening the device. Only one stream runs at a time.
pub struct Session {
    specs: Vec<StageSpec>,
    files: Arc<PipelineFiles>,
    registry: PipelineRegistry,
    backend: Backend,
    token: ChannelToken,
    sources: Vec<(String, FrameFormat)>,
    sinks: Vec<String>,
    warnings: Vec<Diagnostic>,
    discard_artefacts: bool,
    running: Weak<()>,
}

impl Session {
    /// Opens `source` on `backend`, checking that `channels` gives a format for every `channel/frames` source.
    pub fn new(
        source: PipelineSource,
        backend: Backend,
        channels: &[(String, FrameFormat)],
    ) -> Result<Session, SessionError> {
        let PipelineSource {
            mut specs,
            files,
            registry,
        } = source;

        let token = ChannelToken::new();
        let stages = bind_channel_args(&mut specs, &token);
        let matched = match_formats(&stages, channels);
        let sources = match matched {
            Ok(sources) => sources,
            Err(found) => return Err(SessionError::pipeline(files, found)),
        };

        Ok(Session {
            specs,
            files,
            registry,
            backend,
            token,
            sources,
            sinks: stages.sinks,
            warnings: Vec::new(),
            discard_artefacts: false,
            running: Weak::new(),
        })
    }

    /// Reads the pipeline with [PipelineSource::read](crate::PipelineSource::read) and opens it with
    /// [Session::new].
    pub fn load(
        input: PipelineInput,
        vars: &[(String, String)],
        backend: Backend,
        channels: &[(String, FrameFormat)],
    ) -> Result<Session, SessionError> {
        let source = PipelineSource::read(input, vars)?;
        Session::new(source, backend, channels)
    }

    /// The stages the next stream builds, for the CLI's profile passes.
    ///
    /// Channel stages must not be added, removed or renamed through it.
    #[doc(hidden)]
    pub fn specs_mut(&mut self) -> &mut Vec<StageSpec> {
        &mut self.specs
    }

    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// The warnings the last stream's pipeline raised while it compiled.
    pub fn warnings(&self) -> &[Diagnostic] {
        &self.warnings
    }

    /// Writes [Session::warnings] to `writer`, drawn against the pipeline's TOML.
    pub fn render_warnings(&self, writer: &mut dyn WriteColor) {
        report::diagnostics(writer, &self.files, &self.warnings);
    }

    /// Sets whether streams started from now on drop the artefacts their stages make instead of writing them.
    pub fn set_discard_artefacts(&mut self, discard: bool) {
        self.discard_artefacts = discard;
    }

    /// Builds the pipeline afresh and starts running it on a thread of its own.
    ///
    /// The stall deadline follows the backend's tuning unless `settings` gives one. A new stream is refused
    /// while the previous stream's thread is still running.
    pub fn start_stream(&mut self, settings: Settings) -> Result<Stream, SessionError> {
        if self.running.strong_count() > 0 {
            let message = "a stream is still running on this session, join it before starting another";
            return Err(SessionError::message(message));
        }

        let hub = ChannelHub::process();
        hub.release(&self.token);
        self.warnings.clear();

        let stop = ChannelStop::new();
        let senders = bind_sources(hub, &self.token, &self.sources, &stop);
        let receivers = bind_sinks(hub, &self.token, &self.sinks, &stop);

        let started = self.start_engine(settings, stop);
        match started {
            Ok(handle) => Ok(Stream {
                senders,
                receivers,
                handle,
            }),
            Err(error) => {
                hub.release(&self.token);
                Err(error)
            },
        }
    }

    /// Builds the engine and runs it on its own thread, with `stop` set once the run stops.
    fn start_engine(&mut self, settings: Settings, stop: ChannelStop) -> Result<StreamHandle, SessionError> {
        let tuning = self.backend.context().tuning();
        let settings = with_stall_deadline(settings, tuning);

        let built = steps::build(&self.registry, &self.specs);
        let pipeline = match built {
            Ok(pipeline) => pipeline,
            Err(found) => return Err(self.pipeline_error(found)),
        };

        let compiled = steps::compile(pipeline, &self.backend);
        let compiled = match compiled {
            Ok(compiled) => compiled,
            Err(found) => return Err(self.pipeline_error(found)),
        };
        self.warnings = compiled.warnings().to_vec();

        let engine = compiled.engine(settings);
        let mut engine = match engine {
            Ok(engine) => engine,
            Err(error) => {
                let found = error.into_diagnostics().into_inner();
                return Err(self.pipeline_error(found));
            },
        };

        if self.discard_artefacts {
            engine = engine.discarding_artefacts();
        }

        let cancel = engine.cancel_handle();
        cancel.on_stop(move || stop.stop());

        let stages = named_stages(&compiled);
        let graph = compiled.graph();
        let planned = plan::planned_nodes(graph);
        let files = self.files.clone();
        let alive = Arc::new(());
        self.running = Arc::downgrade(&alive);

        let kept = Kept {
            stages,
            planned,
            files,
        };
        StreamHandle::spawn(engine, kept, alive)
    }

    fn pipeline_error(&self, found: Vec<Diagnostic>) -> SessionError {
        let files = self.files.clone();
        SessionError::pipeline(files, found)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let hub = ChannelHub::process();
        hub.release(&self.token);
    }
}

/// Builds `source` without a device, with its channel stages bound as [Session::new] binds them.
pub(crate) fn check_source(
    source: &PipelineSource,
    channels: &[(String, FrameFormat)],
) -> Result<(), SessionError> {
    let mut specs = source.specs.clone();
    let token = ChannelToken::new();
    let stages = bind_channel_args(&mut specs, &token);
    let matched = match_formats(&stages, channels);
    if let Err(found) = matched {
        return Err(source.pipeline_error(found));
    }

    let built = steps::build(&source.registry, &specs);
    match built {
        Ok(_) => Ok(()),
        Err(found) => Err(source.pipeline_error(found)),
    }
}

fn bind_sources(
    hub: &ChannelHub,
    token: &ChannelToken,
    sources: &[(String, FrameFormat)],
    stop: &ChannelStop,
) -> Vec<(String, SyncSender<InputFrame>)> {
    let mut senders = Vec::new();
    for (stage, format) in sources {
        let (sender, receiver) = sync_channel(CHANNEL_DEPTH);
        hub.bind_source(token, stage, *format, receiver, stop.clone());
        senders.push((stage.clone(), sender));
    }

    senders
}

fn bind_sinks(
    hub: &ChannelHub,
    token: &ChannelToken,
    sinks: &[String],
    stop: &ChannelStop,
) -> Vec<(String, Receiver<OutputFrame>)> {
    let mut receivers = Vec::new();
    for stage in sinks {
        let (sender, receiver) = sync_channel(CHANNEL_DEPTH);
        hub.bind_sink(token, stage, sender, stop.clone());
        receivers.push((stage.clone(), receiver));
    }

    receivers
}

/// The `channel/frames` stages of a pipeline, by stage name.
#[derive(Debug, Default)]
struct ChannelStages {
    /// Each source with where its name was written.
    sources: Vec<(String, Span)>,
    sinks: Vec<String>,
}

/// Writes the session's token and each stage's own name into the args of every channel stage.
fn bind_channel_args(specs: &mut [StageSpec], token: &ChannelToken) -> ChannelStages {
    let mut stages = ChannelStages::default();

    let channel_specs = specs
        .iter_mut()
        .filter(|spec| spec.stage_type.value == CHANNEL_STAGE);
    for spec in channel_specs {
        let name = spec.name.value.clone();
        match spec.kind {
            StageKind::Source => stages.sources.push((name.clone(), spec.name.span)),
            StageKind::Sink => stages.sinks.push(name.clone()),
            StageKind::Filter | StageKind::Mask | StageKind::Provider | StageKind::Crossing => continue,
        }

        let args = &mut spec.args.value;
        if !args.is_object() {
            *args = Value::Object(Map::new());
        }

        if let Value::Object(entries) = args {
            let session = Value::String(token.as_str().to_owned());
            entries.insert("session".to_owned(), session);
            entries.insert("stage".to_owned(), Value::String(name));
        }
    }

    stages
}

/// The format of every channel source, or a diagnostic for each source without one, each format naming no
/// channel source and each name given more than once.
fn match_formats(
    stages: &ChannelStages,
    channels: &[(String, FrameFormat)],
) -> Result<Vec<(String, FrameFormat)>, Vec<Diagnostic>> {
    let mut sources = Vec::new();
    let mut found = Vec::new();

    for (stage, span) in &stages.sources {
        let given = channels.iter().find(|(name, _)| name == stage);
        match given {
            Some((_, format)) => sources.push((stage.clone(), *format)),
            None => found.push(missing_format(stage, *span)),
        }
    }

    for (index, (name, _)) in channels.iter().enumerate() {
        let is_source = stages.sources.iter().any(|(stage, _)| stage == name);
        if !is_source {
            found.push(unknown_channel(name));
        }

        let earlier = &channels[..index];
        let first_repeat = earlier.iter().filter(|(other, _)| other == name).count() == 1;
        if first_repeat {
            found.push(repeated_channel(name));
        }
    }

    if !found.is_empty() {
        return Err(found);
    }

    Ok(sources)
}

fn missing_format(stage: &str, span: Span) -> Diagnostic {
    let path = format!("source.{stage}");
    let message = format!(
        "source {stage} reads frames handed over by an av-toolbox library caller, so only the library can run it, and no frame format was given for it"
    );

    Diagnostic::error(DiagnosticCode::InvalidArgs, path, span, message)
}

fn unknown_channel(name: &str) -> Diagnostic {
    let message = format!("a frame format was given for {name}, which is not a {CHANNEL_STAGE} source");

    Diagnostic::error(
        DiagnosticCode::UnknownStageReference,
        "",
        Span::unknown(),
        message,
    )
}

fn repeated_channel(name: &str) -> Diagnostic {
    let message = format!("a frame format was given for {name} more than once");

    Diagnostic::error(DiagnosticCode::InvalidArgs, "", Span::unknown(), message)
}

fn with_stall_deadline(settings: Settings, tuning: Tuning) -> Settings {
    if settings.stall_deadline.is_some() {
        return settings;
    }

    let stall_deadline = steps::stall_deadline_for(tuning);
    Settings {
        stall_deadline,
        ..settings
    }
}

/// Every node of the compiled pipeline that a written or injected stage stands behind.
fn named_stages(compiled: &CompiledPipeline<'_, '_>) -> HashMap<NodeId, NamedStage> {
    let mut stages = HashMap::new();
    for node in compiled.graph().nodes() {
        let id = node.id();
        if let Some(stage) = compiled.stage_of(id) {
            stages.insert(id, NamedStage::new(stage));
        }
    }

    stages
}

/// One run of a session's pipeline, with the caller's ends of its channels.
///
/// Feed the senders and drain the receivers on different threads. Both channels hold only a couple of
/// frames, so a thread doing both blocks once they fill.
pub struct Stream {
    senders: Vec<(String, SyncSender<InputFrame>)>,
    receivers: Vec<(String, Receiver<OutputFrame>)>,
    handle: StreamHandle,
}

impl Stream {
    /// The sender feeding the `channel/frames` source named `stage`.
    ///
    /// Dropping it ends that source's stream once the frames already sent are read.
    pub fn take_sender(&mut self, stage: &str) -> Option<SyncSender<InputFrame>> {
        let position = self.senders.iter().position(|(name, _)| name == stage)?;
        let (_, sender) = self.senders.swap_remove(position);

        Some(sender)
    }

    /// The receiver of the finished frames of the `channel/frames` sink named `stage`, in stream order.
    ///
    /// Dropping it before the stream ends fails the run.
    pub fn take_receiver(&mut self, stage: &str) -> Option<Receiver<OutputFrame>> {
        let position = self.receivers.iter().position(|(name, _)| name == stage)?;
        let (_, receiver) = self.receivers.swap_remove(position);

        Some(receiver)
    }

    /// The handle on the run, dropping any sender or receiver not taken.
    pub fn into_handle(self) -> StreamHandle {
        self.handle
    }
}

/// What a stream keeps of its compiled pipeline for naming its nodes once the run ends.
struct Kept {
    stages: HashMap<NodeId, NamedStage>,
    planned: Vec<PlannedNode>,
    files: Arc<PipelineFiles>,
}

/// The thread a stream runs on.
pub struct StreamHandle {
    thread: Option<JoinHandle<Finish>>,
    cancel: Arc<Cancel>,
    progress: Progress,
    kept: Kept,
}

impl StreamHandle {
    fn spawn(engine: Engine, kept: Kept, alive: Arc<()>) -> Result<Self, SessionError> {
        let cancel = engine.cancel_handle();
        let progress = engine.progress();

        let spawned = thread::Builder::new()
            .name("av-toolbox-stream".to_owned())
            .spawn(move || run_engine(engine, alive));
        let thread = spawned.map_err(spawn_error)?;

        Ok(Self {
            thread: Some(thread),
            cancel,
            progress,
            kept,
        })
    }

    /// Stops the run wherever it has got to. [StreamHandle::join] then reports it as cancelled.
    ///
    /// The channel stages see the stop within [CHANNEL_POLL](av_toolbox_core::CHANNEL_POLL), even while the
    /// caller still holds a sender or a receiver it no longer drains.
    pub fn cancel(&self) {
        self.cancel.stop();
    }

    /// The signal that stops the run, for stopping it from somewhere that cannot hold the handle.
    pub fn cancel_handle(&self) -> Arc<Cancel> {
        self.cancel.clone()
    }

    /// A live view of the run, readable from any thread while it goes.
    pub fn progress(&self) -> Progress {
        self.progress.clone()
    }

    /// The stage behind a node of the run, or `None` for a node no stage stands behind.
    pub fn stage_of(&self, node: NodeId) -> Option<StageRef<'_>> {
        let stage = self.kept.stages.get(&node)?;
        Some(stage.as_ref())
    }

    /// Every node of the run in dependency order, with the width and granularity the plan gave it.
    pub fn plan(&self) -> &[PlannedNode] {
        &self.kept.planned
    }

    /// Waits for the run to end and reports how it went.
    ///
    /// The run ends once every sender is dropped, or after [StreamHandle::cancel]. A failed run names the
    /// stage it failed at.
    pub fn join(mut self) -> Result<RunReport, SessionError> {
        let thread = self.thread.take().expect("a stream is joined once");
        let joined = thread.join();

        match joined {
            Ok(Ok(report)) => Ok(report),
            Ok(Err(error)) => Err(self.run_failure(error)),
            Err(_) => Err(SessionError::message("the stream's thread panicked")),
        }
    }

    fn run_failure(&self, error: RunError) -> SessionError {
        let node = error.node();
        let stage = node.and_then(|node| self.kept.stages.get(&node)).cloned();
        let files = self.kept.files.clone();

        SessionError::run(files, error, stage)
    }
}

impl Drop for StreamHandle {
    /// Cancels a run nobody joined and waits for its thread.
    ///
    /// The channel stages stop waiting on the caller once the run is cancelled, so this does not wait for the
    /// caller to close its senders or drain its receivers.
    fn drop(&mut self) {
        let Some(thread) = self.thread.take() else {
            return;
        };

        self.cancel.stop();
        let _ = thread.join();
    }
}

/// Runs `engine`, holding `alive` until the run ends so the session knows the thread is busy.
fn run_engine(engine: Engine, alive: Arc<()>) -> Finish {
    let finished = engine.run();
    drop(alive);

    finished
}

fn spawn_error(error: io::Error) -> SessionError {
    let message = format!("could not start the stream's thread: {error}");
    SessionError::message(message)
}

#[cfg(test)]
mod tests {
    use av_toolbox_graph::{ArgSpans, Spanned};
    use serde_json::json;

    use super::*;

    fn spec(name: &str, kind: StageKind, stage_type: &str, args: Value) -> StageSpec {
        StageSpec {
            name: Spanned::new(name.to_owned(), Span::new(1, 2)),
            kind,
            stage_type: Spanned::new(stage_type.to_owned(), Span::new(3, 4)),
            inputs: Vec::new(),
            args: Spanned::new(args, Span::new(5, 6)),
            arg_spans: ArgSpans::new(),
            workers: None,
            deferred: Vec::new(),
            span: Span::new(1, 6),
        }
    }

    #[test]
    fn channel_stages_get_the_token_and_their_own_name() {
        let token = ChannelToken::new();
        let mut specs = vec![
            spec("input", StageKind::Source, CHANNEL_STAGE, Value::Null),
            spec("output", StageKind::Sink, CHANNEL_STAGE, json!({})),
            spec("crop", StageKind::Filter, "crop/manual", json!({ "left": 2 })),
        ];

        let stages = bind_channel_args(&mut specs, &token);

        let expected_source = json!({ "session": token.as_str(), "stage": "input" });
        let expected_sink = json!({ "session": token.as_str(), "stage": "output" });
        assert_eq!(specs[0].args.value, expected_source);
        assert_eq!(specs[1].args.value, expected_sink);
        assert_eq!(specs[2].args.value, json!({ "left": 2 }));
        assert_eq!(specs[0].args.span, Span::new(5, 6));
        assert_eq!(stages.sources, [("input".to_owned(), Span::new(1, 2))]);
        assert_eq!(stages.sinks, ["output"]);
    }

    #[test]
    fn channel_args_written_by_hand_are_overwritten() {
        let token = ChannelToken::new();
        let written = json!({ "session": "someone-else", "stage": "elsewhere" });
        let mut specs = vec![spec("input", StageKind::Source, CHANNEL_STAGE, written)];

        bind_channel_args(&mut specs, &token);

        let expected = json!({ "session": token.as_str(), "stage": "input" });
        assert_eq!(specs[0].args.value, expected);
    }

    #[test]
    fn an_explicit_stall_deadline_is_kept() {
        let deadline = std::time::Duration::from_secs(3);
        let settings = Settings {
            stall_deadline: Some(deadline),
            ..Settings::default()
        };

        let kept = with_stall_deadline(settings, Tuning::Auto);

        assert_eq!(kept.stall_deadline, Some(deadline));
    }

    #[test]
    fn auto_tuning_waits_longer_when_no_deadline_is_given() {
        let settings = Settings::default();

        let filled = with_stall_deadline(settings, Tuning::Auto);

        assert_eq!(filled.stall_deadline, steps::stall_deadline_for(Tuning::Auto));
    }
}
