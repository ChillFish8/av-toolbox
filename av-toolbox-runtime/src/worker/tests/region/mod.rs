mod artefacts;
mod chunks;
mod feeds;
mod flushing;
mod frame_routed;
mod paths;
mod steps;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use av_toolbox_core::testing::FixtureChunk;
use av_toolbox_core::{Filter, Inputs, Sink, StageError, StageInstance};
use av_toolbox_gpu::{
    AnyItem,
    ColourInfo,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    LayerId,
    MetaValue,
    SampleFormat,
    StreamItem,
    StreamPolicy,
    Subsampling,
    flush_counting_context,
    testing_context,
};
use av_toolbox_graph::{NodeId, StageBox};

use crate::RunError;
use crate::tests::routes::{meta_in, route};
use crate::worker::{Feed, RegionEdges, RegionRunner};

fn format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

/// A frame at `index`, in `chunk` of layer 0.
fn frame(context: &FrameContext, index: u64, chunk: u64) -> Frame {
    let meta = meta_in(index, 0, chunk);
    context.alloc_frame(&format(), meta)
}

fn item(context: &FrameContext, index: u64, chunk: u64) -> AnyItem {
    StreamItem::Frame(frame(context, index, chunk)).into()
}

/// Restarts each of `chunked` on layer 0, the layer [frame] puts its chunk in.
trait OnLayerZero {
    fn chunked_on_layer_zero(self, chunked: HashSet<NodeId>) -> Self;
}

impl OnLayerZero for RegionEdges {
    fn chunked_on_layer_zero(self, chunked: HashSet<NodeId>) -> Self {
        let routes = chunked.iter().map(|node| (*node, route(0))).collect();

        self.with_chunked(chunked).with_routes(routes)
    }
}

/// A fresh context for a call, since `FrameContext` cannot be cloned or shared mutably
/// across the several calls one test makes.
fn context_of(_seed: &FrameContext) -> FrameContext {
    testing_context()
}

struct PassThrough;

impl StageInstance for PassThrough {}

impl Filter for PassThrough {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// Passes its input through with a fixed timestamp, so two branches from the same producer can be told apart
/// downstream.
struct Retagging {
    seconds: u64,
}

impl StageInstance for Retagging {}

impl Filter for Retagging {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mut frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        frame.meta_mut().timestamp = Some(Duration::from_secs(self.seconds));

        Ok(vec![frame])
    }
}

/// Passes its input through with a fixed frame index.
struct Reindexing {
    index: u64,
}

impl StageInstance for Reindexing {}

impl Filter for Reindexing {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mut frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        frame.set_index(self.index);

        Ok(vec![frame])
    }
}

/// Holds `lag` frames, emitting the oldest once it holds more, and flushes the rest when its chunk ends.
struct Lagging {
    lag: usize,
    held: VecDeque<Frame>,
}

impl Lagging {
    fn new(lag: usize) -> Self {
        Self {
            lag,
            held: VecDeque::new(),
        }
    }
}

impl StageInstance for Lagging {}

impl Filter for Lagging {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        self.held.push_back(frame);

        if self.held.len() <= self.lag {
            return Ok(Vec::new());
        }

        let oldest = self.held.pop_front().expect("more than the lag is held");
        Ok(vec![oldest])
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let flushed = self.held.drain(..).collect();
        Ok(flushed)
    }
}

/// Passes its input through, moving it to chunk 1 of layer 0 from frame `switch_at` onwards.
struct Rechunking {
    switch_at: u64,
}

impl StageInstance for Rechunking {}

impl Filter for Rechunking {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mut frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        let index = frame.meta().index;
        let chunk = u64::from(index >= self.switch_at);
        frame.meta_mut().provided.set(LayerId(0), FixtureChunk(chunk));

        Ok(vec![frame])
    }
}

/// Passes its input through, and fails when its chunk ends.
struct FailingFlush;

impl StageInstance for FailingFlush {}

impl Filter for FailingFlush {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;

        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        Err(StageError::msg("this filter fails to flush"))
    }
}

/// Emits the frame it was given at index `2n`, and a copy of it at index `2n + 1`.
struct Doubling;

impl StageInstance for Doubling {}

impl Filter for Doubling {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mut first = inputs.take_frame(0).ok_or_else(missing_frame)?;
        let index = first.meta().index;
        first.set_index(index * 2);

        let mut second = first.clone();
        second.set_index(index * 2 + 1);

        Ok(vec![first, second])
    }
}

fn missing_frame() -> StageError {
    StageError::msg("expected a frame in slot 0")
}

/// Requires both of its two inputs, and returns the one at slot 0.
struct Joining;

impl StageInstance for Joining {}

impl Filter for Joining {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let left = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        inputs
            .take_frame(1)
            .ok_or_else(|| StageError::msg("expected a frame in slot 1"))?;

        Ok(vec![left])
    }
}

/// A filter that keeps a copy of what it is given and flushes it again when its chunk ends.
struct Holding {
    held: Option<Frame>,
}

impl StageInstance for Holding {}

impl Filter for Holding {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        self.held = Some(frame.clone());
        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        Ok(self.held.take().into_iter().collect())
    }
}

/// Buffers every frame it is given, emitting nothing until its chunk ends.
struct Buffering {
    held: Vec<Frame>,
}

impl StageInstance for Buffering {}

impl Filter for Buffering {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        self.held.push(frame);
        Ok(Vec::new())
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        Ok(std::mem::take(&mut self.held))
    }
}

/// Emits a fixed number of copies of what it was given.
struct Multiply {
    copies: usize,
}

impl StageInstance for Multiply {}

impl Filter for Multiply {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;

        Ok((0..self.copies).map(|_| frame.clone()).collect())
    }
}

/// Records what it was asked to consume, by slot order.
struct Recording {
    seen: Arc<Mutex<Vec<u64>>>,
}

impl StageInstance for Recording {}

impl Sink for Recording {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let mut seen = self.seen.lock().expect("not poisoned");

        for slot in 0..inputs.len() {
            if let Some(item) = inputs.take(slot) {
                seen.push(item.meta().index);
            }
        }

        Ok(())
    }
}

/// Records the timestamp of what it was asked to consume, by slot order.
struct TimestampRecording {
    seen: Arc<Mutex<Vec<Option<Duration>>>>,
}

impl StageInstance for TimestampRecording {}

impl Sink for TimestampRecording {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let mut seen = self.seen.lock().expect("not poisoned");

        for slot in 0..inputs.len() {
            if let Some(item) = inputs.take(slot) {
                seen.push(item.meta().timestamp);
            }
        }

        Ok(())
    }
}

struct Failing;

impl StageInstance for Failing {}

impl Filter for Failing {
    fn process(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        Err(StageError::msg("this filter always fails"))
    }
}

/// Records whether it was started.
struct Starting {
    started: Arc<AtomicBool>,
}

impl StageInstance for Starting {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.started.store(true, Ordering::SeqCst);
        Ok(())
    }
}

impl Sink for Starting {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
        Ok(())
    }
}

/// Attaches `seen` to every item it passes.
struct Tagging;

impl StageInstance for Tagging {}

impl Filter for Tagging {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mut frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        frame.meta_mut().attach("seen", MetaValue::Bool(true));

        Ok(vec![frame])
    }
}

/// Holds each item back one step, releasing the last on end of stream.
struct Delaying {
    held: Option<Frame>,
}

impl StageInstance for Delaying {}

impl Filter for Delaying {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let arrived = inputs.take_frame(0).ok_or_else(missing_frame)?;
        let released = self.held.replace(arrived);

        Ok(released.into_iter().collect())
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let released = self.held.take();

        Ok(released.into_iter().collect())
    }
}

/// Holds every item, releasing it stamped with `flushed` once the stream ends.
struct HoldingUntilEnd {
    held: Option<Frame>,
}

impl StageInstance for HoldingUntilEnd {}

impl Filter for HoldingUntilEnd {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let arrived = inputs.take_frame(0).ok_or_else(missing_frame)?;
        self.held = Some(arrived);

        Ok(vec![])
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let mut released = self.held.take().expect("an item was held");
        released.meta_mut().attach("flushed", MetaValue::Bool(true));

        Ok(vec![released])
    }
}

fn stage_paths(item: &AnyItem) -> Vec<String> {
    item.meta()
        .attachments
        .stages()
        .map(|(path, _)| path.to_string())
        .collect()
}

fn paths(entries: &[(NodeId, &str)]) -> HashMap<NodeId, Arc<str>> {
    entries.iter().map(|(id, path)| (*id, Arc::from(*path))).collect()
}

/// Lets go of whatever it is given as soon as it has it.
struct Releasing;

impl StageInstance for Releasing {}

impl Sink for Releasing {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
        Ok(())
    }
}

/// What [Probing] saw while its region was still running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sighting {
    held_elsewhere: bool,
    flushes: usize,
}

/// Records whether anyone besides the test holds `watched`, and how many flushes there have
/// been so far.
struct Probing {
    watched: Arc<Mutex<Option<Frame>>>,
    flushes: Arc<AtomicUsize>,
    sightings: Arc<Mutex<Vec<Sighting>>>,
}

impl StageInstance for Probing {}

impl Sink for Probing {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
        let held_elsewhere = is_held_elsewhere(&self.watched);
        let flushes = self.flushes.load(Ordering::SeqCst);
        let sighting = Sighting {
            held_elsewhere,
            flushes,
        };

        self.sightings.lock().expect("not poisoned").push(sighting);
        Ok(())
    }
}

/// Whether any holder besides `watched` itself keeps its frame alive.
fn is_held_elsewhere(watched: &Mutex<Option<Frame>>) -> bool {
    let mut slot = watched.lock().expect("not poisoned");
    let frame = slot.take().expect("a frame to watch");

    match frame.try_into_mut() {
        Ok(writable) => {
            *slot = Some(writable.freeze());
            false
        },
        Err(shared) => {
            *slot = Some(shared);
            true
        },
    }
}

/// A region whose head lets go of its inputs at once, followed by a [Probing] node.
fn probed_region(
    watched: &Arc<Mutex<Option<Frame>>>,
    flushes: &Arc<AtomicUsize>,
    sightings: &Arc<Mutex<Vec<Sighting>>>,
) -> RegionRunner {
    let head = NodeId::new(0);
    let probe = NodeId::new(1);
    let probing = Probing {
        watched: watched.clone(),
        flushes: flushes.clone(),
        sightings: sightings.clone(),
    };

    let nodes = vec![
        (head, StageBox::Sink(Box::new(Releasing))),
        (probe, StageBox::Sink(Box::new(probing))),
    ];
    let edges = RegionEdges::new(vec![vec![]], vec![]);

    RegionRunner::new(nodes, edges)
}

/// A head feeding a branch that lags by `lag` and one that does not, joined by a sink recording both slots.
fn lagging_diamond(lag: usize, seen: Arc<Mutex<Vec<u64>>>) -> (RegionRunner, NodeId) {
    lagging_diamond_chunked(lag, seen, HashSet::new())
}

/// The same as [lagging_diamond], restarting each of `chunked` whenever its inputs move to a new chunk.
fn lagging_diamond_chunked(
    lag: usize,
    seen: Arc<Mutex<Vec<u64>>>,
    chunked: HashSet<NodeId>,
) -> (RegionRunner, NodeId) {
    let head = NodeId::new(0);
    let lagging = NodeId::new(1);
    let direct = NodeId::new(2);
    let join = NodeId::new(3);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(PassThrough))),
        (lagging, StageBox::Filter(Box::new(Lagging::new(lag)))),
        (direct, StageBox::Filter(Box::new(PassThrough))),
        (join, StageBox::Sink(Box::new(Recording { seen }))),
    ];
    let head_feed = Feed {
        producer: head,
        slot: 0,
    };
    let join_feeds = vec![
        Feed {
            producer: lagging,
            slot: 0,
        },
        Feed {
            producer: direct,
            slot: 1,
        },
    ];
    let feeds = vec![vec![head_feed], vec![head_feed], join_feeds];
    let edges = RegionEdges::new(feeds, vec![join]).chunked_on_layer_zero(chunked);
    let runner = RegionRunner::new(nodes, edges);

    (runner, join)
}

/// Records every index it consumes, and pushes `u64::MAX` each time it is started.
struct StartRecording {
    seen: Arc<Mutex<Vec<u64>>>,
}

impl StageInstance for StartRecording {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.seen.lock().expect("not poisoned").push(u64::MAX);
        Ok(())
    }
}

impl Sink for StartRecording {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let item = inputs.take(0).ok_or_else(missing_item)?;
        self.seen.lock().expect("not poisoned").push(item.meta().index);
        Ok(())
    }
}

fn missing_item() -> StageError {
    StageError::msg("expected an item in slot 0")
}

/// Runs one item per `(index, chunk)` through `runner`, then ends it, returning what its exits produced.
fn run_all(runner: &mut RegionRunner, items: &[(u64, u64)]) -> Vec<u64> {
    run_steps(runner, items).concat()
}

/// The same as [run_all], with what each `run` produced in its own entry and what `end` produced last.
fn run_steps(runner: &mut RegionRunner, items: &[(u64, u64)]) -> Vec<Vec<u64>> {
    let context = testing_context();
    let mut steps = Vec::new();

    runner.start(&mut context_of(&context)).expect("should start");

    for &(index, chunk) in items {
        let bundle = vec![Some(item(&context, index, chunk))];
        let output = runner.run(&mut context_of(&context), bundle).expect("should run");
        steps.push(output.iter().map(|item| item.meta().index).collect());
    }

    let flushed = runner.end(&mut context_of(&context), None).expect("should end");
    steps.push(flushed.iter().map(|item| item.meta().index).collect());

    steps
}
