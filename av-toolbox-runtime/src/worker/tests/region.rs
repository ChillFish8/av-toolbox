use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use av_toolbox_core::{
    Filter,
    Inputs,
    MetaInputs,
    MetaItem,
    MetaTransform,
    Sink,
    StageError,
    StageInstance,
    Transform,
};
use av_toolbox_gpu::{
    AnyItem,
    ColourInfo,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
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
    }
}

fn frame(context: &FrameContext, index: u64, chunk: u64) -> Frame {
    let meta = FrameMeta::at(index).in_chunk(chunk);
    context.alloc_frame(&format(), meta)
}

fn item(context: &FrameContext, index: u64, chunk: u64) -> AnyItem {
    StreamItem::Frame(frame(context, index, chunk)).into()
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

/// Passes its input through with a fixed frame index, so two branches from the same
/// producer can be told apart downstream.
struct Retagging {
    index: u64,
}

impl StageInstance for Retagging {}

impl Filter for Retagging {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mut frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        frame.set_index(self.index);
        Ok(vec![frame])
    }
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

impl Transform for Buffering {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs,
    ) -> Result<Vec<StreamItem>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        self.held.push(frame);
        Ok(Vec::new())
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<StreamItem>, StageError> {
        Ok(self.held.drain(..).map(StreamItem::Frame).collect())
    }
}

/// Emits a fixed number of copies of what it was given.
struct Multiply {
    copies: usize,
}

impl StageInstance for Multiply {}

impl Transform for Multiply {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs,
    ) -> Result<Vec<StreamItem>, StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;

        Ok((0..self.copies).map(|_| item.clone()).collect())
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

#[test]
fn a_single_node_region_passes_its_bundle_straight_through() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(PassThrough)))];
    let edges = RegionEdges::new(vec![], vec![head]);
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 4, 0))])
        .expect("should run");

    assert_eq!(produced.len(), 1);
    assert_eq!(produced[0].meta().index, 4);
}

#[test]
fn a_chain_threads_one_nodes_output_into_the_next() {
    let context = testing_context();
    let head = NodeId::new(0);
    let second = NodeId::new(1);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(PassThrough))),
        (second, StageBox::Filter(Box::new(PassThrough))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let edges = RegionEdges::new(feeds, vec![second]);
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 4, 0))])
        .expect("should run");

    assert_eq!(produced.len(), 1);
    assert_eq!(produced[0].meta().index, 4);
}

/// A fan out and fan in below one node runs entirely in this one worker.
#[test]
fn a_diamond_feeds_both_branches_from_one_producer_and_joins_them() {
    let context = testing_context();
    let head = NodeId::new(0);
    let left = NodeId::new(1);
    let right = NodeId::new(2);
    let join = NodeId::new(3);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(PassThrough))),
        (left, StageBox::Filter(Box::new(PassThrough))),
        (right, StageBox::Filter(Box::new(PassThrough))),
        (join, StageBox::Filter(Box::new(Joining))),
    ];
    let feeds = vec![
        vec![Feed {
            producer: head,
            slot: 0,
        }],
        vec![Feed {
            producer: head,
            slot: 0,
        }],
        vec![
            Feed {
                producer: left,
                slot: 0,
            },
            Feed {
                producer: right,
                slot: 1,
            },
        ],
    ];
    let edges = RegionEdges::new(feeds, vec![join]);
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 4, 0))])
        .expect("should run");

    assert_eq!(produced.len(), 1);
    assert_eq!(produced[0].meta().index, 4);
}

/// Wires the two producers so slot order and producer-visit order disagree. A runner that
/// assembles inputs in visit order rather than by `Feed::slot` would put the head's frame at
/// slot 0 and the retagged frame at slot 1, the opposite of what is asked for here.
#[test]
fn an_edge_fills_the_input_slot_it_names() {
    let context = testing_context();
    let head = NodeId::new(0);
    let retagged = NodeId::new(1);
    let sink = NodeId::new(2);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let nodes = vec![
        (head, StageBox::Filter(Box::new(PassThrough))),
        (retagged, StageBox::Filter(Box::new(Retagging { index: 8 }))),
        (sink, StageBox::Sink(Box::new(Recording { seen: seen.clone() }))),
    ];
    let feeds = vec![
        vec![Feed {
            producer: head,
            slot: 0,
        }],
        vec![
            Feed {
                producer: head,
                slot: 1,
            },
            Feed {
                producer: retagged,
                slot: 0,
            },
        ],
    ];
    let edges = RegionEdges::new(feeds, vec![sink]);
    let mut runner = RegionRunner::new(nodes, edges);

    runner
        .run(&mut context_of(&context), vec![Some(item(&context, 7, 0))])
        .expect("should run");

    assert_eq!(*seen.lock().expect("not poisoned"), vec![8, 7]);
}

#[test]
fn a_node_producing_several_items_runs_its_consumer_once_per_item() {
    let context = testing_context();
    let head = NodeId::new(0);
    let sink = NodeId::new(1);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let nodes = vec![
        (head, StageBox::Transform(Box::new(Multiply { copies: 3 }))),
        (sink, StageBox::Sink(Box::new(Recording { seen: seen.clone() }))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let edges = RegionEdges::new(feeds, vec![sink]);
    let mut runner = RegionRunner::new(nodes, edges);

    runner
        .run(&mut context_of(&context), vec![Some(item(&context, 0, 0))])
        .expect("should run");

    assert_eq!(seen.lock().expect("not poisoned").len(), 3);
}

#[test]
fn a_node_producing_nothing_does_not_run_its_consumer() {
    let context = testing_context();
    let head = NodeId::new(0);
    let sink = NodeId::new(1);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let nodes = vec![
        (head, StageBox::Transform(Box::new(Multiply { copies: 0 }))),
        (sink, StageBox::Sink(Box::new(Recording { seen: seen.clone() }))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let edges = RegionEdges::new(feeds, vec![sink]);
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 0, 0))])
        .expect("should run");

    assert!(produced.is_empty());
    assert!(seen.lock().expect("not poisoned").is_empty());
}

#[test]
fn starting_a_region_starts_every_node_in_it() {
    let context = testing_context();
    let first_started = Arc::new(AtomicBool::new(false));
    let second_started = Arc::new(AtomicBool::new(false));
    let third_started = Arc::new(AtomicBool::new(false));
    let nodes = vec![
        (
            NodeId::new(0),
            StageBox::Sink(Box::new(Starting {
                started: first_started.clone(),
            })),
        ),
        (
            NodeId::new(1),
            StageBox::Sink(Box::new(Starting {
                started: second_started.clone(),
            })),
        ),
        (
            NodeId::new(2),
            StageBox::Sink(Box::new(Starting {
                started: third_started.clone(),
            })),
        ),
    ];
    let feeds = vec![vec![], vec![]];
    let edges = RegionEdges::new(feeds, vec![]);
    let mut runner = RegionRunner::new(nodes, edges);

    runner.start(&mut context_of(&context)).expect("should start");

    assert!(first_started.load(Ordering::SeqCst));
    assert!(second_started.load(Ordering::SeqCst));
    assert!(third_started.load(Ordering::SeqCst));
}

/// A `Holding` head is pre-seeded with a frame directly, without ever calling `run`, so the
/// only way its downstream `Buffering` node ever sees that frame is through `end` threading
/// the head's flush into it. If that threading were skipped, `Buffering` would still be
/// empty and its own flush would produce nothing.
#[test]
fn ending_a_region_flushes_its_nodes_in_topological_order() {
    let context = testing_context();
    let head = NodeId::new(0);
    let downstream = NodeId::new(1);
    let nodes = vec![
        (
            head,
            StageBox::Filter(Box::new(Holding {
                held: Some(frame(&context, 9, 0)),
            })),
        ),
        (
            downstream,
            StageBox::Transform(Box::new(Buffering { held: Vec::new() })),
        ),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let edges = RegionEdges::new(feeds, vec![downstream]);
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = runner.end(&mut context_of(&context)).expect("should end");

    assert_eq!(produced.len(), 1);
    assert_eq!(produced[0].meta().index, 9);
}

/// `left` multiplies head's one item into 2, `right` multiplies it into 1, so `join`'s two
/// feeds disagree about how many items they supply.
#[test]
fn feeds_that_disagree_on_how_many_items_they_supply_are_an_error() {
    let context = testing_context();
    let head = NodeId::new(0);
    let left = NodeId::new(1);
    let right = NodeId::new(2);
    let join = NodeId::new(3);
    let nodes = vec![
        (head, StageBox::Transform(Box::new(Multiply { copies: 1 }))),
        (left, StageBox::Transform(Box::new(Multiply { copies: 2 }))),
        (right, StageBox::Transform(Box::new(Multiply { copies: 1 }))),
        (
            join,
            StageBox::Sink(Box::new(Recording {
                seen: Arc::new(Mutex::new(Vec::new())),
            })),
        ),
    ];
    let feeds = vec![
        vec![Feed {
            producer: head,
            slot: 0,
        }],
        vec![Feed {
            producer: head,
            slot: 0,
        }],
        vec![
            Feed {
                producer: left,
                slot: 0,
            },
            Feed {
                producer: right,
                slot: 1,
            },
        ],
    ];
    let edges = RegionEdges::new(feeds, vec![join]);
    let mut runner = RegionRunner::new(nodes, edges);

    let result = runner.run(&mut context_of(&context), vec![Some(item(&context, 0, 0))]);

    let Err(RunError::MismatchedFeeds { node, counts }) = result else {
        panic!("feeds supplying different counts should have been an error");
    };

    assert_eq!(node, join);
    assert_eq!(counts, vec![2, 1]);
}

/// `left` and `right` both multiply head's one item into 2, so `join`'s two feeds agree and
/// it runs twice.
#[test]
fn feeds_that_agree_on_how_many_items_they_supply_run_that_many_times() {
    let context = testing_context();
    let head = NodeId::new(0);
    let left = NodeId::new(1);
    let right = NodeId::new(2);
    let join = NodeId::new(3);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let nodes = vec![
        (head, StageBox::Transform(Box::new(Multiply { copies: 1 }))),
        (left, StageBox::Transform(Box::new(Multiply { copies: 2 }))),
        (right, StageBox::Transform(Box::new(Multiply { copies: 2 }))),
        (join, StageBox::Sink(Box::new(Recording { seen: seen.clone() }))),
    ];
    let feeds = vec![
        vec![Feed {
            producer: head,
            slot: 0,
        }],
        vec![Feed {
            producer: head,
            slot: 0,
        }],
        vec![
            Feed {
                producer: left,
                slot: 0,
            },
            Feed {
                producer: right,
                slot: 1,
            },
        ],
    ];
    let edges = RegionEdges::new(feeds, vec![join]);
    let mut runner = RegionRunner::new(nodes, edges);

    runner
        .run(&mut context_of(&context), vec![Some(item(&context, 0, 0))])
        .expect("should run");

    // Two runs of `join`, each recording both of its slots.
    assert_eq!(seen.lock().expect("not poisoned").len(), 4);
}

#[test]
fn a_failure_inside_a_region_names_the_node_it_came_from() {
    let context = testing_context();
    let head = NodeId::new(0);
    let failing = NodeId::new(1);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(PassThrough))),
        (failing, StageBox::Filter(Box::new(Failing))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let edges = RegionEdges::new(feeds, vec![failing]);
    let mut runner = RegionRunner::new(nodes, edges);

    let result = runner.run(&mut context_of(&context), vec![Some(item(&context, 0, 0))]);

    let Err(error) = result else {
        panic!("the failing stage should have failed");
    };

    assert_eq!(error.node(), Some(failing));
}

/// Attaches `seen` to every item it passes.
struct Tagging;

impl StageInstance for Tagging {}

impl MetaTransform for Tagging {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: MetaInputs,
    ) -> Result<Vec<MetaItem>, StageError> {
        let mut item = inputs.take(0).expect("an item");
        item.meta_mut().attach("seen", MetaValue::Bool(true));

        Ok(vec![item])
    }
}

/// Holds each item back one step, releasing the last on end of stream.
struct Delaying {
    held: Option<MetaItem>,
}

impl StageInstance for Delaying {}

impl MetaTransform for Delaying {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: MetaInputs,
    ) -> Result<Vec<MetaItem>, StageError> {
        let arrived = inputs.take(0).expect("an item");
        let released = self.held.replace(arrived);

        Ok(released.into_iter().collect())
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<MetaItem>, StageError> {
        let released = self.held.take();

        Ok(released.into_iter().collect())
    }
}

/// Holds every item, releasing it stamped with `flushed` once the stream ends.
struct HoldingUntilEnd {
    held: Option<MetaItem>,
}

impl StageInstance for HoldingUntilEnd {}

impl MetaTransform for HoldingUntilEnd {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: MetaInputs,
    ) -> Result<Vec<MetaItem>, StageError> {
        let arrived = inputs.take(0).expect("an item");
        self.held = Some(arrived);

        Ok(vec![])
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<MetaItem>, StageError> {
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

#[test]
fn a_heads_output_is_stamped_with_its_path() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::MetaTransform(Box::new(Tagging)))];
    let edges = RegionEdges::new(vec![], vec![head]).with_paths(paths(&[(head, "transform.tag")]));
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 4, 0))])
        .expect("should run");

    assert_eq!(stage_paths(&produced[0]), vec!["transform.tag"]);
}

#[test]
fn each_node_in_a_chain_is_stamped_with_its_own_path() {
    let context = testing_context();
    let head = NodeId::new(0);
    let second = NodeId::new(1);
    let nodes = vec![
        (head, StageBox::MetaTransform(Box::new(Tagging))),
        (second, StageBox::MetaTransform(Box::new(Tagging))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let named = paths(&[(head, "transform.first"), (second, "transform.second")]);
    let edges = RegionEdges::new(feeds, vec![second]).with_paths(named);
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 4, 0))])
        .expect("should run");

    assert_eq!(
        stage_paths(&produced[0]),
        vec!["transform.first", "transform.second"]
    );
}

#[test]
fn a_replayed_item_keeps_its_first_stamp() {
    let context = testing_context();
    let head = NodeId::new(0);
    let second = NodeId::new(1);
    let nodes = vec![
        (head, StageBox::MetaTransform(Box::new(Tagging))),
        (second, StageBox::MetaTransform(Box::new(Delaying { held: None }))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let named = paths(&[(head, "transform.tag"), (second, "transform.delay")]);
    let edges = RegionEdges::new(feeds, vec![second]).with_paths(named);
    let mut runner = RegionRunner::new(nodes, edges);
    let mut stage_context = context_of(&context);

    runner
        .run(&mut stage_context, vec![Some(item(&context, 4, 0))])
        .expect("should run");
    let flushed = runner.end(&mut stage_context).expect("should end");

    assert_eq!(flushed.len(), 1);
    assert_eq!(stage_paths(&flushed[0]), vec!["transform.tag"]);
}

#[test]
fn an_item_released_at_end_of_stream_is_stamped_with_its_path() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(
        head,
        StageBox::MetaTransform(Box::new(HoldingUntilEnd { held: None })),
    )];
    let edges = RegionEdges::new(vec![], vec![head]).with_paths(paths(&[(head, "transform.flush")]));
    let mut runner = RegionRunner::new(nodes, edges);
    let mut stage_context = context_of(&context);

    let produced = runner
        .run(&mut stage_context, vec![Some(item(&context, 4, 0))])
        .expect("should run");
    assert!(produced.is_empty());

    let flushed = runner.end(&mut stage_context).expect("should end");

    assert_eq!(flushed.len(), 1);
    assert_eq!(stage_paths(&flushed[0]), vec!["transform.flush"]);
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

#[test]
fn flushing_at_hand_off_flushes_each_step_before_letting_its_inputs_go() {
    let (mut context, flushes) = flush_counting_context(StreamPolicy::FlushAtHandOff);
    let watched_slot = Mutex::new(None);
    let watched = Arc::new(watched_slot);
    let sighting_log = Mutex::new(Vec::new());
    let sightings = Arc::new(sighting_log);
    let mut runner = probed_region(&watched, &flushes, &sightings);

    for index in 0..3 {
        let frame = frame(&context, index, 0);
        *watched.lock().expect("not poisoned") = Some(frame.clone());
        let before = flushes.load(Ordering::SeqCst);

        let bundle = vec![Some(StreamItem::Frame(frame).into())];
        runner.run(&mut context, bundle).expect("should run");

        let sighting = sightings.lock().expect("not poisoned")[index as usize];
        assert_eq!(
            sighting,
            Sighting {
                held_elsewhere: true,
                flushes: before,
            },
            "the input is still held, unflushed, after the head let it go",
        );
        assert_eq!(flushes.load(Ordering::SeqCst), before + 1, "one flush per step");
        assert!(
            !is_held_elsewhere(&watched),
            "the input is let go once the step is flushed"
        );
    }
}

#[test]
fn a_shared_stream_region_never_flushes_and_lets_inputs_go_at_once() {
    let (mut context, flushes) = flush_counting_context(StreamPolicy::SharedStream);
    let watched_slot = Mutex::new(None);
    let watched = Arc::new(watched_slot);
    let sighting_log = Mutex::new(Vec::new());
    let sightings = Arc::new(sighting_log);
    let mut runner = probed_region(&watched, &flushes, &sightings);

    for index in 0..3 {
        let frame = frame(&context, index, 0);
        *watched.lock().expect("not poisoned") = Some(frame.clone());

        let bundle = vec![Some(StreamItem::Frame(frame).into())];
        runner.run(&mut context, bundle).expect("should run");
    }

    runner.end(&mut context).expect("should end");

    let seen = sightings.lock().expect("not poisoned").clone();
    let expected = Sighting {
        held_elsewhere: false,
        flushes: 0,
    };

    assert!(seen.iter().all(|sighting| *sighting == expected), "{seen:?}");
    assert_eq!(flushes.load(Ordering::SeqCst), 0);
}

#[test]
fn ending_a_flush_at_hand_off_region_flushes_it() {
    let (mut context, flushes) = flush_counting_context(StreamPolicy::FlushAtHandOff);
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(PassThrough)))];
    let edges = RegionEdges::new(vec![], vec![head]);
    let mut runner = RegionRunner::new(nodes, edges);

    runner.end(&mut context).expect("should end");

    assert_eq!(flushes.load(Ordering::SeqCst), 1);
}
