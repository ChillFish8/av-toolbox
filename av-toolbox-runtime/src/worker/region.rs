use std::collections::HashMap;

use av_toolbox_core::Inputs;
use av_toolbox_gpu::{FrameContext, StreamItem};
use av_toolbox_graph::{NodeId, StageBox};

use super::node::{end_node, run_node, start_node};
use crate::RunError;

/// One node's output filling one of another node's input slots.
#[derive(Debug, Clone, Copy)]
pub struct Feed {
    pub producer: NodeId,
    pub slot: usize,
}

/// How a region's nodes feed each other.
///
/// `feeds` holds one entry per node after the head, in the same order those nodes appear in
/// [RegionRunner]'s own node list. Each entry lists which producer fills which of that node's
/// input slots. `exits` names the nodes whose output leaves the region.
#[derive(Debug)]
pub struct RegionEdges {
    feeds: Vec<Vec<Feed>>,
    exits: Vec<NodeId>,
}

impl RegionEdges {
    pub fn new(feeds: Vec<Vec<Feed>>, exits: Vec<NodeId>) -> Self {
        Self { feeds, exits }
    }
}

/// Runs one region's nodes, in topological order, on one thread.
///
/// The head is the only node whose inputs come from outside the region, so [RegionRunner::run]
/// hands it the bundle directly. Every other node's inputs are assembled from other nodes already
/// run in this same list, following the region's edges.
pub struct RegionRunner {
    nodes: Vec<(NodeId, StageBox)>,
    edges: RegionEdges,
    positions: HashMap<NodeId, usize>,
}

impl RegionRunner {
    pub fn new(nodes: Vec<(NodeId, StageBox)>, edges: RegionEdges) -> Self {
        let positions = nodes
            .iter()
            .enumerate()
            .map(|(position, (id, _))| (*id, position))
            .collect();

        Self {
            nodes,
            edges,
            positions,
        }
    }

    /// Signals the start of a chunk to every node in the region.
    pub fn start(&mut self, context: &mut FrameContext) -> Result<(), RunError> {
        for (id, stage) in &mut self.nodes {
            start_node(*id, stage, context)?;
        }

        Ok(())
    }

    /// Runs the region once over `bundle`, the head's inputs, returning what its exits produced.
    pub fn run(
        &mut self,
        context: &mut FrameContext,
        bundle: Vec<Option<StreamItem>>,
    ) -> Result<Vec<StreamItem>, RunError> {
        let mut outputs: Vec<Vec<StreamItem>> = Vec::with_capacity(self.nodes.len());

        let (head_id, head_stage) = &mut self.nodes[0];
        let head_output = run_node(*head_id, head_stage, context, Inputs::new(bundle))?;
        outputs.push(head_output);

        for index in 1..self.nodes.len() {
            let feeds = &self.edges.feeds[index - 1];
            let produced = run_fed(feeds, &outputs, &self.positions, &mut self.nodes[index], context)?;
            outputs.push(produced);
        }

        Ok(collect_exits(&self.edges.exits, &self.positions, &outputs))
    }

    /// Flushes every node in topological order, threading each flush through its consumers.
    ///
    /// A node downstream of the flushing one is run on whatever the upstream flush produced,
    /// exactly like a normal step, before its own flush is taken. That way anything the
    /// downstream node ends up holding because of the upstream flush is captured too.
    pub fn end(&mut self, context: &mut FrameContext) -> Result<Vec<StreamItem>, RunError> {
        let mut outputs: Vec<Vec<StreamItem>> = Vec::with_capacity(self.nodes.len());

        let (head_id, head_stage) = &mut self.nodes[0];
        let head_flush = end_node(*head_id, head_stage, context)?;
        outputs.push(head_flush);

        for index in 1..self.nodes.len() {
            let feeds = &self.edges.feeds[index - 1];
            let flushed = end_fed(feeds, &outputs, &self.positions, &mut self.nodes[index], context)?;
            outputs.push(flushed);
        }

        Ok(collect_exits(&self.edges.exits, &self.positions, &outputs))
    }
}

/// Runs one non-head node once per item its feeds have ready, in the order they arrived.
///
/// See [ready_count] for how many times that is.
fn run_fed(
    feeds: &[Feed],
    outputs: &[Vec<StreamItem>],
    positions: &HashMap<NodeId, usize>,
    node: &mut (NodeId, StageBox),
    context: &mut FrameContext,
) -> Result<Vec<StreamItem>, RunError> {
    let (id, stage) = node;
    let count = ready_count(*id, feeds, outputs, positions)?;

    let mut produced = Vec::new();
    for item_index in 0..count {
        let inputs = inputs_for(feeds, outputs, positions, item_index);
        let mut result = run_node(*id, stage, context, inputs)?;
        produced.append(&mut result);
    }

    Ok(produced)
}

/// Feeds a node whatever its producers flushed, then flushes the node itself.
fn end_fed(
    feeds: &[Feed],
    outputs: &[Vec<StreamItem>],
    positions: &HashMap<NodeId, usize>,
    node: &mut (NodeId, StageBox),
    context: &mut FrameContext,
) -> Result<Vec<StreamItem>, RunError> {
    let mut forwarded = run_fed(feeds, outputs, positions, node, context)?;
    let (id, stage) = node;
    let flushed = end_node(*id, stage, context)?;

    forwarded.extend(flushed);
    Ok(forwarded)
}

/// How many times a node fed by `feeds` is ready to run.
///
/// A node with no feeds runs once, since it needs no input at all. A node with one feed runs
/// once per item that feed made ready. A node with several feeds needs them to agree, since a
/// mismatch means the branches feeding it fell out of step, not that one of them should be
/// guessed at or silently dropped.
fn ready_count(
    node: NodeId,
    feeds: &[Feed],
    outputs: &[Vec<StreamItem>],
    positions: &HashMap<NodeId, usize>,
) -> Result<usize, RunError> {
    if feeds.is_empty() {
        return Ok(1);
    }

    let counts: Vec<usize> = feeds
        .iter()
        .map(|feed| {
            let position = positions[&feed.producer];
            outputs[position].len()
        })
        .collect();

    let first = counts[0];
    if counts.iter().all(|&count| count == first) {
        return Ok(first);
    }

    Err(RunError::MismatchedFeeds { node, counts })
}

/// Builds one `Inputs` from a node's feeds, taking the `item_index`th item each names.
fn inputs_for(
    feeds: &[Feed],
    outputs: &[Vec<StreamItem>],
    positions: &HashMap<NodeId, usize>,
    item_index: usize,
) -> Inputs {
    let width = feeds.iter().map(|feed| feed.slot + 1).max().unwrap_or(0);
    let mut slots: Vec<Option<StreamItem>> = vec![None; width];

    for feed in feeds {
        let position = positions[&feed.producer];
        slots[feed.slot] = Some(outputs[position][item_index].clone());
    }

    Inputs::new(slots)
}

/// Concatenates what every exit node produced, in the order the region names them.
fn collect_exits(
    exits: &[NodeId],
    positions: &HashMap<NodeId, usize>,
    outputs: &[Vec<StreamItem>],
) -> Vec<StreamItem> {
    let mut collected = Vec::new();

    for exit in exits {
        let position = positions[exit];
        collected.extend(outputs[position].iter().cloned());
    }

    collected
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use av_toolbox_core::{Filter, Sink, StageError, StageInstance, Transform};
    use av_toolbox_gpu::{
        ColourInfo,
        Frame,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        SampleFormat,
        Subsampling,
        testing_context,
    };

    use super::*;

    fn format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
        }
    }

    fn frame(context: &FrameContext, index: u64, chunk: u64) -> Frame {
        let meta = FrameMeta::at(index).in_chunk(chunk);
        context.alloc_frame(&format(), meta)
    }

    fn item(context: &FrameContext, index: u64, chunk: u64) -> StreamItem {
        StreamItem::Frame(frame(context, index, chunk))
    }

    /// A fresh context for a call, since `FrameContext` cannot be cloned or shared mutably
    /// across the several calls one test makes.
    fn context_of(_seed: &FrameContext) -> FrameContext {
        testing_context()
    }

    struct PassThrough;

    impl StageInstance for PassThrough {}

    impl Filter for PassThrough {
        fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
            inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
        }
    }

    /// Passes its input through with a fixed frame index, so two branches from the same
    /// producer can be told apart downstream.
    struct Retagging {
        index: u64,
    }

    impl StageInstance for Retagging {}

    impl Filter for Retagging {
        fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
            let mut frame = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
            frame.set_index(self.index);
            Ok(frame)
        }
    }

    /// Requires both of its two inputs, and returns the one at slot 0.
    struct Joining;

    impl StageInstance for Joining {}

    impl Filter for Joining {
        fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
            let left = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
            inputs
                .take_frame(1)
                .ok_or_else(|| StageError::msg("expected a frame in slot 1"))?;

            Ok(left)
        }
    }

    /// A filter that keeps a copy of what it is given and flushes it again when its chunk ends.
    struct Holding {
        held: Option<Frame>,
    }

    impl StageInstance for Holding {}

    impl Filter for Holding {
        fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
            let frame = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
            self.held = Some(frame.clone());
            Ok(frame)
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
        fn process(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<Frame, StageError> {
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
}
