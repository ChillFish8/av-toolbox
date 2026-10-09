use av_toolbox_core::{MetaInputs, MetaSink};
use av_toolbox_gpu::MetaRecord;

use super::*;

/// Passes each frame through, recording a `mark` for it, and a `chunk` counting them when its chunk ends.
#[derive(Default)]
struct Marking {
    marked: i64,
}

impl StageInstance for Marking {}

impl Filter for Marking {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        mark(context, &frame);
        self.marked += 1;

        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let marked = std::mem::take(&mut self.marked);
        context.record("chunk").attach("marked", MetaValue::Int(marked));

        Ok(Vec::new())
    }
}

/// The same as [Marking], also flushing a copy of the last frame it passed when its chunk ends.
#[derive(Default)]
struct FlushingMarking {
    marked: i64,
    last: Option<Frame>,
}

impl StageInstance for FlushingMarking {}

impl Filter for FlushingMarking {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        mark(context, &frame);
        self.marked += 1;
        self.last = Some(frame.clone());

        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let marked = std::mem::take(&mut self.marked);
        context.record("chunk").attach("marked", MetaValue::Int(marked));

        let flushed = self.last.take();

        Ok(flushed.into_iter().collect())
    }
}

fn mark(context: &mut FrameContext, frame: &Frame) {
    let index = frame.meta().index as i64;
    context.record("mark").attach("index", MetaValue::Int(index));
}

/// Swallows each frame, recording a `mark` in its place.
struct Swallowing;

impl StageInstance for Swallowing {}

impl Filter for Swallowing {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        mark(context, &frame);

        Ok(Vec::new())
    }
}

/// Passes each frame through, placing a record whose kind is not well formed.
struct Malformed;

impl StageInstance for Malformed {}

impl Filter for Malformed {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        context.push_record_unchecked(MetaRecord::new("Bad"));

        Ok(vec![frame])
    }
}

/// Writes down every frame and record that reaches it, in order.
struct Watching {
    seen: Arc<Mutex<Vec<String>>>,
}

impl StageInstance for Watching {}

impl MetaSink for Watching {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
        let mut seen = self.seen.lock().expect("not poisoned");

        for slot in 0..inputs.len() {
            if let Some(item) = inputs.take(slot) {
                seen.push(format!("frame {}", item.meta().index));
            }
        }

        Ok(())
    }

    fn consume_record(&mut self, _context: &mut FrameContext, record: &MetaRecord) -> Result<(), StageError> {
        let described = described_record(record);
        self.seen.lock().expect("not poisoned").push(described);

        Ok(())
    }
}

/// A record's kind, followed by the first path it was stamped with when it has one.
fn described_record(record: &MetaRecord) -> String {
    let first_path = record.attachments().stages().next();

    match first_path {
        Some((path, _)) => format!("{} {path}", record.kind()),
        None => record.kind().to_string(),
    }
}

/// `flows` written down the same way [Watching] writes down what reaches it.
fn described_flows(flows: &[Flow]) -> Vec<String> {
    let mut described = Vec::with_capacity(flows.len());

    for flow in flows {
        let entry = match flow {
            Flow::Item(item) => format!("frame {}", item.meta().index),
            Flow::Record(record) => described_record(record),
        };
        described.push(entry);
    }

    described
}

/// `stages` named by their paths, each feeding the next, with a [Watching] exit at the end writing into `seen`.
fn watched_chain(
    stages: Vec<(&str, StageBox)>,
    seen: &Arc<Mutex<Vec<String>>>,
) -> (Vec<(NodeId, StageBox)>, RegionEdges) {
    let mut nodes = Vec::with_capacity(stages.len() + 1);
    let mut named = Vec::with_capacity(stages.len());

    for (position, (path, stage)) in stages.into_iter().enumerate() {
        let node = NodeId::new(position);
        named.push((node, path));
        nodes.push((node, stage));
    }

    let watching = Watching { seen: seen.clone() };
    let sink = NodeId::new(nodes.len());
    nodes.push((sink, StageBox::MetaSink(Box::new(watching))));

    let feeds = (1..nodes.len())
        .map(|position| {
            vec![Feed {
                producer: nodes[position - 1].0,
                slot: 0,
            }]
        })
        .collect();
    let edges = RegionEdges::new(feeds, vec![sink]).with_paths(paths(&named));

    (nodes, edges)
}

/// Runs one frame per `(index, chunk)` through `runner`, then ends it, returning everything its exits produced.
fn run_frames(runner: &mut RegionRunner, frames: &[(u64, u64)]) -> Vec<Flow> {
    let context = testing_context();
    let mut stage_context = context_of(&context);
    let mut produced = Vec::new();

    runner.start(&mut stage_context).expect("should start");

    for &(index, chunk) in frames {
        let bundle = vec![Some(item(&context, index, chunk))];
        let flows = runner.run(&mut stage_context, bundle).expect("should run");
        produced.extend(flows);
    }

    let flushed = runner.end(&mut stage_context, None).expect("should end");
    produced.extend(flushed);

    produced
}

fn watched(seen: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    seen.lock().expect("not poisoned").clone()
}

#[test]
fn a_record_follows_the_output_of_the_call_that_made_it() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stages = vec![("filter.marking", StageBox::Filter(Box::<Marking>::default()))];
    let (nodes, edges) = watched_chain(stages, &seen);
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = run_frames(&mut runner, &[(0, 0), (1, 0)]);

    let expected = vec![
        "frame 0",
        "mark filter.marking",
        "frame 1",
        "mark filter.marking",
        "chunk filter.marking",
    ];
    assert_eq!(watched(&seen), expected);
    assert!(produced.is_empty());
}

#[test]
fn an_end_of_stream_record_follows_the_end_flush() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let flushing = StageBox::Filter(Box::<FlushingMarking>::default());
    let (nodes, edges) = watched_chain(vec![("filter.marking", flushing)], &seen);
    let mut runner = RegionRunner::new(nodes, edges);

    run_frames(&mut runner, &[(0, 0), (1, 0)]);

    let expected = vec![
        "frame 0",
        "mark filter.marking",
        "frame 1",
        "mark filter.marking",
        "frame 1",
        "chunk filter.marking",
    ];
    assert_eq!(watched(&seen), expected);
}

#[test]
fn a_restart_flush_keeps_its_records_ahead_of_the_new_chunk() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let head = NodeId::new(0);
    let stages = vec![("filter.marking", StageBox::Filter(Box::<Marking>::default()))];
    let (nodes, edges) = watched_chain(stages, &seen);
    let chunked_edges = edges.chunked_on_layer_zero(HashSet::from([head]));
    let mut runner = RegionRunner::new(nodes, chunked_edges);

    run_frames(&mut runner, &[(0, 0), (1, 0), (2, 1)]);

    let expected = vec![
        "frame 0",
        "mark filter.marking",
        "frame 1",
        "mark filter.marking",
        "chunk filter.marking",
        "frame 2",
        "mark filter.marking",
        "chunk filter.marking",
    ];
    assert_eq!(watched(&seen), expected);
}

/// The delaying stage hands frame 0 on only when frame 1 arrives, so the first mark reaches the sink
/// before any frame does.
#[test]
fn a_record_passes_a_stage_that_holds_frames_back() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stages = vec![
        ("filter.marking", StageBox::Filter(Box::<Marking>::default())),
        (
            "filter.delay",
            StageBox::Filter(Box::new(Delaying { held: None })),
        ),
    ];
    let (nodes, edges) = watched_chain(stages, &seen);
    let mut runner = RegionRunner::new(nodes, edges);

    run_frames(&mut runner, &[(0, 0), (1, 0)]);

    let expected = vec![
        "mark filter.marking",
        "frame 0",
        "mark filter.marking",
        "chunk filter.marking",
        "frame 1",
    ];
    assert_eq!(watched(&seen), expected);
}

#[test]
fn a_record_reaching_a_join_by_two_feeds_is_forwarded_once() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let head = NodeId::new(0);
    let left = NodeId::new(1);
    let right = NodeId::new(2);
    let join = NodeId::new(3);
    let watching = NodeId::new(4);
    let nodes = vec![
        (head, StageBox::Filter(Box::<Marking>::default())),
        (left, StageBox::Filter(Box::new(PassThrough))),
        (right, StageBox::Filter(Box::new(PassThrough))),
        (join, StageBox::Filter(Box::new(Joining))),
        (
            watching,
            StageBox::MetaSink(Box::new(Watching { seen: seen.clone() })),
        ),
    ];
    let head_feed = Feed {
        producer: head,
        slot: 0,
    };
    let join_feeds = vec![
        Feed {
            producer: left,
            slot: 0,
        },
        Feed {
            producer: right,
            slot: 1,
        },
    ];
    let watching_feed = Feed {
        producer: join,
        slot: 0,
    };
    let feeds = vec![vec![head_feed], vec![head_feed], join_feeds, vec![watching_feed]];
    let named = paths(&[
        (head, "filter.marking"),
        (left, "filter.left"),
        (right, "filter.right"),
        (join, "filter.join"),
    ]);
    let edges = RegionEdges::new(feeds, vec![watching]).with_paths(named);
    let mut runner = RegionRunner::new(nodes, edges);

    run_frames(&mut runner, &[(0, 0), (1, 0)]);

    let expected = vec![
        "frame 0",
        "mark filter.marking",
        "frame 1",
        "mark filter.marking",
        "chunk filter.marking",
    ];
    assert_eq!(watched(&seen), expected);
}

#[test]
fn a_record_offered_to_the_region_leaves_ahead_of_the_head_output() {
    let context = testing_context();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stages = vec![("filter.marking", StageBox::Filter(Box::<Marking>::default()))];
    let (nodes, edges) = watched_chain(stages, &seen);
    let mut runner = RegionRunner::new(nodes, edges);
    let mut stage_context = context_of(&context);

    runner.start(&mut stage_context).expect("should start");
    runner.offer_records(vec![MetaRecord::new("outside")]);

    let bundle = vec![Some(item(&context, 0, 0))];
    runner.run(&mut stage_context, bundle).expect("should run");

    let expected = vec!["outside", "frame 0", "mark filter.marking"];
    assert_eq!(watched(&seen), expected);
}

#[test]
fn a_record_offered_before_the_end_leaves_after_the_end_flush() {
    let context = testing_context();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let flushing = StageBox::Filter(Box::<FlushingMarking>::default());
    let (nodes, edges) = watched_chain(vec![("filter.marking", flushing)], &seen);
    let mut runner = RegionRunner::new(nodes, edges);
    let mut stage_context = context_of(&context);

    runner.start(&mut stage_context).expect("should start");

    let bundle = vec![Some(item(&context, 0, 0))];
    runner.run(&mut stage_context, bundle).expect("should run");

    runner.offer_records(vec![MetaRecord::new("outside")]);
    runner.end(&mut stage_context, None).expect("should end");

    let expected = vec![
        "frame 0",
        "mark filter.marking",
        "frame 0",
        "chunk filter.marking",
        "outside",
    ];
    assert_eq!(watched(&seen), expected);
}

#[test]
fn a_record_offered_twice_leaves_once() {
    let context = testing_context();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stages = vec![("filter.marking", StageBox::Filter(Box::<Marking>::default()))];
    let (nodes, edges) = watched_chain(stages, &seen);
    let mut runner = RegionRunner::new(nodes, edges);
    let mut stage_context = context_of(&context);
    let outside = MetaRecord::new("outside");

    runner.start(&mut stage_context).expect("should start");
    runner.offer_records(vec![outside.clone(), outside]);

    let bundle = vec![Some(item(&context, 0, 0))];
    runner.run(&mut stage_context, bundle).expect("should run");

    let expected = vec!["outside", "frame 0", "mark filter.marking"];
    assert_eq!(watched(&seen), expected);
}

#[test]
fn a_malformed_kind_fails_the_run_naming_the_node() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(Malformed)))];
    let edges = RegionEdges::new(Vec::new(), vec![head]);
    let mut runner = RegionRunner::new(nodes, edges);

    let bundle = vec![Some(item(&context, 0, 0))];
    let result = runner.run(&mut context_of(&context), bundle);

    let Err(RunError::Stage { node, source }) = result else {
        panic!("a record with a malformed kind should have failed the run");
    };

    let message = source.to_string();
    assert_eq!(node, head);
    assert!(message.contains("\"Bad\""), "{message}");
    assert!(message.contains("a-z, 0-9 and _"), "{message}");
}

#[test]
fn records_only_from_a_stage_in_a_frame_routed_region_are_not_held_in_the_region() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::<Marking>::default()))];
    let edges = RegionEdges::new(Vec::new(), vec![head])
        .with_paths(paths(&[(head, "filter.marking")]))
        .frame_routed();
    let mut runner = RegionRunner::new(nodes, edges);

    let bundle = vec![Some(item(&context, 3, 0))];
    let produced = runner.run(&mut context_of(&context), bundle).expect("should run");

    assert_eq!(described_flows(&produced), vec!["frame 3", "mark filter.marking"]);
}

#[test]
fn a_stage_returning_only_records_in_a_frame_routed_region_still_holds_its_frame() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(Swallowing)))];
    let edges = RegionEdges::new(Vec::new(), vec![head]).frame_routed();
    let mut runner = RegionRunner::new(nodes, edges);

    let bundle = vec![Some(item(&context, 0, 0))];
    let result = runner.run(&mut context_of(&context), bundle);

    let Err(RunError::HeldInFrameRegion { node }) = result else {
        panic!("a filter returning only records in a frame routed region should have been an error");
    };

    assert_eq!(node, head);
}
