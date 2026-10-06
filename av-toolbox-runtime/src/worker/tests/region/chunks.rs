use super::*;

#[test]
fn a_chunk_node_flushes_its_tail_before_the_next_chunk() {
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(Lagging::new(2))))];
    let chunked = HashSet::from([head]);
    let edges = RegionEdges::new(Vec::new(), vec![head]).chunked_on_layer_zero(chunked);
    let mut runner = RegionRunner::new(nodes, edges);

    let items = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1)];
    let steps = run_steps(&mut runner, &items);

    // Chunk 0 holds 0, 1 and 2. Its third frame releases 0, and the change to chunk 1 flushes 1 and 2
    // in that same step. Chunk 1's two frames never exceed the lag and come out at the end.
    let expected = vec![vec![], vec![], vec![0], vec![1, 2], vec![], vec![3, 4]];
    assert_eq!(steps, expected);
}

#[test]
fn a_chunk_node_is_started_again_for_each_chunk() {
    let head = NodeId::new(0);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = StartRecording { seen: seen.clone() };
    let nodes = vec![(head, StageBox::Sink(Box::new(sink)))];
    let chunked = HashSet::from([head]);
    let edges = RegionEdges::new(Vec::new(), vec![]).chunked_on_layer_zero(chunked);
    let mut runner = RegionRunner::new(nodes, edges);

    run_all(&mut runner, &[(0, 0), (1, 1), (2, 1), (3, 2)]);

    let started = u64::MAX;
    let expected = vec![started, 0, started, 1, 2, started, 3];
    assert_eq!(*seen.lock().expect("not poisoned"), expected);
}

#[test]
fn a_stream_node_is_never_restarted() {
    let head = NodeId::new(0);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = StartRecording { seen: seen.clone() };
    let nodes = vec![(head, StageBox::Sink(Box::new(sink)))];
    let edges = RegionEdges::new(Vec::new(), vec![]);
    let mut runner = RegionRunner::new(nodes, edges);

    run_all(&mut runner, &[(0, 0), (1, 1), (2, 1), (3, 2)]);

    let started = u64::MAX;
    let expected = vec![started, 0, 1, 2, 3];
    assert_eq!(*seen.lock().expect("not poisoned"), expected);
}

#[test]
fn chained_chunk_nodes_each_flush_in_turn() {
    let head = NodeId::new(0);
    let next = NodeId::new(1);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(Lagging::new(1)))),
        (next, StageBox::Filter(Box::new(Lagging::new(2)))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let chunked = HashSet::from([head, next]);
    let edges = RegionEdges::new(feeds, vec![next]).chunked_on_layer_zero(chunked);
    let mut runner = RegionRunner::new(nodes, edges);

    let items = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)];
    let steps = run_steps(&mut runner, &items);

    // The head's flush of frame 1 reaches `next` in chunk 0, so `next` flushes 0 and 1 as chunk 1 arrives.
    let expected = vec![vec![], vec![], vec![], vec![0, 1], vec![], vec![2, 3, 4]];
    assert_eq!(steps, expected);
}

#[test]
fn a_lagging_branch_flushes_its_chunk_before_the_join_pairs_the_next() {
    let context = testing_context();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let chunked = HashSet::from([NodeId::new(1)]);
    let (mut runner, _) = lagging_diamond_chunked(2, seen.clone(), chunked);

    runner.start(&mut context_of(&context)).expect("should start");

    for (index, chunk) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
        let bundle = vec![Some(item(&context, index, chunk))];
        runner.run(&mut context_of(&context), bundle).expect("should run");

        // Frame 2 opens chunk 1, which flushes the lagging branch's frames 0 and 1 at once.
        if index == 2 {
            assert_eq!(*seen.lock().expect("not poisoned"), vec![0, 0, 1, 1]);
        }
    }

    runner.end(&mut context_of(&context)).expect("should end");

    let expected = vec![0, 0, 1, 1, 2, 2, 3, 3];
    assert_eq!(*seen.lock().expect("not poisoned"), expected);
}

#[test]
fn a_chunk_nodes_flush_comes_ahead_of_what_the_new_chunk_produces() {
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(Holding { held: None })))];
    let chunked = HashSet::from([head]);
    let edges = RegionEdges::new(Vec::new(), vec![head]).chunked_on_layer_zero(chunked);
    let mut runner = RegionRunner::new(nodes, edges);

    let produced = run_all(&mut runner, &[(0, 0), (1, 0), (2, 1)]);

    // Frame 1 is flushed again as chunk 0 ends, ahead of frame 2, and frame 2 once more at the very end.
    assert_eq!(produced, vec![0, 1, 1, 2, 2]);
}

#[test]
fn a_runner_started_again_does_not_restart_on_its_first_chunk() {
    let head = NodeId::new(0);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = StartRecording { seen: seen.clone() };
    let nodes = vec![(head, StageBox::Sink(Box::new(sink)))];
    let chunked = HashSet::from([head]);
    let edges = RegionEdges::new(Vec::new(), vec![]).chunked_on_layer_zero(chunked);
    let mut runner = RegionRunner::new(nodes, edges);

    run_all(&mut runner, &[(0, 3)]);
    run_all(&mut runner, &[(1, 4)]);

    let started = u64::MAX;
    let expected = vec![started, 0, started, 1];
    assert_eq!(*seen.lock().expect("not poisoned"), expected);
}

#[test]
fn a_head_restarts_when_the_chunk_in_its_authority_slot_changes() {
    let context = testing_context();
    let head = NodeId::new(0);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = StartRecording { seen: seen.clone() };
    let nodes = vec![(head, StageBox::Sink(Box::new(sink)))];
    let chunked = HashSet::from([head]);
    let edges = RegionEdges::new(Vec::new(), vec![])
        .chunked_on_layer_zero(chunked)
        .with_authorities(HashMap::from([(head, 1)]));
    let mut runner = RegionRunner::new(nodes, edges);

    runner.start(&mut context_of(&context)).expect("should start");

    for (index, authority_chunk) in [(0, 0), (1, 1), (2, 1)] {
        let bundle = vec![
            Some(item(&context, index, 0)),
            Some(item(&context, index, authority_chunk)),
        ];
        runner.run(&mut context_of(&context), bundle).expect("should run");
    }

    let started = u64::MAX;
    let expected = vec![started, 0, started, 1, 2];
    assert_eq!(*seen.lock().expect("not poisoned"), expected);
}

#[test]
fn a_non_head_node_restarts_when_the_chunk_in_its_authority_slot_changes() {
    let head = NodeId::new(0);
    let rechunk = NodeId::new(1);
    let target = NodeId::new(2);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = StartRecording { seen: seen.clone() };
    let nodes = vec![
        (head, StageBox::Filter(Box::new(PassThrough))),
        (rechunk, StageBox::Filter(Box::new(Rechunking { switch_at: 1 }))),
        (target, StageBox::Sink(Box::new(sink))),
    ];
    let feeds = vec![
        vec![Feed {
            producer: head,
            slot: 0,
        }],
        vec![
            Feed {
                producer: head,
                slot: 0,
            },
            Feed {
                producer: rechunk,
                slot: 1,
            },
        ],
    ];
    let edges = RegionEdges::new(feeds, vec![])
        .chunked_on_layer_zero(HashSet::from([target]))
        .with_authorities(HashMap::from([(target, 1)]));
    let mut runner = RegionRunner::new(nodes, edges);

    run_all(&mut runner, &[(0, 0), (1, 0), (2, 0)]);

    let started = u64::MAX;
    let expected = vec![started, 0, started, 1, 2];
    assert_eq!(*seen.lock().expect("not poisoned"), expected);
}

#[test]
fn a_failing_restart_flush_names_the_chunk_it_was_flushing() {
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(FailingFlush)))];
    let chunked = HashSet::from([head]);
    let edges = RegionEdges::new(Vec::new(), vec![head]).chunked_on_layer_zero(chunked);
    let mut runner = RegionRunner::new(nodes, edges);
    let context = testing_context();

    runner.start(&mut context_of(&context)).expect("should start");

    let first = vec![Some(item(&context, 0, 0))];
    runner.run(&mut context_of(&context), first).expect("should run");

    let second = vec![Some(item(&context, 1, 1))];
    let error = runner
        .run(&mut context_of(&context), second)
        .expect_err("the flush should fail");

    assert_eq!(error.chunk(), Some(0));
}

/// Passes its input through, counting each time it is started.
struct StartCounting {
    starts: Arc<AtomicUsize>,
}

impl StageInstance for StartCounting {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl Filter for StartCounting {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_item)?;
        Ok(vec![frame])
    }
}

#[test]
fn a_node_restarts_on_its_own_layer() {
    let context = testing_context();
    let head = NodeId::new(0);
    let next = NodeId::new(1);
    let head_starts = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let counting = StartCounting {
        starts: head_starts.clone(),
    };
    let recording = StartRecording { seen: seen.clone() };
    let nodes = vec![
        (head, StageBox::Filter(Box::new(counting))),
        (next, StageBox::Sink(Box::new(recording))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let layers = HashMap::from([(head, LayerId(0)), (next, LayerId(1))]);
    let edges = RegionEdges::new(feeds, vec![])
        .with_chunked(HashSet::from([head, next]))
        .with_layers(layers);
    let mut runner = RegionRunner::new(nodes, edges);

    runner.start(&mut context_of(&context)).expect("should start");

    for (index, chunk) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
        let meta = FrameMeta::at(index)
            .in_layer(LayerId(0), 0)
            .in_layer(LayerId(1), chunk);
        let frame = context.alloc_frame(&format(), meta);
        let bundle = vec![Some(StreamItem::Frame(frame).into())];
        runner.run(&mut context_of(&context), bundle).expect("should run");
    }

    runner.end(&mut context_of(&context)).expect("should end");

    let started = u64::MAX;
    let expected = vec![started, 0, 1, started, 2, 3];
    assert_eq!(*seen.lock().expect("not poisoned"), expected);
    assert_eq!(head_starts.load(Ordering::SeqCst), 1);
}
