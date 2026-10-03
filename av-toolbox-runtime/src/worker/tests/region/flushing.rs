use super::*;

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
