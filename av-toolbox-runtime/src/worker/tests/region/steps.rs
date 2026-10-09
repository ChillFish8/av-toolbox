use super::*;

#[test]
fn a_single_node_region_passes_its_bundle_straight_through() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(PassThrough)))];
    let edges = RegionEdges::new(vec![], vec![head]);
    let mut runner = RegionRunner::new(nodes, edges);

    let flows = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 4, 0))])
        .expect("should run");
    let produced = items(flows);

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

    let flows = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 4, 0))])
        .expect("should run");
    let produced = items(flows);

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

    let flows = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 4, 0))])
        .expect("should run");
    let produced = items(flows);

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
        (retagged, StageBox::Filter(Box::new(Retagging { seconds: 8 }))),
        (
            sink,
            StageBox::Sink(Box::new(TimestampRecording { seen: seen.clone() })),
        ),
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

    assert_eq!(
        *seen.lock().expect("not poisoned"),
        vec![Some(Duration::from_secs(8)), None]
    );
}

#[test]
fn a_node_producing_several_items_runs_its_consumer_once_per_item() {
    let context = testing_context();
    let head = NodeId::new(0);
    let sink = NodeId::new(1);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let nodes = vec![
        (head, StageBox::Filter(Box::new(Multiply { copies: 3 }))),
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
        (head, StageBox::Filter(Box::new(Multiply { copies: 0 }))),
        (sink, StageBox::Sink(Box::new(Recording { seen: seen.clone() }))),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let edges = RegionEdges::new(feeds, vec![sink]);
    let mut runner = RegionRunner::new(nodes, edges);

    let flows = runner
        .run(&mut context_of(&context), vec![Some(item(&context, 0, 0))])
        .expect("should run");
    let produced = items(flows);

    assert!(produced.is_empty());
    assert!(seen.lock().expect("not poisoned").is_empty());
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
