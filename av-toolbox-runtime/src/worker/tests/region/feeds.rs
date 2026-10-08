use super::*;

/// `left` multiplies head's one item into 2 and `right` into 1, so `join` runs once and one of `left`'s
/// items is never paired.
#[test]
fn feeds_that_still_disagree_when_the_region_ends_are_an_error() {
    let context = testing_context();
    let head = NodeId::new(0);
    let left = NodeId::new(1);
    let right = NodeId::new(2);
    let join = NodeId::new(3);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(Multiply { copies: 1 }))),
        (left, StageBox::Filter(Box::new(Multiply { copies: 2 }))),
        (right, StageBox::Filter(Box::new(Multiply { copies: 1 }))),
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

    runner
        .run(&mut context_of(&context), vec![Some(item(&context, 0, 0))])
        .expect("a step whose feeds lag each other still runs");

    let result = runner.end(&mut context_of(&context), None);

    let Err(RunError::UnpairedFeeds { node, leftover }) = result else {
        panic!("feeds left holding items at the end should have been an error");
    };

    assert_eq!(node, join);
    assert_eq!(leftover, vec![1, 0]);
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
        (head, StageBox::Filter(Box::new(Multiply { copies: 1 }))),
        (left, StageBox::Filter(Box::new(Multiply { copies: 2 }))),
        (right, StageBox::Filter(Box::new(Multiply { copies: 2 }))),
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
fn a_join_waits_for_its_lagging_feed_and_pairs_by_index() {
    let context = testing_context();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (mut runner, _) = lagging_diamond(2, seen.clone());

    for index in 0..5 {
        let bundle = vec![Some(item(&context, index, 0))];
        runner.run(&mut context_of(&context), bundle).expect("should run");
    }

    runner.end(&mut context_of(&context), None).expect("should end");

    let expected = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    assert_eq!(*seen.lock().expect("not poisoned"), expected);
}

#[test]
fn a_join_runs_nothing_while_its_lagging_feed_fills() {
    let context = testing_context();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (mut runner, _) = lagging_diamond(2, seen.clone());

    for index in 0..2 {
        let bundle = vec![Some(item(&context, index, 0))];
        runner.run(&mut context_of(&context), bundle).expect("should run");
    }

    assert!(seen.lock().expect("not poisoned").is_empty());
}

#[test]
fn paired_items_carrying_different_indices_are_an_error() {
    let context = testing_context();
    let head = NodeId::new(0);
    let left = NodeId::new(1);
    let right = NodeId::new(2);
    let join = NodeId::new(3);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(PassThrough))),
        (left, StageBox::Filter(Box::new(PassThrough))),
        (right, StageBox::Filter(Box::new(Reindexing { index: 8 }))),
        (join, StageBox::Filter(Box::new(Joining))),
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
    let feeds = vec![vec![head_feed], vec![head_feed], join_feeds];
    let edges = RegionEdges::new(feeds, vec![join]);
    let mut runner = RegionRunner::new(nodes, edges);

    let result = runner.run(&mut context_of(&context), vec![Some(item(&context, 4, 0))]);

    let Err(RunError::OutOfStepFeeds { node, indices }) = result else {
        panic!("items for different frames should not have been paired");
    };

    assert_eq!(node, join);
    assert_eq!(indices, vec![4, 8]);
}
