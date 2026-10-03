use super::*;

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
