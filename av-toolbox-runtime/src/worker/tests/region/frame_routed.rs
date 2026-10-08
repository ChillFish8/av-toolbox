use super::*;

#[test]
fn a_filter_emitting_nothing_in_a_frame_routed_region_is_an_error() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(Lagging::new(1))))];
    let edges = RegionEdges::new(Vec::new(), vec![head]).frame_routed();
    let mut runner = RegionRunner::new(nodes, edges);

    let bundle = vec![Some(item(&context, 0, 0))];
    let result = runner.run(&mut context_of(&context), bundle);

    let Err(RunError::HeldInFrameRegion { node }) = result else {
        panic!("a filter emitting nothing in a frame routed region should have been an error");
    };

    assert_eq!(node, head);
}

#[test]
fn a_filter_emitting_several_frames_in_a_frame_routed_region_runs() {
    let context = testing_context();
    let head = NodeId::new(0);
    let nodes = vec![(head, StageBox::Filter(Box::new(Doubling)))];
    let edges = RegionEdges::new(Vec::new(), vec![head]).frame_routed();
    let mut runner = RegionRunner::new(nodes, edges);

    let bundle = vec![Some(item(&context, 3, 0))];
    let produced = runner.run(&mut context_of(&context), bundle).expect("should run");

    let indices: Vec<u64> = produced.iter().map(|item| item.meta().index).collect();
    assert_eq!(indices, vec![6, 7]);
}

#[test]
fn ending_a_frame_routed_region_does_not_check_flushes() {
    let context = testing_context();
    let head = NodeId::new(0);
    let second = NodeId::new(1);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(PassThrough))),
        (second, StageBox::Filter(Box::new(PassThrough))),
    ];
    let feed = Feed {
        producer: head,
        slot: 0,
    };
    let edges = RegionEdges::new(vec![vec![feed]], vec![second]).frame_routed();
    let mut runner = RegionRunner::new(nodes, edges);

    let bundle = vec![Some(item(&context, 0, 0))];
    runner.run(&mut context_of(&context), bundle).expect("should run");
    let flushed = runner.end(&mut context_of(&context), None).expect("should end");

    assert!(flushed.is_empty());
}
