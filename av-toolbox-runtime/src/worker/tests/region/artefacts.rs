use av_toolbox_core::ArtefactData;
use av_toolbox_graph::CompiledGraph;
use serde_json::json;

use super::*;
use crate::artefacts::ArtefactCollector;
use crate::engine::tests::{
    ArtefactingArgs,
    ArtefactingInstance,
    artefacting_spec,
    collecting_spec,
    planned_graph,
    sink_spec,
    source_spec,
};

fn counting_graph() -> (CompiledGraph, NodeId) {
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        artefacting_spec("count", "input", 1, true, "counting"),
        sink_spec("out", "count"),
        collecting_spec("files", &["filter.count.frames"]),
    ];
    let (graph, _) = planned_graph(&specs);
    let count = graph
        .nodes()
        .iter()
        .find(|node| node.name() == "count")
        .expect("a count node")
        .id();

    (graph, count)
}

fn counting_stage() -> StageBox {
    let args = ArtefactingArgs::default();
    let built = ArtefactingInstance::from_args(args);

    StageBox::Filter(Box::new(built))
}

fn merged_frames(collector: &ArtefactCollector, count: NodeId) -> ArtefactData {
    let mut merged = collector.merge().expect("merges");
    let artefact = merged.remove(&(count, "frames")).expect("a frames artefact");

    artefact.data
}

#[test]
fn a_chunked_head_hands_over_each_chunk_as_it_ends() {
    let (graph, count) = counting_graph();
    let collector = Arc::new(ArtefactCollector::new(&graph));
    let nodes = vec![(count, counting_stage())];
    let chunked = HashSet::from([count]);
    let edges = RegionEdges::new(Vec::new(), vec![count]).chunked_on_layer_zero(chunked);
    let mut runner = RegionRunner::new(nodes, edges).collecting(collector.clone());

    run_all(&mut runner, &[(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)]);

    let expected = ArtefactData::Structured(json!([[0, 2], [1, 3]]));
    assert_eq!(merged_frames(&collector, count), expected);
}

#[test]
fn a_chunked_fed_node_hands_over_each_chunk_as_it_ends() {
    let (graph, count) = counting_graph();
    let collector = Arc::new(ArtefactCollector::new(&graph));
    let head = NodeId::new(1000);
    let nodes = vec![
        (head, StageBox::Filter(Box::new(Lagging::new(1)))),
        (count, counting_stage()),
    ];
    let feeds = vec![vec![Feed {
        producer: head,
        slot: 0,
    }]];
    let chunked = HashSet::from([head, count]);
    let edges = RegionEdges::new(feeds, vec![count]).chunked_on_layer_zero(chunked);
    let mut runner = RegionRunner::new(nodes, edges).collecting(collector.clone());

    run_all(&mut runner, &[(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)]);

    let expected = ArtefactData::Structured(json!([[0, 2], [1, 3]]));
    assert_eq!(merged_frames(&collector, count), expected);
}

#[test]
fn a_stream_node_hands_over_once_without_a_chunk() {
    let (graph, count) = counting_graph();
    let collector = Arc::new(ArtefactCollector::new(&graph));
    let nodes = vec![(count, counting_stage())];
    let edges = RegionEdges::new(Vec::new(), vec![count]);
    let mut runner = RegionRunner::new(nodes, edges).collecting(collector.clone());

    run_all(&mut runner, &[(0, 0), (1, 0), (2, 1)]);

    let expected = ArtefactData::Structured(json!([[null, 3]]));
    assert_eq!(merged_frames(&collector, count), expected);
}
