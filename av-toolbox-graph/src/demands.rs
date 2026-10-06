use av_toolbox_core::{
    AnyDemand,
    DemandKind,
    InputDemand,
    LayerDemand,
    PipelineRegistry,
    Placement,
    PreparedStage,
    StageKind,
};
use av_toolbox_gpu::LayerId;

use crate::graph::{CompiledGraph, EdgeId, Node, NodeId, NodeRole, ProviderNode};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// Where the graph placed one provider, and the layers it serves there.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderPlacement {
    pub kind: DemandKind,
    pub layers: Vec<ServedLayer>,
}

/// One layer a provider serves.
#[derive(Debug, Clone, PartialEq)]
pub struct ServedLayer {
    pub layer: LayerId,
    pub demand: AnyDemand,
    /// The demand as `PipelineRegistry::describe` puts it, such as `natural`.
    pub description: String,
    /// The TOML names of the stages that demanded this layer.
    pub demanded_by: Vec<String>,
}

/// One demand a stage made, as the pass settled it.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeDemand {
    pub input: &'static str,
    pub kind: DemandKind,
    pub description: String,
    pub layer: LayerId,
    /// Whether this demand is the stage's chunk request.
    pub chunks: bool,
}

/// One demand a stage made on an input it has bound.
struct Asked {
    node: NodeId,
    demand: InputDemand,
    chunks: bool,
}

/// One demand traced back to its source and given its layer.
struct Traced {
    asked: Asked,
    source: NodeId,
    /// The edges from the source to the demanding node, source end first.
    path: Vec<EdgeId>,
    layer: LayerId,
}

/// Every traced demand of one kind reaching one source, which one provider serves.
struct Group {
    source: NodeId,
    kind: DemandKind,
    /// Indices into the traced demands, in the graph's order.
    members: Vec<usize>,
}

/// Inserts a provider for every source some stage demands something of, and gives each demanding node its layer.
///
/// Each demand is traced upstream to its source, following a node's chunk authority where it has
/// several inputs. Demands are grouped by source and kind, every distinct demand becomes one
/// layer, and one provider per group is placed where its registration asks. A demanding node
/// records its demands, and routes on the layer of its chunk request.
pub fn satisfy_demands(graph: &mut CompiledGraph, registry: &PipelineRegistry) -> Result<(), Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let asked = collect_demands(graph, &prepared);
    if asked.is_empty() {
        return Ok(());
    }

    let traced = trace_demands(graph, registry, &prepared, asked, &mut diagnostics);
    let groups = group_by_source(&traced);
    check_placements(graph, &traced, &groups, &mut diagnostics);

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    record_demands(graph, registry, &traced);

    for group in &groups {
        place_provider(graph, registry, &prepared, &traced, group);
    }

    Ok(())
}

/// Adds a note to `diagnostic` naming every stage whose demand placed `node`, when it is a provider.
pub(crate) fn note_demanders(graph: &CompiledGraph, node: &Node, diagnostic: Diagnostic) -> Diagnostic {
    let NodeRole::Provider(placement) = node.role() else {
        return diagnostic;
    };

    let mut noted = diagnostic;
    for served in &placement.layers {
        for name in &served.demanded_by {
            let span = graph
                .nodes()
                .iter()
                .find(|candidate| candidate.name() == name)
                .map_or(node.span(), Node::span);
            let message = format!(
                "this provider serves `{name}`, which demands {} ({})",
                placement.kind.name(),
                served.description,
            );

            noted = noted.with_note(message, span);
        }
    }

    noted
}

/// Every demand every stage makes on an input it has bound, in the graph's order.
fn collect_demands(graph: &CompiledGraph, prepared: &[Option<PreparedStage>]) -> Vec<Asked> {
    let mut asked = Vec::new();

    for id in graph.order() {
        let Some(stage) = prepared[id.index()].as_ref() else {
            continue;
        };

        let node = graph.node(*id);
        let parallelism = stage.parallelism();
        let chunk_demand = parallelism.demand();

        for demand in stage.demands() {
            let Some(edge) = bound_edge(node, demand.input) else {
                continue;
            };

            let on_authority = stage.chunk_authority() == Some(node.inputs()[edge].slot());
            let chunks = on_authority && chunk_demand == Some(&demand.demand);

            asked.push(Asked {
                node: *id,
                demand,
                chunks,
            });
        }
    }

    asked
}

/// Traces every demand to its source and numbers a layer per distinct demand of each source.
///
/// A demand of a kind nothing serves, or one that reaches no source, is reported and dropped.
fn trace_demands(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    prepared: &[Option<PreparedStage>],
    asked: Vec<Asked>,
    diagnostics: &mut Diagnostics,
) -> Vec<Traced> {
    let mut layers: Vec<(NodeId, AnyDemand, LayerId)> = Vec::new();
    let mut traced = Vec::new();

    for each in asked {
        let node = graph.node(each.node);
        let kind = each.demand.demand.kind();

        if let Some(problem) = unserved(registry, kind) {
            let message = format!(
                "`{}` demands {} on `{}`, {problem}",
                node.name(),
                kind.name(),
                each.demand.input,
            );
            diagnostics.push(error_at(node, DiagnosticCode::UnsatisfiedDemand, message));
            continue;
        }

        let Some(path) = trace_to_source(graph, prepared, each.node, each.demand.input) else {
            let message = format!(
                "`{}` demands {} on `{}`, which traces to no source",
                node.name(),
                kind.name(),
                each.demand.input,
            );
            diagnostics.push(error_at(node, DiagnosticCode::UnsatisfiedDemand, message));
            continue;
        };

        let source = source_of(graph, path[0]);
        let layer = layer_for(&mut layers, source, &each.demand.demand);

        traced.push(Traced {
            asked: each,
            source,
            path,
            layer,
        });
    }

    traced
}

/// Why nothing can serve a demand of `kind`, or `None` when a provider can.
fn unserved(registry: &PipelineRegistry, kind: DemandKind) -> Option<&'static str> {
    if registry.demand_kind(kind.name()).is_none() {
        return Some("which is not a registered demand kind");
    }

    if registry.provider_for(kind).is_none() {
        return Some("which no provider serves");
    }

    None
}

/// The layer serving `demand` at `source`, numbering a new one when it is the first of its kind.
fn layer_for(layers: &mut Vec<(NodeId, AnyDemand, LayerId)>, source: NodeId, demand: &AnyDemand) -> LayerId {
    let existing = layers
        .iter()
        .find(|(at, served, _)| *at == source && served == demand);
    if let Some((_, _, layer)) = existing {
        return *layer;
    }

    let layer = LayerId(layers.len() as u32);
    layers.push((source, demand.clone(), layer));
    layer
}

/// The edges a demand on `input` of `node` travels along from its source, source end first.
///
/// A node with one input is followed through it, and a node with several through its chunk
/// authority. The trace ends at a node with no inputs.
fn trace_to_source(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    node: NodeId,
    input: &str,
) -> Option<Vec<EdgeId>> {
    let start = bound_edge(graph.node(node), input)?;
    let mut path = vec![EdgeId::new(node, start)];
    let mut current = graph.node(node).inputs()[start].source();

    loop {
        let producer = graph.node(current);
        let followed = match producer.inputs().len() {
            0 => break,
            1 => 0,
            _ => authority_edge(producer, prepared)?,
        };

        path.push(EdgeId::new(current, followed));
        current = producer.inputs()[followed].source();
    }

    path.reverse();
    Some(path)
}

/// The position of the edge bound to `input`, by its signature param or its variadic name.
fn bound_edge(node: &Node, input: &str) -> Option<usize> {
    node.inputs().iter().position(|edge| match edge.param() {
        Some(param) => param == input,
        None => edge.name() == Some(input),
    })
}

/// The position of the edge filling `node`'s chunk authority.
fn authority_edge(node: &Node, prepared: &[Option<PreparedStage>]) -> Option<usize> {
    let stage = prepared.get(node.id().index())?.as_ref()?;
    let slot = stage.chunk_authority()?;

    node.inputs().iter().position(|edge| edge.slot() == slot)
}

fn source_of(graph: &CompiledGraph, edge: EdgeId) -> NodeId {
    graph.node(edge.node()).inputs()[edge.index()].source()
}

/// Groups traced demands by the source they reach and their kind, in the order first seen.
fn group_by_source(traced: &[Traced]) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();

    for (index, each) in traced.iter().enumerate() {
        let kind = each.asked.demand.demand.kind();
        let existing = groups
            .iter_mut()
            .find(|group| group.source == each.source && group.kind == kind);

        match existing {
            Some(group) => group.members.push(index),
            None => groups.push(Group {
                source: each.source,
                kind,
                members: vec![index],
            }),
        }
    }

    groups
}

/// Refuses groups whose demands name different stages to go after, or one off a demand's path.
fn check_placements(
    graph: &CompiledGraph,
    traced: &[Traced],
    groups: &[Group],
    diagnostics: &mut Diagnostics,
) {
    for group in groups {
        let Some(first) = group
            .members
            .iter()
            .find(|member| after_of(&traced[**member]).is_some())
        else {
            continue;
        };

        let first = &traced[*first];
        let after = after_of(first).expect("the first member names a stage");
        let first_name = graph.node(first.asked.node).name();

        for member in &group.members {
            let each = &traced[*member];
            let node = graph.node(each.asked.node);
            let kind = group.kind.name();

            if let Some(other) = after_of(each)
                && other != after
            {
                let message = format!(
                    "`{}` asks for {kind} after `{other}`, but `{first_name}` asks for them after \
                     `{after}`, and one provider serves both",
                    node.name(),
                );
                diagnostics.push(error_at(node, DiagnosticCode::ConflictingPlacement, message));
                continue;
            }

            let passes = each
                .path
                .iter()
                .any(|edge| graph.node(source_of(graph, *edge)).name() == after);
            if passes {
                continue;
            }

            let message = format!(
                "`{}` needs {kind} from after `{after}`, which its `{}` input does not pass through",
                node.name(),
                each.asked.demand.input,
            );
            diagnostics.push(error_at(node, DiagnosticCode::UnknownAfter, message));
        }
    }
}

/// The stage a demand asks its provider to go after, read from its request's `after` field.
fn after_of(traced: &Traced) -> Option<&str> {
    let request = traced.asked.demand.demand.raw();

    request.get("after")?.as_str()
}

/// Gives every demanding node its demands, and the layer of its chunk request.
fn record_demands(graph: &mut CompiledGraph, registry: &PipelineRegistry, traced: &[Traced]) {
    let mut settled: Vec<(NodeId, Vec<NodeDemand>)> = Vec::new();

    for each in traced {
        let demand = &each.asked.demand.demand;
        let node_demand = NodeDemand {
            input: each.asked.demand.input,
            kind: demand.kind(),
            description: registry.describe(demand),
            layer: each.layer,
            chunks: each.asked.chunks,
        };

        if each.asked.chunks {
            graph.node_mut(each.asked.node).set_layer(each.layer);
        }

        match settled.iter_mut().find(|(node, _)| *node == each.asked.node) {
            Some((_, demands)) => demands.push(node_demand),
            None => settled.push((each.asked.node, vec![node_demand])),
        }
    }

    for (node, demands) in settled {
        graph.node_mut(node).set_demands(demands);
    }
}

/// Inserts the provider serving `group`, where its placement or an `after` puts it.
///
/// Its demands are traced again first, so a provider an earlier group placed is on their paths.
fn place_provider(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    prepared: &[Option<PreparedStage>],
    traced: &[Traced],
    group: &Group,
) {
    let (identifier, placement) = registry
        .provider_for(group.kind)
        .expect("every grouped kind has a provider");

    let mut paths = Vec::with_capacity(group.members.len());
    for member in &group.members {
        let asked = &traced[*member].asked;
        let path = trace_to_source(graph, prepared, asked.node, asked.demand.input)
            .expect("a demand traced once still traces");
        paths.push(path);
    }

    let after = group.members.iter().find_map(|member| after_of(&traced[*member]));
    let (producer, moved) = match (after, placement) {
        (Some(name), _) => after_named(graph, &paths, name),
        (None, Placement::AfterSource) => after_source(graph, group.source),
        (None, Placement::Shared) => before_common_node(graph, &paths, group.source),
    };

    let layers = served_layers(graph, registry, traced, group);
    let demands: Vec<LayerDemand> = layers
        .iter()
        .map(|served| LayerDemand {
            layer: served.layer,
            demand: served.demand.clone(),
        })
        .collect();
    let args = serde_json::to_value(demands).expect("layer demands serialise");

    let provider = ProviderNode {
        identifier: identifier.to_string(),
        input: provider_input(registry, identifier),
        source: graph.node(group.source).name().to_string(),
        args,
        placement: ProviderPlacement {
            kind: group.kind,
            layers,
        },
    };

    graph.push_provider(producer, provider, &moved);
    graph.finish_edit();
}

/// After the stage named `name`, taking the edges leaving it on every demanding path.
fn after_named(graph: &CompiledGraph, paths: &[Vec<EdgeId>], name: &str) -> (NodeId, Vec<EdgeId>) {
    let mut producer = None;
    let mut moved = Vec::new();

    for edge in paths.iter().flatten() {
        let source = source_of(graph, *edge);
        if graph.node(source).name() != name {
            continue;
        }

        producer = Some(source);
        if !moved.contains(edge) {
            moved.push(*edge);
        }
    }

    let producer = producer.expect("an `after` stage was checked to be on every path");
    (producer, moved)
}

/// After the source, taking every edge reading it.
fn after_source(graph: &CompiledGraph, source: NodeId) -> (NodeId, Vec<EdgeId>) {
    let mut moved = Vec::new();

    for node in graph.nodes() {
        for (index, edge) in node.inputs().iter().enumerate() {
            if edge.source() == source {
                moved.push(EdgeId::new(node.id(), index));
            }
        }
    }

    (source, moved)
}

/// Directly before the latest node every demanding path passes through.
///
/// That is the last edge every path shares. Paths that share none part at the source, so the
/// provider goes after it and takes the first edge of each.
fn before_common_node(graph: &CompiledGraph, paths: &[Vec<EdgeId>], source: NodeId) -> (NodeId, Vec<EdgeId>) {
    let first = &paths[0];
    let shared = first
        .iter()
        .enumerate()
        .take_while(|(position, edge)| paths.iter().all(|path| path.get(*position) == Some(edge)))
        .count();

    if shared > 0 {
        let edge = first[shared - 1];
        let producer = source_of(graph, edge);

        return (producer, vec![edge]);
    }

    let mut moved = Vec::new();
    for path in paths {
        if !moved.contains(&path[0]) {
            moved.push(path[0]);
        }
    }

    (source, moved)
}

/// The layers one group's provider serves, each with the stages that demanded it.
fn served_layers(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    traced: &[Traced],
    group: &Group,
) -> Vec<ServedLayer> {
    let mut layers: Vec<ServedLayer> = Vec::new();

    for member in &group.members {
        let each = &traced[*member];
        let name = graph.node(each.asked.node).name().to_string();

        let existing = layers.iter().position(|served| served.layer == each.layer);
        let position = existing.unwrap_or_else(|| {
            let demand = each.asked.demand.demand.clone();
            let description = registry.describe(&demand);

            layers.push(ServedLayer {
                layer: each.layer,
                demand,
                description,
                demanded_by: Vec::new(),
            });
            layers.len() - 1
        });

        let served = &mut layers[position];
        if !served.demanded_by.contains(&name) {
            served.demanded_by.push(name);
        }
    }

    layers
}

/// The name of a registered provider's one input.
fn provider_input(registry: &PipelineRegistry, identifier: &str) -> &'static str {
    let info = registry
        .stages()
        .find(|info| info.kind() == StageKind::Provider && info.full_identifier() == identifier)
        .expect("a provider registered for a kind is a registered stage");

    info.inputs()
        .params()
        .first()
        .map(|param| param.name())
        .expect("a provider takes one input")
}

fn error_at(node: &Node, code: DiagnosticCode, message: String) -> Diagnostic {
    Diagnostic::error(code, node.path(), node.span(), message)
}
