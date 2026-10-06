use av_toolbox_core::{
    AnyDemand,
    DemandKind,
    InputDemand,
    LayerDemand,
    PathNode,
    PathView,
    PipelineRegistry,
    PreparedStage,
    StageKind,
};
use av_toolbox_gpu::{ChunkRoute, LayerId};

use crate::graph::{CompiledGraph, EdgeId, Node, NodeId, NodeRole, ProviderNode};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// Where the graph placed one provider, and the layers it serves there.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderPlacement {
    pub kind: DemandKind,
    /// The real node the provider sits behind.
    pub node: NodeId,
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

/// One demand its provider placed, with the instance serving it and its layer there.
struct Placed {
    asked: Asked,
    instance: usize,
    layer: LayerId,
}

/// One provider, serving every demand of one kind placed behind one real node.
struct Instance {
    kind: DemandKind,
    /// The source the real node traces back to.
    source: NodeId,
    /// The real node the provider sits behind.
    node: NodeId,
    /// The edges leaving the real node on its demanders' paths, which the provider takes over.
    edges: Vec<EdgeId>,
    /// Each distinct request this instance serves, with its layer.
    layers: Vec<(AnyDemand, LayerId)>,
}

/// Every demand placed so far, and the instances serving them.
#[derive(Default)]
struct Placements {
    placed: Vec<Placed>,
    instances: Vec<Instance>,
    next_layer: u32,
}

/// Inserts a provider for every group of demands that place themselves behind the same node, and
/// gives each demanding node its layer.
///
/// Each demand is traced upstream to its source, following a node's chunk authority where it has
/// several inputs. The demand's provider picks an edge on that path. Demands of one kind placed
/// behind the same node share one provider, which takes only the edges its demanders read
/// through. Every distinct request on a provider becomes one layer, numbered across the graph. A
/// demanding node records its demands, and routes on the layer of its chunk request.
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

    let placements = place_demands(graph, registry, &prepared, asked, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    record_demands(graph, registry, &prepared, &placements.placed);
    insert_instances(graph, registry, &placements);

    Ok(())
}

/// Adds notes to `diagnostic` naming what `node` reads and every stage whose demand placed it, when
/// it is a provider.
pub(crate) fn note_demanders(graph: &CompiledGraph, node: &Node, diagnostic: Diagnostic) -> Diagnostic {
    let NodeRole::Provider(placement) = node.role() else {
        return diagnostic;
    };

    let producer_id = node.inputs()[0].source();
    let producer = graph.node(producer_id);
    let message = format!("`{}` reads the frames of `{}`", node.name(), producer.name());
    let mut noted = diagnostic.with_note(message, producer.span());

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

/// Traces every demand to its source, lets its provider place it, and files it under the instance
/// behind the node it was placed after.
///
/// A demand of a kind nothing serves, one that reaches no source, and one its provider refuses to
/// place are reported and dropped.
fn place_demands(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    prepared: &[Option<PreparedStage>],
    asked: Vec<Asked>,
    diagnostics: &mut Diagnostics,
) -> Placements {
    let mut placements = Placements::default();

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

        let (_, place) = registry.provider_for(kind).expect("a served kind has a provider");
        let view = path_view(graph, &path);

        let chosen = match place(&each.demand.demand, &view) {
            Ok(chosen) => chosen,
            Err(error) => {
                let message = format!(
                    "`{}` demands {} on `{}`, but {error}",
                    node.name(),
                    kind.name(),
                    each.demand.input,
                );
                diagnostics.push(error_at(node, DiagnosticCode::PlaceFailed, message));
                continue;
            },
        };

        let edge = path[chosen.node()];
        let source = source_of(graph, path[0]);
        placements.file(each, source, edge, graph);
    }

    placements
}

impl Placements {
    /// Files `asked`, placed on `edge`, under the instance of its kind behind that edge's node.
    fn file(&mut self, asked: Asked, source: NodeId, edge: EdgeId, graph: &CompiledGraph) {
        let kind = asked.demand.demand.kind();
        let node = source_of(graph, edge);

        let existing = self
            .instances
            .iter()
            .position(|instance| instance.kind == kind && instance.node == node);
        let index = existing.unwrap_or_else(|| {
            self.instances.push(Instance {
                kind,
                source,
                node,
                edges: Vec::new(),
                layers: Vec::new(),
            });
            self.instances.len() - 1
        });

        let instance = &mut self.instances[index];
        if !instance.edges.contains(&edge) {
            instance.edges.push(edge);
        }

        let layer = layer_for(instance, &mut self.next_layer, &asked.demand.demand);
        self.placed.push(Placed {
            asked,
            instance: index,
            layer,
        });
    }
}

/// The layer serving `demand` on `instance`, numbering a new one when it is the first of its kind.
fn layer_for(instance: &mut Instance, next_layer: &mut u32, demand: &AnyDemand) -> LayerId {
    let existing = instance.layers.iter().find(|(served, _)| served == demand);
    if let Some((_, layer)) = existing {
        return *layer;
    }

    let layer = LayerId(*next_layer);
    *next_layer += 1;
    instance.layers.push((demand.clone(), layer));

    layer
}

/// The real nodes a demand travels through along `path`, as its provider sees them.
fn path_view<'g>(graph: &'g CompiledGraph, path: &[EdgeId]) -> PathView<'g> {
    let nodes = path
        .iter()
        .map(|edge| {
            let producer = source_of(graph, *edge);
            let node = graph.node(producer);

            PathNode {
                name: node.name(),
                identifier: node.identifier(),
                args: node.args(),
            }
        })
        .collect();

    PathView::new(nodes)
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

/// The route a node takes for its chunk request, reading the layer with the node's own chunk key.
fn route_of(prepared: &[Option<PreparedStage>], placed: &Placed) -> ChunkRoute {
    let stage = prepared[placed.asked.node.index()]
        .as_ref()
        .expect("a stage that made a chunk request was prepared");
    let key = stage
        .parallelism()
        .chunk_key()
        .expect("a stage with a chunk request chunks by a demand");

    ChunkRoute {
        layer: placed.layer,
        key,
    }
}

/// Gives every demanding node its demands, and the route of its chunk request.
fn record_demands(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    prepared: &[Option<PreparedStage>],
    placed: &[Placed],
) {
    let mut settled: Vec<(NodeId, Vec<NodeDemand>)> = Vec::new();

    for each in placed {
        let demand = &each.asked.demand.demand;
        let node_demand = NodeDemand {
            input: each.asked.demand.input,
            kind: demand.kind(),
            description: registry.describe(demand),
            layer: each.layer,
            chunks: each.asked.chunks,
        };

        if each.asked.chunks {
            let route = route_of(prepared, each);
            graph.node_mut(each.asked.node).set_route(route);
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

/// Inserts every instance behind its real node, taking only its demanders' edges.
fn insert_instances(graph: &mut CompiledGraph, registry: &PipelineRegistry, placements: &Placements) {
    for (index, instance) in placements.instances.iter().enumerate() {
        let (identifier, _) = registry
            .provider_for(instance.kind)
            .expect("every placed kind has a provider");

        let layers = served_layers(graph, registry, placements, index);
        let demands: Vec<LayerDemand> = layers
            .iter()
            .map(|served| LayerDemand {
                layer: served.layer,
                demand: served.demand.clone(),
            })
            .collect();
        let args = serde_json::to_value(demands).expect("layer demands serialise");

        let first_demander = placements
            .placed
            .iter()
            .find(|placed| placed.instance == index)
            .map(|placed| placed.asked.node)
            .expect("an instance has a demand");
        let span = graph.node(first_demander).span();

        let provider = ProviderNode {
            identifier: identifier.to_string(),
            input: provider_input(registry, identifier),
            name: instance_name(graph, &placements.instances, instance),
            span,
            args,
            placement: ProviderPlacement {
                kind: instance.kind,
                node: instance.node,
                layers,
            },
        };

        let edges = reading_edges(graph, instance);
        graph.push_provider(instance.node, provider, &edges);
    }

    graph.finish_edit();
}

/// The edges `instance` takes over, each one moved up to read its real node.
///
/// An instance of another kind behind the same node may already have taken one of the edges. This
/// instance then takes that provider's input instead, so it sits between the node and that
/// provider.
fn reading_edges(graph: &CompiledGraph, instance: &Instance) -> Vec<EdgeId> {
    let mut edges = Vec::with_capacity(instance.edges.len());

    for edge in &instance.edges {
        let mut current = *edge;
        loop {
            let source = source_of(graph, current);
            if source == instance.node {
                break;
            }

            current = EdgeId::new(source, 0);
        }

        if !edges.contains(&current) {
            edges.push(current);
        }
    }

    edges
}

/// The provider's name, such as `scenes(input)`.
///
/// An instance behind a later node, while another of its kind shares the source, is named
/// `scenes(input after denoise)`, so the two stay apart.
fn instance_name(graph: &CompiledGraph, instances: &[Instance], instance: &Instance) -> String {
    let kind = instance.kind.name();
    let source = graph.node(instance.source).name();

    if instance.node == instance.source {
        return format!("{kind}({source})");
    }

    let shares_source = instances.iter().any(|other| {
        other.kind == instance.kind && other.source == instance.source && other.node != instance.node
    });
    if !shares_source {
        return format!("{kind}({source})");
    }

    let after = graph.node(instance.node).name();
    format!("{kind}({source} after {after})")
}

/// The layers one instance serves, each with the stages that demanded it.
fn served_layers(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    placements: &Placements,
    instance: usize,
) -> Vec<ServedLayer> {
    let mut layers: Vec<ServedLayer> = Vec::new();

    let members = placements
        .placed
        .iter()
        .filter(|placed| placed.instance == instance);
    for each in members {
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
