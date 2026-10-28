mod insert;
mod place;
mod trace;

use av_toolbox_core::{
    AnyDemand,
    DemandKind,
    InputDemand,
    LayerDemand,
    PipelineRegistry,
    PreparedStage,
    Residency,
    StageKind,
};
use av_toolbox_gpu::{ChunkRoute, LayerId};

use self::insert::insert_instances;
use self::place::{place_demand, provider_demands};
pub(crate) use self::trace::bound_edge;
use crate::graph::{CompiledGraph, EdgeId, Node, NodeId, NodeRole};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// Where the graph placed one provider, and the layers it serves there.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderPlacement {
    pub kind: DemandKind,
    /// The real node the provider sits behind.
    pub node: NodeId,
    /// The side the provider was registered for.
    pub side: Residency,
    pub layers: Vec<ServedLayer>,
}

/// One layer a provider serves.
#[derive(Debug, Clone, PartialEq)]
pub struct ServedLayer {
    pub layer: LayerId,
    pub demand: AnyDemand,
    /// The demand as `PipelineRegistry::describe` puts it, such as `natural`.
    pub description: String,
    /// The names of the stages and providers that demanded this layer.
    pub demanded_by: Vec<String>,
}

/// One demand a stage or provider made, as the pass settled it.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeDemand {
    pub input: &'static str,
    pub kind: DemandKind,
    pub description: String,
    pub layer: LayerId,
    /// Whether this demand is the stage's chunk request.
    pub chunks: bool,
}

/// What made a demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Demander {
    Stage(NodeId),
    /// The instance at this position in [Placements].
    Instance(usize),
}

/// One demand made on a bound input.
struct Asked {
    demander: Demander,
    demand: InputDemand,
    chunks: bool,
}

/// One demand its provider placed, with the instance serving it, its layer there, and the edges
/// it reads through.
struct Placed {
    asked: Asked,
    instance: usize,
    layer: LayerId,
    edges: Vec<EdgeId>,
    /// The round this demand was filed in.
    round: u32,
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
    /// The instances that demanded this one.
    demanded_by: Vec<usize>,
}

impl Instance {
    /// Whether any demand is still filed under this instance.
    fn is_live(&self) -> bool {
        !self.layers.is_empty()
    }
}

/// One real node a demand travels through, with the edges leaving it toward the demander.
struct Step {
    node: NodeId,
    edges: Vec<EdgeId>,
}

/// Every demand placed so far, and the instances serving them.
#[derive(Default)]
struct Placements {
    placed: Vec<Placed>,
    instances: Vec<Instance>,
    next_layer: u32,
    /// The round being filed.
    round: u32,
    /// Each stage a provider refused to place a demand for, with the reason it gave.
    place_failures: Vec<(NodeId, String)>,
}

/// Inserts a provider for every group of demands that place themselves behind the same node, and
/// gives each demanding node its layer.
pub fn satisfy_demands(graph: &mut CompiledGraph, registry: &PipelineRegistry) -> Result<(), Diagnostics> {
    let nodes = graph.filled_order();
    satisfy_demands_for(graph, registry, &nodes)
}

/// Inserts a provider for every group of demands the stages in `nodes` make, and gives each
/// demanding node its layer.
///
/// Each demand is traced upstream to its source, following a node's chunk authority where it has
/// several inputs. The demand's provider picks an edge on that path. Demands of one kind placed
/// behind the same node share one provider, which takes only the edges its demanders read
/// through. Every distinct request on a provider becomes one layer, numbered across the graph.
///
/// Providers may demand of their own input. Their demands are traced from the node they sit
/// behind and placed the same way, round after round, until a round changes nothing. A provider
/// whose layers or edges changed is asked again, and its new demands replace its old ones. A demand
/// that would make a provider serve itself, directly or through others, is refused. The providers
/// behind one node are stacked so each sits after every provider it demands.
///
/// Only demands from `nodes` share providers with each other.
pub fn satisfy_demands_for(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    nodes: &[NodeId],
) -> Result<(), Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let mut asked = collect_demands(graph, &prepared, nodes);
    if asked.is_empty() {
        return Ok(());
    }

    let next_layer = next_free_layer(graph);
    let mut placements = Placements {
        next_layer,
        ..Placements::default()
    };
    let mut before = placements.snapshot();
    let mut reasked = Vec::new();
    loop {
        for each in asked {
            place_demand(
                graph,
                registry,
                &prepared,
                &mut placements,
                each,
                &mut diagnostics,
            );
        }

        placements.replace_earlier(&reasked);
        placements.settle();

        let changed = placements.changed_since(&before);
        if changed.is_empty() {
            break;
        }

        before = placements.snapshot();
        asked = provider_demands(graph, registry, &placements, &changed, &mut diagnostics);
        reasked = changed;
        placements.round += 1;
    }

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let inserted = insert_instances(graph, registry, &placements);
    record_demands(graph, registry, &prepared, &placements, &inserted);

    Ok(())
}

/// One past the highest layer any node in `graph` already serves or reads.
fn next_free_layer(graph: &CompiledGraph) -> u32 {
    let mut next = 0;

    for node in graph.nodes() {
        for demand in node.demands() {
            next = next.max(demand.layer.0 + 1);
        }

        if let NodeRole::Provider(placement) = node.role() {
            for served in &placement.layers {
                next = next.max(served.layer.0 + 1);
            }
        }
    }

    next
}

/// Adds notes to `diagnostic` naming what `node` reads and how each demand it serves reached it,
/// when it is a provider.
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
            let chain = demand_chain(graph, placement.kind, name);
            let span = chain.stage.map_or(node.span(), Node::span);
            let message = format!(
                "this provider serves {}, asking for {}",
                chain.text, served.description
            );

            noted = noted.with_note(message, span);
        }
    }

    noted
}

/// How a demand reached a provider, and the stage whose demand started it.
pub(crate) struct DemandChain<'g> {
    /// The chain spelled out, such as "`scenes` demanded by `scene-split(input)`, demanded by `bwdif`".
    pub(crate) text: String,
    pub(crate) stage: Option<&'g Node>,
}

/// How a demand of `kind` made by the node named `demander` reached its provider.
///
/// A demander that is itself a provider is followed back through its first demander, until a
/// stage is reached.
pub(crate) fn demand_chain<'g>(
    graph: &'g CompiledGraph,
    kind: DemandKind,
    demander: &str,
) -> DemandChain<'g> {
    let mut names = vec![demander.to_string()];
    let mut current = named_node(graph, demander);

    while let Some(node) = current {
        let NodeRole::Provider(placement) = node.role() else {
            break;
        };

        let first = placement
            .layers
            .first()
            .and_then(|served| served.demanded_by.first());
        let Some(next) = first else {
            break;
        };

        names.push(next.clone());
        current = named_node(graph, next);
    }

    DemandChain {
        text: chain_text(kind, &names),
        stage: current,
    }
}

fn named_node<'g>(graph: &'g CompiledGraph, name: &str) -> Option<&'g Node> {
    graph.nodes().iter().find(|node| node.name() == name)
}

/// Spells out a chain of demanders, such as "`scenes` demanded by `scene-split(input)`, demanded
/// by `bwdif`".
fn chain_text(kind: DemandKind, names: &[String]) -> String {
    let links: Vec<String> = names.iter().map(|name| format!("demanded by `{name}`")).collect();

    format!("`{}` {}", kind.name(), links.join(", "))
}

/// Every demand each stage in `nodes` makes on an input it has bound, in the order given.
fn collect_demands(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    nodes: &[NodeId],
) -> Vec<Asked> {
    let mut asked = Vec::new();

    for id in nodes {
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
                demander: Demander::Stage(*id),
                demand,
                chunks,
            });
        }
    }

    asked
}

/// The args an instance is built from, one entry per layer it serves.
fn instance_args(instance: &Instance) -> serde_json::Value {
    let demands: Vec<LayerDemand> = instance
        .layers
        .iter()
        .map(|(demand, layer)| LayerDemand {
            layer: *layer,
            demand: demand.clone(),
        })
        .collect();

    serde_json::to_value(demands).expect("layer demands serialise")
}

/// The route a stage takes for its chunk request, reading the layer with the stage's own chunk key.
fn route_of(prepared: &[Option<PreparedStage>], node: NodeId, layer: LayerId) -> ChunkRoute {
    let stage = prepared[node.index()]
        .as_ref()
        .expect("a stage that made a chunk request was prepared");
    let key = stage
        .parallelism()
        .chunk_key()
        .expect("a stage with a chunk request chunks by a demand");

    ChunkRoute { layer, key }
}

/// Gives every demanding node its demands, and the route of its chunk request.
///
/// `inserted` names the provider node each instance became.
fn record_demands(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    prepared: &[Option<PreparedStage>],
    placements: &Placements,
    inserted: &[NodeId],
) {
    let mut settled: Vec<(NodeId, Vec<NodeDemand>)> = Vec::new();

    for each in &placements.placed {
        let demand = &each.asked.demand.demand;
        let node_demand = NodeDemand {
            input: each.asked.demand.input,
            kind: demand.kind(),
            description: registry.describe(demand),
            layer: each.layer,
            chunks: each.asked.chunks,
        };

        let node = match each.asked.demander {
            Demander::Stage(node) => node,
            Demander::Instance(index) => inserted[index],
        };

        if each.asked.chunks {
            let route = route_of(prepared, node, each.layer);
            graph.node_mut(node).set_route(route);
        }

        match settled.iter_mut().find(|(settled_node, _)| *settled_node == node) {
            Some((_, demands)) => demands.push(node_demand),
            None => settled.push((node, vec![node_demand])),
        }
    }

    for (node, demands) in settled {
        graph.node_mut(node).set_demands(demands);
    }
}

/// The provider's name, such as `scenes(input)`.
///
/// An instance behind a later node, while another of its kind shares the source, is named
/// `scenes(input after denoise)`, so the two stay apart.
fn instance_name(graph: &CompiledGraph, instances: &[Instance], index: usize) -> String {
    let instance = &instances[index];
    let kind = instance.kind.name();
    let source = graph.node(instance.source).name();

    if instance.node == instance.source {
        return format!("{kind}({source})");
    }

    let shares_source = instances.iter().any(|other| {
        let same_source = other.kind == instance.kind && other.source == instance.source;

        same_source && other.node != instance.node && other.is_live()
    });
    if !shares_source {
        return format!("{kind}({source})");
    }

    let after = graph.node(instance.node).name();

    format!("{kind}({source} after {after})")
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
