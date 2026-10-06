use av_toolbox_core::PipelineRegistry;

use super::{
    Instance,
    Placements,
    ProviderPlacement,
    ServedLayer,
    instance_args,
    instance_name,
    provider_input,
};
use crate::graph::{CompiledGraph, EdgeId, NodeId, ProviderNode};

/// Inserts every live instance behind its real node, and returns the node each one became.
///
/// When the edge sets in a stack nest, the stack is wired as a tree. Each instance reads the
/// smallest instance before it that holds all its edges, or the real node, and takes exactly its
/// demanders' edges. When edge sets only partly overlap, the stack is wired as one chain instead,
/// so a branch may pass through a provider it did not demand.
pub(super) fn insert_instances(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    placements: &Placements,
) -> Vec<NodeId> {
    let mut inserted = vec![NodeId::new(0); placements.instances.len()];

    for stack in stacks(placements) {
        let nested = nests(placements, &stack);

        for (position, index) in stack.iter().enumerate() {
            let instance = &placements.instances[*index];
            let earlier = &stack[..position];
            let parent = match nested {
                true => enclosing(placements, earlier, instance),
                false => earlier.last().copied(),
            };
            let after = parent.map_or(instance.node, |parent| inserted[parent]);

            let provider = provider_node(graph, registry, placements, *index);
            inserted[*index] = graph.push_provider(after, provider, &instance.edges);
        }
    }

    graph.finish_edit();
    inserted
}

/// Whether every two edge sets in `stack` are either apart or one inside the other.
fn nests(placements: &Placements, stack: &[usize]) -> bool {
    stack.iter().all(|first| {
        stack.iter().all(|second| {
            let first_edges = &placements.instances[*first].edges;
            let second_edges = &placements.instances[*second].edges;
            let apart = !first_edges.iter().any(|edge| second_edges.contains(edge));

            apart || contains_all(first_edges, second_edges) || contains_all(second_edges, first_edges)
        })
    })
}

/// The last instance in `earlier` holding every edge of `instance`.
fn enclosing(placements: &Placements, earlier: &[usize], instance: &Instance) -> Option<usize> {
    earlier
        .iter()
        .rev()
        .copied()
        .find(|candidate| contains_all(&placements.instances[*candidate].edges, &instance.edges))
}

fn contains_all(outer: &[EdgeId], inner: &[EdgeId]) -> bool {
    inner.iter().all(|edge| outer.contains(edge))
}

/// The live instances behind each real node, each stack ordered upstream first.
///
/// An instance comes after every instance it demands, and after every instance whose edges are a
/// strict superset of its own. Instances with neither between them are ordered by kind name, then
/// by the position of the stage whose demand started their chain.
fn stacks(placements: &Placements) -> Vec<Vec<usize>> {
    let mut grouped: Vec<(NodeId, Vec<usize>)> = Vec::new();

    let live = placements
        .instances
        .iter()
        .enumerate()
        .filter(|(_, instance)| instance.is_live());
    for (index, instance) in live {
        match grouped.iter_mut().find(|(node, _)| *node == instance.node) {
            Some((_, members)) => members.push(index),
            None => grouped.push((instance.node, vec![index])),
        }
    }

    grouped
        .into_iter()
        .map(|(_, members)| stack_order(placements, members))
        .collect()
}

/// Orders the instances of one stack so each comes after every instance that must sit upstream of it.
fn stack_order(placements: &Placements, members: Vec<usize>) -> Vec<usize> {
    let mut ordered = Vec::with_capacity(members.len());
    let mut waiting = members;

    while !waiting.is_empty() {
        let next = waiting
            .iter()
            .copied()
            .filter(|candidate| !waits_on_any(placements, *candidate, &waiting))
            .min_by_key(|candidate| stack_key(placements, *candidate))
            .expect("providers never demand each other in a cycle");

        ordered.push(next);
        waiting.retain(|index| *index != next);
    }

    ordered
}

/// Whether one of `others` must sit upstream of `instance`, because `instance` demands it or its
/// edges are a strict superset of those of `instance`.
fn waits_on_any(placements: &Placements, instance: usize, others: &[usize]) -> bool {
    let edges = &placements.instances[instance].edges;

    others.iter().any(|other| {
        let other_instance = &placements.instances[*other];
        let demanded = other_instance.demanded_by.contains(&instance);
        let wider = other_instance.edges.len() > edges.len() && contains_all(&other_instance.edges, edges);

        *other != instance && (demanded || wider)
    })
}

/// How instances with no demand between them are ordered in a stack.
fn stack_key(placements: &Placements, instance: usize) -> (&'static str, usize) {
    let kind = placements.instances[instance].kind.name();
    let first = placements.first_demander(instance);
    let root = placements.root_stage(first);

    (kind, root.index())
}

/// The provider node `index` becomes, carrying its layers and pointing at the stage that started
/// its chain.
fn provider_node(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    placements: &Placements,
    index: usize,
) -> ProviderNode {
    let instance = &placements.instances[index];
    let (identifier, _) = registry
        .provider_for(instance.kind)
        .expect("every placed kind has a provider");

    let first = placements.first_demander(index);
    let root = placements.root_stage(first);
    let span = graph.node(root).span();

    ProviderNode {
        identifier: identifier.to_string(),
        input: provider_input(registry, identifier),
        name: instance_name(graph, &placements.instances, index),
        span,
        args: instance_args(instance),
        placement: ProviderPlacement {
            kind: instance.kind,
            node: instance.node,
            layers: served_layers(graph, registry, placements, index),
        },
    }
}

/// The layers one instance serves, each with the stages and providers that demanded it.
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
        let name = placements.demander_name(graph, each.asked.demander);

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
