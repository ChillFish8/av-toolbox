use av_toolbox_core::{AnyDemand, DemandKind, PipelineRegistry, PreparedStage};
use av_toolbox_gpu::LayerId;

use super::trace::{path_view, trace};
use super::{
    Asked,
    Demander,
    Instance,
    Placed,
    Placements,
    Step,
    chain_text,
    error_at,
    instance_args,
    instance_name,
};
use crate::graph::{CompiledGraph, EdgeId, NodeId};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// What every live instance in `changed` demands of its own input, for the next round.
pub(super) fn provider_demands(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    placements: &Placements,
    changed: &[usize],
    diagnostics: &mut Diagnostics,
) -> Vec<Asked> {
    let mut asked = Vec::new();

    for index in changed {
        let instance = &placements.instances[*index];
        if !instance.is_live() {
            continue;
        }

        let demands_of = registry
            .provider_demands(instance.kind)
            .expect("every placed kind has a provider");
        let args = instance_args(instance);

        let demands = match demands_of(&args) {
            Ok(demands) => demands,
            Err(error) => {
                let name = instance_name(graph, &placements.instances, *index);
                let message = format!("`{name}` cannot read its own demands, because {error}");
                let demander = placements.first_demander(*index);
                let code = DiagnosticCode::InvalidArgs;
                let diagnostic = placements.error_for(graph, instance.kind, demander, code, message);

                push_once(diagnostics, diagnostic);
                continue;
            },
        };

        for demand in demands {
            asked.push(Asked {
                demander: Demander::Instance(*index),
                demand,
                chunks: false,
            });
        }
    }

    asked
}

/// Traces `asked` to its source, lets its provider place it, and files it under the instance
/// behind the node it was placed after.
///
/// A demand of a kind nothing serves, one that reaches no source, one its provider refuses to
/// place, and one that would close a cycle of providers are reported and dropped.
pub(super) fn place_demand(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    prepared: &[Option<PreparedStage>],
    placements: &mut Placements,
    asked: Asked,
    diagnostics: &mut Diagnostics,
) {
    let kind = asked.demand.demand.kind();
    let demander = placements.demander_name(graph, asked.demander);
    let input = asked.demand.input;

    if let Some(problem) = unserved(registry, kind) {
        let message = format!("`{demander}` demands {} on `{input}`, {problem}", kind.name());
        let code = DiagnosticCode::UnsatisfiedDemand;
        let diagnostic = placements.error_for(graph, kind, asked.demander, code, message);
        push_once(diagnostics, diagnostic);
        return;
    }

    let Some(steps) = trace(graph, prepared, placements, &asked) else {
        let message = format!(
            "`{demander}` demands {} on `{input}`, which traces to no source",
            kind.name(),
        );
        let code = DiagnosticCode::UnsatisfiedDemand;
        let diagnostic = placements.error_for(graph, kind, asked.demander, code, message);
        push_once(diagnostics, diagnostic);
        return;
    };

    let (_, place) = registry.provider_for(kind).expect("a served kind has a provider");
    let view = path_view(graph, &steps);

    let chosen = match place(&asked.demand.demand, &view) {
        Ok(chosen) => chosen,
        Err(error) => {
            let reason = error.to_string();
            if !placements.first_place_failure(asked.demander, reason) {
                return;
            }

            let message = format!("`{demander}` demands {} on `{input}`, but {error}", kind.name());
            let code = DiagnosticCode::PlaceFailed;
            let diagnostic = placements.error_for(graph, kind, asked.demander, code, message);
            push_once(diagnostics, diagnostic);
            return;
        },
    };

    let step = &steps[chosen.node()];
    let source = steps[0].node;

    if let Some(cycle) = placements.cycle(&asked, step.node) {
        let names: Vec<String> = cycle
            .iter()
            .map(|index| instance_name(graph, &placements.instances, *index))
            .collect();
        let message = format!(
            "the providers {} demand each other in a cycle, so none of them can be served",
            names.join(" → "),
        );
        let code = DiagnosticCode::ProviderCycle;
        let diagnostic = placements.error_for(graph, kind, asked.demander, code, message);
        push_once(diagnostics, diagnostic);
        return;
    }

    placements.file(asked, source, step);
}

impl Placements {
    /// The edges and layers of each instance, to tell later which ones changed.
    pub(super) fn snapshot(&self) -> Vec<(Vec<EdgeId>, Vec<LayerId>)> {
        self.instances
            .iter()
            .map(|instance| {
                let layers = instance.layers.iter().map(|(_, layer)| *layer).collect();

                (instance.edges.clone(), layers)
            })
            .collect()
    }

    /// The instances created since `before` was taken, or whose edges or layers changed since.
    pub(super) fn changed_since(&self, before: &[(Vec<EdgeId>, Vec<LayerId>)]) -> Vec<usize> {
        let now = self.snapshot();

        (0..now.len())
            .filter(|index| before.get(*index) != Some(&now[*index]))
            .collect()
    }

    /// Withdraws every demand the instances in `demanders` filed.
    fn retract(&mut self, demanders: &[usize]) {
        self.placed.retain(|placed| match placed.asked.demander {
            Demander::Stage(_) => true,
            Demander::Instance(index) => !demanders.contains(&index),
        });
    }

    /// Withdraws the demands the instances in `demanders` filed before this round, once their
    /// replacements are filed.
    ///
    /// Withdrawing only after the round keeps every instance filed under something while its
    /// demands are placed, so a chain of demanders always leads back to a stage.
    pub(super) fn replace_earlier(&mut self, demanders: &[usize]) {
        let round = self.round;

        self.placed.retain(|placed| match placed.asked.demander {
            Demander::Stage(_) => true,
            Demander::Instance(index) => placed.round == round || !demanders.contains(&index),
        });
    }

    /// Drops whatever withdrawn demands left behind.
    ///
    /// An instance left serving nothing withdraws its own demands in turn. Every instance then
    /// keeps only the edges, layers and demanding instances its filed demands still use.
    pub(super) fn settle(&mut self) {
        loop {
            let dead: Vec<usize> = (0..self.instances.len())
                .filter(|index| !self.placed.iter().any(|placed| placed.instance == *index))
                .collect();
            let filed = self.placed.len();

            self.retract(&dead);
            if self.placed.len() == filed {
                break;
            }
        }

        for (index, instance) in self.instances.iter_mut().enumerate() {
            let members: Vec<&Placed> = self
                .placed
                .iter()
                .filter(|placed| placed.instance == index)
                .collect();

            instance
                .edges
                .retain(|edge| members.iter().any(|placed| placed.edges.contains(edge)));
            instance
                .layers
                .retain(|(_, layer)| members.iter().any(|placed| placed.layer == *layer));
            instance.demanded_by.retain(|demander| {
                let filed_by = Demander::Instance(*demander);

                members.iter().any(|placed| placed.asked.demander == filed_by)
            });
        }
    }

    fn find(&self, kind: DemandKind, node: NodeId) -> Option<usize> {
        self.instances
            .iter()
            .position(|instance| instance.kind == kind && instance.node == node)
    }

    /// Files `asked`, placed on `step`, under the instance of its kind behind that step's node.
    fn file(&mut self, asked: Asked, source: NodeId, step: &Step) {
        let kind = asked.demand.demand.kind();

        let existing = self.find(kind, step.node);
        let index = existing.unwrap_or_else(|| {
            self.instances.push(Instance {
                kind,
                source,
                node: step.node,
                edges: Vec::new(),
                layers: Vec::new(),
                demanded_by: Vec::new(),
            });
            self.instances.len() - 1
        });

        let instance = &mut self.instances[index];
        for edge in &step.edges {
            if !instance.edges.contains(edge) {
                instance.edges.push(*edge);
            }
        }

        if let Demander::Instance(demanding) = asked.demander
            && !instance.demanded_by.contains(&demanding)
        {
            instance.demanded_by.push(demanding);
        }

        let layer = layer_for(instance, &mut self.next_layer, &asked.demand.demand);
        if self.repeats(&asked, layer) {
            return;
        }

        self.placed.push(Placed {
            asked,
            instance: index,
            layer,
            edges: step.edges.clone(),
            round: self.round,
        });
    }

    /// Whether `asked` is a provider's demand already filed on `layer` this round, as it is when
    /// the provider lists one demand twice.
    fn repeats(&self, asked: &Asked, layer: LayerId) -> bool {
        let Demander::Instance(_) = asked.demander else {
            return false;
        };

        self.placed.iter().any(|placed| {
            placed.round == self.round
                && placed.asked.demander == asked.demander
                && placed.asked.demand.input == asked.demand.input
                && placed.layer == layer
        })
    }

    /// The instances `asked` would close a cycle through, when it is a provider's demand that
    /// would be filed under that provider or one of the providers that demanded it.
    ///
    /// The cycle starts and ends at the instance the demand would reach, such as
    /// `[scene-split, scene-score, scene-split]`.
    fn cycle(&self, asked: &Asked, node: NodeId) -> Option<Vec<usize>> {
        let Demander::Instance(demanding) = asked.demander else {
            return None;
        };

        let kind = asked.demand.demand.kind();
        let target = self.find(kind, node)?;
        let mut cycle = self.demand_path(target, demanding)?;
        cycle.push(target);

        Some(cycle)
    }

    /// The instances from `target` to `from`, each demanded by the one before it, when `target`
    /// is `from` or demanded it through others.
    fn demand_path(&self, target: usize, from: usize) -> Option<Vec<usize>> {
        if from == target {
            return Some(vec![from]);
        }

        for demander in &self.instances[from].demanded_by {
            if let Some(mut path) = self.demand_path(target, *demander) {
                path.push(from);
                return Some(path);
            }
        }

        None
    }

    /// Records that a provider refused to place a demand of `demander` because of `reason`, and
    /// returns whether this is the first such refusal for the stage that started its chain.
    ///
    /// One mistake, such as a misspelt `after`, is often behind demands of several kinds.
    fn first_place_failure(&mut self, demander: Demander, reason: String) -> bool {
        let root = self.root_stage(demander);
        let failure = (root, reason);
        if self.place_failures.contains(&failure) {
            return false;
        }

        self.place_failures.push(failure);
        true
    }

    /// What made the first demand filed under `instance`.
    pub(super) fn first_demander(&self, instance: usize) -> Demander {
        self.placed
            .iter()
            .find(|placed| placed.instance == instance)
            .map(|placed| placed.asked.demander)
            .expect("a live instance keeps a filed demand until its replacements are filed")
    }

    /// The stage whose demand started the chain that led to `demander`.
    pub(super) fn root_stage(&self, demander: Demander) -> NodeId {
        let mut current = demander;

        loop {
            match current {
                Demander::Stage(node) => return node,
                Demander::Instance(index) => current = self.first_demander(index),
            }
        }
    }

    pub(super) fn demander_name(&self, graph: &CompiledGraph, demander: Demander) -> String {
        match demander {
            Demander::Stage(node) => graph.node(node).name().to_string(),
            Demander::Instance(index) => instance_name(graph, &self.instances, index),
        }
    }

    /// An error about a demand of `kind` that `demander` made, at the stage that started its chain.
    ///
    /// A provider's demand also gets a note spelling the chain out.
    fn error_for(
        &self,
        graph: &CompiledGraph,
        kind: DemandKind,
        demander: Demander,
        code: DiagnosticCode,
        message: String,
    ) -> Diagnostic {
        let root = self.root_stage(demander);
        let stage = graph.node(root);
        let diagnostic = error_at(stage, code, message);

        if let Demander::Stage(_) = demander {
            return diagnostic;
        }

        let mut names = Vec::new();
        let mut current = demander;
        while let Demander::Instance(link) = current {
            names.push(instance_name(graph, &self.instances, link));
            current = self.first_demander(link);
        }
        names.push(stage.name().to_string());

        let text = chain_text(kind, &names);
        diagnostic.with_note(text, stage.span())
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

/// Pushes `diagnostic` unless the same one was already reported, as it is when a provider is asked
/// for its demands again.
fn push_once(diagnostics: &mut Diagnostics, diagnostic: Diagnostic) {
    let repeated = diagnostics.iter().any(|reported| {
        reported.code() == diagnostic.code()
            && reported.path() == diagnostic.path()
            && reported.message() == diagnostic.message()
    });

    if !repeated {
        diagnostics.push(diagnostic);
    }
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
