use std::collections::HashMap;
use std::sync::Mutex;

use av_toolbox_core::{Artefact, ArtefactPiece, MergedArtefact, StageArtefacts, StageError, WantedArtefacts};
use av_toolbox_gpu::FrameContext;
use av_toolbox_graph::{CompiledGraph, NodeId, StageBox};

use crate::RunError;

/// What one node declares, and which of it some sink will write.
struct NodeArtefacts {
    path: String,
    declared: StageArtefacts,
    wanted: WantedArtefacts,
}

/// One piece a node handed over, waiting to be merged.
struct Collected {
    node: NodeId,
    name: &'static str,
    piece: ArtefactPiece,
}

/// Gathers the artefact pieces every worker's nodes hand over, keeping only those a sink will write.
pub(crate) struct ArtefactCollector {
    nodes: HashMap<NodeId, NodeArtefacts>,
    pieces: Mutex<Vec<Collected>>,
}

impl ArtefactCollector {
    /// A collector for `graph`, wanting what its artefact sinks selected.
    pub(crate) fn new(graph: &CompiledGraph) -> Self {
        let mut wanted_names: HashMap<NodeId, Vec<&'static str>> = HashMap::new();

        for sink in graph.artefact_sinks() {
            for (node, name) in sink.selected() {
                let names = wanted_names.entry(*node).or_default();
                if !names.contains(name) {
                    names.push(name);
                }
            }
        }

        let mut nodes = HashMap::new();
        for node in graph.nodes() {
            let declared = node.artefacts();
            if declared.declared().is_empty() {
                continue;
            }

            let names = wanted_names.remove(&node.id()).unwrap_or_default();
            let entry = NodeArtefacts {
                path: node.path(),
                declared,
                wanted: WantedArtefacts::new(names),
            };
            nodes.insert(node.id(), entry);
        }

        Self {
            nodes,
            pieces: Mutex::new(Vec::new()),
        }
    }

    /// Tells `stage` which of its artefacts some sink will write.
    pub(crate) fn announce(&self, node: NodeId, stage: &mut StageBox) {
        let Some(instance) = stage.instance_mut() else {
            return;
        };

        let wanted = match self.nodes.get(&node) {
            Some(entry) => entry.wanted.clone(),
            None => WantedArtefacts::default(),
        };
        instance.wants_artefacts(&wanted);
    }

    /// Takes whatever `stage` built since it last handed artefacts over, filed under `chunk`.
    pub(crate) fn take(
        &self,
        node: NodeId,
        chunk: Option<u64>,
        stage: &mut StageBox,
        context: &mut FrameContext,
    ) -> Result<(), RunError> {
        let Some(instance) = stage.instance_mut() else {
            return Ok(());
        };

        let taken = instance.take_artefacts(context);
        let artefacts = taken.map_err(|source| RunError::Stage { node, source })?;
        if artefacts.is_empty() {
            return Ok(());
        }

        let entry = self.nodes.get(&node);
        let mut kept = Vec::with_capacity(artefacts.len());

        for artefact in artefacts {
            check_declared(node, entry, &artefact)?;

            let wanted = entry.is_some_and(|entry| entry.wanted.contains(artefact.name));
            if !wanted {
                continue;
            }

            let piece = ArtefactPiece {
                chunk,
                data: artefact.data,
            };
            kept.push(Collected {
                node,
                name: artefact.name,
                piece,
            });
        }

        let mut pieces = self
            .pieces
            .lock()
            .expect("nothing panics while holding the artefact pieces");
        pieces.extend(kept);

        Ok(())
    }

    /// Merges every piece collected so far, by node and artefact, with each node's pieces in chunk order.
    pub(crate) fn merge(&self) -> Result<HashMap<(NodeId, &'static str), MergedArtefact>, RunError> {
        let collected = {
            let mut pieces = self
                .pieces
                .lock()
                .expect("nothing panics while holding the artefact pieces");
            std::mem::take(&mut *pieces)
        };

        let mut groups: HashMap<(NodeId, &'static str), Vec<ArtefactPiece>> = HashMap::new();
        for entry in collected {
            let group = groups.entry((entry.node, entry.name)).or_default();
            group.push(entry.piece);
        }

        let mut merged = HashMap::with_capacity(groups.len());
        for ((node, name), mut pieces) in groups {
            pieces.sort_by_key(|piece| piece.chunk);

            let entry = &self.nodes[&node];
            let declared = entry
                .declared
                .find(name)
                .expect("only declared artefacts are collected");
            let combined = entry.declared.merge(name, pieces);
            let data = combined.map_err(|source| RunError::Stage { node, source })?;

            let artefact = MergedArtefact {
                stage: entry.path.clone(),
                name,
                kind: declared.kind,
                data,
            };
            merged.insert((node, name), artefact);
        }

        Ok(merged)
    }
}

/// Fails the run when `artefact` is not one `node` declared, or holds data of another kind.
fn check_declared(node: NodeId, entry: Option<&NodeArtefacts>, artefact: &Artefact) -> Result<(), RunError> {
    let declared = entry.and_then(|entry| entry.declared.find(artefact.name));
    let Some(declared) = declared else {
        let message = format!(
            "the stage handed over an artefact named {} that it never declared",
            artefact.name
        );
        let source = StageError::msg(message);
        return Err(RunError::Stage { node, source });
    };

    if artefact.data.is_kind(declared.kind) {
        return Ok(());
    }

    let message = format!(
        "the stage handed over artefact {} holding another kind of data than it declared",
        artefact.name
    );
    let source = StageError::msg(message);

    Err(RunError::Stage { node, source })
}
