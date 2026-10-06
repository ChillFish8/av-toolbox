use serde_json::Value;
use thiserror::Error;

use crate::AnyDemand;

/// The real nodes on a demand's path, the source first and the node feeding the demander last.
///
/// The demanding stage itself is never on the path.
#[derive(Debug, Clone)]
pub struct PathView<'g> {
    nodes: Vec<PathNode<'g>>,
    /// The name of every stage in the graph, on the path or not.
    stages: Vec<&'g str>,
}

/// One stage on a [PathView].
#[derive(Debug, Clone, Copy)]
pub struct PathNode<'g> {
    pub name: &'g str,
    /// The stage's full identifier, such as `denoise/avd-nlmeans`.
    pub identifier: &'g str,
    pub args: &'g Value,
}

/// The edge leaving one node of a [PathView].
///
/// Only a path can make one.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PathEdge(usize);

impl PathEdge {
    /// The position in its path of the node this edge leaves.
    pub fn node(self) -> usize {
        self.0
    }
}

impl<'g> PathView<'g> {
    /// Builds a path from its nodes, the source first, in a graph holding the stages named `stages`.
    ///
    /// # Panics
    ///
    /// When `nodes` is empty, since every path starts at a source.
    pub fn new(nodes: Vec<PathNode<'g>>, stages: Vec<&'g str>) -> Self {
        assert!(!nodes.is_empty(), "a demand's path always holds its source");

        Self { nodes, stages }
    }

    pub fn nodes(&self) -> &[PathNode<'g>] {
        &self.nodes
    }

    /// The edge leaving the node at position `node`.
    ///
    /// # Panics
    ///
    /// When `node` is past the end of the path.
    pub fn after(&self, node: usize) -> PathEdge {
        assert!(
            node < self.nodes.len(),
            "position {node} is past the end of a path of {} nodes",
            self.nodes.len(),
        );

        PathEdge(node)
    }

    pub fn after_source(&self) -> PathEdge {
        PathEdge(0)
    }

    /// The edge entering the demanding stage.
    pub fn before_demander(&self) -> PathEdge {
        PathEdge(self.nodes.len() - 1)
    }

    /// The edge leaving the stage named `name`.
    ///
    /// A name no stage in the graph has is refused differently from a stage that is off this path.
    pub fn after_named(&self, name: &str) -> Result<PathEdge, PlaceError> {
        let position = self.nodes.iter().position(|node| node.name == name);
        if let Some(position) = position {
            return Ok(PathEdge(position));
        }

        if self.stages.contains(&name) {
            return Err(off_path(name));
        }

        Err(no_stage_named(name))
    }

    /// The edge leaving the last stage whose identifier is one of `identifiers`.
    pub fn after_last_of(&self, identifiers: &[&str]) -> Option<PathEdge> {
        let position = self
            .nodes
            .iter()
            .rposition(|node| identifiers.contains(&node.identifier))?;

        Some(PathEdge(position))
    }
}

/// Why a provider could not be placed on a demand's path.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct PlaceError(String);

impl PlaceError {
    pub fn msg(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// A provider's `place` with its request still serialised.
pub type PlaceFn = fn(&AnyDemand, &PathView<'_>) -> Result<PathEdge, PlaceError>;

fn off_path(name: &str) -> PlaceError {
    let message = format!(
        "`{name}` is not a stage on the path this demand travels, so the provider cannot go after it"
    );

    PlaceError::msg(message)
}

fn no_stage_named(name: &str) -> PlaceError {
    let message = format!("there is no stage named `{name}`");

    PlaceError::msg(message)
}
