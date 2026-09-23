use av_toolbox_graph::NodeId;
use thiserror::Error;

/// Something a run noticed that did not stop it.
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum RunWarning {
    /// A stage was widened, but every frame reaching it belonged to one chunk.
    ///
    /// Its other inputs did carry chunk ids, so the stage that marks them is very likely on the
    /// wrong branch. Only one worker had anything to do.
    #[error(
        "node {node:?} saw every frame belong to one chunk while another input carried chunk ids, so the stage marking chunks is likely on the wrong branch"
    )]
    NeverChunked { node: NodeId },
}
