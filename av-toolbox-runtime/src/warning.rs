use av_toolbox_graph::NodeId;
use thiserror::Error;

/// Something a run noticed that did not stop it.
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum RunWarning {
    /// A widened stage saw one chunk while another of its inputs was chunked.
    ///
    /// Its other inputs did carry chunk ids, so the stage that marks them is very likely on the
    /// wrong branch. Only one worker had anything to do.
    #[error(
        "node {node:?} saw every frame belong to one chunk while another of its inputs was chunked, so the stage marking chunks is likely on the wrong branch"
    )]
    DetectorOnWrongBranch { node: NodeId },
    /// A widened stage saw one chunk, and so did everything reaching it.
    ///
    /// A user who lands here either forgot to mark chunks upstream or gave the stage a
    /// granularity it does not need, and the engine cannot tell which, so the message names
    /// both fixes.
    #[error(
        "node {node:?} was widened but nothing reaching it marks chunks, so only one worker had anything to do. Mark chunks upstream, or declare the stage frame parallel if it does not need them"
    )]
    NothingChunked { node: NodeId },
}
