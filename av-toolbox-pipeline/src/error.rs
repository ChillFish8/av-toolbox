use av_toolbox_graph::Diagnostics;

/// A pipeline that failed to build, compile or start, with every problem found.
#[derive(Debug, thiserror::Error)]
#[error("{diagnostics}")]
pub struct PipelineError {
    diagnostics: Diagnostics,
}

impl PipelineError {
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    pub fn into_diagnostics(self) -> Diagnostics {
        self.diagnostics
    }
}

impl From<Diagnostics> for PipelineError {
    fn from(diagnostics: Diagnostics) -> Self {
        Self { diagnostics }
    }
}
