use thiserror::Error;

/// Something a run noticed that did not stop it.
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum RunWarning {}
