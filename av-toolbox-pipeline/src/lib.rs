//! Building and running av-toolbox pipelines
//!
//! A pipeline is read from TOML with [toml::parse], or built in code as a list of
//! [StageSpec](av_toolbox_graph::StageSpec) values. [Pipeline::build] checks it against a registry, such as
//! [registry], and [Pipeline::compile] plans it on a backend from [open_backend], ready to run.

mod backend;
mod error;
mod pipeline;
mod registry;
pub mod toml;

pub use self::backend::{OpenBackendError, open_backend};
pub use self::error::PipelineError;
pub use self::pipeline::{CompiledPipeline, LiveSource, Pipeline, StageRef};
pub use self::registry::registry;
