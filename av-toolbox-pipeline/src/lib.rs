//! Building and running av-toolbox pipelines
//!
//! A pipeline is read from its TOML files with [load], which fills in variables and expands calls
//! to other files, or built in code as a list of [StageSpec](av_toolbox_graph::StageSpec) values.
//! [Pipeline::build] checks it against a registry, such as [registry], and [Pipeline::compile]
//! plans it on a backend from [open_backend], ready to run.

mod backend;
mod error;
mod load;
mod pipeline;
mod registry;
#[cfg(test)]
mod testing;
pub mod toml;
pub mod variables;

pub use self::backend::{OpenBackendError, open_backend};
pub use self::error::PipelineError;
pub use self::load::{DiskFiles, Files, LoadFailure, Loaded, SourceFile, SourceMap, load};
pub use self::pipeline::{CompiledPipeline, LiveSource, Pipeline, StageRef};
pub use self::registry::registry;
pub use self::variables::Supplied;
