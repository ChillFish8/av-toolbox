//! Building and running av-toolbox pipelines
//!
//! [registry] holds every stage the toolbox ships, and [open_backend] opens a device to run them on.

mod backend;
mod registry;

pub use self::backend::{OpenBackendError, open_backend};
pub use self::registry::registry;
