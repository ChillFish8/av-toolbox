//! Weight files for av-toolbox stages.
//!
//! A [Store] finds a pack's files in the models directory, falling back to the few built into the binary. Packs
//! are installed from zip archives, checking every file against the sha256 the [Registry] lists for it.

mod lookup;
mod registry;
mod root;

pub use self::lookup::{Located, MissingModel, Store, builtin};
pub use self::registry::{Pack, PackFile, Registry, RegistryError, registry, sha256_hex};
pub use self::root::{MODELS_ENV, TAG, default_root, installed_tags, models_root};
