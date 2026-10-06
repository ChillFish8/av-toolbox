use std::fmt;
use std::marker::PhantomData;

use av_toolbox_gpu::{FrameMeta, LayerId};
use thiserror::Error;

use crate::DemandDefinition;

/// A provided value that names the chunk a frame belongs to.
pub trait ChunkKey {
    fn chunk(&self) -> u64;
}

/// Reads one demand's value from frames.
///
/// It is built from a stage's bound inputs.
pub struct Provision<D: DemandDefinition> {
    layer: LayerId,
    marker: PhantomData<fn() -> D>,
}

impl<D: DemandDefinition> Clone for Provision<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: DemandDefinition> Copy for Provision<D> {}

impl<D: DemandDefinition> fmt::Debug for Provision<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Provision")
            .field("kind", &D::kind())
            .field("layer", &self.layer)
            .finish()
    }
}

impl<D: DemandDefinition> Provision<D> {
    pub(crate) const fn new(layer: LayerId) -> Self {
        Self {
            layer,
            marker: PhantomData,
        }
    }

    pub const fn layer(&self) -> LayerId {
        self.layer
    }

    /// The value provided on `meta`, if the frame was given one.
    pub fn get<'a>(&self, meta: &'a FrameMeta) -> Option<&'a D::Output> {
        meta.provided.get::<D::Output>(self.layer)
    }
}

/// Why a stage could not read a demand's provided value.
#[derive(Debug, Error)]
pub enum ProvisionError {
    #[error("`{input}` demanded no {kind}")]
    NotDemanded { input: String, kind: &'static str },
    #[error("`{input}` demanded {kind} more than once, so which one to read is ambiguous")]
    Ambiguous { input: String, kind: &'static str },
}
