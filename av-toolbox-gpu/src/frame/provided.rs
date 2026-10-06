use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::frame::gpu::FrameMeta;
use crate::frame::layers::LayerId;

/// A value a provider wrote onto a frame, carried without its type.
pub trait ProvidedValue: Any + Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn eq_value(&self, other: &dyn ProvidedValue) -> bool;

    fn hash_value(&self, state: &mut dyn Hasher);
}

impl<T: Any + Clone + Eq + Hash + Debug + Send + Sync> ProvidedValue for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn eq_value(&self, other: &dyn ProvidedValue) -> bool {
        match other.as_any().downcast_ref::<T>() {
            Some(other) => self == other,
            None => false,
        }
    }

    fn hash_value(&self, state: &mut dyn Hasher) {
        let mut adaptor = HasherWrite(state);
        self.hash(&mut adaptor);
    }
}

struct HasherWrite<'a>(&'a mut dyn Hasher);

impl Hasher for HasherWrite<'_> {
    fn finish(&self) -> u64 {
        self.0.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}

type Entries = Vec<(LayerId, Arc<dyn ProvidedValue>)>;

/// The values providers wrote onto a frame, one for each layer it carries.
///
/// Entries are kept sorted by layer, so two stores holding equal values compare and hash equal
/// whatever order they were set in. A clone shares the entries until either side writes.
#[derive(Debug, Default, Clone)]
pub struct Provided(Option<Arc<Entries>>);

impl Provided {
    pub const fn new() -> Self {
        Self(None)
    }

    /// Stores `value` for `layer`, replacing any value already there.
    pub fn set<T: ProvidedValue>(&mut self, layer: LayerId, value: T) {
        let shared = self.0.get_or_insert_with(Default::default);
        let entries = Arc::make_mut(shared);
        let value: Arc<dyn ProvidedValue> = Arc::new(value);

        match position(entries, layer) {
            Ok(position) => entries[position].1 = value,
            Err(position) => entries.insert(position, (layer, value)),
        }
    }

    /// The value held for `layer`, without its type.
    pub fn get_raw(&self, layer: LayerId) -> Option<&dyn ProvidedValue> {
        let entries = self.entries();

        match position(entries, layer) {
            Ok(position) => Some(entries[position].1.as_ref()),
            Err(_) => None,
        }
    }

    /// The value held for `layer`, when it is a `T`.
    pub fn get<T: ProvidedValue>(&self, layer: LayerId) -> Option<&T> {
        let value = self.get_raw(layer)?;
        value.as_any().downcast_ref::<T>()
    }

    /// The layers that hold a value, in order.
    pub fn layers(&self) -> impl Iterator<Item = LayerId> + '_ {
        self.entries().iter().map(|(layer, _)| *layer)
    }

    fn entries(&self) -> &[(LayerId, Arc<dyn ProvidedValue>)] {
        match &self.0 {
            Some(entries) => entries,
            None => &[],
        }
    }
}

fn position(entries: &[(LayerId, Arc<dyn ProvidedValue>)], layer: LayerId) -> Result<usize, usize> {
    entries.binary_search_by_key(&layer, |(entry, _)| *entry)
}

impl PartialEq for Provided {
    fn eq(&self, other: &Self) -> bool {
        let ours = self.entries();
        let theirs = other.entries();
        if ours.len() != theirs.len() {
            return false;
        }

        ours.iter()
            .zip(theirs)
            .all(|((our_layer, our_value), (their_layer, their_value))| {
                our_layer == their_layer && our_value.eq_value(their_value.as_ref())
            })
    }
}

impl Eq for Provided {}

impl Hash for Provided {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let entries = self.entries();
        entries.len().hash(state);

        for (layer, value) in entries {
            layer.hash(state);
            value.hash_value(state);
        }
    }
}

/// Reads the chunk a provided value names.
pub type ChunkKeyFn = fn(&dyn ProvidedValue) -> u64;

/// The layer a node routes its chunks on, and how to read a chunk from that layer's value.
///
/// Two routes are equal when their layers are, since a layer always holds one kind of value.
#[derive(Debug, Copy, Clone)]
pub struct ChunkRoute {
    pub layer: LayerId,
    pub key: ChunkKeyFn,
}

impl ChunkRoute {
    /// The chunk `meta` is in on this route, 0 when it carries no value for the layer.
    pub fn chunk_of(&self, meta: &FrameMeta) -> u64 {
        match meta.provided.get_raw(self.layer) {
            Some(value) => (self.key)(value),
            None => 0,
        }
    }
}

impl PartialEq for ChunkRoute {
    fn eq(&self, other: &Self) -> bool {
        self.layer == other.layer
    }
}

impl Eq for ChunkRoute {}
