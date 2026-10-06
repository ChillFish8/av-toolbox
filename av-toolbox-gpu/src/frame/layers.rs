use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// One distinct request on one provider instance, numbered by the graph.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize, JsonSchema)]
pub struct LayerId(pub u32);

/// A frame's chunk in each scene layer it carries.
///
/// Entries are kept sorted by layer, so two sets holding the same chunks compare and hash
/// equal whatever order they were set in. A clone shares the entries until either side writes.
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash)]
pub struct Layers(Option<Arc<Vec<(LayerId, u64)>>>);

impl Layers {
    pub const fn new() -> Self {
        Self(None)
    }

    /// The frame's chunk in `layer`, 0 when the frame carries none for it.
    pub fn get(&self, layer: LayerId) -> u64 {
        let entries = self.entries();

        match position(entries, layer) {
            Ok(position) => entries[position].1,
            Err(_) => 0,
        }
    }

    pub fn set(&mut self, layer: LayerId, chunk: u64) {
        let shared = self.0.get_or_insert_with(Default::default);
        let entries = Arc::make_mut(shared);

        match position(entries, layer) {
            Ok(position) => entries[position].1 = chunk,
            Err(position) => entries.insert(position, (layer, chunk)),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (LayerId, u64)> + '_ {
        self.entries().iter().copied()
    }

    fn entries(&self) -> &[(LayerId, u64)] {
        match &self.0 {
            Some(entries) => entries,
            None => &[],
        }
    }
}

fn position(entries: &[(LayerId, u64)], layer: LayerId) -> Result<usize, usize> {
    entries.binary_search_by_key(&layer, |(entry, _)| *entry)
}
