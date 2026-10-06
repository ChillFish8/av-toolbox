use std::collections::VecDeque;

use av_toolbox_core::MetaItem;
use av_toolbox_gpu::LayerId;

use super::layers::LayerBreaks;
use crate::scene::{SceneChunk, SceneCut};

/// One layer with the chunks it has placed that no item has been stamped with yet.
struct HeldLayer {
    layer: LayerId,
    breaks: LayerBreaks,
    placed: VecDeque<u64>,
}

/// Holds scored items until every layer has placed them, then stamps and hands them on in order.
///
/// Items are held once however many layers there are, so they wait only on the slowest layer.
pub(crate) struct LayerHold {
    items: VecDeque<MetaItem>,
    layers: Vec<HeldLayer>,
}

impl LayerHold {
    /// Holds items for `layers`, which are placed in the order given.
    pub(crate) fn new(layers: Vec<(LayerId, LayerBreaks)>) -> Self {
        let layers = layers
            .into_iter()
            .map(|(layer, breaks)| HeldLayer {
                layer,
                breaks,
                placed: VecDeque::new(),
            })
            .collect();

        Self {
            items: VecDeque::new(),
            layers,
        }
    }

    /// Takes the next item with its cut, returning every item now placed by every layer.
    pub(crate) fn place(&mut self, item: MetaItem, cut: SceneCut) -> Vec<MetaItem> {
        self.items.push_back(item);

        for held in &mut self.layers {
            let placed = held.breaks.place(cut.score, cut.starts_scene);
            held.placed.extend(placed);
        }

        self.release()
    }

    /// Hands on every item still held, as the stream has ended.
    pub(crate) fn finish(&mut self) -> Vec<MetaItem> {
        for held in &mut self.layers {
            let placed = held.breaks.finish();
            held.placed.extend(placed);
        }

        self.release()
    }

    fn release(&mut self) -> Vec<MetaItem> {
        let mut released = Vec::new();
        while self.every_layer_placed_the_next() {
            let mut item = self
                .items
                .pop_front()
                .expect("an item is held while every layer has placed it");

            for held in &mut self.layers {
                let chunk = held
                    .placed
                    .pop_front()
                    .expect("every layer placed the item being released");
                let meta = item.meta_mut();
                meta.provided.set(held.layer, SceneChunk(chunk));
            }

            released.push(item);
        }

        released
    }

    fn every_layer_placed_the_next(&self) -> bool {
        if self.items.is_empty() {
            return false;
        }

        self.layers.iter().all(|held| !held.placed.is_empty())
    }
}
