use std::collections::VecDeque;

use av_toolbox_gpu::{HostFrame, LayerId};

use super::layers::LayerBreaks;

/// One layer with the chunks it has placed that no frame has been stamped with yet.
struct HeldLayer {
    layer: LayerId,
    breaks: LayerBreaks,
    placed: VecDeque<u64>,
}

/// Holds decided frames until every layer has placed them, then stamps and hands them on in order.
///
/// Frames are held once however many layers there are, so they wait only on the slowest layer.
pub(crate) struct LayerHold {
    frames: VecDeque<HostFrame>,
    layers: Vec<HeldLayer>,
}

impl LayerHold {
    /// Holds frames for `layers`, which are placed in the order given.
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
            frames: VecDeque::new(),
            layers,
        }
    }

    /// Takes the next decided frame, returning every frame now placed by every layer.
    pub(crate) fn place(&mut self, frame: HostFrame, starts_scene: bool) -> Vec<HostFrame> {
        let score = frame.meta().scene_score;
        self.frames.push_back(frame);

        for held in &mut self.layers {
            let placed = held.breaks.place(score, starts_scene);
            held.placed.extend(placed);
        }

        self.release()
    }

    /// Hands on every frame still held, as the stream has ended.
    pub(crate) fn finish(&mut self) -> Vec<HostFrame> {
        for held in &mut self.layers {
            let placed = held.breaks.finish();
            held.placed.extend(placed);
        }

        self.release()
    }

    fn release(&mut self) -> Vec<HostFrame> {
        let mut released = Vec::new();
        while self.every_layer_placed_the_next() {
            let mut frame = self
                .frames
                .pop_front()
                .expect("a frame is held while every layer has placed it");

            for held in &mut self.layers {
                let chunk = held
                    .placed
                    .pop_front()
                    .expect("every layer placed the frame being released");
                frame.meta_mut().chunks.set(held.layer, chunk);
            }

            released.push(frame);
        }

        released
    }

    fn every_layer_placed_the_next(&self) -> bool {
        if self.frames.is_empty() {
            return false;
        }

        self.layers.iter().all(|held| !held.placed.is_empty())
    }
}
