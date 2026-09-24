use av_toolbox_core::{MetaItem, StageError};
use av_toolbox_gpu::{FrameMeta, HostFrame, SceneScore};

use super::limit::SplitLimit;
use super::rule::split_lengths;

/// An item whose place in the stream can be read and restamped.
pub(crate) trait Stamped {
    fn meta(&self) -> &FrameMeta;

    fn meta_mut(&mut self) -> &mut FrameMeta;
}

impl Stamped for HostFrame {
    fn meta(&self) -> &FrameMeta {
        HostFrame::meta(self)
    }

    fn meta_mut(&mut self) -> &mut FrameMeta {
        HostFrame::meta_mut(self)
    }
}

impl Stamped for MetaItem {
    fn meta(&self) -> &FrameMeta {
        MetaItem::meta(self)
    }

    fn meta_mut(&mut self) -> &mut FrameMeta {
        MetaItem::meta_mut(self)
    }
}

/// Holds each scene whole until it ends, then cuts it into chunks no longer than a limit.
pub(crate) struct Scenes<I> {
    limit: SplitLimit,
    open: Vec<I>,
    next_chunk: u64,
}

impl<I: Stamped> Scenes<I> {
    pub(crate) fn new(limit: SplitLimit) -> Self {
        Self {
            limit,
            open: Vec::new(),
            next_chunk: 0,
        }
    }

    /// Adds the next item, first handing on the open scene when `item` starts a new one.
    pub(crate) fn push(&mut self, item: I, starts_scene: bool) -> Result<Vec<I>, StageError> {
        let mut emitted = Vec::new();
        if starts_scene && !self.open.is_empty() {
            emitted = self.close()?;
        }

        self.open.push(item);

        Ok(emitted)
    }

    /// Hands on the open scene, as the stream has ended.
    pub(crate) fn finish(&mut self) -> Result<Vec<I>, StageError> {
        self.close()
    }

    fn close(&mut self) -> Result<Vec<I>, StageError> {
        let scene = std::mem::take(&mut self.open);
        if scene.is_empty() {
            return Ok(scene);
        }

        let scores: Vec<Option<SceneScore>> = scene.iter().map(|item| item.meta().scene_score).collect();
        let first = scene.first().and_then(|item| item.meta().timestamp);
        let last = scene.last().and_then(|item| item.meta().timestamp);
        let limit = self.limit.frames_for(scene.len(), first, last)?;
        let lengths = split_lengths(&scores, limit);

        let mut emitted = Vec::with_capacity(scene.len());
        let mut items = scene.into_iter();
        for length in lengths {
            for mut item in items.by_ref().take(length) {
                item.meta_mut().chunk = self.next_chunk;
                emitted.push(item);
            }

            self.next_chunk += 1;
        }

        Ok(emitted)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    impl Stamped for FrameMeta {
        fn meta(&self) -> &FrameMeta {
            self
        }

        fn meta_mut(&mut self) -> &mut FrameMeta {
            self
        }
    }

    fn frames_limit(frames: usize) -> SplitLimit {
        SplitLimit::new(Some(frames), None).expect("a valid limit")
    }

    fn chunks_of(items: &[FrameMeta]) -> Vec<u64> {
        items.iter().map(|meta| meta.chunk).collect()
    }

    #[test]
    fn a_scene_is_held_until_the_next_one_starts() {
        let mut scenes = Scenes::new(frames_limit(10));

        let first = scenes.push(FrameMeta::at(0), true).expect("pushes");
        let second = scenes.push(FrameMeta::at(1), false).expect("pushes");
        let third = scenes.push(FrameMeta::at(2), false).expect("pushes");
        let next = scenes.push(FrameMeta::at(3), true).expect("pushes");

        assert!(first.is_empty() && second.is_empty() && third.is_empty());
        assert_eq!(chunks_of(&next), vec![0, 0, 0]);
    }

    #[test]
    fn a_long_scene_is_cut_into_chunks() {
        let mut scenes = Scenes::new(frames_limit(10));
        for index in 0..25 {
            let pushed = scenes.push(FrameMeta::at(index), index == 0).expect("pushes");
            assert!(pushed.is_empty());
        }

        let emitted = scenes.finish().expect("finishes");

        let mut expected = vec![0; 8];
        expected.extend([1; 8]);
        expected.extend([2; 9]);
        assert_eq!(chunks_of(&emitted), expected);
    }

    #[test]
    fn chunks_count_on_across_scenes() {
        let mut scenes = Scenes::new(frames_limit(10));
        let mut emitted = Vec::new();
        for index in 0..10 {
            let pushed = scenes.push(FrameMeta::at(index), index % 5 == 0).expect("pushes");
            emitted.extend(pushed);
        }

        let rest = scenes.finish().expect("finishes");
        emitted.extend(rest);

        assert_eq!(chunks_of(&emitted), vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn finishing_with_nothing_held_emits_nothing() {
        let mut scenes: Scenes<FrameMeta> = Scenes::new(frames_limit(10));

        let emitted = scenes.finish().expect("finishes");

        assert!(emitted.is_empty());
    }

    #[test]
    fn a_scene_measured_in_seconds_uses_its_timestamps() {
        let limit = SplitLimit::new(None, Some(Some(0.2))).expect("a valid limit");
        let mut scenes = Scenes::new(limit);
        for index in 0..10 {
            let timestamp = Duration::from_millis(index * 50);
            let meta = FrameMeta {
                timestamp: Some(timestamp),
                ..FrameMeta::at(index)
            };
            let pushed = scenes.push(meta, index == 0).expect("pushes");
            assert!(pushed.is_empty());
        }

        let emitted = scenes.finish().expect("finishes");

        assert_eq!(chunks_of(&emitted), vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn a_scene_without_timestamps_fails_on_seconds_alone() {
        let limit = SplitLimit::new(None, Some(Some(0.2))).expect("a valid limit");
        let mut scenes = Scenes::new(limit);
        for index in 0..10 {
            let pushed = scenes.push(FrameMeta::at(index), index == 0).expect("pushes");
            assert!(pushed.is_empty());
        }

        let emitted = scenes.finish();

        assert!(emitted.is_err());
    }
}
