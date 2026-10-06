use std::collections::VecDeque;

use av_toolbox_core::StageError;
use av_toolbox_gpu::{FrameFormat, HostFrame};

use crate::scene::args::{DetectArgs, Mode, Speed};
use crate::scene::window::{Decision, LumaWindow, Settings};

/// A frame the detector has decided, scored, with whether it starts a scene.
pub(crate) struct Decided {
    pub(crate) frame: HostFrame,
    pub(crate) starts_scene: bool,
}

/// Scene detection over the provider's input, keeping every cut and every frame's score.
pub(crate) struct Detection {
    window: LumaWindow,
    undecided: VecDeque<HostFrame>,
}

impl Detection {
    pub(crate) fn new(format: &FrameFormat, speed: Speed) -> Result<Self, StageError> {
        let args = DetectArgs {
            mode: Mode::Natural,
            speed,
            min_scene_frames: 0,
            max_scene_frames: None,
        };
        let settings = Settings::for_stream(format, &args)?;
        let window = LumaWindow::new(settings);

        Ok(Self {
            window,
            undecided: VecDeque::new(),
        })
    }

    /// Takes the next frame, returning every frame that can now be decided.
    pub(crate) fn push(&mut self, frame: HostFrame) -> Result<Vec<Decided>, StageError> {
        let decisions = self.window.push(&frame)?;
        self.undecided.push_back(frame);

        Ok(self.score(decisions))
    }

    /// Decides every frame still waiting, once the stream has ended.
    pub(crate) fn finish(&mut self) -> Vec<Decided> {
        let decisions = self.window.finish();

        self.score(decisions)
    }

    /// Scores the oldest undecided frames with `decisions`, in order.
    fn score(&mut self, decisions: Vec<Decision>) -> Vec<Decided> {
        let mut decided = Vec::with_capacity(decisions.len());
        for decision in decisions {
            let mut frame = self
                .undecided
                .pop_front()
                .expect("every decision is for a frame still held");
            frame.meta_mut().scene_score = decision.score;

            decided.push(Decided {
                frame,
                starts_scene: decision.starts_scene,
            });
        }

        decided
    }
}
