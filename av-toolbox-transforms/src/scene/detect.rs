use std::collections::VecDeque;

use av_toolbox_core::{Inputs, StageError, StageInstance, Transform};
use av_toolbox_gpu::{FrameContext, FrameFormat, HostFrame, HostSide, HostStreamItem};

use super::args::{DetectArgs, Mode};
use super::window::{Decision, LumaWindow, Settings};
use crate::split::Scenes;

/// Detects scene changes and stamps each frame with its chunk and score.
pub(super) struct Detect {
    window: LumaWindow,
    undecided: VecDeque<HostFrame>,
    chunks: Chunks,
}

/// How decided frames become chunks.
enum Chunks {
    /// One chunk per scene, counting the scenes started so far.
    Natural { scenes: u64 },
    /// Each scene held whole and split to a limit.
    Split(Scenes<HostFrame>),
}

impl Chunks {
    /// Places a decided frame, returning every frame now ready to hand on.
    fn place(&mut self, mut frame: HostFrame, starts_scene: bool) -> Result<Vec<HostFrame>, StageError> {
        match self {
            Self::Natural { scenes } => {
                if starts_scene {
                    *scenes += 1;
                }

                frame.meta_mut().chunk = *scenes - 1;

                Ok(vec![frame])
            },
            Self::Split(held) => held.push(frame, starts_scene),
        }
    }

    fn finish(&mut self) -> Result<Vec<HostFrame>, StageError> {
        match self {
            Self::Natural { .. } => Ok(Vec::new()),
            Self::Split(held) => held.finish(),
        }
    }
}

impl Detect {
    pub(super) fn new(args: &DetectArgs, format: &FrameFormat) -> Result<Self, StageError> {
        let settings = Settings::for_stream(format, args)?;
        let window = LumaWindow::new(settings);
        let chunks = match args.mode {
            Mode::Natural => Chunks::Natural { scenes: 0 },
            Mode::Split(limit) => Chunks::Split(Scenes::new(limit)),
        };

        Ok(Self {
            window,
            undecided: VecDeque::new(),
            chunks,
        })
    }

    /// Scores the oldest undecided frames with `decisions`, in order, and places them in chunks.
    fn stamp(&mut self, decisions: Vec<Decision>) -> Result<Vec<HostStreamItem>, StageError> {
        let mut emitted = Vec::new();
        for decision in decisions {
            let mut frame = self
                .undecided
                .pop_front()
                .expect("every decision is for a frame still held");
            frame.meta_mut().scene_score = decision.score;

            let ready = self.chunks.place(frame, decision.starts_scene)?;
            let items = ready.into_iter().map(HostStreamItem::Frame);
            emitted.extend(items);
        }

        Ok(emitted)
    }
}

impl StageInstance for Detect {}

impl Transform<HostSide> for Detect {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<Vec<HostStreamItem>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("scene/detect expected a frame in slot 0"))?;

        let decisions = self.window.push(&frame)?;
        self.undecided.push_back(frame);

        self.stamp(decisions)
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<HostStreamItem>, StageError> {
        let decisions = self.window.finish();
        let mut emitted = self.stamp(decisions)?;

        let held = self.chunks.finish()?;
        let items = held.into_iter().map(HostStreamItem::Frame);
        emitted.extend(items);

        Ok(emitted)
    }
}
