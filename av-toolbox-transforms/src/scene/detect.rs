use std::collections::VecDeque;

use av_toolbox_core::{Inputs, StageError, StageInstance, Transform};
use av_toolbox_gpu::{FrameContext, FrameFormat, HostFrame, HostSide, HostStreamItem};

use super::args::DetectArgs;
use super::window::{Decision, LumaWindow, Settings};

/// Detects scene changes and stamps each frame with its chunk and score.
pub(super) struct Detect {
    window: LumaWindow,
    undecided: VecDeque<HostFrame>,
    scenes: u64,
}

impl Detect {
    pub(super) fn new(args: &DetectArgs, format: &FrameFormat) -> Result<Self, StageError> {
        let settings = Settings::for_stream(format, args)?;
        let window = LumaWindow::new(settings);

        Ok(Self {
            window,
            undecided: VecDeque::new(),
            scenes: 0,
        })
    }

    /// Stamps the oldest undecided frames with `decisions`, in order, and hands them on.
    fn stamp(&mut self, decisions: Vec<Decision>) -> Vec<HostStreamItem> {
        let mut emitted = Vec::with_capacity(decisions.len());
        for decision in decisions {
            let mut frame = self
                .undecided
                .pop_front()
                .expect("every decision is for a frame still held");

            if decision.starts_scene {
                self.scenes += 1;
            }

            let meta = frame.meta_mut();
            meta.chunk = self.scenes - 1;
            meta.scene_score = decision.score;

            emitted.push(HostStreamItem::Frame(frame));
        }

        emitted
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

        let emitted = self.stamp(decisions);

        Ok(emitted)
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<HostStreamItem>, StageError> {
        let decisions = self.window.finish();
        let emitted = self.stamp(decisions);

        Ok(emitted)
    }
}
