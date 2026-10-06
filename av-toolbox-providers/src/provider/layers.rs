use av_toolbox_core::StageError;
use av_toolbox_gpu::{FrameRate, SceneScore};

use super::demand::SceneBreaks;
use crate::split::{frames_in, split_lengths};

/// How many times its limit a limited layer lets a scene run before closing it.
pub(crate) const LONGEST_SCENE: usize = 4;

/// One scene layer's breaks, derived from the raw cuts as frames are decided.
///
/// A layer sees each frame's score and whether it starts a scene, and answers with the chunk of
/// every frame it has now placed, in order. It may place a frame long after it saw it.
pub(crate) enum LayerBreaks {
    /// One chunk per scene, counting the scenes started so far.
    Natural {
        scenes: u64,
    },
    Limited(LimitedLayer),
}

impl LayerBreaks {
    /// The layer serving `breaks`, its limit worked out with `frame_rate`.
    pub(crate) fn new(breaks: &SceneBreaks, frame_rate: Option<FrameRate>) -> Result<Self, StageError> {
        let SceneBreaks::Limited {
            min_frames,
            max_frames,
            max_seconds,
        } = *breaks
        else {
            return Ok(Self::Natural { scenes: 0 });
        };

        let limit = limit_of(max_frames, max_seconds, frame_rate)?;
        let layer = LimitedLayer::new(min_frames, limit);

        Ok(Self::Limited(layer))
    }

    /// Takes the next decided frame, returning the chunk of every frame now placed.
    pub(crate) fn place(&mut self, score: Option<SceneScore>, starts_scene: bool) -> Vec<u64> {
        match self {
            Self::Natural { scenes } => {
                if starts_scene || *scenes == 0 {
                    *scenes += 1;
                }

                vec![*scenes - 1]
            },
            Self::Limited(layer) => layer.place(score, starts_scene),
        }
    }

    /// Places every frame still held, as the stream has ended.
    pub(crate) fn finish(&mut self) -> Vec<u64> {
        match self {
            Self::Natural { .. } => Vec::new(),
            Self::Limited(layer) => layer.finish(),
        }
    }
}

/// A `Limited` layer, which drops cuts too close to the last one and splits scenes to its limit.
pub(crate) struct LimitedLayer {
    min_frames: u64,
    limit: Option<usize>,
    /// How many frames this layer has seen.
    seen: u64,
    /// The frame the last kept break fell on.
    last_break: Option<u64>,
    /// The scores of the open scene's frames, while a limit holds it.
    open: Vec<Option<SceneScore>>,
    /// How many chunks have been handed out.
    chunks: u64,
}

impl LimitedLayer {
    pub(crate) fn new(min_frames: u32, limit: Option<usize>) -> Self {
        Self {
            min_frames: u64::from(min_frames),
            limit,
            seen: 0,
            last_break: None,
            open: Vec::new(),
            chunks: 0,
        }
    }

    fn place(&mut self, score: Option<SceneScore>, starts_scene: bool) -> Vec<u64> {
        let position = self.seen;
        self.seen += 1;

        let kept = match self.last_break {
            None => true,
            Some(last) => starts_scene && position - last >= self.min_frames,
        };

        let Some(limit) = self.limit else {
            if kept {
                self.last_break = Some(position);
                self.chunks += 1;
            }

            return vec![self.chunks - 1];
        };

        let full = self.open.len() >= LONGEST_SCENE * limit;
        let mut placed = Vec::new();
        if kept || full {
            self.last_break = Some(position);
            placed = self.close(limit);
        }

        self.open.push(score);

        placed
    }

    fn finish(&mut self) -> Vec<u64> {
        match self.limit {
            Some(limit) => self.close(limit),
            None => Vec::new(),
        }
    }

    /// Splits the open scene into chunks no longer than `limit`, returning each frame's chunk.
    fn close(&mut self, limit: usize) -> Vec<u64> {
        let scores = std::mem::take(&mut self.open);
        let lengths = split_lengths(&scores, limit);

        let mut placed = Vec::with_capacity(scores.len());
        for length in lengths {
            let chunk = std::iter::repeat_n(self.chunks, length);
            placed.extend(chunk);
            self.chunks += 1;
        }

        placed
    }
}

/// The longest chunk a limited layer allows, in frames, or `None` when it sets no limit.
///
/// Seconds become frames through `frame_rate`. When both limits are given, the smaller wins.
pub(crate) fn limit_of(
    max_frames: Option<u32>,
    max_seconds: Option<f64>,
    frame_rate: Option<FrameRate>,
) -> Result<Option<usize>, StageError> {
    if max_frames == Some(0) {
        return Err(StageError::msg("a stage asks for scenes of at most 0 frames"));
    }

    let frame_limit = max_frames.map(|frames| frames as usize);
    let Some(seconds) = max_seconds else {
        return Ok(frame_limit);
    };

    if !(seconds.is_finite() && seconds > 0.0) {
        let message = format!("a stage asks for scenes of at most {seconds} seconds, which is not above 0");
        return Err(StageError::msg(message));
    }

    let Some(rate) = frame_rate else {
        return match frame_limit {
            Some(frames) => Ok(Some(frames)),
            None => Err(no_frame_rate(seconds)),
        };
    };

    let frame_duration = f64::from(rate.denominator()) / f64::from(rate.numerator());
    let seconds_limit = frames_in(seconds, frame_duration);

    let limit = match frame_limit {
        Some(frames) => frames.min(seconds_limit),
        None => seconds_limit,
    };

    Ok(Some(limit))
}

fn no_frame_rate(seconds: f64) -> StageError {
    let message = format!(
        "a stage asks for scenes of at most {seconds} seconds, but the stream has no frame rate. Give `max_frames` too."
    );

    StageError::msg(message)
}
