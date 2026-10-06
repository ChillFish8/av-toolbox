use std::borrow::Cow;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use av_scenechange::{Rational32, SceneChangeDetector, SceneDetectionSpeed, ScenecutResult};
use av_toolbox_core::StageError;
use av_toolbox_gpu::{Channel, FrameFormat, FrameLayout, HostFrame, SampleFormat, SceneScore};
use v_frame::chroma::ChromaSubsampling;
use v_frame::frame::{Frame, FrameBuilder};
use v_frame::pixel::Pixel;

use super::args::DetectArgs;

/// How many frames past the one being analysed the detector reads.
pub(super) const LOOKAHEAD: usize = 5;

/// The precision float luma is read at.
const FLOAT_BITS: u8 = 16;

/// Padding around each luma plane, wide enough for the detector's motion search to read past
/// every edge without going out of bounds.
const LUMA_PADDING: usize = 88;

/// What the detector decided about one frame.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) struct Decision {
    pub(super) starts_scene: bool,
    pub(super) score: Option<SceneScore>,
}

impl Decision {
    /// The first frame, which always starts a scene and is never scored.
    pub(super) const FIRST: Self = Self {
        starts_scene: true,
        score: None,
    };
}

/// The detector's settings for one stream.
#[derive(Debug, Copy, Clone)]
pub(super) struct Settings {
    width: usize,
    height: usize,
    bit_depth: u8,
    speed: SceneDetectionSpeed,
    min_scene_frames: usize,
    max_scene_frames: usize,
}

impl Settings {
    /// The settings for frames of `format`, or an error when the detector cannot read them.
    pub(super) fn for_stream(format: &FrameFormat, args: &DetectArgs) -> Result<Self, StageError> {
        check_layout(format)?;

        let bit_depth = match format.sample {
            SampleFormat::U8 => 8,
            SampleFormat::U16 { depth } => depth.bits(),
            SampleFormat::F32 => FLOAT_BITS,
            SampleFormat::F16 => {
                let message = format!("scene/detect reads integer samples, not {:?}", format.sample);
                return Err(StageError::msg(message));
            },
        };

        let max_scene_frames = args.max_scene_frames.map_or(usize::MAX, NonZeroUsize::get);

        Ok(Self {
            width: format.width as usize,
            height: format.height as usize,
            bit_depth,
            speed: args.speed.detection_speed(),
            min_scene_frames: args.min_scene_frames,
            max_scene_frames,
        })
    }

    fn detector<T: Pixel>(&self, frame_rate: Rational32) -> SceneChangeDetector<T> {
        let resolution = (self.width, self.height);
        let bit_depth = usize::from(self.bit_depth);

        SceneChangeDetector::new(
            resolution,
            bit_depth,
            frame_rate,
            ChromaSubsampling::Monochrome,
            LOOKAHEAD,
            self.speed,
            self.min_scene_frames,
            self.max_scene_frames,
        )
    }
}

/// Refuses a layout with no luma to read.
pub(super) fn check_layout(format: &FrameFormat) -> Result<(), StageError> {
    if matches!(format.layout, FrameLayout::Rgb) {
        return Err(StageError::msg(
            "scene/detect reads luma, which an RGB frame does not carry",
        ));
    }

    if matches!(format.layout, FrameLayout::Chroma { .. }) {
        return Err(StageError::msg(
            "scene/detect reads YUV or Grey frames, not chroma planes",
        ));
    }

    Ok(())
}

/// The detector's frame rate, from the gap between the first two frames, or 30 fps without one.
pub(super) fn frame_rate(first: Option<Duration>, second: Option<Duration>) -> Rational32 {
    let fallback = Rational32::new(30, 1);
    let (Some(first), Some(second)) = (first, second) else {
        return fallback;
    };

    let gap = second.saturating_sub(first);
    let micros = gap.as_micros();
    let Ok(micros) = i32::try_from(micros) else {
        return fallback;
    };

    if micros == 0 {
        return fallback;
    }

    Rational32::new(1_000_000, micros)
}

/// Scene detection over a sliding window of luma planes, for whichever sample type the stream has.
pub(super) enum LumaWindow {
    Eight(Window<u8>),
    Deep(Window<u16>),
}

impl LumaWindow {
    pub(super) fn new(settings: Settings) -> Self {
        if settings.bit_depth == 8 {
            return Self::Eight(Window::new(settings));
        }

        Self::Deep(Window::new(settings))
    }

    /// Takes the next frame, returning a decision for every frame that can now be decided.
    pub(super) fn push(&mut self, frame: &HostFrame) -> Result<Vec<Decision>, StageError> {
        match self {
            Self::Eight(window) => window.push(frame),
            Self::Deep(window) => window.push(frame),
        }
    }

    /// Decides every frame still waiting, once the stream has ended.
    pub(super) fn finish(&mut self) -> Vec<Decision> {
        match self {
            Self::Eight(window) => window.finish(),
            Self::Deep(window) => window.finish(),
        }
    }
}

/// Scene detection over a sliding window of luma planes.
///
/// A frame is decided once the lookahead past it has arrived, or once the stream ends.
pub(super) struct Window<T: Pixel> {
    settings: Settings,
    detector: Option<SceneChangeDetector<T>>,
    first_timestamp: Option<Duration>,
    /// Luma from the frame before the next one to analyse onward.
    frames: VecDeque<Arc<Frame<T>>>,
    received: usize,
    next: usize,
    last_cut: usize,
}

impl<T: Pixel> Window<T> {
    fn new(settings: Settings) -> Self {
        Self {
            settings,
            detector: None,
            first_timestamp: None,
            frames: VecDeque::new(),
            received: 0,
            next: 0,
            last_cut: 0,
        }
    }

    fn push(&mut self, frame: &HostFrame) -> Result<Vec<Decision>, StageError> {
        let luma = self.luma_of(frame)?;
        let timestamp = frame.meta().timestamp;
        self.frames.push_back(luma);
        self.received += 1;

        if self.received == 1 {
            self.first_timestamp = timestamp;
            self.next = 1;
            return Ok(vec![Decision::FIRST]);
        }

        if self.detector.is_none() {
            let rate = frame_rate(self.first_timestamp, timestamp);
            let detector = self.settings.detector(rate);
            self.detector = Some(detector);
        }

        let mut decisions = Vec::new();
        while self.next + LOOKAHEAD < self.received {
            let decision = self.analyse_next();
            decisions.push(decision);
        }

        Ok(decisions)
    }

    fn finish(&mut self) -> Vec<Decision> {
        let mut decisions = Vec::new();
        while self.next < self.received {
            let decision = self.analyse_next();
            decisions.push(decision);
        }

        decisions
    }

    fn analyse_next(&mut self) -> Decision {
        let detector = self
            .detector
            .as_mut()
            .expect("the detector is built once a second frame arrives");
        let frame_set: Vec<&Arc<Frame<T>>> = self.frames.iter().take(LOOKAHEAD + 2).collect();

        let (starts_scene, result) = detector.analyze_next_frame(&frame_set, self.next, self.last_cut);
        if starts_scene {
            self.last_cut = self.next;
        }

        self.frames.pop_front();
        self.next += 1;

        let score = result.as_ref().and_then(score_of);

        Decision { starts_scene, score }
    }

    fn luma_of(&self, frame: &HostFrame) -> Result<Arc<Frame<T>>, StageError> {
        let plane = frame
            .plane(Channel::Y)
            .ok_or_else(|| StageError::msg("scene/detect needs a luma plane"))?;

        let builder = FrameBuilder::new(
            self.settings.width,
            self.settings.height,
            ChromaSubsampling::Monochrome,
            self.settings.bit_depth,
        )
        .luma_padding_left(LUMA_PADDING)
        .luma_padding_right(LUMA_PADDING)
        .luma_padding_top(LUMA_PADDING)
        .luma_padding_bottom(LUMA_PADDING);
        let built = builder.build::<T>();
        let mut luma =
            built.map_err(|error| StageError::msg(format!("could not hold a frame's luma: {error:?}")))?;

        let samples = luma_samples(frame, plane.bytes());
        let copied = luma.y_plane.copy_from_u8_slice(&samples);
        copied.map_err(|error| {
            StageError::msg(format!("a frame's luma does not match its format: {error:?}"))
        })?;

        Ok(Arc::new(luma))
    }
}

/// The luma samples the detector reads, with float luma quantised to [FLOAT_BITS] bits.
fn luma_samples<'a>(frame: &HostFrame, bytes: &'a [u8]) -> Cow<'a, [u8]> {
    if frame.format().sample != SampleFormat::F32 {
        return Cow::Borrowed(bytes);
    }

    let peak = f32::from(u16::MAX);
    let (floats, _) = bytes.as_chunks::<4>();

    let mut quantised = Vec::with_capacity(floats.len() * 2);
    for float in floats {
        let value = f32::from_le_bytes(*float);
        let clamped = value.clamp(0.0, 1.0);
        let code = (clamped * peak).round() as u16;
        quantised.extend_from_slice(&code.to_le_bytes());
    }

    Cow::Owned(quantised)
}

/// The frame's cost as a cut relative to the cost the detector needs to call one.
///
/// `None` when the threshold is 0, such as on a flat frame, which would otherwise divide out to
/// an infinite or NaN score.
fn score_of(result: &ScenecutResult) -> Option<SceneScore> {
    let ratio = result.inter_cost / result.threshold;
    if !ratio.is_finite() {
        return None;
    }

    Some(SceneScore::new(ratio as f32))
}
