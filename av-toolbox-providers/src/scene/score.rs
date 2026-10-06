use std::collections::VecDeque;

use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    LayerDemand,
    PathEdge,
    PathView,
    PlaceError,
    Provider,
    ProviderDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    FrameFormat,
    HostFrame,
    HostSide,
    HostStreamItem,
    LayerId,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
};

use super::demand::{SceneCut, SceneScores, SceneSpeed, ScoreRequest, layer_requests};
use super::place::place;
use super::window::{Decision, DetectOptions, LOOKAHEAD, LumaWindow, Settings, check_layout};

/// The name the score provider gives itself in its errors.
const PROVIDER: &str = "scene-score";

/// Scores every frame as a scene cut once, for every layer demanding scores.
pub(crate) struct SceneScoreProvider;

impl StageDefinition for SceneScoreProvider {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "provider"
    }

    fn identifier() -> &'static str {
        "scene-score"
    }

    /// The detector's lookahead.
    fn held_frames(_args: &Self::Args, _formats: &ResolvedFormats) -> usize {
        LOOKAHEAD
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl ProviderDefinition for SceneScoreProvider {
    type Side = HostSide;
    type Demand = SceneScores;

    fn place(request: &ScoreRequest, path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
        let after = request.after.as_deref();

        place(after, path)
    }

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        offered: StreamFormat,
    ) -> Result<FormatSet, StageError> {
        let Some(format) = offered.as_frames() else {
            return Err(masks_offered());
        };

        check_layout(format)?;

        let accepted = match format.sample {
            SampleFormat::U8 | SampleFormat::U16 { .. } | SampleFormat::F32 => FormatSet::of([format.sample]),
            SampleFormat::F16 => FormatSet::new(),
        };

        Ok(accepted)
    }

    fn check(args: &Self::Args, _input: &FrameFormat) -> Result<(), StageError> {
        layer_requests::<SceneScores>(args, PROVIDER)?;

        Ok(())
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: &BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Provider<HostSide>>, StageError> {
        let input = formats.input(0).ok_or_else(missing_input_format)?;
        let format = input.as_frames().ok_or_else(masks_offered)?;

        let provider = ServeScores::new(&args, format)?;

        Ok(Box::new(provider))
    }
}

/// The score provider's instance, detecting cuts and writing each frame's [SceneCut] into every layer.
pub(crate) struct ServeScores {
    detection: Detection,
    layers: Vec<LayerId>,
}

impl ServeScores {
    pub(crate) fn new(args: &[LayerDemand], format: &FrameFormat) -> Result<Self, StageError> {
        let requests = layer_requests::<SceneScores>(args, PROVIDER)?;

        let speeds = requests.iter().map(|(_, request)| request.speed);
        let speed = SceneSpeed::slowest(speeds);
        let detection = Detection::new(format, speed)?;

        let layers = requests.iter().map(|(layer, _)| *layer).collect();

        Ok(Self { detection, layers })
    }

    /// Writes each decided frame's cut into every layer, in order.
    fn stamp(&self, decided: Vec<Decided>) -> Vec<HostStreamItem> {
        let mut stamped = Vec::with_capacity(decided.len());
        for Decided { mut frame, cut } in decided {
            let meta = frame.meta_mut();
            for layer in &self.layers {
                meta.provided.set(*layer, cut);
            }

            stamped.push(HostStreamItem::Frame(frame));
        }

        stamped
    }
}

impl StageInstance for ServeScores {}

impl Provider<HostSide> for ServeScores {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<Vec<HostStreamItem>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;

        let decided = self.detection.push(frame)?;

        Ok(self.stamp(decided))
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<HostStreamItem>, StageError> {
        let decided = self.detection.finish();

        Ok(self.stamp(decided))
    }
}

/// A frame the detector has decided, with its cut.
pub(crate) struct Decided {
    pub(crate) frame: HostFrame,
    pub(crate) cut: SceneCut,
}

/// Scene detection over the provider's input, keeping every cut and every frame's score.
pub(crate) struct Detection {
    window: LumaWindow,
    undecided: VecDeque<HostFrame>,
}

impl Detection {
    pub(crate) fn new(format: &FrameFormat, speed: SceneSpeed) -> Result<Self, StageError> {
        let options = DetectOptions {
            speed,
            min_scene_frames: 0,
            max_scene_frames: None,
        };
        let settings = Settings::for_stream(format, &options)?;
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

        Ok(self.pair(decisions))
    }

    /// Decides every frame still waiting, once the stream has ended.
    pub(crate) fn finish(&mut self) -> Vec<Decided> {
        let decisions = self.window.finish();

        self.pair(decisions)
    }

    /// Pairs the oldest undecided frames with `decisions`, in order.
    fn pair(&mut self, decisions: Vec<Decision>) -> Vec<Decided> {
        let mut decided = Vec::with_capacity(decisions.len());
        for decision in decisions {
            let frame = self
                .undecided
                .pop_front()
                .expect("every decision is for a frame still held");
            let cut = SceneCut {
                score: decision.score,
                starts_scene: decision.starts_scene,
            };

            decided.push(Decided { frame, cut });
        }

        decided
    }
}

fn missing_input_format() -> StageError {
    StageError::msg("the scene-score provider was built without its input's format")
}

fn masks_offered() -> StageError {
    StageError::msg("scene detection reads frames, not masks")
}

fn missing_frame() -> StageError {
    StageError::msg("the scene-score provider expected a frame in slot 0")
}
