use av_toolbox_core::{
    AnyDemand,
    BoundInputs,
    InputDemand,
    InputParam,
    InputSignature,
    LayerDemand,
    MetaItem,
    MetaProvider,
    MetaSide,
    PathEdge,
    PathView,
    PlaceError,
    ProviderDefinition,
    Provision,
    ProvisionError,
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
    FrameRate,
    LayerId,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};

use super::hold::LayerHold;
use super::layers::{LONGEST_SCENE, LayerBreaks, limit_of};
use crate::scene::{
    SceneBreaks,
    SceneRequest,
    SceneScores,
    SceneSpeed,
    Scenes,
    ScoreRequest,
    layer_requests,
    place,
};

/// The name the split provider gives itself in its errors.
const PROVIDER: &str = "scene-split";

/// Breaks the stream into scenes for every layer demanding them, from the cuts `scene-score` found.
pub(crate) struct SceneSplitProvider;

impl StageDefinition for SceneSplitProvider {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "provider"
    }

    fn identifier() -> &'static str {
        "scene-split"
    }

    /// The longest scene any limited layer holds.
    fn held_frames(args: &Self::Args, formats: &ResolvedFormats) -> usize {
        let input = formats.input(0);
        let frame_rate = input
            .as_ref()
            .and_then(StreamFormat::as_frames)
            .and_then(|format| format.frame_rate);

        let requests = requests(args).unwrap_or_default();
        let longest = requests
            .iter()
            .filter_map(|(_, request)| limit_for(request, frame_rate))
            .max()
            .unwrap_or(0);

        LONGEST_SCENE * longest
    }

    /// One score request, at the slowest speed any layer asks for and after the first layer's stage.
    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        let requests = requests(args).unwrap_or_default();
        let Some((_, first)) = requests.first() else {
            return Vec::new();
        };

        let speeds = requests.iter().map(|(_, request)| request.speed);
        let request = ScoreRequest {
            speed: SceneSpeed::slowest(speeds),
            after: first.after.clone(),
        };

        vec![InputDemand {
            input: "input",
            demand: AnyDemand::new::<SceneScores>(request),
        }]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl ProviderDefinition for SceneSplitProvider {
    type Side = MetaSide;
    type Demand = Scenes;

    fn place(request: &SceneRequest, path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
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

        Ok(FormatSet::of([format.sample]))
    }

    fn check(args: &Self::Args, input: &FrameFormat) -> Result<(), StageError> {
        for (_, request) in requests(args)? {
            request.validate()?;
            LayerBreaks::new(&request.breaks, input.frame_rate)?;
        }

        Ok(())
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        inputs: &BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaProvider>, StageError> {
        let input = formats.input(0).ok_or_else(missing_input_format)?;
        let format = input.as_frames().ok_or_else(masks_offered)?;
        let scores = inputs
            .provision::<SceneScores>("input")
            .map_err(unprovided_scores)?;

        let provider = ServeSplits::new(&args, format.frame_rate, scores)?;

        Ok(Box::new(provider))
    }
}

/// The split provider's instance, placing every layer's chunks from each item's cut.
pub(crate) struct ServeSplits {
    scores: Provision<SceneScores>,
    hold: LayerHold,
}

impl ServeSplits {
    pub(crate) fn new(
        args: &[LayerDemand],
        frame_rate: Option<FrameRate>,
        scores: Provision<SceneScores>,
    ) -> Result<Self, StageError> {
        let requests = requests(args)?;

        let mut layers = Vec::with_capacity(requests.len());
        for (layer, request) in &requests {
            let breaks = LayerBreaks::new(&request.breaks, frame_rate)?;
            layers.push((*layer, breaks));
        }

        let hold = LayerHold::new(layers);

        Ok(Self { scores, hold })
    }
}

impl StageInstance for ServeSplits {}

impl MetaProvider for ServeSplits {
    fn process(&mut self, _context: &mut FrameContext, item: MetaItem) -> Result<Vec<MetaItem>, StageError> {
        let meta = item.meta();
        let cut = self.scores.get(meta).copied();
        let cut = cut.ok_or_else(unscored_frame)?;

        Ok(self.hold.place(item, cut))
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<MetaItem>, StageError> {
        Ok(self.hold.finish())
    }
}

/// Every layer's scene request, in order of layer.
fn requests(args: &[LayerDemand]) -> Result<Vec<(LayerId, SceneRequest)>, StageError> {
    layer_requests::<Scenes>(args, PROVIDER)
}

/// The limit of a request's layer, when it has one it can work out.
fn limit_for(request: &SceneRequest, frame_rate: Option<FrameRate>) -> Option<usize> {
    let SceneBreaks::Limited {
        max_frames,
        max_seconds,
        ..
    } = request.breaks
    else {
        return None;
    };

    limit_of(max_frames, max_seconds, frame_rate).ok().flatten()
}

fn unprovided_scores(error: ProvisionError) -> StageError {
    StageError::with_source("scene-split was not given the scores it demands", error)
}

fn unscored_frame() -> StageError {
    StageError::msg("scene-split was given a frame scene-score never scored")
}

fn missing_input_format() -> StageError {
    StageError::msg("the scene-split provider was built without its input's format")
}

fn masks_offered() -> StageError {
    StageError::msg("scene splitting reads frames, not masks")
}
