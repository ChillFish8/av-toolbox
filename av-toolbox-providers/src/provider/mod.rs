mod demand;
mod detection;
mod hold;
mod layers;
#[cfg(test)]
mod tests;

use av_toolbox_core::{
    InputParam,
    InputSignature,
    Inputs,
    LayerDemand,
    Placement,
    Provider,
    ProviderDefinition,
    RequestError,
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
    HostSide,
    HostStreamItem,
    LayerId,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
};

pub use self::demand::{SceneBreaks, SceneChunk, SceneRequest, SceneSpeed, Scenes};
use self::detection::Detection;
use self::hold::LayerHold;
use self::layers::{LONGEST_SCENE, LayerBreaks, limit_of};
use crate::scene::window::{LOOKAHEAD, check_layout};

/// Finds scene cuts once and breaks the stream into scenes for every layer demanding them.
pub(crate) struct SceneProvider;

impl StageDefinition for SceneProvider {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "provider"
    }

    fn identifier() -> &'static str {
        "scenes"
    }

    /// The detector's lookahead, plus the longest scene any limited layer holds.
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

        LOOKAHEAD + LONGEST_SCENE * longest
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl ProviderDefinition for SceneProvider {
    type Side = HostSide;
    type Demand = Scenes;

    fn placement() -> Placement {
        Placement::AfterSource
    }

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        offered: StreamFormat,
    ) -> Result<FormatSet, StageError> {
        let Some(format) = offered.as_frames() else {
            return Err(StageError::msg("scene detection reads frames, not masks"));
        };

        check_layout(format)?;

        let accepted = match format.sample {
            SampleFormat::U8 | SampleFormat::U16 { .. } | SampleFormat::F32 => FormatSet::of([format.sample]),
            SampleFormat::F16 => FormatSet::new(),
        };

        Ok(accepted)
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
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Provider<HostSide>>, StageError> {
        let input = formats.input(0).ok_or_else(missing_input_format)?;
        let format = input.as_frames().ok_or_else(masks_offered)?;

        let provider = ServeScenes::new(&args, format)?;

        Ok(Box::new(provider))
    }
}

/// The scene provider's instance, detecting cuts and stamping every layer's chunks.
pub(crate) struct ServeScenes {
    detection: Detection,
    hold: LayerHold,
}

impl ServeScenes {
    pub(crate) fn new(args: &[LayerDemand], format: &FrameFormat) -> Result<Self, StageError> {
        let requests = requests(args)?;

        let any_standard = requests
            .iter()
            .any(|(_, request)| request.speed == SceneSpeed::Standard);
        let speed = if any_standard {
            SceneSpeed::Standard
        } else {
            SceneSpeed::Fast
        };
        let detection = Detection::new(format, speed)?;

        let mut layers = Vec::with_capacity(requests.len());
        for (layer, request) in &requests {
            let breaks = LayerBreaks::new(&request.breaks, format.frame_rate)?;
            layers.push((*layer, breaks));
        }

        let hold = LayerHold::new(layers);

        Ok(Self { detection, hold })
    }
}

impl StageInstance for ServeScenes {}

impl Provider<HostSide> for ServeScenes {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<Vec<HostStreamItem>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;

        let decided = self.detection.push(frame)?;

        let mut emitted = Vec::new();
        for decision in decided {
            let placed = self.hold.place(decision.frame, decision.starts_scene);
            let items = placed.into_iter().map(HostStreamItem::Frame);
            emitted.extend(items);
        }

        Ok(emitted)
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<HostStreamItem>, StageError> {
        let decided = self.detection.finish();

        let mut emitted = Vec::new();
        for decision in decided {
            let placed = self.hold.place(decision.frame, decision.starts_scene);
            emitted.extend(placed);
        }

        let rest = self.hold.finish();
        emitted.extend(rest);

        let items = emitted.into_iter().map(HostStreamItem::Frame).collect();
        Ok(items)
    }
}

/// Every layer's scene request, in order of layer.
fn requests(args: &[LayerDemand]) -> Result<Vec<(LayerId, SceneRequest)>, StageError> {
    let mut requests = Vec::with_capacity(args.len());
    for served in args {
        let request = served.demand.try_request::<Scenes>();
        let request = request.map_err(|error| unreadable_request(served, error))?;
        requests.push((served.layer, request));
    }

    requests.sort_by_key(|(layer, _)| *layer);

    Ok(requests)
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

fn unreadable_request(served: &LayerDemand, error: RequestError) -> StageError {
    let message = match error {
        RequestError::OtherKind { found, .. } => format!(
            "the scene provider serves scenes demands, but layer {} asks for a {} demand",
            served.layer.0,
            found.name(),
        ),
        RequestError::Malformed { source, .. } => {
            format!("layer {}'s scenes request is malformed: {source}", served.layer.0)
        },
    };

    StageError::msg(message)
}

fn missing_input_format() -> StageError {
    StageError::msg("the scene provider was built without its input's format")
}

fn masks_offered() -> StageError {
    StageError::msg("scene detection reads frames, not masks")
}

fn missing_frame() -> StageError {
    StageError::msg("the scene provider expected a frame in slot 0")
}
