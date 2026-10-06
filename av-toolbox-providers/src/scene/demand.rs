use std::hash::{Hash, Hasher};

use av_scenechange::SceneDetectionSpeed;
use av_toolbox_core::{ChunkKey, DemandDefinition, LayerDemand, RequestError, StageError};
use av_toolbox_gpu::LayerId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Scene breaks, supplied by the `scene-split` provider.
pub struct Scenes;

impl DemandDefinition for Scenes {
    type Request = SceneRequest;
    type Output = SceneChunk;

    fn kind() -> &'static str {
        "scenes"
    }

    /// `natural`, or `limited` followed by its bounds, such as `limited ≥24 frames ≤10 s`.
    fn describe(request: &SceneRequest) -> String {
        let SceneBreaks::Limited {
            min_frames,
            max_frames,
            max_seconds,
        } = request.breaks
        else {
            return "natural".to_string();
        };

        let mut parts = vec!["limited".to_string()];
        if min_frames > 0 {
            parts.push(format!("≥{min_frames} frames"));
        }

        if let Some(frames) = max_frames {
            parts.push(format!("≤{frames} frames"));
        }

        if let Some(seconds) = max_seconds {
            parts.push(format!("≤{seconds} s"));
        }

        parts.join(" ")
    }
}

/// The scene a frame belongs to, counted from zero.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct SceneChunk(pub u64);

impl ChunkKey for SceneChunk {
    fn chunk(&self) -> u64 {
        self.0
    }
}

/// Which scene breaks a stage asks for.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SceneRequest {
    pub breaks: SceneBreaks,
    #[serde(default)]
    pub speed: SceneSpeed,
    /// The stage to detect scenes after, in place of where the provider would go.
    #[serde(default)]
    pub after: Option<String>,
}

impl SceneRequest {
    pub const fn natural() -> Self {
        Self {
            breaks: SceneBreaks::Natural,
            speed: SceneSpeed::Standard,
            after: None,
        }
    }

    /// Refuses a seconds limit that is not a finite number, which would otherwise read as no limit.
    pub fn validate(&self) -> Result<(), StageError> {
        let SceneBreaks::Limited {
            max_seconds: Some(seconds),
            ..
        } = self.breaks
        else {
            return Ok(());
        };

        if seconds.is_finite() {
            return Ok(());
        }

        let message = format!("a scenes limit of {seconds} seconds is not a finite number of seconds");
        Err(StageError::msg(message))
    }

    /// Scenes split to `max_seconds`, with no shortest scene.
    pub fn limited_to_seconds(max_seconds: f64) -> Self {
        let breaks = SceneBreaks::Limited {
            min_frames: 0,
            max_frames: None,
            max_seconds: Some(max_seconds),
        };

        Self {
            breaks,
            ..Self::natural()
        }
    }
}

/// How a stream is broken into scenes.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SceneBreaks {
    /// Every cut the detector finds, with no shortest or longest scene.
    Natural,
    /// Cuts closer than `min_frames` to the last one are dropped, then scenes are split to the limit.
    ///
    /// When both `max_frames` and `max_seconds` are given, the smaller wins.
    Limited {
        #[serde(default)]
        min_frames: u32,
        #[serde(default)]
        max_frames: Option<u32>,
        #[serde(default)]
        max_seconds: Option<f64>,
    },
}

/// Which of the detector's analyses to run.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SceneSpeed {
    /// Compares frame costs and motion.
    #[default]
    Standard,
    /// Compares downscaled pixels.
    Fast,
}

impl SceneSpeed {
    pub(crate) const fn detection_speed(self) -> SceneDetectionSpeed {
        match self {
            Self::Standard => SceneDetectionSpeed::Standard,
            Self::Fast => SceneDetectionSpeed::Fast,
        }
    }

    /// The slowest of `speeds`, which is [SceneSpeed::Fast] when there are none.
    pub(crate) fn slowest(speeds: impl IntoIterator<Item = Self>) -> Self {
        let mut speeds = speeds.into_iter();
        let any_standard = speeds.any(|speed| speed == Self::Standard);

        if any_standard { Self::Standard } else { Self::Fast }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Fast => "fast",
        }
    }
}

/// Each frame's scene cut score, supplied by the `scene-score` provider.
pub struct SceneScores;

impl DemandDefinition for SceneScores {
    type Request = ScoreRequest;
    type Output = SceneCut;

    fn kind() -> &'static str {
        "scene-score"
    }

    /// The speed the detector runs at, such as `standard`.
    fn describe(request: &ScoreRequest) -> String {
        request.speed.name().to_string()
    }
}

/// Which scene detection a stage asks to score frames with.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScoreRequest {
    /// Which of the detector's analyses to score frames with.
    #[serde(default)]
    pub speed: SceneSpeed,
    /// The stage to score frames after, in place of where the provider would go.
    #[serde(default)]
    pub after: Option<String>,
}

/// What scene detection decided about one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneCut {
    /// How strongly the detector favoured a cut just before the frame, when it could score one.
    pub score: Option<SceneScore>,
    pub starts_scene: bool,
}

/// How strongly scene detection favoured a cut just before a frame.
///
/// Higher is a better place to cut. Two scores are equal when their bits are, so metadata carrying
/// one can still be compared and hashed.
#[derive(Debug, Copy, Clone)]
pub struct SceneScore(f32);

impl SceneScore {
    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for SceneScore {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for SceneScore {}

impl Hash for SceneScore {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Every layer's request of kind `D`, in order of layer.
///
/// `provider` names the provider reading them in the error for a request it cannot read.
pub(crate) fn layer_requests<D: DemandDefinition>(
    args: &[LayerDemand],
    provider: &str,
) -> Result<Vec<(LayerId, D::Request)>, StageError> {
    let mut requests = Vec::with_capacity(args.len());
    for served in args {
        let request = served.demand.try_request::<D>();
        let request = request.map_err(|error| unreadable_request(provider, served, error))?;
        requests.push((served.layer, request));
    }

    requests.sort_by_key(|(layer, _)| *layer);

    Ok(requests)
}

fn unreadable_request(provider: &str, served: &LayerDemand, error: RequestError) -> StageError {
    let message = match error {
        RequestError::OtherKind { expected, found } => format!(
            "the {provider} provider serves {} demands, but layer {} asks for a {} demand",
            expected.name(),
            served.layer.0,
            found.name(),
        ),
        RequestError::Malformed { kind, source } => {
            format!(
                "layer {}'s {} request is malformed: {source}",
                served.layer.0,
                kind.name()
            )
        },
    };

    StageError::msg(message)
}
