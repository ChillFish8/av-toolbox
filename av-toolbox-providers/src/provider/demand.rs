use av_scenechange::SceneDetectionSpeed;
use av_toolbox_core::DemandDefinition;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Scene breaks, supplied by the scene provider.
pub struct Scenes;

impl DemandDefinition for Scenes {
    type Request = SceneRequest;

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

    /// Scenes split to whichever of `max_frames` and `max_seconds` is shorter, with no shortest scene.
    ///
    /// A stream with no frame rate is still split, to `max_frames`.
    pub fn limited_to(max_frames: u32, max_seconds: f64) -> Self {
        let breaks = SceneBreaks::Limited {
            min_frames: 0,
            max_frames: Some(max_frames),
            max_seconds: Some(max_seconds),
        };

        Self {
            breaks,
            ..Self::natural()
        }
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
}
