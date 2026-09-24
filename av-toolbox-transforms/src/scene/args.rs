use std::num::NonZeroUsize;

use av_scenechange::SceneDetectionSpeed;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::split::{SplitLimit, explicit};

const DEFAULT_MIN_SCENE_FRAMES: usize = 24;

/// Which of the detector's analyses to run.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Speed {
    /// Compares frame costs and motion.
    #[default]
    Standard,
    /// Compares downscaled pixels.
    Fast,
}

impl Speed {
    pub(crate) const fn detection_speed(self) -> SceneDetectionSpeed {
        match self {
            Self::Standard => SceneDetectionSpeed::Standard,
            Self::Fast => SceneDetectionSpeed::Fast,
        }
    }
}

/// Whether scenes are kept whole or split to a limit.
#[derive(Debug, Copy, Clone, PartialEq, JsonSchema)]
pub(crate) enum Mode {
    Natural,
    Split(SplitLimit),
}

/// How `scene/detect` divides the stream.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
#[serde(try_from = "RawDetectArgs")]
pub(crate) struct DetectArgs {
    pub(crate) mode: Mode,
    pub(crate) speed: Speed,
    pub(crate) min_scene_frames: usize,
    pub(crate) max_scene_frames: Option<NonZeroUsize>,
}

/// The arguments of `scene/detect` as it was given them.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum RawDetectArgs {
    /// One chunk per scene.
    Natural {
        #[serde(default)]
        speed: Speed,
        #[serde(default = "default_min_scene_frames")]
        min_scene_frames: usize,
        #[serde(default)]
        max_scene_frames: Option<usize>,
    },
    /// Scenes split so no chunk is longer than a limit.
    Split {
        #[serde(default)]
        speed: Speed,
        #[serde(default = "default_min_scene_frames")]
        min_scene_frames: usize,
        #[serde(default)]
        max_scene_frames: Option<usize>,
        #[serde(default)]
        max_frames: Option<usize>,
        #[serde(default, deserialize_with = "explicit")]
        #[schemars(with = "Option<f64>")]
        max_seconds: Option<Option<f64>>,
    },
}

impl TryFrom<RawDetectArgs> for DetectArgs {
    type Error = String;

    fn try_from(raw: RawDetectArgs) -> Result<Self, String> {
        let (mode, speed, min_scene_frames, max_scene_frames) = match raw {
            RawDetectArgs::Natural {
                speed,
                min_scene_frames,
                max_scene_frames,
            } => (Mode::Natural, speed, min_scene_frames, max_scene_frames),
            RawDetectArgs::Split {
                speed,
                min_scene_frames,
                max_scene_frames,
                max_frames,
                max_seconds,
            } => {
                let limit = SplitLimit::new(max_frames, max_seconds)?;
                (Mode::Split(limit), speed, min_scene_frames, max_scene_frames)
            },
        };

        let max_scene_frames = scene_limit(min_scene_frames, max_scene_frames)?;

        Ok(Self {
            mode,
            speed,
            min_scene_frames,
            max_scene_frames,
        })
    }
}

fn default_min_scene_frames() -> usize {
    DEFAULT_MIN_SCENE_FRAMES
}

/// Checks the scene frame limits agree, returning the maximum.
fn scene_limit(
    min_scene_frames: usize,
    max_scene_frames: Option<usize>,
) -> Result<Option<NonZeroUsize>, String> {
    let Some(max_scene_frames) = max_scene_frames else {
        return Ok(None);
    };

    let Some(maximum) = NonZeroUsize::new(max_scene_frames) else {
        return Err("max_scene_frames must be at least 1".to_string());
    };

    if min_scene_frames > max_scene_frames {
        return Err(format!(
            "min_scene_frames ({min_scene_frames}) is above max_scene_frames ({max_scene_frames})"
        ));
    }

    Ok(Some(maximum))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::split::SplitLimit;

    fn parse(value: serde_json::Value) -> Result<DetectArgs, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn natural_mode_takes_the_defaults() {
        let args = parse(json!({ "mode": "natural" })).expect("parses");

        assert_eq!(args.mode, Mode::Natural);
        assert_eq!(args.speed, Speed::Standard);
        assert_eq!(args.min_scene_frames, 24);
        assert_eq!(args.max_scene_frames, None);
    }

    #[test]
    fn every_field_can_be_given() {
        let value =
            json!({ "mode": "natural", "speed": "fast", "min_scene_frames": 4, "max_scene_frames": 90 });

        let args = parse(value).expect("parses");

        assert_eq!(args.speed, Speed::Fast);
        assert_eq!(args.min_scene_frames, 4);
        assert_eq!(args.max_scene_frames, NonZeroUsize::new(90));
    }

    #[test]
    fn a_minimum_above_the_maximum_is_refused() {
        let parsed = parse(json!({ "mode": "natural", "min_scene_frames": 30, "max_scene_frames": 20 }));

        let error = parsed.expect_err("refused").to_string();
        assert!(error.contains("above max_scene_frames"), "{error}");
    }

    #[test]
    fn a_zero_maximum_is_refused() {
        let parsed = parse(json!({ "mode": "natural", "min_scene_frames": 0, "max_scene_frames": 0 }));

        assert!(parsed.is_err());
    }

    #[test]
    fn a_missing_or_unknown_mode_is_refused() {
        let missing = parse(json!({}));
        let unknown = parse(json!({ "mode": "sideways" }));

        assert!(missing.is_err());
        assert!(unknown.is_err());
    }

    #[test]
    fn unknown_fields_are_refused() {
        let parsed = parse(json!({ "mode": "natural", "max_frames": 12 }));

        assert!(parsed.is_err());
    }

    #[test]
    fn split_mode_defaults_to_ten_seconds() {
        let args = parse(json!({ "mode": "split" })).expect("parses");

        let expected = SplitLimit::new(None, None).expect("a valid limit");
        assert_eq!(args.mode, Mode::Split(expected));
        assert_eq!(args.min_scene_frames, 24);
    }

    #[test]
    fn split_mode_takes_its_limits() {
        let args = parse(json!({ "mode": "split", "max_frames": 48, "max_seconds": 2.0 })).expect("parses");

        let expected = SplitLimit::new(Some(48), Some(Some(2.0))).expect("a valid limit");
        assert_eq!(args.mode, Mode::Split(expected));
    }

    #[test]
    fn split_mode_refuses_bad_limits() {
        let zero = parse(json!({ "mode": "split", "max_frames": 0 }));
        let both_null = parse(json!({ "mode": "split", "max_frames": null, "max_seconds": null }));
        let negative = parse(json!({ "mode": "split", "max_seconds": -2.0 }));

        assert!(zero.is_err());
        assert!(both_null.is_err());
        assert!(negative.is_err());
    }

    #[test]
    fn split_mode_checks_the_scene_limits_too() {
        let parsed = parse(json!({ "mode": "split", "min_scene_frames": 30, "max_scene_frames": 20 }));

        assert!(parsed.is_err());
    }
}
