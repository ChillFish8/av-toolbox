use std::num::NonZeroUsize;
use std::time::Duration;

use av_toolbox_core::StageError;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer};

const DEFAULT_MAX_SECONDS: f64 = 10.0;

/// Keeps float error from losing a whole frame when seconds convert to frames.
const FRAME_TOLERANCE: f64 = 1e-9;

/// How long a piece of a split scene may be, as the stage was given it.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct LimitArgs {
    #[serde(default)]
    max_frames: Option<usize>,
    #[serde(default, deserialize_with = "explicit")]
    #[schemars(with = "Option<f64>")]
    max_seconds: Option<Option<f64>>,
}

/// The longest a piece of a split scene may be, in frames, seconds, or both.
#[derive(Debug, Copy, Clone, PartialEq, Deserialize, JsonSchema)]
#[serde(try_from = "LimitArgs")]
pub(crate) struct SplitLimit {
    max_frames: Option<NonZeroUsize>,
    max_seconds: Option<f64>,
}

impl SplitLimit {
    /// A limit from the frames and seconds a stage was given.
    ///
    /// Leaving out both means 10 seconds. Giving `null` for both is refused.
    pub(crate) fn new(max_frames: Option<usize>, max_seconds: Option<Option<f64>>) -> Result<Self, String> {
        let max_seconds = match (max_frames, max_seconds) {
            (None, None) => Some(DEFAULT_MAX_SECONDS),
            (_, seconds) => seconds.flatten(),
        };

        let max_frames = match max_frames {
            Some(frames) => {
                let frames = NonZeroUsize::new(frames).ok_or("max_frames must be at least 1")?;
                Some(frames)
            },
            None => None,
        };

        if let Some(seconds) = max_seconds
            && !(seconds.is_finite() && seconds > 0.0)
        {
            return Err(format!(
                "max_seconds must be a finite number above 0, found {seconds}"
            ));
        }

        if max_frames.is_none() && max_seconds.is_none() {
            return Err("a split needs max_frames or max_seconds".to_string());
        }

        Ok(Self {
            max_frames,
            max_seconds,
        })
    }

    /// The most frames one piece of a scene of `length` frames may hold.
    ///
    /// Seconds convert using the scene's average frame duration, from the timestamps of its first
    /// and last frames. Without them, the frame limit is used alone.
    pub(crate) fn frames_for(
        &self,
        length: usize,
        first: Option<Duration>,
        last: Option<Duration>,
    ) -> Result<usize, StageError> {
        let frame_limit = self.max_frames.map(NonZeroUsize::get);
        let Some(max_seconds) = self.max_seconds else {
            return Ok(frame_limit.expect("a limit without seconds has frames"));
        };

        if length <= 1 {
            return Ok(frame_limit.unwrap_or(1));
        }

        let (Some(first), Some(last)) = (first, last) else {
            return frame_limit.ok_or_else(|| {
                StageError::msg(
                    "the frames carry no timestamps, so max_seconds cannot be measured. Set max_frames",
                )
            });
        };

        let span = last.saturating_sub(first);
        if span.is_zero() {
            return Err(StageError::msg(
                "the frames' timestamps do not advance, so max_seconds cannot be measured",
            ));
        }

        let gaps = (length - 1) as f64;
        let frame_duration = span.as_secs_f64() / gaps;
        let frames = max_seconds / frame_duration + FRAME_TOLERANCE;
        let seconds_limit = frames.floor() as usize;
        let seconds_limit = seconds_limit.max(1);

        let limit = match frame_limit {
            Some(frame_limit) => frame_limit.min(seconds_limit),
            None => seconds_limit,
        };

        Ok(limit)
    }
}

impl TryFrom<LimitArgs> for SplitLimit {
    type Error = String;

    fn try_from(args: LimitArgs) -> Result<Self, String> {
        Self::new(args.max_frames, args.max_seconds)
    }
}

/// Reads a field that was given, so a `null` stays apart from a field left out.
pub(crate) fn explicit<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let value = Option::<T>::deserialize(deserializer)?;

    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(value: serde_json::Value) -> Result<SplitLimit, serde_json::Error> {
        serde_json::from_value(value)
    }

    fn limit(max_frames: Option<usize>, max_seconds: Option<f64>) -> SplitLimit {
        let max_frames = max_frames.and_then(NonZeroUsize::new);

        SplitLimit {
            max_frames,
            max_seconds,
        }
    }

    fn millis(value: u64) -> Option<Duration> {
        Some(Duration::from_millis(value))
    }

    #[test]
    fn leaving_both_limits_out_means_ten_seconds() {
        let parsed = parse(json!({})).expect("parses");

        assert_eq!(parsed, limit(None, Some(10.0)));
    }

    #[test]
    fn frames_alone_set_no_seconds() {
        let parsed = parse(json!({ "max_frames": 48 })).expect("parses");

        assert_eq!(parsed, limit(Some(48), None));
    }

    #[test]
    fn both_given_keeps_both() {
        let parsed = parse(json!({ "max_frames": 48, "max_seconds": 2.5 })).expect("parses");

        assert_eq!(parsed, limit(Some(48), Some(2.5)));
    }

    #[test]
    fn null_seconds_beside_frames_keeps_the_frames_alone() {
        let parsed = parse(json!({ "max_frames": 48, "max_seconds": null })).expect("parses");

        assert_eq!(parsed, limit(Some(48), None));
    }

    #[test]
    fn null_for_both_is_refused() {
        let parsed = parse(json!({ "max_frames": null, "max_seconds": null }));

        let error = parsed.expect_err("refused").to_string();
        assert!(error.contains("needs max_frames or max_seconds"), "{error}");
    }

    #[test]
    fn null_seconds_alone_is_refused() {
        let parsed = parse(json!({ "max_seconds": null }));

        assert!(parsed.is_err());
    }

    #[test]
    fn zero_frames_is_refused() {
        let parsed = parse(json!({ "max_frames": 0 }));

        let error = parsed.expect_err("refused").to_string();
        assert!(error.contains("at least 1"), "{error}");
    }

    #[test]
    fn seconds_at_or_below_zero_are_refused() {
        let zero = parse(json!({ "max_seconds": 0.0 }));
        let negative = parse(json!({ "max_seconds": -1.0 }));

        let error = zero.expect_err("refused").to_string();
        assert!(error.contains("finite number above 0"), "{error}");
        assert!(negative.is_err());
    }

    #[test]
    fn seconds_that_are_not_finite_are_refused() {
        let not_a_number = SplitLimit::new(None, Some(Some(f64::NAN)));
        let infinite = SplitLimit::new(None, Some(Some(f64::INFINITY)));

        assert!(not_a_number.is_err());
        assert!(infinite.is_err());
    }

    #[test]
    fn unknown_fields_are_refused() {
        let parsed = parse(json!({ "max_frame": 4 }));

        assert!(parsed.is_err());
    }

    #[test]
    fn frames_alone_are_the_frame_limit() {
        let frames = limit(Some(48), None).frames_for(100, None, None);

        assert_eq!(frames.expect("converts"), 48);
    }

    #[test]
    fn seconds_convert_by_the_average_frame_duration() {
        let frames = limit(None, Some(2.0)).frames_for(101, millis(0), millis(6250));

        assert_eq!(frames.expect("converts"), 32);
    }

    #[test]
    fn ten_seconds_at_twenty_five_frames_a_second_is_two_hundred_and_fifty_frames() {
        let frames = limit(None, Some(10.0)).frames_for(251, millis(0), millis(10_000));

        assert_eq!(frames.expect("converts"), 250);
    }

    #[test]
    fn the_smaller_limit_wins() {
        let frames_smaller = limit(Some(20), Some(2.0)).frames_for(101, millis(0), millis(6250));
        let seconds_smaller = limit(Some(40), Some(2.0)).frames_for(101, millis(0), millis(6250));

        assert_eq!(frames_smaller.expect("converts"), 20);
        assert_eq!(seconds_smaller.expect("converts"), 32);
    }

    #[test]
    fn missing_timestamps_fall_back_to_frames() {
        let frames = limit(Some(48), Some(2.0)).frames_for(101, None, millis(6250));

        assert_eq!(frames.expect("converts"), 48);
    }

    #[test]
    fn missing_timestamps_without_frames_fail() {
        let frames = limit(None, Some(2.0)).frames_for(101, millis(0), None);

        let error = frames.expect_err("fails").to_string();
        assert!(error.contains("max_frames"), "{error}");
    }

    #[test]
    fn timestamps_that_do_not_advance_fail() {
        let frames = limit(None, Some(2.0)).frames_for(10, millis(40), millis(40));

        let error = frames.expect_err("fails").to_string();
        assert!(error.contains("do not advance"), "{error}");
    }

    #[test]
    fn a_single_frame_scene_needs_no_timestamps() {
        let frames = limit(None, Some(2.0)).frames_for(1, None, None);

        assert_eq!(frames.expect("converts"), 1);
    }

    #[test]
    fn a_tiny_seconds_limit_is_still_one_frame() {
        let frames = limit(None, Some(0.001)).frames_for(101, millis(0), millis(6250));

        assert_eq!(frames.expect("converts"), 1);
    }
}
