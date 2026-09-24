use std::borrow::Cow;

use av_toolbox_gpu::FrameRate;
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::Deserialize;

/// A frame rate argument, written as `"30000/1001"` or a whole number such as `"25"`.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize)]
#[serde(try_from = "String")]
pub(crate) struct RateArg(pub(crate) FrameRate);

impl TryFrom<String> for RateArg {
    type Error = String;

    fn try_from(written: String) -> Result<Self, Self::Error> {
        let parsed = parse_rate(&written);

        parsed
            .map(Self)
            .ok_or_else(|| format!("{written:?} is not a frame rate, write it as \"30000/1001\" or \"25\""))
    }
}

impl JsonSchema for RateArg {
    fn schema_name() -> Cow<'static, str> {
        "FrameRate".into()
    }

    fn json_schema(_generator: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "string",
            "pattern": "^[0-9]+(/[0-9]+)?$"
        })
    }
}

fn parse_rate(written: &str) -> Option<FrameRate> {
    let (numerator, denominator) = written.split_once('/').unwrap_or((written, "1"));
    let numerator = numerator.parse().ok()?;
    let denominator = denominator.parse().ok()?;

    FrameRate::new(numerator, denominator)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parsed(written: &str) -> Result<RateArg, serde_json::Error> {
        serde_json::from_value(json!(written))
    }

    #[test]
    fn a_fraction_parses() {
        let rate = parsed("30000/1001").expect("parses");

        assert_eq!(Some(rate.0), FrameRate::new(30000, 1001));
    }

    #[test]
    fn a_whole_number_parses_over_one() {
        let rate = parsed("25").expect("parses");

        assert_eq!(Some(rate.0), FrameRate::new(25, 1));
    }

    #[test]
    fn a_parsed_rate_is_reduced() {
        let rate = parsed("50/2").expect("parses");

        assert_eq!(Some(rate.0), FrameRate::new(25, 1));
    }

    #[test]
    fn malformed_rates_are_refused_with_the_expected_form() {
        for written in ["0", "25/0", "/1", "25/", "abc", "1.5", "25/1/1", ""] {
            let error = parsed(written).expect_err(written);

            assert!(error.to_string().contains("30000/1001"), "{written}");
        }
    }
}
