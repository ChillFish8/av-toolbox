use av_toolbox_core::StageError;
use schemars::JsonSchema;
use serde::Deserialize;

use super::picture::{Picture, RgbPicture};

/// A solid colour, as RGB or as raw sample values per plane.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
#[serde(
    untagged,
    expecting = "a colour as \"#rrggbb\", [r, g, b] or { planes = [..] }"
)]
pub(crate) enum Colour {
    /// An RGB colour written as `#rrggbb`.
    Hex(String),
    /// An RGB colour with each channel between 0 and 255.
    Rgb([u8; 3]),
    /// Raw sample values in the output format, one per plane in plane order.
    Planes(PlaneValues),
}

/// Raw sample values, one per plane in plane order.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlaneValues {
    pub(crate) planes: Vec<u16>,
}

fn bad_hex(stage: &str, text: &str) -> StageError {
    let message = format!("{stage} needs a colour written as #rrggbb, got {text:?}");

    StageError::msg(message)
}

/// Reads `#rrggbb` into its three channels.
fn parse_hex(text: &str) -> Option<[u8; 3]> {
    let digits = text.strip_prefix('#')?;
    let all_hex = digits.chars().all(|digit| digit.is_ascii_hexdigit());
    if digits.len() != 6 || !all_hex {
        return None;
    }

    let red = u8::from_str_radix(&digits[0..2], 16).ok()?;
    let green = u8::from_str_radix(&digits[2..4], 16).ok()?;
    let blue = u8::from_str_radix(&digits[4..6], 16).ok()?;

    Some([red, green, blue])
}

fn one_pixel(channels: [u8; 3]) -> Picture {
    let [red, green, blue] = channels;
    let rgb = RgbPicture {
        width: 1,
        height: 1,
        red: vec![f32::from(red) / 255.0],
        green: vec![f32::from(green) / 255.0],
        blue: vec![f32::from(blue) / 255.0],
    };

    Picture::Rgb(rgb)
}

impl Colour {
    /// This colour as a one pixel picture, or its raw plane values.
    pub(crate) fn picture(&self, stage: &str) -> Result<Picture, StageError> {
        match self {
            Self::Hex(text) => {
                let Some(channels) = parse_hex(text) else {
                    return Err(bad_hex(stage, text));
                };

                Ok(one_pixel(channels))
            },
            Self::Rgb(channels) => Ok(one_pixel(*channels)),
            Self::Planes(values) => Ok(Picture::Planes(values.planes.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(value: serde_json::Value) -> Result<Colour, serde_json::Error> {
        serde_json::from_value(value)
    }

    fn single_rgb(picture: Picture) -> (f32, f32, f32) {
        let Picture::Rgb(rgb) = picture else {
            panic!("expected an RGB picture");
        };

        assert_eq!((rgb.width, rgb.height), (1, 1));

        (rgb.red[0], rgb.green[0], rgb.blue[0])
    }

    #[test]
    fn hex_parses_in_either_case() {
        let lower = parse(json!("#ff8000")).expect("parses");
        let upper = parse(json!("#FF8000")).expect("parses");

        let lower_rgb = lower.picture("solid/fill").expect("a colour");
        let upper_rgb = upper.picture("solid/fill").expect("a colour");

        assert_eq!(single_rgb(lower_rgb), (1.0, 128.0 / 255.0, 0.0));
        assert_eq!(single_rgb(upper_rgb), (1.0, 128.0 / 255.0, 0.0));
    }

    #[test]
    fn a_list_parses_as_rgb() {
        let colour = parse(json!([255, 128, 0])).expect("parses");

        let picture = colour.picture("solid/fill").expect("a colour");

        assert_eq!(single_rgb(picture), (1.0, 128.0 / 255.0, 0.0));
    }

    #[test]
    fn planes_parse_as_raw_values() {
        let colour = parse(json!({ "planes": [16, 128, 128] })).expect("parses");

        let picture = colour.picture("solid/fill").expect("a colour");

        let Picture::Planes(values) = picture else {
            panic!("expected raw planes");
        };
        assert_eq!(values, vec![16, 128, 128]);
    }

    #[test]
    fn bad_hex_is_refused() {
        let cases = ["#ff80", "ff8000", "#gg8000", "#+f8000", "#ff80000"];

        for text in cases {
            let colour = parse(json!(text)).expect("any string parses");
            let error = colour.picture("solid/fill").expect_err("not a colour");

            assert!(error.to_string().contains("solid/fill"), "{text}");
            assert!(error.to_string().contains(text), "{text}");
        }
    }

    #[test]
    fn other_shapes_name_the_three_forms() {
        let cases = [
            json!(12),
            json!([255, 128]),
            json!([256, 0, 0]),
            json!({ "red": 1 }),
        ];

        for value in cases {
            let error = parse(value).expect_err("not a colour");

            assert!(error.to_string().contains("#rrggbb"), "{error}");
        }
    }

    #[test]
    fn planes_with_an_extra_key_are_refused() {
        let value = json!({ "planes": [16, 128, 128], "extra": 1 });

        let error = parse(value).expect_err("extra is not a colour setting");

        assert!(error.to_string().contains("#rrggbb"), "{error}");
    }
}
