use std::io;
use std::io::Write;

use av_toolbox_gpu::{FrameMeta, MetaValue};
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};

use super::filter::KeyFilter;

/// Writes one frame's metadata as a JSON object followed by a newline.
pub(crate) fn write_line<W: Write>(output: &mut W, meta: &FrameMeta, filter: &KeyFilter) -> io::Result<()> {
    let line = Line { meta, filter };
    serde_json::to_writer(&mut *output, &line)?;

    output.write_all(b"\n")
}

/// One frame's metadata, written with its keys in a fixed order.
struct Line<'a> {
    meta: &'a FrameMeta,
    filter: &'a KeyFilter,
}

impl Serialize for Line<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let timestamp = self.meta.timestamp.map(|timestamp| timestamp.as_secs_f64());
        let scene_score = self.meta.scene_score.map(|score| score.get());

        let mut map = serializer.serialize_map(None)?;
        self.entry(&mut map, "index", &self.meta.index)?;
        self.entry(&mut map, "chunk", &self.meta.chunk)?;
        self.entry(&mut map, "timestamp", &timestamp)?;
        self.entry(&mut map, "scene_score", &scene_score)?;

        for (path, values) in self.meta.attachments.stages() {
            let values = StageValues(values);
            self.entry(&mut map, path, &values)?;
        }

        map.end()
    }
}

impl Line<'_> {
    fn entry<M: SerializeMap, V: Serialize>(
        &self,
        map: &mut M,
        key: &str,
        value: &V,
    ) -> Result<(), M::Error> {
        if !self.filter.keeps(key) {
            return Ok(());
        }

        map.serialize_entry(key, value)
    }
}

/// One stage's values as an object, in the order they were attached.
struct StageValues<'a>(&'a [(String, MetaValue)]);

impl Serialize for StageValues<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;

        for (key, value) in self.0 {
            map.serialize_entry(key, &Value(value))?;
        }

        map.end()
    }
}

struct Value<'a>(&'a MetaValue);

impl Serialize for Value<'_> {
    /// A NaN or infinite float is written as `null`, which `serde_json` does for any non-finite float.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            MetaValue::Bool(value) => serializer.serialize_bool(*value),
            MetaValue::Int(value) => serializer.serialize_i64(*value),
            MetaValue::Float(value) => serializer.serialize_f64(*value),
            MetaValue::String(value) => serializer.serialize_str(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use av_toolbox_gpu::{FrameMeta, MetaValue, SceneScore};
    use serde_json::json;

    use super::*;
    use crate::metadata::filter::{KeyFilter, Pattern};

    fn line_of(meta: &FrameMeta, filter: &KeyFilter) -> String {
        let mut output = Vec::new();
        write_line(&mut output, meta, filter).expect("writes");

        String::from_utf8(output).expect("utf-8")
    }

    fn patterns(texts: &[&str]) -> Vec<Pattern> {
        texts
            .iter()
            .map(|text| {
                let value = json!(text);
                serde_json::from_value(value).expect("a pattern")
            })
            .collect()
    }

    fn stamped_meta() -> FrameMeta {
        let mut meta = FrameMeta::at(12);
        meta.timestamp = Some(Duration::from_millis(500));
        meta.scene_score = Some(SceneScore::new(2.5));
        meta.attach("attr1", MetaValue::Int(123));
        meta.attach("clipped", MetaValue::Bool(false));
        meta.attachments.stamp("filter.convert_colour");
        meta.attach("cut", MetaValue::Bool(true));
        meta.attachments.stamp("transform.scenes.detect");

        meta
    }

    #[test]
    fn built_in_keys_come_first_then_stages_in_order() {
        let meta = stamped_meta();
        let filter = KeyFilter::default();
        let line = line_of(&meta, &filter);

        let expected = concat!(
            r#"{"index":12,"chunk":0,"timestamp":0.5,"scene_score":2.5,"#,
            r#""filter.convert_colour":{"attr1":123,"clipped":false},"#,
            r#""transform.scenes.detect":{"cut":true}}"#,
            "\n",
        );
        assert_eq!(line, expected);
    }

    #[test]
    fn unknown_built_ins_are_null() {
        let meta = FrameMeta::at(0);
        let filter = KeyFilter::default();
        let line = line_of(&meta, &filter);

        assert_eq!(
            line,
            "{\"index\":0,\"chunk\":0,\"timestamp\":null,\"scene_score\":null}\n"
        );
    }

    #[test]
    fn nan_and_infinite_floats_are_null() {
        let mut meta = FrameMeta::at(0);
        meta.attach("nan", MetaValue::Float(f64::NAN));
        meta.attach("infinite", MetaValue::Float(f64::INFINITY));
        meta.attachments.stamp("filter.measure");

        let filter = KeyFilter::default();
        let line = line_of(&meta, &filter);

        assert!(
            line.contains(r#""filter.measure":{"nan":null,"infinite":null}"#),
            "{line}"
        );
    }

    #[test]
    fn a_string_with_quotes_and_newlines_stays_on_one_line() {
        let mut meta = FrameMeta::at(0);
        let awkward = "say \"hi\"\nthen \\ leave";
        meta.attach("note", MetaValue::String(awkward.to_string()));
        meta.attachments.stamp("filter.note");

        let filter = KeyFilter::default();
        let line = line_of(&meta, &filter);

        assert_eq!(line.matches('\n').count(), 1);
        assert!(line.ends_with('\n'));
        let trimmed = line.trim_end();
        let parsed: serde_json::Value = serde_json::from_str(trimmed).expect("valid json");
        assert_eq!(parsed["filter.note"]["note"], awkward);
    }

    #[test]
    fn the_filter_drops_built_ins_and_stages() {
        let pattern_list = patterns(&["index", "*.scenes.*"]);
        let filter = KeyFilter::new(Some(pattern_list), None);
        let meta = stamped_meta();
        let line = line_of(&meta, &filter);

        assert_eq!(
            line,
            "{\"index\":12,\"transform.scenes.detect\":{\"cut\":true}}\n"
        );
    }
}
