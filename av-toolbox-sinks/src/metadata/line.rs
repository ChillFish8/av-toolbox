use std::io;
use std::io::Write;

use av_toolbox_core::KeyFilter;
use av_toolbox_gpu::{FrameMeta, MetaRecord, MetaValue};
use av_toolbox_providers::SceneScore;
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};

/// Writes one frame's metadata as a JSON object followed by a newline.
///
/// `chunk` is the frame's scene chunk, and is left out when there is none.
/// `score` is the frame's scene cut score, written as `null` when there is none.
pub(crate) fn write_line<W: Write>(
    output: &mut W,
    meta: &FrameMeta,
    chunk: Option<u64>,
    score: Option<SceneScore>,
    filter: &KeyFilter,
) -> io::Result<()> {
    let line = Line {
        meta,
        chunk,
        score,
        filter,
    };
    serde_json::to_writer(&mut *output, &line)?;

    output.write_all(b"\n")
}

/// Writes one record as a JSON object followed by a newline.
///
/// `after` is the index of the last frame written, and is `null` before any.
/// `chunk` is the scene chunk of that frame, and is left out unless scenes were asked for.
pub(crate) fn write_record_line<W: Write>(
    output: &mut W,
    record: &MetaRecord,
    after: Option<u64>,
    chunk: Option<Option<u64>>,
    filter: &KeyFilter,
) -> io::Result<()> {
    let line = RecordLine {
        record,
        after,
        chunk,
        filter,
    };
    serde_json::to_writer(&mut *output, &line)?;

    output.write_all(b"\n")
}

/// One record, written with its keys in a fixed order.
struct RecordLine<'a> {
    record: &'a MetaRecord,
    after: Option<u64>,
    chunk: Option<Option<u64>>,
    filter: &'a KeyFilter,
}

impl Serialize for RecordLine<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("kind", self.record.kind())?;
        entry(self.filter, &mut map, "after", &self.after)?;

        if let Some(chunk) = self.chunk {
            entry(self.filter, &mut map, "chunk", &chunk)?;
        }

        for (path, values) in self.record.attachments().stages() {
            let values = StageValues(values);
            entry(self.filter, &mut map, path, &values)?;
        }

        map.end()
    }
}

/// One frame's metadata, written with its keys in a fixed order.
struct Line<'a> {
    meta: &'a FrameMeta,
    chunk: Option<u64>,
    score: Option<SceneScore>,
    filter: &'a KeyFilter,
}

impl Serialize for Line<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let timestamp = self.meta.timestamp.map(|timestamp| timestamp.as_secs_f64());
        let scene_score = self.score.map(SceneScore::get);

        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("kind", "frame")?;
        self.entry(&mut map, "index", &self.meta.index)?;

        if let Some(chunk) = self.chunk {
            self.entry(&mut map, "chunk", &chunk)?;
        }

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
        entry(self.filter, map, key, value)
    }
}

fn entry<M: SerializeMap, V: Serialize>(
    filter: &KeyFilter,
    map: &mut M,
    key: &str,
    value: &V,
) -> Result<(), M::Error> {
    if !filter.keeps(key) {
        return Ok(());
    }

    map.serialize_entry(key, value)
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

    use av_toolbox_core::{KeyFilter, Pattern};
    use av_toolbox_gpu::{FrameMeta, MetaRecord, MetaValue};
    use serde_json::json;

    use super::*;

    fn line_of(
        meta: &FrameMeta,
        chunk: Option<u64>,
        score: Option<SceneScore>,
        filter: &KeyFilter,
    ) -> String {
        let mut output = Vec::new();
        write_line(&mut output, meta, chunk, score, filter).expect("writes");

        String::from_utf8(output).expect("utf-8")
    }

    fn record_line_of(
        record: &MetaRecord,
        after: Option<u64>,
        chunk: Option<Option<u64>>,
        filter: &KeyFilter,
    ) -> String {
        let mut output = Vec::new();
        write_record_line(&mut output, record, after, chunk, filter).expect("writes");

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
        meta.attach("attr1", MetaValue::Int(123));
        meta.attach("clipped", MetaValue::Bool(false));
        meta.attachments.stamp("filter.convert_colour");
        meta.attach("cut", MetaValue::Bool(true));
        meta.attachments.stamp("filter.scenes.detect");

        meta
    }

    #[test]
    fn built_in_keys_come_first_then_stages_in_order() {
        let meta = stamped_meta();
        let filter = KeyFilter::default();
        let score = Some(SceneScore::new(2.5));
        let line = line_of(&meta, Some(4), score, &filter);

        let expected = concat!(
            r#"{"kind":"frame","index":12,"chunk":4,"timestamp":0.5,"scene_score":2.5,"#,
            r#""filter.convert_colour":{"attr1":123,"clipped":false},"#,
            r#""filter.scenes.detect":{"cut":true}}"#,
            "\n",
        );
        assert_eq!(line, expected);
    }

    #[test]
    fn unknown_built_ins_are_null() {
        let meta = FrameMeta::at(0);
        let filter = KeyFilter::default();
        let line = line_of(&meta, Some(0), None, &filter);

        assert_eq!(
            line,
            "{\"kind\":\"frame\",\"index\":0,\"chunk\":0,\"timestamp\":null,\"scene_score\":null}\n"
        );
    }

    #[test]
    fn no_chunk_is_written_without_one() {
        let meta = FrameMeta::at(0);
        let filter = KeyFilter::default();
        let line = line_of(&meta, None, None, &filter);

        assert_eq!(
            line,
            "{\"kind\":\"frame\",\"index\":0,\"timestamp\":null,\"scene_score\":null}\n"
        );
    }

    #[test]
    fn nan_and_infinite_floats_are_null() {
        let mut meta = FrameMeta::at(0);
        meta.attach("nan", MetaValue::Float(f64::NAN));
        meta.attach("infinite", MetaValue::Float(f64::INFINITY));
        meta.attachments.stamp("filter.measure");

        let filter = KeyFilter::default();
        let line = line_of(&meta, None, None, &filter);

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
        let line = line_of(&meta, None, None, &filter);

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
        let line = line_of(&meta, None, None, &filter);

        assert_eq!(
            line,
            "{\"kind\":\"frame\",\"index\":12,\"filter.scenes.detect\":{\"cut\":true}}\n"
        );
    }

    #[test]
    fn a_frame_line_leads_with_its_kind() {
        let meta = stamped_meta();

        let line = line_of(&meta, Some(4), None, &KeyFilter::default());

        assert!(line.starts_with("{\"kind\":\"frame\",\"index\":12,"));
    }

    #[test]
    fn the_kind_survives_an_include_that_would_drop_it() {
        let meta = stamped_meta();
        let filter = KeyFilter::new(Some(patterns(&["index"])), None);

        let line = line_of(&meta, None, None, &filter);

        assert_eq!(line, "{\"kind\":\"frame\",\"index\":12}\n");
    }

    #[test]
    fn a_record_line_holds_kind_after_chunk_then_its_stages() {
        let mut record = MetaRecord::new("chunk");
        record.attachments_mut().attach("frames", MetaValue::Int(24));
        record.attachments_mut().stamp("filter.stats");

        let line = record_line_of(&record, Some(47), Some(Some(3)), &KeyFilter::default());

        let expected = "{\"kind\":\"chunk\",\"after\":47,\"chunk\":3,\"filter.stats\":{\"frames\":24}}\n";
        assert_eq!(line, expected);
    }

    #[test]
    fn a_record_before_any_frame_is_after_null() {
        let record = MetaRecord::new("start");

        let line = record_line_of(&record, None, None, &KeyFilter::default());

        assert_eq!(line, "{\"kind\":\"start\",\"after\":null}\n");
    }

    #[test]
    fn a_record_line_is_filtered_but_keeps_its_kind() {
        let record = MetaRecord::new("start");
        let filter = KeyFilter::new(Some(patterns(&["chunk"])), None);

        let line = record_line_of(&record, Some(1), Some(Some(2)), &filter);

        assert_eq!(line, "{\"kind\":\"start\",\"chunk\":2}\n");
    }
}
