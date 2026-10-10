use serde_json::{Value, json};

use super::export::{ExportedStage, FORMAT_VERSION};

/// The `sha256:<hex>` hash of `stages`.
///
/// The stages are sorted by key and written as compact JSON with sorted keys.
/// So neither the order of the stages nor the order of their args changes the hash.
pub(crate) fn pipeline_hash(stages: &[ExportedStage]) -> String {
    let mut sorted = stages.to_vec();
    sorted.sort_by(|left, right| left.key.cmp(&right.key));

    let document = json!({
        "format_version": FORMAT_VERSION,
        "stages": sorted,
    });

    let mut canonical = String::new();
    write_canonical(&document, &mut canonical);

    let digest = av_toolbox_models::sha256_hex(canonical.as_bytes());
    format!("sha256:{digest}")
}

/// Writes `value` as compact JSON with every object's keys sorted.
fn write_canonical(value: &Value, written: &mut String) {
    match value {
        Value::Array(items) => {
            written.push('[');

            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    written.push(',');
                }

                write_canonical(item, written);
            }

            written.push(']');
        },
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));

            written.push('{');

            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    written.push(',');
                }

                let quoted = Value::String(key.clone()).to_string();
                written.push_str(&quoted);
                written.push(':');
                write_canonical(item, written);
            }

            written.push('}');
        },
        scalar => {
            let text = scalar.to_string();
            written.push_str(&text);
        },
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::inspect::export::ExportedInput;

    fn stage(name: &str, args: Value, workers: Option<usize>) -> ExportedStage {
        let source = ExportedInput {
            name: None,
            stage: "src".to_string(),
        };

        ExportedStage {
            key: format!("filter.{name}"),
            name: name.to_string(),
            kind: "filter".to_string(),
            stage_type: "test/blur".to_string(),
            inputs: vec![source],
            args,
            workers,
        }
    }

    fn canonical(value: &Value) -> String {
        let mut written = String::new();
        write_canonical(value, &mut written);

        written
    }

    #[test]
    fn the_canonical_form_sorts_keys_and_has_no_whitespace() {
        let value = json!({ "b": 1, "a": [1, { "d": 2.5, "c": null }] });

        assert_eq!(canonical(&value), r#"{"a":[1,{"c":null,"d":2.5}],"b":1}"#);
    }

    #[test]
    fn the_canonical_form_escapes_strings_like_json() {
        let value = json!({ "say \"hi\"": "/clips/é\n.mkv" });

        assert_eq!(canonical(&value), r#"{"say \"hi\"":"/clips/é\n.mkv"}"#);
    }

    #[test]
    fn a_known_stage_list_hashes_to_a_fixed_value() {
        let stages = [stage("blur", json!({ "sigma": 1.5 }), Some(2))];

        let expected = "sha256:14bc2cb66658a8c3e725eccb8a6ecb76408106391a3066b3b29f894839a1e209";
        assert_eq!(pipeline_hash(&stages), expected);
    }

    #[test]
    fn reordered_stages_hash_the_same() {
        let blur = stage("blur", json!({}), None);
        let sharpen = stage("sharpen", json!({}), None);
        let forwards = [blur.clone(), sharpen.clone()];
        let backwards = [sharpen, blur];

        assert_eq!(pipeline_hash(&forwards), pipeline_hash(&backwards));
    }

    #[test]
    fn a_changed_arg_changes_the_hash() {
        let before = [stage("blur", json!({ "sigma": 1.5 }), None)];
        let after = [stage("blur", json!({ "sigma": 2.0 }), None)];

        assert_ne!(pipeline_hash(&before), pipeline_hash(&after));
    }

    #[test]
    fn a_changed_worker_count_changes_the_hash() {
        let before = [stage("blur", json!({}), None)];
        let after = [stage("blur", json!({}), Some(3))];

        assert_ne!(pipeline_hash(&before), pipeline_hash(&after));
    }

    #[test]
    fn a_changed_input_changes_the_hash() {
        let before = [stage("blur", json!({}), None)];
        let mut moved = stage("blur", json!({}), None);
        moved.inputs[0].stage = "other".to_string();
        let after = [moved];

        assert_ne!(pipeline_hash(&before), pipeline_hash(&after));
    }
}
