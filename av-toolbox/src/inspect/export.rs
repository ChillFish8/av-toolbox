use std::collections::BTreeMap;

use av_toolbox_graph::{InputRef, StageSpec};
use av_toolbox_pipeline::variables::ResolvedVariable;
use serde::Serialize;
use serde_json::Value;

use crate::profile::file::stage_key;

/// The version of the export's shape and of the hash input.
pub(crate) const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExportedVariable {
    #[serde(rename = "type")]
    pub(crate) variable_type: &'static str,
    pub(crate) value: Value,
}

/// One stage after variables are filled in and calls are expanded.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExportedStage {
    /// The stage's `<kind>.<name>`, as profile files name it.
    pub(crate) key: String,
    pub(crate) name: String,
    pub(crate) kind: String,
    #[serde(rename = "type")]
    pub(crate) stage_type: String,
    pub(crate) inputs: Vec<ExportedInput>,
    pub(crate) args: Value,
    pub(crate) workers: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExportedInput {
    /// The input's name, or `None` when it was given positionally.
    pub(crate) name: Option<String>,
    pub(crate) stage: String,
}

pub(crate) fn variables(resolved: &[ResolvedVariable]) -> BTreeMap<String, ExportedVariable> {
    let mut exported = BTreeMap::new();

    for variable in resolved {
        let value = variable.value.to_json();
        let entry = ExportedVariable {
            variable_type: variable.variable_type.as_str(),
            value,
        };
        exported.insert(variable.name.clone(), entry);
    }

    exported
}

pub(crate) fn stages(specs: &[StageSpec]) -> Vec<ExportedStage> {
    specs.iter().map(stage).collect()
}

fn stage(spec: &StageSpec) -> ExportedStage {
    let inputs = spec
        .inputs
        .iter()
        .map(|reference| input(&reference.value))
        .collect();

    ExportedStage {
        key: stage_key(spec),
        name: spec.name.value.clone(),
        kind: spec.kind.to_string(),
        stage_type: spec.stage_type.value.clone(),
        inputs,
        args: spec.args.value.clone(),
        workers: spec.workers,
    }
}

fn input(reference: &InputRef) -> ExportedInput {
    ExportedInput {
        name: reference.name.clone(),
        stage: reference.target.clone(),
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use av_toolbox_graph::{ArgSpans, InputRef, Span, Spanned, StageSpec};
    use av_toolbox_pipeline::variables::{ResolvedVariable, VariableType, VariableValue};
    use serde_json::json;

    use super::*;

    fn input(name: Option<&str>, target: &str) -> Spanned<InputRef> {
        let reference = InputRef {
            name: name.map(str::to_string),
            target: target.to_string(),
        };

        Spanned::unknown(reference)
    }

    fn spec(kind: StageKind, name: &str, inputs: Vec<Spanned<InputRef>>) -> StageSpec {
        StageSpec {
            name: Spanned::unknown(name.to_string()),
            kind,
            stage_type: Spanned::unknown("test/stage".to_string()),
            inputs,
            args: Spanned::unknown(json!({ "sigma": 1.5 })),
            arg_spans: ArgSpans::default(),
            workers: Some(2),
            span: Span::unknown(),
        }
    }

    #[test]
    fn a_stage_exports_its_key_type_args_and_workers() {
        let specs = [spec(StageKind::Filter, "scenes.denoise", Vec::new())];

        let exported = stages(&specs);

        assert_eq!(exported[0].key, "filter.scenes.denoise");
        assert_eq!(exported[0].name, "scenes.denoise");
        assert_eq!(exported[0].kind, "filter");
        assert_eq!(exported[0].stage_type, "test/stage");
        assert_eq!(exported[0].args, json!({ "sigma": 1.5 }));
        assert_eq!(exported[0].workers, Some(2));
    }

    #[test]
    fn inputs_keep_their_order_and_names() {
        let inputs = vec![input(None, "clip"), input(Some("mask"), "lines")];
        let specs = [spec(StageKind::Filter, "blend", inputs)];

        let exported = stages(&specs);

        let expected = vec![
            ExportedInput {
                name: None,
                stage: "clip".to_string(),
            },
            ExportedInput {
                name: Some("mask".to_string()),
                stage: "lines".to_string(),
            },
        ];
        assert_eq!(exported[0].inputs, expected);
    }

    #[test]
    fn a_stage_serialises_its_type_under_type() {
        let specs = [spec(StageKind::Sink, "out", Vec::new())];

        let exported = stages(&specs);
        let value = serde_json::to_value(&exported[0]).expect("plain JSON");

        assert_eq!(value["type"], json!("test/stage"));
        assert!(value.get("stage_type").is_none());
    }

    #[test]
    fn variables_export_their_type_and_value() {
        let resolved = [
            ResolvedVariable {
                name: "width".to_string(),
                variable_type: VariableType::Integer,
                value: VariableValue::Integer(1280),
            },
            ResolvedVariable {
                name: "input".to_string(),
                variable_type: VariableType::Path,
                value: VariableValue::Path("/clips/in.mkv".to_string()),
            },
        ];

        let exported = variables(&resolved);
        let value = serde_json::to_value(&exported).expect("plain JSON");

        let expected = json!({
            "input": { "type": "path", "value": "/clips/in.mkv" },
            "width": { "type": "integer", "value": 1280 },
        });
        assert_eq!(value, expected);
    }
}
