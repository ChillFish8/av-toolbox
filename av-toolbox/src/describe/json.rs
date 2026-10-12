use av_toolbox_core::{ArtefactDecl, InputParam, InputSignature, StageInfo, StreamType};
use serde::Serialize;
use serde_json::Value;

use super::text::artefact_kind_name;
use crate::list_stages::tag_name;

/// One stage as `describe --json` prints it.
#[derive(Debug, Serialize)]
struct Described<'a> {
    identifier: &'a str,
    kind: String,
    tags: Vec<&'static str>,
    summary: &'static str,
    details: Option<&'static str>,
    inputs: Inputs,
    artefacts: Vec<Artefact>,
    schema: &'a Value,
}

#[derive(Debug, Serialize)]
struct Inputs {
    params: Vec<Param>,
    variadic: Option<Variadic>,
}

#[derive(Debug, Serialize)]
struct Param {
    name: &'static str,
    accepts: Option<String>,
    required: bool,
}

#[derive(Debug, Serialize)]
struct Variadic {
    accepts: Option<String>,
}

#[derive(Debug, Serialize)]
struct Artefact {
    name: &'static str,
    kind: &'static str,
    extension: &'static str,
    description: &'static str,
}

/// Every stage as one pretty-printed JSON array.
pub(super) fn render(stages: &[StageInfo<'_>]) -> String {
    let described: Vec<Described<'_>> = stages.iter().map(described).collect();
    serde_json::to_string_pretty(&described).expect("a description is plain JSON")
}

fn described<'a>(info: &StageInfo<'a>) -> Described<'a> {
    let tags = info.tags().iter().map(tag_name).collect();

    let signature = info.inputs();
    let inputs = inputs(signature);

    let stage_artefacts = info.artefacts();
    let artefacts = stage_artefacts.declared().iter().map(artefact).collect();

    Described {
        identifier: info.full_identifier(),
        kind: info.kind().to_string(),
        tags,
        summary: info.summary(),
        details: info.details(),
        inputs,
        artefacts,
        schema: info.schema(),
    }
}

fn inputs(signature: &InputSignature) -> Inputs {
    let params = signature.params().iter().map(param).collect();

    let variadic = if signature.is_variadic() {
        let accepts = signature.variadic_accepts();
        let stream = stream_name(accepts);
        Some(Variadic { accepts: stream })
    } else {
        None
    };

    Inputs { params, variadic }
}

fn param(param: &InputParam) -> Param {
    let accepts = param.accepts();
    let stream = stream_name(accepts);

    Param {
        name: param.name(),
        accepts: stream,
        required: param.is_required(),
    }
}

fn stream_name(accepts: Option<StreamType>) -> Option<String> {
    accepts.map(|stream| stream.to_string())
}

fn artefact(decl: &ArtefactDecl) -> Artefact {
    Artefact {
        name: decl.name,
        kind: artefact_kind_name(&decl.kind),
        extension: decl.kind.extension(),
        description: decl.description,
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::PipelineRegistry;

    use super::*;

    fn built_in() -> PipelineRegistry {
        av_toolbox_pipeline::registry().expect("the built-in stages register")
    }

    fn rendered(stage_type: &str) -> Value {
        let registry = built_in();
        let stages = crate::describe::find(&registry, stage_type).expect("registered");
        let text = render(&stages);

        serde_json::from_str(&text).expect("valid JSON")
    }

    #[test]
    fn a_stage_renders_as_an_array_of_one() {
        let value = rendered("av-denoise/nl4d");
        let stage = &value[0];

        assert_eq!(value.as_array().map(Vec::len), Some(1));
        assert_eq!(stage["identifier"], "av-denoise/nl4d");
        assert_eq!(stage["kind"], "filter");
        assert_eq!(stage["tags"], serde_json::json!(["denoise"]));
        assert!(stage["schema"]["properties"]["preset"].is_object());
    }

    #[test]
    fn the_summary_and_details_are_split() {
        let value = rendered("av-denoise/nl4d");
        let stage = &value[0];

        let summary = stage["summary"].as_str().expect("a summary");
        let details = stage["details"].as_str().expect("details");
        assert!(summary.starts_with("Removes noise"), "{summary}");
        assert!(details.contains("film grain table"), "{details}");
    }

    #[test]
    fn inputs_and_artefacts_are_listed() {
        let value = rendered("av-denoise/nl4d");
        let stage = &value[0];

        let input = &stage["inputs"]["params"][0];
        assert_eq!(input["name"], "input");
        assert_eq!(input["accepts"], "frames");
        assert_eq!(input["required"], true);
        assert!(stage["inputs"]["variadic"].is_null());

        let artefact = &stage["artefacts"][0];
        assert_eq!(artefact["name"], "grain_table");
        assert_eq!(artefact["kind"], "opaque");
        assert_eq!(artefact["extension"], "tbl");
    }

    #[test]
    fn a_type_registered_twice_renders_both() {
        let value = rendered("y4m/pipe");

        let kinds: Vec<&str> = value
            .as_array()
            .expect("an array")
            .iter()
            .filter_map(|stage| stage["kind"].as_str())
            .collect();

        assert_eq!(kinds, ["sink", "source"]);
    }

    #[test]
    fn a_variadic_signature_lists_what_it_accepts() {
        let value = rendered("null/discard");

        assert_eq!(
            value[0]["inputs"]["variadic"],
            serde_json::json!({ "accepts": null })
        );
    }
}
