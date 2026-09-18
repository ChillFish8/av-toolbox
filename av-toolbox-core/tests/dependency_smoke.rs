// This file tests schemars and jsonschema rather than this crate. It is kept deliberately, it
// pins the third party API shapes the crate depends on. It is what caught that
// `ValidationError::instance_path` is a method, not a field.

use schemars::{JsonSchema, SchemaGenerator};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct ExampleArgs {
    sigma_scale: f32,
    label: Option<String>,
}

fn example_schema() -> serde_json::Value {
    let generator = SchemaGenerator::default();
    let schema = generator.into_root_schema_for::<ExampleArgs>();
    serde_json::to_value(&schema).expect("schema should serialise to JSON")
}

#[test]
fn schema_generation_produces_an_object_schema() {
    let schema = example_schema();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["sigma_scale"].is_object());
}

#[test]
fn validator_accepts_matching_arguments() {
    let schema = example_schema();
    let validator = jsonschema::validator_for(&schema).expect("schema should compile");

    let args = serde_json::json!({ "sigma_scale": 0.8 });
    let violations = validator.iter_errors(&args).count();

    assert_eq!(violations, 0);
}

#[test]
fn validator_reports_every_violation_with_a_path() {
    let schema = example_schema();
    let validator = jsonschema::validator_for(&schema).expect("schema should compile");

    let args = serde_json::json!({ "sigma_scale": "not a number" });
    let violations: Vec<(String, String)> = validator
        .iter_errors(&args)
        .map(|error| (error.instance_path().to_string(), error.to_string()))
        .collect();

    assert!(!violations.is_empty());
    assert_eq!(violations[0].0, "/sigma_scale");
}
