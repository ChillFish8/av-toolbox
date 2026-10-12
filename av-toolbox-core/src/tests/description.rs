use schemars::JsonSchema;
use serde::Deserialize;

use crate::factory::Definition;
use crate::{InputSignature, RegistryError, StageDefinition, StageKind};

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct Blank;

impl StageDefinition for Blank {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "blank"
    }

    fn description() -> &'static str {
        "  \n\nAn explanation without a summary."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

#[test]
fn a_stage_without_a_summary_is_refused() {
    let created = Definition::<Blank>::new(StageKind::Filter);
    let Err(error) = created else {
        panic!("a stage without a summary should be refused");
    };

    assert!(
        matches!(&error, RegistryError::MissingDescription { identifier } if identifier == "test/blank"),
        "{error}"
    );
}
