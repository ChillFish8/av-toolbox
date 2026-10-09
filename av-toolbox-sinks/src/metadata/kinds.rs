use av_toolbox_gpu::MetaRecord;
use schemars::JsonSchema;
use serde::de::Error;
use serde::{Deserialize, Deserializer};

const EMPTY_KINDS: &str = "kinds cannot be empty, expected a list such as [\"frame\", \"chunk\"]";
const EXPECTED_NAME: &str = "expected a name such as \"frame\" or \"chunk\"";

/// The kinds of line to write.
#[derive(Debug, Clone, PartialEq, Eq, JsonSchema)]
pub(crate) struct Kinds(Vec<String>);

impl Kinds {
    pub(crate) fn keeps(&self, kind: &str) -> bool {
        self.0.iter().any(|kept| kept == kind)
    }
}

impl<'de> Deserialize<'de> for Kinds {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let names = Vec::<String>::deserialize(deserializer)?;
        if names.is_empty() {
            return Err(D::Error::custom(EMPTY_KINDS));
        }

        for name in &names {
            if let Err(reason) = check_name(name) {
                return Err(D::Error::custom(reason));
            }
        }

        Ok(Self(names))
    }
}

/// Checks `name` is a well formed record kind, or `frame`, the kind of frame lines.
fn check_name(name: &str) -> Result<(), String> {
    if name == "frame" {
        return Ok(());
    }

    match MetaRecord::check_kind(name) {
        Ok(()) => Ok(()),
        Err(reason) => Err(format!("{reason}, {EXPECTED_NAME}")),
    }
}
