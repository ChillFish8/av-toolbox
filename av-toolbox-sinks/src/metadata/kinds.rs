use schemars::JsonSchema;
use serde::de::Error;
use serde::{Deserialize, Deserializer};

const EMPTY_KINDS: &str = "kinds cannot be empty, expected a list such as [\"frame\", \"chunk\"]";
const EMPTY_KIND: &str = "a kind cannot be empty, expected a name such as \"frame\" or \"chunk\"";

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

        let has_empty = names.iter().any(String::is_empty);
        if has_empty {
            return Err(D::Error::custom(EMPTY_KIND));
        }

        Ok(Self(names))
    }
}
