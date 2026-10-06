use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::{Mutex, OnceLock};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// One kind of thing a stage can ask of its input, such as scene breaks.
pub trait DemandDefinition: 'static {
    /// What a stage asks for, such as which scene breaks.
    type Request: Clone + PartialEq + Debug + Serialize + DeserializeOwned + JsonSchema + Send + Sync;

    /// The name the kind is registered under, such as `scenes`.
    fn kind() -> &'static str;

    /// A short description of a request for `explain`, such as `natural`. The kind's name by default.
    fn describe(_request: &Self::Request) -> String {
        Self::kind().to_string()
    }
}

/// The name a demand kind is registered under.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DemandKind(&'static str);

impl DemandKind {
    pub fn of<D: DemandDefinition>() -> Self {
        Self(D::kind())
    }

    pub const fn name(self) -> &'static str {
        self.0
    }
}

/// A demand of any registered kind, carried without its type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnyDemand {
    #[serde(serialize_with = "serialize_kind", deserialize_with = "deserialize_kind")]
    kind: DemandKind,
    request: Value,
}

impl AnyDemand {
    pub fn new<D: DemandDefinition>(request: D::Request) -> Self {
        let request = serde_json::to_value(request).expect("a demand request serialises");

        Self {
            kind: DemandKind::of::<D>(),
            request,
        }
    }

    pub fn kind(&self) -> DemandKind {
        self.kind
    }

    /// The request, when this demand is of kind `D`.
    pub fn request<D: DemandDefinition>(&self) -> Option<D::Request> {
        if self.kind != DemandKind::of::<D>() {
            return None;
        }

        let request = self.request.clone();
        serde_json::from_value(request).ok()
    }

    /// The request as it was serialised, for rendering and for provider args.
    pub fn raw(&self) -> &Value {
        &self.request
    }
}

/// One demand on a named input.
#[derive(Debug, Clone, PartialEq)]
pub struct InputDemand {
    pub input: &'static str,
    pub demand: AnyDemand,
}

fn serialize_kind<S: serde::Serializer>(kind: &DemandKind, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(kind.name())
}

fn deserialize_kind<'de, D: Deserializer<'de>>(deserializer: D) -> Result<DemandKind, D::Error> {
    let name = String::deserialize(deserializer)?;
    let interned = intern(&name);

    Ok(DemandKind(interned))
}

/// Hands back a `'static` copy of `name`, leaking each distinct name once per process.
pub(crate) fn intern(name: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();

    let names = NAMES.get_or_init(Mutex::default);
    let mut names = names.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

    if let Some(existing) = names.get(name) {
        return existing;
    }

    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(leaked);
    leaked
}
