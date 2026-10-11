use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use av_toolbox_gpu::{FrameFormat, HostFrame};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::StageError;

/// One kind of thing a stage can learn by reading its source before the run starts.
pub trait ProbeDefinition: 'static {
    /// What a stage asks for.
    type Request: Clone + PartialEq + Debug + Serialize + DeserializeOwned + JsonSchema + Send + Sync;

    /// What the probe found for one request.
    type Output: Clone + Debug + Send + Sync + 'static;

    /// The name the kind is registered under, such as `crop`.
    fn kind() -> &'static str;

    /// A short description of a request for `explain`. The kind's name by default.
    fn describe(_request: &Self::Request) -> String {
        Self::kind().to_string()
    }

    /// Reads the source through `seeker` and answers one output per request, in request order.
    ///
    /// Frame numbers are the source's own, before any stage changes them.
    fn probe(seeker: &mut dyn Seeker, requests: &[Self::Request]) -> Result<Vec<Self::Output>, StageError>;

    /// Merges what several sources found for one request. Refused by default.
    fn combine(_outputs: Vec<Self::Output>) -> Result<Self::Output, StageError> {
        Err(cannot_merge(Self::kind()))
    }
}

/// Reads any frame of an opened source by its number.
pub trait Seeker: Send {
    fn format(&self) -> &FrameFormat;

    fn frame_count(&self) -> usize;

    /// The frame at `index`, refused at or past [Seeker::frame_count].
    fn read(&mut self, index: usize) -> Result<HostFrame, StageError>;
}

/// The name a probe kind is registered under.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ProbeKind(&'static str);

impl ProbeKind {
    pub fn of<P: ProbeDefinition>() -> Self {
        Self(P::kind())
    }

    pub const fn name(self) -> &'static str {
        self.0
    }
}

/// A probe of any registered kind, carried without its type.
#[derive(Debug, Clone, PartialEq)]
pub struct AnyProbe {
    kind: ProbeKind,
    request: Value,
}

impl AnyProbe {
    pub fn new<P: ProbeDefinition>(request: P::Request) -> Self {
        let request = serde_json::to_value(request).expect("a probe request serialises");

        Self {
            kind: ProbeKind::of::<P>(),
            request,
        }
    }

    pub fn kind(&self) -> ProbeKind {
        self.kind
    }

    /// The request, when this probe is of kind `P`.
    pub fn request<P: ProbeDefinition>(&self) -> Option<P::Request> {
        if self.kind != ProbeKind::of::<P>() {
            return None;
        }

        let request = self.request.clone();
        serde_json::from_value(request).ok()
    }

    /// The request as it was serialised.
    pub fn raw(&self) -> &Value {
        &self.request
    }
}

/// One probe on a named input.
#[derive(Debug, Clone, PartialEq)]
pub struct InputProbe {
    pub input: &'static str,
    pub probe: AnyProbe,
}

/// What one probe found, carried without its type.
pub type ProbedValue = Arc<dyn Any + Send + Sync>;

/// Runs a batch of probes of one kind against a seeker.
pub type RunProbeFn = fn(&mut dyn Seeker, &[AnyProbe]) -> Result<Vec<ProbedValue>, StageError>;

/// Merges what one probe kind found at several sources.
pub type CombineProbeFn = fn(Vec<ProbedValue>) -> Result<ProbedValue, StageError>;

/// What a stage's probes found, one output per input and kind.
#[derive(Clone, Default)]
pub struct Probed {
    found: Vec<Found>,
}

#[derive(Clone)]
struct Found {
    input: &'static str,
    kind: ProbeKind,
    output: ProbedValue,
}

impl Probed {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, input: &'static str, kind: ProbeKind, output: ProbedValue) {
        let found = Found { input, kind, output };
        self.found.push(found);
    }

    /// What the probe of kind `P` on `input` found.
    pub fn get<P: ProbeDefinition>(&self, input: &str) -> Option<&P::Output> {
        let kind = ProbeKind::of::<P>();
        let found = self
            .found
            .iter()
            .find(|found| found.input == input && found.kind == kind)?;

        found.output.downcast_ref::<P::Output>()
    }

    pub fn is_empty(&self) -> bool {
        self.found.is_empty()
    }
}

impl fmt::Debug for Probed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self.found.iter().map(|found| (found.input, found.kind.name()));

        formatter.debug_list().entries(entries).finish()
    }
}

fn cannot_merge(kind: &str) -> StageError {
    let message = format!("`{kind}` probes cannot merge what several sources found");

    StageError::msg(message)
}

fn malformed(kind: &str) -> StageError {
    let message = format!("a `{kind}` probe was handed a request it cannot read");

    StageError::msg(message)
}

fn miscounted(kind: &str, asked: usize, answered: usize) -> StageError {
    let message = format!("a `{kind}` probe answered {answered} of {asked} requests");

    StageError::msg(message)
}

fn wrong_output(kind: &str) -> StageError {
    let message = format!("a `{kind}` probe was handed an output of another kind to merge");

    StageError::msg(message)
}

fn erase<P: ProbeDefinition>(output: P::Output) -> ProbedValue {
    Arc::new(output)
}

pub(crate) fn run_as<P: ProbeDefinition>(
    seeker: &mut dyn Seeker,
    probes: &[AnyProbe],
) -> Result<Vec<ProbedValue>, StageError> {
    let mut requests = Vec::with_capacity(probes.len());
    for probe in probes {
        let Some(request) = probe.request::<P>() else {
            return Err(malformed(P::kind()));
        };

        requests.push(request);
    }

    let outputs = P::probe(seeker, &requests)?;
    if outputs.len() != requests.len() {
        return Err(miscounted(P::kind(), requests.len(), outputs.len()));
    }

    let erased = outputs.into_iter().map(erase::<P>).collect();
    Ok(erased)
}

pub(crate) fn combine_as<P: ProbeDefinition>(values: Vec<ProbedValue>) -> Result<ProbedValue, StageError> {
    let mut outputs = Vec::with_capacity(values.len());
    for value in &values {
        let Some(output) = value.downcast_ref::<P::Output>() else {
            return Err(wrong_output(P::kind()));
        };

        outputs.push(output.clone());
    }

    let combined = P::combine(outputs)?;
    Ok(Arc::new(combined))
}

pub(crate) fn describe_as<P: ProbeDefinition>(probe: &AnyProbe) -> String {
    match probe.request::<P>() {
        Some(request) => P::describe(&request),
        None => P::kind().to_string(),
    }
}

pub(crate) fn show_as<P: ProbeDefinition>(value: &ProbedValue) -> String {
    match value.downcast_ref::<P::Output>() {
        Some(output) => format!("{output:?}"),
        None => P::kind().to_string(),
    }
}
