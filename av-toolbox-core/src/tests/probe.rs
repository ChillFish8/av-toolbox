use av_toolbox_gpu::{FrameFormat, FrameMeta, HostFrame, HostPool};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AnyProbe,
    PipelineRegistry,
    ProbeDefinition,
    ProbeKind,
    Probed,
    RegistryError,
    Seeker,
    StageError,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct Mark {
    pub(crate) value: u32,
}

/// Answers each request with its value plus the source's frame count.
pub(crate) struct Counting;

impl ProbeDefinition for Counting {
    type Request = Mark;
    type Output = u32;

    fn kind() -> &'static str {
        "counting"
    }

    fn describe(request: &Self::Request) -> String {
        format!("mark {}", request.value)
    }

    fn probe(seeker: &mut dyn Seeker, requests: &[Self::Request]) -> Result<Vec<Self::Output>, StageError> {
        let count = seeker.frame_count() as u32;
        let answers = requests.iter().map(|request| request.value + count).collect();

        Ok(answers)
    }

    fn combine(outputs: Vec<Self::Output>) -> Result<Self::Output, StageError> {
        let largest = outputs.into_iter().max().unwrap_or(0);

        Ok(largest)
    }
}

pub(crate) struct Plain;

impl ProbeDefinition for Plain {
    type Request = Mark;
    type Output = u32;

    fn kind() -> &'static str {
        "plain"
    }

    fn probe(_seeker: &mut dyn Seeker, requests: &[Self::Request]) -> Result<Vec<Self::Output>, StageError> {
        let answers = requests.iter().map(|request| request.value).collect();

        Ok(answers)
    }
}

pub(crate) struct Blank {
    pub(crate) format: FrameFormat,
    pub(crate) frames: usize,
}

impl Seeker for Blank {
    fn format(&self) -> &FrameFormat {
        &self.format
    }

    fn frame_count(&self) -> usize {
        self.frames
    }

    fn read(&mut self, index: usize) -> Result<HostFrame, StageError> {
        let pool = HostPool::new();
        let meta = FrameMeta::at(index as u64);
        let frame = pool.alloc_frame(&self.format, meta);

        Ok(frame.freeze())
    }
}

fn blank() -> Blank {
    let format = crate::tests::factory::frame_format();

    Blank { format, frames: 8 }
}

#[test]
fn a_probe_reads_back_its_typed_request() {
    let probe = AnyProbe::new::<Counting>(Mark { value: 3 });

    assert_eq!(probe.kind(), ProbeKind::of::<Counting>());
    assert_eq!(probe.request::<Counting>(), Some(Mark { value: 3 }));
}

#[test]
fn a_probe_reads_as_none_for_another_kind() {
    let probe = AnyProbe::new::<Counting>(Mark { value: 3 });

    assert_eq!(probe.request::<Plain>(), None);
}

#[test]
fn probes_with_equal_requests_are_equal() {
    let first = AnyProbe::new::<Counting>(Mark { value: 3 });
    let second = AnyProbe::new::<Counting>(Mark { value: 3 });
    let other = AnyProbe::new::<Counting>(Mark { value: 4 });

    assert_eq!(first, second);
    assert_ne!(first, other);
}

#[test]
fn a_probe_kind_registers_once() {
    let mut registry = PipelineRegistry::new();
    registry.register_probe::<Counting>().expect("registers");

    let refused = registry.register_probe::<Counting>();

    assert!(matches!(
        refused,
        Err(RegistryError::DuplicateProbe { kind: "counting" })
    ));
}

#[test]
fn the_registry_runs_a_kind_through_its_erased_runner() {
    let mut registry = PipelineRegistry::new();
    registry.register_probe::<Counting>().expect("registers");
    let kind = registry.probe_kind("counting").expect("registered");
    let runner = registry.probe_runner(kind).expect("a runner");
    let probes = [AnyProbe::new::<Counting>(Mark { value: 3 })];
    let mut seeker = blank();

    let found = runner(&mut seeker, &probes).expect("runs");

    let answer = found[0].downcast_ref::<u32>();
    assert_eq!(answer, Some(&11));
}

#[test]
fn a_kind_without_combine_refuses_to_merge() {
    let mut registry = PipelineRegistry::new();
    registry.register_probe::<Plain>().expect("registers");
    let kind = registry.probe_kind("plain").expect("registered");
    let combiner = registry.probe_combiner(kind).expect("a combiner");
    let first: crate::ProbedValue = std::sync::Arc::new(1u32);
    let second: crate::ProbedValue = std::sync::Arc::new(2u32);

    let refused = combiner(vec![first, second]);

    let error = refused.expect_err("plain cannot merge");
    assert!(error.to_string().contains("`plain`"), "{error}");
}

#[test]
fn probed_hands_back_an_output_by_input_and_kind() {
    let mut probed = Probed::new();
    let output: crate::ProbedValue = std::sync::Arc::new(7u32);
    probed.insert("input", ProbeKind::of::<Counting>(), output);

    assert_eq!(probed.get::<Counting>("input"), Some(&7));
    assert_eq!(probed.get::<Counting>("other"), None);
    assert_eq!(probed.get::<Plain>("input"), None);
}

#[test]
fn a_probe_is_described_by_its_kind() {
    let mut registry = PipelineRegistry::new();
    registry.register_probe::<Counting>().expect("registers");
    let probe = AnyProbe::new::<Counting>(Mark { value: 3 });

    assert_eq!(registry.describe_probe(&probe), "mark 3");
}
