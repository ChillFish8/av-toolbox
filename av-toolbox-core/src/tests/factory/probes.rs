use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::{ExampleFilter, capabilities, frame_format, reference_input};
use crate::tests::probe::{Blank, Counting, Mark};
use crate::{
    AnyProbe,
    ArgsError,
    BoundInputs,
    BuildStageError,
    Filter,
    FilterDefinition,
    InputParam,
    InputProbe,
    InputSignature,
    PipelineRegistry,
    PreparedStage,
    ProbeKind,
    Probed,
    ProbedValue,
    Seeker,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageKind,
    StreamType,
};

#[derive(Debug, Deserialize, JsonSchema)]
struct ProbingArgs {
    #[serde(default)]
    missing: bool,
    #[serde(default)]
    twice: bool,
    #[serde(skip)]
    found: Option<u32>,
}

struct Probing;

impl StageDefinition for Probing {
    type Args = ProbingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "probing"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn probes(args: &Self::Args) -> Vec<InputProbe> {
        let input = if args.missing { "missing" } else { "reference" };
        let probe = InputProbe {
            input,
            probe: AnyProbe::new::<Counting>(Mark { value: 1 }),
        };

        match args.twice {
            true => vec![probe.clone(), probe],
            false => vec![probe],
        }
    }

    fn settle(args: &mut Self::Args, probed: &Probed) -> Result<(), StageError> {
        args.found = probed.get::<Counting>("reference").copied();

        Ok(())
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Probing {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    /// Refuses to produce until settled, and then reports the settled value as its width.
    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        _chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let Some(found) = args.found else {
            return Err(StageError::msg("not settled"));
        };

        let mut format = frame_format();
        format.width = found;
        let stream = StreamFormat::Frames(format);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&stream),
            samples: FormatSet::of([format.sample]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(ExampleFilter))
    }
}

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();
    registry.register_filter::<Probing>().expect("registers");
    registry.register_source::<Seekable>().expect("registers");
    registry
}

fn prepare_probing(args: serde_json::Value) -> Result<PreparedStage, BuildStageError> {
    let registry = registry();
    let inputs = reference_input();

    registry.prepare(StageKind::Filter, "test/probing", &args, &inputs)
}

fn counted(value: u32) -> Probed {
    let mut probed = Probed::new();
    let output: ProbedValue = std::sync::Arc::new(value);
    probed.insert("reference", ProbeKind::of::<Counting>(), output);
    probed
}

#[test]
fn a_stage_lists_its_probes() {
    let prepared = prepare_probing(serde_json::json!({})).expect("prepares");

    let expected = InputProbe {
        input: "reference",
        probe: AnyProbe::new::<Counting>(Mark { value: 1 }),
    };
    assert_eq!(prepared.probes(), vec![expected]);
}

#[test]
fn a_probe_on_an_unknown_input_is_refused() {
    let args = serde_json::json!({ "missing": true });

    let Err(error) = prepare_probing(args) else {
        panic!("should refuse the probe");
    };

    let BuildStageError::InvalidArgs { source, .. } = error else {
        panic!("expected invalid arguments, got {error:?}");
    };
    assert!(matches!(source, ArgsError::InvalidProbe(_)));
    assert!(source.to_string().contains("missing"));
}

#[test]
fn two_probes_of_one_kind_on_one_input_are_refused() {
    let args = serde_json::json!({ "twice": true });

    let Err(error) = prepare_probing(args) else {
        panic!("should refuse the second probe");
    };

    assert!(error.to_string().contains("probing"), "{error}");
}

#[test]
fn settling_reaches_the_stage_args() {
    let mut prepared = prepare_probing(serde_json::json!({})).expect("prepares");
    let chosen = [Some(StreamFormat::Frames(frame_format()))];
    let unsettled = prepared.produces(&capabilities(), &chosen);
    assert!(unsettled.is_err());

    let probed = counted(42);
    prepared.settle(&probed).expect("settles");

    let produced = prepared.produces(&capabilities(), &chosen).expect("settled");
    let format = produced.template.with_sample(frame_format().sample);
    assert_eq!(format.as_frames().map(|frames| frames.width), Some(42));
}

#[test]
fn a_stage_that_does_not_probe_settles_as_a_no_op() {
    let registry = super::registry_with_example();
    let inputs = reference_input();
    let args = serde_json::json!({ "sigma_scale": 1.0 });
    let mut prepared = registry
        .prepare(StageKind::Filter, "av-denoise/nl4d", &args, &inputs)
        .expect("prepares");

    assert!(prepared.probes().is_empty());
    prepared.settle(&Probed::new()).expect("a no-op");
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct Seekable;

impl StageDefinition for Seekable {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "seekable"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Seekable {
    type Side = DeviceSide;
    type Opened = ();

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        Ok(())
    }

    fn seeker(_opened: &Self::Opened) -> Result<Option<Box<dyn Seeker>>, StageError> {
        let format = frame_format();
        let seeker = Blank { format, frames: 5 };

        Ok(Some(Box::new(seeker)))
    }

    fn produces(_capabilities: &Capabilities, _opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let stream = StreamFormat::Frames(frame_format());

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&stream),
            samples: FormatSet::of([frame_format().sample]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<DeviceSide>>, StageError> {
        Err(StageError::msg("never built"))
    }
}

#[test]
fn a_source_hands_out_its_seeker() {
    let registry = registry();
    let args = serde_json::json!({});
    let prepared = registry
        .prepare(StageKind::Source, "test/seekable", &args, &[])
        .expect("prepares");

    let seeker = prepared.seeker().expect("opens").expect("seekable");

    assert_eq!(seeker.frame_count(), 5);
}

#[test]
fn a_filter_has_no_seeker() {
    let prepared = prepare_probing(serde_json::json!({})).expect("prepares");

    let seeker = prepared.seeker().expect("no error");

    assert!(seeker.is_none());
}
