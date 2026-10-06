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

use super::{ExampleFilter, reference_input};
use crate::tests::demand::{Breaks, Cuts};
use crate::{
    AnyDemand,
    ArgsError,
    BoundInputs,
    BuildStageError,
    Filter,
    FilterDefinition,
    InputArgument,
    InputDemand,
    InputParam,
    InputSignature,
    Parallelism,
    PipelineRegistry,
    PreparedStage,
    StageDefinition,
    StageError,
    StageHandle,
    StageKind,
    StreamType,
};

/// Writes a filter definition that passes its first input through, for the demand fixtures.
macro_rules! pass_through_filter {
    ($definition:ident) => {
        impl FilterDefinition for $definition {
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

            fn produces(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                chosen: &[Option<StreamFormat>],
            ) -> Result<OutputFormats, StageError> {
                let format = chosen.iter().flatten().next().copied().expect("an input");
                Ok(OutputFormats {
                    template: StreamTemplate::from_format(&format),
                    samples: FormatSet::of([format.sample()]),
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
    };
}

fn cuts() -> AnyDemand {
    AnyDemand::new::<Cuts>(Breaks { every: 4 })
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DemandingArgs {
    #[serde(default)]
    demand_missing: bool,
    #[serde(default)]
    workers: Option<usize>,
}

struct DemandingExample;

impl StageDefinition for DemandingExample {
    type Args = DemandingArgs;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        "demanding"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Chunk(Some(cuts()))
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        if !args.demand_missing {
            return Vec::new();
        }

        let demand = InputDemand {
            input: "missing",
            demand: cuts(),
        };
        vec![demand]
    }

    fn workers(args: &Self::Args) -> Option<usize> {
        args.workers
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

pass_through_filter!(DemandingExample);

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct VariadicExample;

impl StageDefinition for VariadicExample {
    type Args = NoArgs;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        "variadic"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Chunk(Some(cuts()))
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::with_variadic(&[]);
        SIGNATURE
    }
}

pass_through_filter!(VariadicExample);

fn prepare_demanding(args: serde_json::Value) -> Result<PreparedStage, BuildStageError> {
    let mut registry = PipelineRegistry::new();
    registry
        .register_filter::<DemandingExample>()
        .expect("should register");

    registry.prepare(
        StageKind::Filter,
        "av-denoise/demanding",
        &args,
        &reference_input(),
    )
}

#[test]
fn a_chunk_stage_demands_its_chunks_on_its_authority() {
    let prepared = prepare_demanding(serde_json::json!({})).expect("should prepare");

    let expected = InputDemand {
        input: "input",
        demand: cuts(),
    };
    assert_eq!(prepared.demands(), vec![expected]);
}

#[test]
fn a_chunk_stage_on_a_variadic_input_demands_on_it() {
    let mut registry = PipelineRegistry::new();
    registry
        .register_filter::<VariadicExample>()
        .expect("should register");

    let inputs = [InputArgument::named(
        "clip",
        StageHandle::new(1),
        StreamType::Frames,
    )];
    let args = serde_json::json!({});
    let Ok(prepared) = registry.prepare(StageKind::Filter, "av-denoise/variadic", &args, &inputs) else {
        panic!("should prepare");
    };

    let expected = InputDemand {
        input: "clip",
        demand: cuts(),
    };
    assert_eq!(prepared.demands(), vec![expected]);
}

#[test]
fn a_demand_on_an_unknown_input_is_refused() {
    let args = serde_json::json!({ "demand_missing": true });

    let Err(error) = prepare_demanding(args) else {
        panic!("should refuse the demand");
    };

    let BuildStageError::InvalidArgs { source, .. } = error else {
        panic!("expected invalid arguments, got {error:?}");
    };
    assert!(matches!(source, ArgsError::InvalidDemand(_)));
    assert!(source.to_string().contains("missing"));
}

#[test]
fn a_stage_suggests_its_workers() {
    let prepared = prepare_demanding(serde_json::json!({ "workers": 2 })).expect("should prepare");

    assert_eq!(prepared.suggested_workers(), Some(2));
}
