use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    FrameFormat,
    FrameRate,
    HostSide,
    HostStreamItem,
    ResolvedFormats,
    StreamFormat,
};
use serde_json::json;

use super::demand::{Breaks, Cuts, Other};
use super::factory::{capabilities, frame_format};
use crate::{
    AnyDemand,
    BoundInputs,
    DemandKind,
    InputArgument,
    InputDemand,
    InputParam,
    InputSignature,
    Inputs,
    LayerDemand,
    Parallelism,
    PathEdge,
    PathNode,
    PathView,
    PipelineRegistry,
    PlaceError,
    Provider,
    ProviderDefinition,
    RegistryError,
    Residency,
    StageDefinition,
    StageError,
    StageHandle,
    StageInstance,
    StageKind,
    StreamType,
};

struct PassThrough;

impl StageInstance for PassThrough {}

impl Provider<HostSide> for PassThrough {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<Vec<HostStreamItem>, StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;

        Ok(vec![item])
    }
}

/// Writes a provider of cuts that passes every item through and needs a frame rate.
macro_rules! cuts_provider {
    ($definition:ident, $identifier:literal) => {
        struct $definition;

        impl StageDefinition for $definition {
            type Args = Vec<LayerDemand>;

            fn package() -> &'static str {
                "provider"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn description() -> &'static str {
                "A stage used by tests."
            }

            fn held_frames(args: &Self::Args, _formats: &ResolvedFormats) -> usize {
                args.len()
            }

            fn demands(args: &Self::Args) -> Vec<InputDemand> {
                args.iter()
                    .filter_map(|served| served.demand.request::<Cuts>())
                    .map(|request| InputDemand {
                        input: "input",
                        demand: AnyDemand::new::<Other>(request),
                    })
                    .collect()
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature =
                    &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
                SIGNATURE
            }
        }

        impl ProviderDefinition for $definition {
            type Side = HostSide;
            type Demand = Cuts;

            fn place(_request: &Breaks, path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
                Ok(path.before_demander())
            }

            fn accepts_input(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                offered: StreamFormat,
            ) -> Result<FormatSet, StageError> {
                Ok(FormatSet::of([offered.sample()]))
            }

            fn check(_args: &Self::Args, input: &FrameFormat) -> Result<(), StageError> {
                match input.frame_rate {
                    Some(_) => Ok(()),
                    None => Err(StageError::msg("the stream has no frame rate")),
                }
            }

            fn build(
                _context: &BuildContext,
                _args: Self::Args,
                _inputs: &BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Provider<HostSide>>, StageError> {
                Ok(Box::new(PassThrough))
            }
        }
    };
}

cuts_provider!(CutsProvider, "cuts");
cuts_provider!(OtherCutsProvider, "other-cuts");

fn registry_with_cuts() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();
    registry
        .register_provider::<CutsProvider>()
        .expect("should register");

    registry
}

fn provider_args() -> serde_json::Value {
    json!([
        {
            "layer": 0,
            "demand": { "kind": "cuts", "request": { "every": 4 } },
        },
    ])
}

fn path_of<'g>(names: &[&'g str], args: &'g serde_json::Value) -> PathView<'g> {
    let nodes = names
        .iter()
        .map(|name| PathNode {
            name,
            identifier: "test/blur",
            args,
        })
        .collect();

    PathView::new(nodes, names.to_vec())
}

fn input() -> Vec<InputArgument> {
    vec![InputArgument::named(
        "input",
        StageHandle::new(1),
        StreamType::Frames,
    )]
}

#[test]
fn a_provider_registers_under_its_demand_kind() {
    let registry = registry_with_cuts();

    let found = registry.provider_for(DemandKind::of::<Cuts>());
    let identifier = found.map(|(identifier, _)| identifier);

    assert_eq!(identifier, Some("provider/cuts"));

    let args = provider_args();
    let prepared = registry
        .prepare(StageKind::Provider, "provider/cuts", &args, &input())
        .expect("should prepare");

    assert_eq!(prepared.kind(), StageKind::Provider);
    assert_eq!(prepared.parallelism(), Parallelism::Stream);
    assert_eq!(prepared.chunk_authority(), Some(0));
    assert_eq!(prepared.residency(), Residency::Host);
}

#[test]
fn registering_a_provider_registers_its_demand_kind() {
    let registry = registry_with_cuts();

    assert_eq!(registry.demand_kind("cuts"), Some(DemandKind::of::<Cuts>()));

    let demand = AnyDemand::new::<Cuts>(Breaks { every: 4 });
    assert_eq!(registry.describe(&demand), "cuts");
}

#[test]
fn a_provider_registers_beside_its_kind_registered_first() {
    let mut registry = PipelineRegistry::new();
    registry.register_demand::<Cuts>().expect("should register");

    let registered = registry.register_provider::<CutsProvider>();

    assert!(registered.is_ok(), "{registered:?}");
}

#[test]
fn a_provider_cannot_be_named_twice_for_one_kind() {
    let mut registry = registry_with_cuts();

    let Err(error) = registry.register_provider::<OtherCutsProvider>() else {
        panic!("a second provider of cuts should be refused");
    };

    assert!(
        matches!(error, RegistryError::DuplicateProvider { kind: "cuts" }),
        "{error:?}"
    );
}

#[test]
fn a_provider_produces_what_it_is_given_once_checked() {
    let registry = registry_with_cuts();
    let args = provider_args();
    let prepared = registry
        .prepare(StageKind::Provider, "provider/cuts", &args, &input())
        .expect("should prepare");

    let without_rate = StreamFormat::Frames(frame_format());
    let refused = prepared.produces(&capabilities(), &[Some(without_rate)]);
    assert!(refused.is_err());

    let rate = FrameRate::new(25, 1);
    let format = FrameFormat {
        frame_rate: rate,
        ..frame_format()
    };
    let with_rate = StreamFormat::Frames(format);
    let produced = prepared
        .produces(&capabilities(), &[Some(with_rate)])
        .expect("should produce");

    assert_eq!(produced.samples, FormatSet::of([format.sample]));
}

#[test]
fn a_provider_holds_what_its_definition_says() {
    let registry = registry_with_cuts();
    let args = provider_args();
    let prepared = registry
        .prepare(StageKind::Provider, "provider/cuts", &args, &input())
        .expect("should prepare");

    let formats = ResolvedFormats::new(vec![], None);

    assert_eq!(prepared.held_frames(&formats), 1);
}

#[test]
fn a_registered_provider_places_through_its_definition() {
    let registry = registry_with_cuts();
    let (_, place) = registry
        .provider_for(DemandKind::of::<Cuts>())
        .expect("cuts has a provider");
    let args = json!({});
    let path = path_of(&["input", "soften"], &args);
    let demand = AnyDemand::new::<Cuts>(Breaks { every: 4 });

    let edge = place(&demand, &path).expect("should place");

    assert_eq!(edge, path.before_demander());
}

#[test]
fn a_registered_provider_refuses_a_request_it_cannot_read() {
    let registry = registry_with_cuts();
    let (_, place) = registry
        .provider_for(DemandKind::of::<Cuts>())
        .expect("cuts has a provider");
    let args = json!({});
    let path = path_of(&["input"], &args);
    let value = json!({ "kind": "cuts", "request": { "every": "often" } });
    let malformed: AnyDemand = serde_json::from_value(value).expect("deserialises");

    let Err(error) = place(&malformed, &path) else {
        panic!("an unreadable request should be refused");
    };

    assert!(error.to_string().contains("cannot be read"), "{error}");
}

#[test]
fn a_registered_provider_reports_its_demands() {
    let registry = registry_with_cuts();
    let demands_of = registry
        .provider_demands(DemandKind::of::<Cuts>())
        .expect("cuts has a provider");

    let demands = demands_of(&provider_args()).expect("should read its args");

    let expected = vec![InputDemand {
        input: "input",
        demand: AnyDemand::new::<Other>(Breaks { every: 4 }),
    }];
    assert_eq!(demands, expected);
}

#[test]
fn a_registered_provider_refuses_args_it_cannot_read() {
    let registry = registry_with_cuts();
    let demands_of = registry
        .provider_demands(DemandKind::of::<Cuts>())
        .expect("cuts has a provider");

    let refused = demands_of(&json!({ "layer": "first" }));

    assert!(refused.is_err());
}

#[test]
fn a_prepared_provider_reports_its_demands_on_its_input() {
    let registry = registry_with_cuts();
    let args = provider_args();
    let prepared = registry
        .prepare(StageKind::Provider, "provider/cuts", &args, &input())
        .expect("should prepare");

    let expected = vec![InputDemand {
        input: "input",
        demand: AnyDemand::new::<Other>(Breaks { every: 4 }),
    }];
    assert_eq!(prepared.demands(), expected);
}
