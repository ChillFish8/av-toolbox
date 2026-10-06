use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AnyDemand, DemandDefinition, DemandKind, Granularity, Parallelism};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct Breaks {
    pub(crate) every: u32,
}

pub(crate) struct Cuts;

impl DemandDefinition for Cuts {
    type Request = Breaks;

    fn kind() -> &'static str {
        "cuts"
    }
}

struct Other;

impl DemandDefinition for Other {
    type Request = Breaks;

    fn kind() -> &'static str {
        "other"
    }
}

#[test]
fn a_demand_round_trips_its_request() {
    let demand = AnyDemand::new::<Cuts>(Breaks { every: 4 });

    assert_eq!(demand.kind(), DemandKind::of::<Cuts>());
    assert_eq!(demand.request::<Cuts>(), Some(Breaks { every: 4 }));
}

#[test]
fn a_demand_of_another_kind_reads_as_none() {
    let demand = AnyDemand::new::<Cuts>(Breaks { every: 4 });

    assert_eq!(demand.request::<Other>(), None);
}

#[test]
fn equal_requests_are_equal_demands() {
    let first = AnyDemand::new::<Cuts>(Breaks { every: 4 });
    let second = AnyDemand::new::<Cuts>(Breaks { every: 4 });
    let different = AnyDemand::new::<Cuts>(Breaks { every: 5 });

    assert_eq!(first, second);
    assert_ne!(first, different);
}

#[test]
fn a_demand_survives_json() {
    let demand = AnyDemand::new::<Cuts>(Breaks { every: 4 });

    let value = serde_json::to_value(&demand).expect("serialises");
    let back: AnyDemand = serde_json::from_value(value).expect("deserialises");

    assert_eq!(back, demand);
}

#[test]
fn a_chunk_request_carries_its_demand() {
    let demand = AnyDemand::new::<Cuts>(Breaks { every: 4 });
    let chunked = Parallelism::Chunk(demand.clone());

    assert_eq!(chunked.granularity(), Granularity::Chunk);
    assert_eq!(chunked.demand(), Some(&demand));
    assert_eq!(Parallelism::Frame.demand(), None);
}

#[test]
fn granularities_order_by_how_much_they_demand() {
    assert!(Granularity::Frame < Granularity::Chunk);
    assert!(Granularity::Chunk < Granularity::Stream);
}
