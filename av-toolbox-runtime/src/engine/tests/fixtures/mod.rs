pub(crate) mod delay;
pub(crate) mod double;
pub(crate) mod failing;
pub(crate) mod filters;
pub(crate) mod gated;
pub(crate) mod joins;
pub(crate) mod shaping;
pub(crate) mod sinks;
pub(crate) mod sources;

use av_toolbox_core::testing::{FixtureCuts, HostFixtureCuts};
use av_toolbox_core::{AnyDemand, Parallelism};

/// One chunk at a time, chunked by the fixture demand kind.
pub(crate) fn chunked() -> Parallelism {
    let demand = AnyDemand::new::<FixtureCuts>(());

    Parallelism::Chunk(demand)
}

/// One chunk at a time, chunked by the host fixture demand kind, so the provider runs on the host.
pub(crate) fn host_chunked() -> Parallelism {
    let demand = AnyDemand::new::<HostFixtureCuts>(());

    Parallelism::Chunk(demand)
}
