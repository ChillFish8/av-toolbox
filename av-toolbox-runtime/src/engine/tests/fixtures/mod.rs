pub(crate) mod delay;
pub(crate) mod double;
pub(crate) mod failing;
pub(crate) mod filters;
pub(crate) mod gated;
pub(crate) mod joins;
pub(crate) mod shaping;
pub(crate) mod sinks;
pub(crate) mod sources;
pub(crate) mod tags;

use av_toolbox_core::Parallelism;
use av_toolbox_core::testing::{FixtureCuts, HostFixtureCuts};

/// One chunk at a time, chunked by the fixture demand kind.
pub(crate) fn chunked() -> Parallelism {
    Parallelism::chunk::<FixtureCuts>(())
}

/// One chunk at a time, chunked by the host fixture demand kind, so the provider runs on the host.
pub(crate) fn host_chunked() -> Parallelism {
    Parallelism::chunk::<HostFixtureCuts>(())
}
