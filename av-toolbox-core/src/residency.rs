use std::fmt;

use av_toolbox_gpu::{DeviceSide, HostSide, Side};

use crate::{Filter, Mask, MetaProvider, Provider, Sink, Source, StageInstance};

/// Where a stage runs.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Residency {
    Host,
    Device,
    Either,
}

impl fmt::Display for Residency {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Host => "host",
            Self::Device => "gpu",
            Self::Either => "either",
        };

        formatter.write_str(name)
    }
}

impl Residency {
    /// The name a provider registered for this side is shown under.
    ///
    /// A provider that runs on metadata alone is shown as `meta`.
    pub fn provider_tag(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Device => "device",
            Self::Either => "meta",
        }
    }
}

/// A built stage of one kind, on whichever side its definition named.
pub enum Sided<D: ?Sized, H: ?Sized> {
    Device(Box<D>),
    Host(Box<H>),
}

impl<D: ?Sized, H: ?Sized> Sided<D, H> {
    pub fn residency(&self) -> Residency {
        match self {
            Self::Device(_) => Residency::Device,
            Self::Host(_) => Residency::Host,
        }
    }

    pub fn into_device(self) -> Option<Box<D>> {
        match self {
            Self::Device(stage) => Some(stage),
            Self::Host(_) => None,
        }
    }

    pub fn into_host(self) -> Option<Box<H>> {
        match self {
            Self::Host(stage) => Some(stage),
            Self::Device(_) => None,
        }
    }
}

pub type BuiltSource = Sided<dyn Source<DeviceSide>, dyn Source<HostSide>>;
pub type BuiltFilter = Sided<dyn Filter<DeviceSide>, dyn Filter<HostSide>>;
pub type BuiltMask = Sided<dyn Mask<DeviceSide>, dyn Mask<HostSide>>;
pub type BuiltSink = Sided<dyn Sink<DeviceSide>, dyn Sink<HostSide>>;

/// A side a stage can be built for, which knows how to file what was built for it.
pub trait StageSide: Side {
    const RESIDENCY: Residency;

    fn source(stage: Box<dyn Source<Self>>) -> BuiltSource;

    fn filter(stage: Box<dyn Filter<Self>>) -> BuiltFilter;

    fn mask(stage: Box<dyn Mask<Self>>) -> BuiltMask;

    fn sink(stage: Box<dyn Sink<Self>>) -> BuiltSink;
}

impl StageSide for DeviceSide {
    const RESIDENCY: Residency = Residency::Device;

    fn source(stage: Box<dyn Source<Self>>) -> BuiltSource {
        Sided::Device(stage)
    }

    fn filter(stage: Box<dyn Filter<Self>>) -> BuiltFilter {
        Sided::Device(stage)
    }

    fn mask(stage: Box<dyn Mask<Self>>) -> BuiltMask {
        Sided::Device(stage)
    }

    fn sink(stage: Box<dyn Sink<Self>>) -> BuiltSink {
        Sided::Device(stage)
    }
}

impl StageSide for HostSide {
    const RESIDENCY: Residency = Residency::Host;

    fn source(stage: Box<dyn Source<Self>>) -> BuiltSource {
        Sided::Host(stage)
    }

    fn filter(stage: Box<dyn Filter<Self>>) -> BuiltFilter {
        Sided::Host(stage)
    }

    fn mask(stage: Box<dyn Mask<Self>>) -> BuiltMask {
        Sided::Host(stage)
    }

    fn sink(stage: Box<dyn Sink<Self>>) -> BuiltSink {
        Sided::Host(stage)
    }
}

/// A built provider, on whichever side its definition named.
pub enum BuiltProvider {
    Device(Box<dyn Provider<DeviceSide>>),
    Host(Box<dyn Provider<HostSide>>),
    Meta(Box<dyn MetaProvider>),
}

/// Where a provider runs, on the host, on the device, or on metadata alone.
pub trait ProviderSide: 'static {
    const RESIDENCY: Residency;

    /// What a provider on this side is built as.
    type Instance: ?Sized + StageInstance;

    fn provider(stage: Box<Self::Instance>) -> BuiltProvider;
}

impl ProviderSide for DeviceSide {
    const RESIDENCY: Residency = Residency::Device;

    type Instance = dyn Provider<DeviceSide>;

    fn provider(stage: Box<Self::Instance>) -> BuiltProvider {
        BuiltProvider::Device(stage)
    }
}

impl ProviderSide for HostSide {
    const RESIDENCY: Residency = Residency::Host;

    type Instance = dyn Provider<HostSide>;

    fn provider(stage: Box<Self::Instance>) -> BuiltProvider {
        BuiltProvider::Host(stage)
    }
}

/// A provider that only reads and writes metadata, so it runs on whichever side its input is on.
#[derive(Debug, Copy, Clone)]
pub struct MetaSide;

impl ProviderSide for MetaSide {
    const RESIDENCY: Residency = Residency::Either;

    type Instance = dyn MetaProvider;

    fn provider(stage: Box<Self::Instance>) -> BuiltProvider {
        BuiltProvider::Meta(stage)
    }
}
