use std::fmt;

use av_toolbox_gpu::{DeviceSide, HostSide, Side};

use crate::{Filter, Mask, Sink, Source, Transform};

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
pub type BuiltTransform = Sided<dyn Transform<DeviceSide>, dyn Transform<HostSide>>;
pub type BuiltSink = Sided<dyn Sink<DeviceSide>, dyn Sink<HostSide>>;

/// A side a stage can be built for, which knows how to file what was built for it.
pub trait StageSide: Side {
    const RESIDENCY: Residency;

    fn source(stage: Box<dyn Source<Self>>) -> BuiltSource;

    fn filter(stage: Box<dyn Filter<Self>>) -> BuiltFilter;

    fn mask(stage: Box<dyn Mask<Self>>) -> BuiltMask;

    fn transform(stage: Box<dyn Transform<Self>>) -> BuiltTransform;

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

    fn transform(stage: Box<dyn Transform<Self>>) -> BuiltTransform {
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

    fn transform(stage: Box<dyn Transform<Self>>) -> BuiltTransform {
        Sided::Host(stage)
    }

    fn sink(stage: Box<dyn Sink<Self>>) -> BuiltSink {
        Sided::Host(stage)
    }
}
