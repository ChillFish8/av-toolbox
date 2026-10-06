//! Core interfaces for building video processing pipelines
//!
//! A pipeline is made of stages, and every stage falls into one of five kinds:
//!
//! - [Source], which produces frames.
//! - [Filter], which changes the frames.
//! - [Mask], which selects pixels from two or more inputs.
//! - [Provider], which supplies what other stages demand. The graph places these itself.
//! - [Sink], which consumes the finished stream.
//!
//! Every kind trait is typed over the side it runs on, defaulting to the device.
//!
//! A stage type is written as a [StageDefinition] plus the matching kind trait, one of
//! [SourceDefinition], [FilterDefinition], [MaskDefinition], [ProviderDefinition] or
//! [SinkDefinition]. [StageDefinition] declares its arguments as a type, which supplies a JSON
//! schema, and its inputs as an [InputSignature]. The kind trait declares what formats it accepts
//! and produces, and how to build it. Stages are registered with a [PipelineRegistry], which
//! validates arguments against the schema and binds inputs to the signature before handing them
//! to the stage.

mod chroma;
mod claims;
mod crossing;
mod demand;
mod error;
mod factory;
mod filter;
mod inputs;
mod mask;
mod meta;
mod open;
mod parallelism;
mod provider;
mod provision;
mod registry;
mod residency;
mod sink;
mod source;
mod stage;
mod stream;
mod tag;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;

#[cfg(test)]
mod tests;

pub use av_toolbox_gpu::{AnyItem, DeviceSide, HostSide, Side};

pub use self::claims::{ClaimError, Claims, DescriptorClaim};
pub use self::crossing::Crossing;
pub use self::demand::{AnyDemand, DemandDefinition, DemandKind, InputDemand, RequestError};
pub use self::error::{
    ArgsError,
    BindInputsError,
    BuildStageError,
    RegistryError,
    SchemaViolation,
    StageError,
};
pub use self::factory::{
    PreparedFilter,
    PreparedMask,
    PreparedMetaSink,
    PreparedProvider,
    PreparedSink,
    PreparedSource,
    PreparedStage,
    StageFactory,
    StageInfo,
};
pub use self::filter::Filter;
pub use self::inputs::{
    BoundInputs,
    InputArgument,
    InputParam,
    InputSignature,
    Inputs,
    StageHandle,
    VariadicInput,
    bind_inputs,
};
pub use self::mask::Mask;
pub use self::meta::{MetaInputs, MetaItem, MetaSink, end_meta_sink, run_meta_sink};
pub use self::open::OpenSlot;
pub use self::parallelism::{Granularity, Parallelism};
pub use self::provider::{LayerDemand, Placement, Provider, ProviderDefinition};
pub use self::provision::{ChunkKey, Provision, ProvisionError};
pub use self::registry::PipelineRegistry;
pub use self::residency::{
    BuiltFilter,
    BuiltMask,
    BuiltProvider,
    BuiltSink,
    BuiltSource,
    Residency,
    Sided,
    StageSide,
};
pub use self::sink::Sink;
pub use self::source::Source;
pub use self::stage::{
    FilterDefinition,
    MaskDefinition,
    MetaSinkDefinition,
    SinkDefinition,
    SourceDefinition,
    StageDefinition,
    StageInstance,
    StageKind,
};
pub use self::stream::StreamType;
pub use self::tag::Tag;
