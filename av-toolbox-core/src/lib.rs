//! Core interfaces for building video processing pipelines
//!
//! A pipeline is made of stages, and every stage falls into one of five kinds:
//!
//! - [Source], which produces frames.
//! - [Filter], which changes the frames.
//! - [Mask], which selects pixels from two or more inputs.
//! - [Transform], which changes the stream rather than the frames.
//! - [Sink], which consumes the finished stream.
//!
//! Every kind trait is typed over the side it runs on, defaulting to the device, so a plain
//! `impl Filter for X` reads exactly as it did before sides existed.
//!
//! A stage type is written as a [StageDefinition] plus the matching kind trait, one of
//! [SourceDefinition], [FilterDefinition], [MaskDefinition], [TransformDefinition] or
//! [SinkDefinition]. [StageDefinition] declares its arguments as a type, which supplies a JSON
//! schema, and its inputs as an [InputSignature]. The kind trait declares what formats it accepts
//! and produces, and how to build it. Stages are registered with a [PipelineRegistry], which
//! validates arguments against the schema and binds inputs to the signature before handing them
//! to the stage.

mod crossing;
mod error;
mod factory;
mod filter;
mod inputs;
mod mask;
mod meta;
mod parallelism;
mod registry;
mod residency;
mod sink;
mod source;
mod stage;
mod stream;
mod tag;
mod transform;

#[cfg(test)]
mod tests;

pub use av_toolbox_gpu::{AnyItem, DeviceSide, HostSide, Side};

pub use self::crossing::Crossing;
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
    PreparedMetaTransform,
    PreparedSink,
    PreparedSource,
    PreparedStage,
    PreparedTransform,
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
pub use self::meta::{MetaInputs, MetaItem, MetaTransform, end_meta_transform, run_meta_transform};
pub use self::parallelism::Parallelism;
pub use self::registry::PipelineRegistry;
pub use self::residency::{
    BuiltFilter,
    BuiltMask,
    BuiltSink,
    BuiltSource,
    BuiltTransform,
    Residency,
    Sided,
    StageSide,
};
pub use self::sink::Sink;
pub use self::source::Source;
pub use self::stage::{
    FilterDefinition,
    MaskDefinition,
    MetaTransformDefinition,
    SinkDefinition,
    SourceDefinition,
    StageDefinition,
    StageInstance,
    StageKind,
    TransformDefinition,
};
pub use self::stream::StreamType;
pub use self::tag::Tag;
pub use self::transform::Transform;
