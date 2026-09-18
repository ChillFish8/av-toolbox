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
//! A stage type is written as a [StageDefinition]. It declares its arguments as a type, which
//! supplies a JSON schema, and its inputs as an [InputSignature]. Stages are registered with a
//! [PipelineRegistry], which validates arguments against the schema and binds inputs to the
//! signature before handing them to the stage.

mod error;
mod factory;
mod filter;
mod inputs;
mod mask;
mod registry;
mod sink;
mod source;
mod stage;
mod tag;
mod transform;

pub use self::error::{
    ArgsError,
    BindInputsError,
    BuildStageError,
    RegistryError,
    SchemaViolation,
    StageError,
};
pub use self::factory::{StageFactory, StageInfo};
pub use self::filter::Filter;
pub use self::inputs::{
    BoundInputs,
    InputArgument,
    InputParam,
    InputSignature,
    StageHandle,
    VariadicInput,
    bind_inputs,
};
pub use self::mask::Mask;
pub use self::registry::PipelineRegistry;
pub use self::sink::Sink;
pub use self::source::Source;
pub use self::stage::{StageDefinition, StageInstance, StageKind};
pub use self::tag::Tag;
pub use self::transform::Transform;
