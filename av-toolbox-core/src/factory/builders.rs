use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    Capabilities,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use serde_json::Value;

use super::definition::{Definition, Prepared, keep};
use super::prepared::{PreparedFilter, PreparedMask, PreparedSink, PreparedSource, PreparedTransform};
use super::{StageFactory, StageInfo};
use crate::chroma::add_chroma_plane;
use crate::{
    BoundInputs,
    BuildStageError,
    BuiltFilter,
    BuiltMask,
    BuiltSink,
    BuiltSource,
    BuiltTransform,
    FilterDefinition,
    InputArgument,
    InputDemand,
    MaskDefinition,
    OpenSlot,
    Parallelism,
    Residency,
    SinkDefinition,
    SourceDefinition,
    StageError,
    StageSide,
    TransformDefinition,
};

/// Writes a stage kind's build task and factory, handing each built stage to `$finish`.
macro_rules! impl_stage_factory {
    (
        $kind_trait:ident,
        $definition_trait:ident,
        $prepared_trait:ident,
        $task:ident,
        $built:ty,
        $wrap:ident,
        $finish:path
    ) => {
        /// Carries a stage's pieces into whichever runtime the backend opened.
        struct $task<'a, D: $definition_trait> {
            args: D::Args,
            inputs: BoundInputs,
            formats: &'a ResolvedFormats,
        }

        impl<'a, D: $definition_trait> BuildTask for $task<'a, D> {
            type Output = Result<$built, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                let stage = D::build::<R>(context, self.args, self.inputs, self.formats)?;
                let built = <D::Side as StageSide>::$wrap(stage);

                $finish(context, self.formats, built)
            }
        }

        impl<D: $definition_trait> $prepared_trait for Prepared<D> {
            fn parallelism(&self) -> Parallelism {
                D::parallelism(&self.args)
            }

            fn chunk_authority(&self) -> Option<usize> {
                self.inputs.chunk_authority()
            }

            fn demands(&self) -> Vec<InputDemand> {
                self.stage_demands()
            }

            fn suggested_workers(&self) -> Option<usize> {
                D::workers(&self.args)
            }

            fn held_frames(&self, formats: &ResolvedFormats) -> usize {
                D::held_frames(&self.args, formats)
            }

            fn accepts_input(
                &self,
                capabilities: &Capabilities,
                index: usize,
                offered: StreamFormat,
                inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                D::accepts_input(capabilities, &self.args, index, offered, inputs)
            }

            fn produces(
                &self,
                capabilities: &Capabilities,
                chosen: &[Option<StreamFormat>],
            ) -> Result<OutputFormats, StageError> {
                D::produces(capabilities, &self.args, chosen)
            }

            fn residency(&self) -> Residency {
                <D::Side as StageSide>::RESIDENCY
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<$built, BuildStageError> {
                let identifier = self.identifier;
                let task = $task::<D> {
                    args: self.args,
                    inputs: self.inputs,
                    formats,
                };

                backend
                    .build(task)
                    .map_err(|source| BuildStageError::Build { identifier, source })
            }
        }

        impl<D: $definition_trait> StageFactory<dyn $prepared_trait> for Definition<D> {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn prepare(
                &self,
                args: &Value,
                inputs: &[InputArgument],
                slot: &OpenSlot,
            ) -> Result<Box<dyn $prepared_trait>, BuildStageError> {
                let prepared = self.prepared(args, inputs, slot)?;

                Ok(Box::new(prepared))
            }
        }
    };
}

/// Writes a source's build task and factory, since a source has no `accepts_input` to negotiate.
macro_rules! impl_source_factory {
    ($task:ident) => {
        struct $task<'a, D: SourceDefinition> {
            opened: D::Opened,
            inputs: BoundInputs,
            formats: &'a ResolvedFormats,
        }

        impl<'a, D: SourceDefinition> BuildTask for $task<'a, D> {
            type Output = Result<BuiltSource, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.opened, self.inputs, self.formats)
                    .map(<D::Side as StageSide>::source)
            }
        }

        impl<D: SourceDefinition> PreparedSource for Prepared<D> {
            fn parallelism(&self) -> Parallelism {
                D::parallelism(&self.args)
            }

            fn chunk_authority(&self) -> Option<usize> {
                self.inputs.chunk_authority()
            }

            fn demands(&self) -> Vec<InputDemand> {
                self.stage_demands()
            }

            fn suggested_workers(&self) -> Option<usize> {
                D::workers(&self.args)
            }

            fn held_frames(&self, formats: &ResolvedFormats) -> usize {
                D::held_frames(&self.args, formats)
            }

            fn produces(&self, capabilities: &Capabilities) -> Result<OutputFormats, StageError> {
                let open = || D::open(&self.args);
                let produced = self
                    .slot
                    .read(open, |opened: &D::Opened| D::produces(capabilities, opened))?;

                produced
            }

            fn residency(&self) -> Residency {
                <D::Side as StageSide>::RESIDENCY
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<BuiltSource, BuildStageError> {
                let identifier = self.identifier;
                let open = || D::open(&self.args);
                let opened = self
                    .slot
                    .take(open)
                    .map_err(|source| BuildStageError::Build {
                        identifier: identifier.clone(),
                        source,
                    })?;

                let task = $task::<D> {
                    opened,
                    inputs: self.inputs,
                    formats,
                };

                backend
                    .build(task)
                    .map_err(|source| BuildStageError::Build { identifier, source })
            }
        }

        impl<D: SourceDefinition> StageFactory<dyn PreparedSource> for Definition<D> {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn prepare(
                &self,
                args: &Value,
                inputs: &[InputArgument],
                slot: &OpenSlot,
            ) -> Result<Box<dyn PreparedSource>, BuildStageError> {
                let prepared = self.prepared(args, inputs, slot)?;

                Ok(Box::new(prepared))
            }

            fn reads_stream(&self, args: &Value) -> Result<bool, BuildStageError> {
                let args = self.deserialize_args(args)?;

                Ok(D::reads_stream(&args))
            }
        }
    };
}

/// Writes a sink's build task and factory, since a sink has no `produces` to negotiate.
macro_rules! impl_sink_factory {
    ($task:ident) => {
        struct $task<'a, D: SinkDefinition> {
            args: D::Args,
            inputs: BoundInputs,
            formats: &'a ResolvedFormats,
        }

        impl<'a, D: SinkDefinition> BuildTask for $task<'a, D> {
            type Output = Result<BuiltSink, StageError>;

            fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
                D::build::<R>(context, self.args, self.inputs, self.formats).map(<D::Side as StageSide>::sink)
            }
        }

        impl<D: SinkDefinition> PreparedSink for Prepared<D> {
            fn parallelism(&self) -> Parallelism {
                D::parallelism(&self.args)
            }

            fn chunk_authority(&self) -> Option<usize> {
                self.inputs.chunk_authority()
            }

            fn demands(&self) -> Vec<InputDemand> {
                self.stage_demands()
            }

            fn suggested_workers(&self) -> Option<usize> {
                D::workers(&self.args)
            }

            fn held_frames(&self, formats: &ResolvedFormats) -> usize {
                D::held_frames(&self.args, formats)
            }

            fn accepts_input(
                &self,
                capabilities: &Capabilities,
                index: usize,
                offered: StreamFormat,
                inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                D::accepts_input(capabilities, &self.args, index, offered, inputs)
            }

            fn residency(&self) -> Residency {
                <D::Side as StageSide>::RESIDENCY
            }

            fn build(
                self: Box<Self>,
                backend: &Backend,
                formats: &ResolvedFormats,
            ) -> Result<BuiltSink, BuildStageError> {
                let identifier = self.identifier;
                let task = $task::<D> {
                    args: self.args,
                    inputs: self.inputs,
                    formats,
                };

                backend
                    .build(task)
                    .map_err(|source| BuildStageError::Build { identifier, source })
            }
        }

        impl<D: SinkDefinition> StageFactory<dyn PreparedSink> for Definition<D> {
            fn info(&self) -> StageInfo<'_> {
                self.stage_info()
            }

            fn prepare(
                &self,
                args: &Value,
                inputs: &[InputArgument],
                slot: &OpenSlot,
            ) -> Result<Box<dyn PreparedSink>, BuildStageError> {
                let prepared = self.prepared(args, inputs, slot)?;

                Ok(Box::new(prepared))
            }
        }
    };
}

impl_source_factory!(BuildSource);
impl_stage_factory!(
    Filter,
    FilterDefinition,
    PreparedFilter,
    BuildFilter,
    BuiltFilter,
    filter,
    keep
);
impl_stage_factory!(
    Mask,
    MaskDefinition,
    PreparedMask,
    BuildMask,
    BuiltMask,
    mask,
    add_chroma_plane
);
impl_stage_factory!(
    Transform,
    TransformDefinition,
    PreparedTransform,
    BuildTransform,
    BuiltTransform,
    transform,
    keep
);
impl_sink_factory!(BuildSink);
