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

use super::definition::{Definition, Prepared};
use super::prepared::{PreparedMetaSink, PreparedMetaTransform};
use super::{StageFactory, StageInfo};
use crate::{
    BoundInputs,
    BuildStageError,
    InputArgument,
    InputDemand,
    MetaSink,
    MetaSinkDefinition,
    MetaTransform,
    MetaTransformDefinition,
    OpenSlot,
    Parallelism,
    StageError,
    bind_inputs,
};

struct BuildMetaTransform<'a, D: MetaTransformDefinition> {
    args: D::Args,
    inputs: BoundInputs,
    formats: &'a ResolvedFormats,
}

impl<'a, D: MetaTransformDefinition> BuildTask for BuildMetaTransform<'a, D> {
    type Output = Result<Box<dyn MetaTransform>, StageError>;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        D::build::<R>(context, self.args, self.inputs, self.formats)
    }
}

impl<D: MetaTransformDefinition> PreparedMetaTransform for Prepared<D> {
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

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaTransform>, BuildStageError> {
        let identifier = self.identifier;
        let task = BuildMetaTransform::<D> {
            args: self.args,
            inputs: self.inputs,
            formats,
        };

        backend
            .build(task)
            .map_err(|source| BuildStageError::Build { identifier, source })
    }
}

impl<D: MetaTransformDefinition> StageFactory<dyn PreparedMetaTransform> for Definition<D> {
    fn info(&self) -> StageInfo<'_> {
        self.stage_info()
    }

    fn prepare(
        &self,
        args: &Value,
        inputs: &[InputArgument],
        slot: &OpenSlot,
    ) -> Result<Box<dyn PreparedMetaTransform>, BuildStageError> {
        let args = self.deserialize_args(args)?;
        self.check_demand_inputs(&args)?;
        let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

        Ok(Box::new(Prepared::<D> {
            identifier: self.full_identifier().to_string(),
            args,
            inputs: bound,
            slot: slot.clone(),
        }))
    }
}

struct BuildMetaSink<'a, D: MetaSinkDefinition> {
    args: D::Args,
    inputs: BoundInputs,
    formats: &'a ResolvedFormats,
}

impl<'a, D: MetaSinkDefinition> BuildTask for BuildMetaSink<'a, D> {
    type Output = Result<Box<dyn MetaSink>, StageError>;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        D::build::<R>(context, self.args, self.inputs, self.formats)
    }
}

impl<D: MetaSinkDefinition> PreparedMetaSink for Prepared<D> {
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

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaSink>, BuildStageError> {
        let identifier = self.identifier;
        let task = BuildMetaSink::<D> {
            args: self.args,
            inputs: self.inputs,
            formats,
        };

        backend
            .build(task)
            .map_err(|source| BuildStageError::Build { identifier, source })
    }
}

impl<D: MetaSinkDefinition> StageFactory<dyn PreparedMetaSink> for Definition<D> {
    fn info(&self) -> StageInfo<'_> {
        self.stage_info()
    }

    fn prepare(
        &self,
        args: &Value,
        inputs: &[InputArgument],
        slot: &OpenSlot,
    ) -> Result<Box<dyn PreparedMetaSink>, BuildStageError> {
        let args = self.deserialize_args(args)?;
        self.check_demand_inputs(&args)?;
        let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;

        Ok(Box::new(Prepared::<D> {
            identifier: self.full_identifier().to_string(),
            args,
            inputs: bound,
            slot: slot.clone(),
        }))
    }
}
