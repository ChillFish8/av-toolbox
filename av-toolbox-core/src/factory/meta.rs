use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    Capabilities,
    FormatSet,
    Offered,
    ResolvedFormats,
    StreamFormat,
};
use serde_json::Value;

use super::definition::{Definition, Prepared};
use super::prepared::PreparedMetaSink;
use super::{StageFactory, StageInfo};
use crate::{
    BoundInputs,
    BuildStageError,
    InputArgument,
    InputDemand,
    InputProbe,
    MetaSink,
    MetaSinkDefinition,
    OpenSlot,
    Parallelism,
    Probed,
    StageError,
};

struct BuildMetaSink<'a, D: MetaSinkDefinition> {
    args: D::Args,
    inputs: BoundInputs,
    formats: &'a ResolvedFormats,
}

impl<'a, D: MetaSinkDefinition> BuildTask for BuildMetaSink<'a, D> {
    type Output = Result<Box<dyn MetaSink>, StageError>;

    fn run(self, context: &BuildContext) -> Self::Output {
        D::build(context, self.args, self.inputs, self.formats)
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

    fn probes(&self) -> Vec<InputProbe> {
        D::probes(&self.args)
    }

    fn settle(&mut self, probed: &Probed) -> Result<(), StageError> {
        D::settle(&mut self.args, probed)
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
        let prepared = self.prepared(args, inputs, slot)?;

        Ok(Box::new(prepared))
    }
}
