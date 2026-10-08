use serde_json::Value;

use super::definition::{Definition, Prepared};
use super::prepared::PreparedArtefactSink;
use super::{StageFactory, StageInfo};
use crate::{ArtefactSink, ArtefactSinkDefinition, BuildStageError, InputArgument, KeyFilter, OpenSlot};

impl<D: ArtefactSinkDefinition> PreparedArtefactSink for Prepared<D> {
    fn selection(&self) -> KeyFilter {
        D::selection(&self.args)
    }

    fn build(self: Box<Self>) -> Result<Box<dyn ArtefactSink>, BuildStageError> {
        let identifier = self.identifier;
        let built = D::build(self.args);

        built.map_err(|source| BuildStageError::Build { identifier, source })
    }
}

impl<D: ArtefactSinkDefinition> StageFactory<dyn PreparedArtefactSink> for Definition<D> {
    fn info(&self) -> StageInfo<'_> {
        self.stage_info()
    }

    fn prepare(
        &self,
        args: &Value,
        inputs: &[InputArgument],
        slot: &OpenSlot,
    ) -> Result<Box<dyn PreparedArtefactSink>, BuildStageError> {
        let prepared = self.prepared(args, inputs, slot)?;

        Ok(Box::new(prepared))
    }
}
