use std::marker::PhantomData;

use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    Capabilities,
    FormatSet,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use serde_json::Value;

use super::definition::{Definition, Prepared};
use super::prepared::PreparedProvider;
use super::{StageFactory, StageInfo};
use crate::{
    BuildStageError,
    BuiltProvider,
    InputArgument,
    LayerDemand,
    OpenSlot,
    ProviderDefinition,
    Residency,
    StageError,
    StageSide,
};

/// Carries a provider's demands into whichever runtime the backend opened.
struct BuildProvider<'a, D: ProviderDefinition> {
    args: Vec<LayerDemand>,
    formats: &'a ResolvedFormats,
    marker: PhantomData<fn() -> D>,
}

impl<'a, D: ProviderDefinition> BuildTask for BuildProvider<'a, D> {
    type Output = Result<BuiltProvider, StageError>;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        let stage = D::build::<R>(context, self.args, self.formats)?;

        Ok(<D::Side as StageSide>::provider(stage))
    }
}

impl<D: ProviderDefinition> PreparedProvider for Prepared<D> {
    fn chunk_authority(&self) -> Option<usize> {
        self.inputs.chunk_authority()
    }

    fn held_frames(&self, formats: &ResolvedFormats) -> usize {
        D::held_frames(&self.args, formats)
    }

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        offered: StreamFormat,
    ) -> Result<FormatSet, StageError> {
        D::accepts_input(capabilities, &self.args, offered)
    }

    fn produces(&self, chosen: &[Option<StreamFormat>]) -> Result<OutputFormats, StageError> {
        let chosen = chosen.first().copied().flatten();
        let format = chosen.ok_or_else(unchosen_input)?;
        let frames = format.as_frames().ok_or_else(masks_offered)?;

        D::check(&self.args, frames)?;

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn residency(&self) -> Residency {
        <D::Side as StageSide>::RESIDENCY
    }

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltProvider, BuildStageError> {
        let identifier = self.identifier;
        let task = BuildProvider::<D> {
            args: self.args,
            formats,
            marker: PhantomData,
        };

        backend
            .build(task)
            .map_err(|source| BuildStageError::Build { identifier, source })
    }
}

impl<D: ProviderDefinition> StageFactory<dyn PreparedProvider> for Definition<D> {
    fn info(&self) -> StageInfo<'_> {
        self.stage_info()
    }

    fn prepare(
        &self,
        args: &Value,
        inputs: &[InputArgument],
        slot: &OpenSlot,
    ) -> Result<Box<dyn PreparedProvider>, BuildStageError> {
        let prepared = self.prepared(args, inputs, slot)?;

        Ok(Box::new(prepared))
    }
}

fn unchosen_input() -> StageError {
    StageError::msg("a provider was asked what it produces before its input was chosen")
}

fn masks_offered() -> StageError {
    StageError::msg("a provider reads frames, not masks")
}
