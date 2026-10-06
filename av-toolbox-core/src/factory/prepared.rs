use av_toolbox_gpu::{
    Backend,
    Capabilities,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
};

use crate::{
    BuildStageError,
    BuiltFilter,
    BuiltMask,
    BuiltProvider,
    BuiltSink,
    BuiltSource,
    BuiltTransform,
    InputDemand,
    MetaSink,
    MetaTransform,
    Parallelism,
    Residency,
    StageError,
};

/// A source whose arguments are deserialized, ready to negotiate and build.
pub trait PreparedSource: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    /// What this stage asks of the frames reaching its inputs.
    fn demands(&self) -> Vec<InputDemand>;

    /// How many workers this stage would like when the pipeline does not say.
    fn suggested_workers(&self) -> Option<usize>;

    /// The most frames this stage may hold back at once, given its resolved formats.
    fn held_frames(&self, formats: &ResolvedFormats) -> usize;

    fn produces(&self, capabilities: &Capabilities) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltSource, BuildStageError>;
}

/// A filter whose arguments are deserialized and whose inputs are bound.
pub trait PreparedFilter: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    /// What this stage asks of the frames reaching its inputs.
    fn demands(&self) -> Vec<InputDemand>;

    /// How many workers this stage would like when the pipeline does not say.
    fn suggested_workers(&self) -> Option<usize>;

    /// The most frames this stage may hold back at once, given its resolved formats.
    fn held_frames(&self, formats: &ResolvedFormats) -> usize;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltFilter, BuildStageError>;
}

/// A mask whose arguments are deserialized and whose inputs are bound.
pub trait PreparedMask: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    /// What this stage asks of the frames reaching its inputs.
    fn demands(&self) -> Vec<InputDemand>;

    /// How many workers this stage would like when the pipeline does not say.
    fn suggested_workers(&self) -> Option<usize>;

    /// The most frames this stage may hold back at once, given its resolved formats.
    fn held_frames(&self, formats: &ResolvedFormats) -> usize;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltMask, BuildStageError>;
}

/// A transform whose arguments are deserialized and whose inputs are bound.
pub trait PreparedTransform: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    /// What this stage asks of the frames reaching its inputs.
    fn demands(&self) -> Vec<InputDemand>;

    /// How many workers this stage would like when the pipeline does not say.
    fn suggested_workers(&self) -> Option<usize>;

    /// The most frames this stage may hold back at once, given its resolved formats.
    fn held_frames(&self, formats: &ResolvedFormats) -> usize;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltTransform, BuildStageError>;
}

/// A provider whose demands are deserialized and whose input is bound.
pub trait PreparedProvider: Send + Sync {
    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    /// The most frames this stage may hold back at once, given its resolved formats.
    fn held_frames(&self, formats: &ResolvedFormats) -> usize;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        offered: StreamFormat,
    ) -> Result<FormatSet, StageError>;

    /// The format the provider was given, once it has checked it can serve its demands on it.
    fn produces(&self, chosen: &[Option<StreamFormat>]) -> Result<OutputFormats, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltProvider, BuildStageError>;
}

/// A metadata-only transform whose arguments are deserialized and whose inputs are bound.
pub trait PreparedMetaTransform: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    /// What this stage asks of the frames reaching its inputs.
    fn demands(&self) -> Vec<InputDemand>;

    /// How many workers this stage would like when the pipeline does not say.
    fn suggested_workers(&self) -> Option<usize>;

    /// The most frames this stage may hold back at once, given its resolved formats.
    fn held_frames(&self, formats: &ResolvedFormats) -> usize;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn produces(
        &self,
        capabilities: &Capabilities,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError>;

    /// Builds the instance against a real backend. A metadata-only transform has no side to build
    /// for.
    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaTransform>, BuildStageError>;
}

/// A sink whose arguments are deserialized and whose inputs are bound.
pub trait PreparedSink: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    /// What this stage asks of the frames reaching its inputs.
    fn demands(&self) -> Vec<InputDemand>;

    /// How many workers this stage would like when the pipeline does not say.
    fn suggested_workers(&self) -> Option<usize>;

    /// The most frames this stage may hold back at once, given its resolved formats.
    fn held_frames(&self, formats: &ResolvedFormats) -> usize;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    fn residency(&self) -> Residency;

    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<BuiltSink, BuildStageError>;
}

/// A metadata-only sink whose arguments are deserialized and whose inputs are bound.
pub trait PreparedMetaSink: Send + Sync {
    /// The smallest unit of a stream this stage can be given on its own.
    fn parallelism(&self) -> Parallelism;

    /// Which input's chunk ids line this stage's inputs up.
    fn chunk_authority(&self) -> Option<usize>;

    /// What this stage asks of the frames reaching its inputs.
    fn demands(&self) -> Vec<InputDemand>;

    /// How many workers this stage would like when the pipeline does not say.
    fn suggested_workers(&self) -> Option<usize>;

    /// The most frames this stage may hold back at once, given its resolved formats.
    fn held_frames(&self, formats: &ResolvedFormats) -> usize;

    fn accepts_input(
        &self,
        capabilities: &Capabilities,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError>;

    /// Builds the instance against a real backend. A metadata-only sink has no side to build for.
    fn build(
        self: Box<Self>,
        backend: &Backend,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaSink>, BuildStageError>;
}
