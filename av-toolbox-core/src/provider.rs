use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    FrameFormat,
    LayerId,
    ResolvedFormats,
    Runtime,
    Side,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AnyDemand,
    DemandDefinition,
    Inputs,
    PathEdge,
    PathView,
    PlaceError,
    StageDefinition,
    StageError,
    StageInstance,
    StageSide,
};

/// The args every provider is built from, one entry per layer it serves.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
pub struct LayerDemand {
    pub layer: LayerId,
    pub demand: AnyDemand,
}

/// Supplies what demands of one kind ask for. The graph places it, so the TOML never names it.
///
/// A provider takes one input, runs over the whole stream, and passes every frame through.
pub trait ProviderDefinition: StageDefinition<Args = Vec<LayerDemand>> {
    /// The side this provider runs on.
    type Side: StageSide;

    /// The demand kind this provider serves.
    ///
    /// A provider reads its requests with `demand.request::<Self::Demand>()`.
    type Demand: DemandDefinition;

    /// Picks the edge on a demand's path to serve `request` from.
    ///
    /// Demands of one kind placed after the same stage share one provider.
    fn place(
        request: &<Self::Demand as DemandDefinition>::Request,
        path: &PathView<'_>,
    ) -> Result<PathEdge, PlaceError>;

    /// Which sample formats this provider will take, most preferred first.
    fn accepts_input(
        capabilities: &Capabilities,
        args: &Self::Args,
        offered: StreamFormat,
    ) -> Result<FormatSet, StageError>;

    /// Refuses demands the provider cannot bound, such as a seconds limit with no frame rate.
    fn check(args: &Self::Args, input: &FrameFormat) -> Result<(), StageError>;

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Provider<Self::Side>>, StageError>;
}

/// A built provider. It passes every item through and writes what it supplies into their metadata.
pub trait Provider<S: Side = DeviceSide>: StageInstance {
    /// What this emits for one input, which may be nothing while it holds frames back.
    fn process(&mut self, context: &mut FrameContext, inputs: Inputs<S>) -> Result<Vec<S::Item>, StageError>;

    /// What this emits once the stream has ended, which may be nothing.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<S::Item>, StageError> {
        Ok(Vec::new())
    }
}
