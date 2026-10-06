use av_toolbox_gpu::{
    AnyItem,
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
    BoundInputs,
    DemandDefinition,
    InputDemand,
    Inputs,
    MetaItem,
    PathEdge,
    PathView,
    PlaceError,
    ProviderSide,
    StageDefinition,
    StageError,
    StageInstance,
};

/// Reads a provider's args from JSON and reports what it demands of its own input.
pub type DemandsFn = fn(&serde_json::Value) -> Result<Vec<InputDemand>, StageError>;

/// The args every provider is built from, one entry per layer it serves.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
pub struct LayerDemand {
    pub layer: LayerId,
    pub demand: AnyDemand,
}

/// Supplies what demands of one kind ask for. The graph places it, so the TOML never names it.
///
/// A provider takes one input, runs over the whole stream, and passes every frame through.
/// What it demands of that input is declared through
/// [StageDefinition::demands](crate::StageDefinition::demands).
pub trait ProviderDefinition: StageDefinition<Args = Vec<LayerDemand>> {
    /// The side this provider runs on, or [MetaSide](crate::MetaSide) to run on metadata alone.
    type Side: ProviderSide;

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

    /// Builds the provider.
    ///
    /// `inputs` hands out what was provided for its own demands.
    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        inputs: &BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<<Self::Side as ProviderSide>::Instance>, StageError>;
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

/// A provider that passes items through, touching only their metadata.
///
/// Having no way to reach a plane is what lets it run on whichever side its input already sits on.
pub trait MetaProvider: StageInstance {
    /// What this emits for one item, which may be nothing while it holds items back.
    fn process(&mut self, context: &mut FrameContext, item: MetaItem) -> Result<Vec<MetaItem>, StageError>;

    /// What this emits once the stream has ended, which may be nothing.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<MetaItem>, StageError> {
        Ok(Vec::new())
    }
}

/// Hands `item` to `provider`, giving back what it emits on the side `item` came from.
pub fn run_meta_provider(
    provider: &mut dyn MetaProvider,
    context: &mut FrameContext,
    item: AnyItem,
) -> Result<Vec<AnyItem>, StageError> {
    let item = MetaItem::new(item);
    let emitted = provider.process(context, item)?;

    Ok(emitted.into_iter().map(MetaItem::into_any).collect())
}

/// Ends `provider`'s stream, giving back what it flushes.
pub fn end_meta_provider(
    provider: &mut dyn MetaProvider,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, StageError> {
    let flushed = provider.end_of_stream(context)?;

    Ok(flushed.into_iter().map(MetaItem::into_any).collect())
}
