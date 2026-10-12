use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    FrameFormat,
    FrameMeta,
    HostSide,
    LayerId,
    MetaValue,
    ResolvedFormats,
    Runtime,
    Side,
    StreamFormat,
};

use crate::{
    BoundInputs,
    ChunkKey,
    DemandDefinition,
    InputParam,
    InputSignature,
    Inputs,
    LayerDemand,
    PathEdge,
    PathView,
    PlaceError,
    Provider,
    ProviderDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};

/// The attachment key a fixture's chunk is kept under, for [Stamping] and [HostStamping] to copy.
pub const FIXTURE_CHUNK: &str = "fixture_chunk";

/// The path a fixture's chunk is filed under.
const FIXTURE_PATH: &str = "fixture";

/// `meta` carrying `chunk` for [Stamping] or [HostStamping] to copy into every layer they serve.
pub fn fixture_chunk(mut meta: FrameMeta, chunk: u64) -> FrameMeta {
    let value = i64::try_from(chunk).expect("a fixture chunk fits an attachment");
    meta.attach(FIXTURE_CHUNK, MetaValue::Int(value));
    meta.attachments.stamp(FIXTURE_PATH);

    meta
}

/// The chunk a fixture gave `meta`, or 0 when it gave none.
pub fn fixture_chunk_of(meta: &FrameMeta) -> u64 {
    let values = meta
        .attachments
        .stages()
        .filter(|(path, _)| *path == FIXTURE_PATH)
        .flat_map(|(_, values)| values.iter());

    for (key, value) in values {
        if let (FIXTURE_CHUNK, MetaValue::Int(chunk)) = (key.as_str(), value) {
            return u64::try_from(*chunk).expect("a fixture chunk is never negative");
        }
    }

    0
}

/// The value fixture providers write for their demands.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FixtureChunk(pub u64);

impl ChunkKey for FixtureChunk {
    fn chunk(&self) -> u64 {
        self.0
    }
}

/// A demand kind for test fixtures that need a chunk request but no particular breaks.
pub struct FixtureCuts;

impl DemandDefinition for FixtureCuts {
    type Request = ();
    type Output = FixtureChunk;

    fn kind() -> &'static str {
        "fixture-cuts"
    }
}

/// A demand kind for host-side test fixtures, served on the host so they need no crossing.
pub struct HostFixtureCuts;

impl DemandDefinition for HostFixtureCuts {
    type Request = ();
    type Output = FixtureChunk;

    fn kind() -> &'static str {
        "host-fixture-cuts"
    }
}

/// Writes a provider serving `$demand` directly after the source, on `$side`.
macro_rules! stamping_provider {
    ($definition:ident, $side:ty, $demand:ty, $identifier:literal) => {
        impl StageDefinition for $definition {
            type Args = Vec<LayerDemand>;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn description() -> &'static str {
                "A stage used by tests."
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature =
                    &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
                SIGNATURE
            }
        }

        impl ProviderDefinition for $definition {
            type Side = $side;
            type Demand = $demand;

            fn place(_request: &(), path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
                Ok(path.after_source())
            }

            fn accepts_input(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                offered: StreamFormat,
            ) -> Result<FormatSet, StageError> {
                Ok(FormatSet::of([offered.sample()]))
            }

            fn check(_args: &Self::Args, _input: &FrameFormat) -> Result<(), StageError> {
                Ok(())
            }

            fn build<R: Runtime>(
                _context: &BuildContext<R>,
                args: Self::Args,
                _inputs: &BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Provider<$side>>, StageError> {
                let instance = StampingInstance::new(&args);

                Ok(Box::new(instance))
            }
        }
    };
}

/// Serves [FixtureCuts] directly after the source, on the device.
///
/// Every layer it serves takes the chunk [fixture_chunk] gave each frame, so a fixture that stamps
/// chunks at its source keeps the same chunks once the graph routes on layers.
pub struct Stamping;

stamping_provider!(Stamping, DeviceSide, FixtureCuts, "stamping");

/// Serves [HostFixtureCuts] the way [Stamping] serves [FixtureCuts], on the host.
pub struct HostStamping;

stamping_provider!(HostStamping, HostSide, HostFixtureCuts, "host-stamping");

/// The instance [Stamping] and [HostStamping] build.
pub struct StampingInstance {
    layers: Vec<LayerId>,
}

impl StampingInstance {
    pub fn new(args: &[LayerDemand]) -> Self {
        let layers = args.iter().map(|served| served.layer).collect();

        Self { layers }
    }
}

impl StageInstance for StampingInstance {}

impl<S: Side> Provider<S> for StampingInstance {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<S>,
    ) -> Result<Vec<S::Item>, StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;

        let mut any = S::into_any(item);
        let meta = any.meta_mut();
        let chunk = fixture_chunk_of(meta);
        for layer in &self.layers {
            meta.provided.set(*layer, FixtureChunk(chunk));
        }

        let stamped = S::from_any(any).map_err(|_| other_side())?;

        Ok(vec![stamped])
    }
}

fn other_side() -> StageError {
    StageError::msg("an item changed sides while its layers were stamped")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fixture_chunk_reads_back() {
        let meta = fixture_chunk(FrameMeta::at(3), 7);

        assert_eq!(fixture_chunk_of(&meta), 7);
    }

    #[test]
    fn metadata_without_a_fixture_chunk_is_in_chunk_zero() {
        let meta = FrameMeta::at(3);

        assert_eq!(fixture_chunk_of(&meta), 0);
    }
}
