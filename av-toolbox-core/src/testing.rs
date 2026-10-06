use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    FrameFormat,
    HostSide,
    LayerId,
    ResolvedFormats,
    Runtime,
    Side,
    StreamFormat,
};

use crate::{
    DemandDefinition,
    InputParam,
    InputSignature,
    Inputs,
    LayerDemand,
    Placement,
    Provider,
    ProviderDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};

/// A demand kind for test fixtures that need a chunk request but no particular breaks.
pub struct FixtureCuts;

impl DemandDefinition for FixtureCuts {
    type Request = ();

    fn kind() -> &'static str {
        "fixture-cuts"
    }
}

/// A demand kind for host-side test fixtures, served on the host so they need no crossing.
pub struct HostFixtureCuts;

impl DemandDefinition for HostFixtureCuts {
    type Request = ();

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

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature =
                    &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
                SIGNATURE
            }
        }

        impl ProviderDefinition for $definition {
            type Side = $side;
            type Demand = $demand;

            fn placement() -> Placement {
                Placement::AfterSource
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
/// Every layer it serves takes each frame's `meta.chunk`, so a fixture that stamps chunks at its
/// source keeps the same chunks once the graph routes on layers.
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
        for layer in &self.layers {
            meta.chunks.set(*layer, meta.chunk);
        }

        let stamped = S::from_any(any).map_err(|_| other_side())?;

        Ok(vec![stamped])
    }
}

fn other_side() -> StageError {
    StageError::msg("an item changed sides while its layers were stamped")
}
