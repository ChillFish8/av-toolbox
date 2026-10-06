use av_toolbox_core::testing::{FixtureChunk, HostFixtureCuts};
use av_toolbox_core::{
    AnyDemand,
    BoundInputs,
    DemandDefinition,
    Filter,
    FilterDefinition,
    InputDemand,
    InputParam,
    InputSignature,
    Inputs,
    LayerDemand,
    Parallelism,
    Placement,
    Provider,
    ProviderDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamItem,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::shared::{first_bound, frames_template};

/// Asks for a break every `every` frames, optionally placed after a named stage.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
pub(crate) struct BreaksRequest {
    pub(crate) every: u32,
    pub(crate) after: Option<String>,
}

/// A demand kind with requests that can differ, served by [Breaking].
pub(crate) struct FixtureBreaks;

impl DemandDefinition for FixtureBreaks {
    type Request = BreaksRequest;
    type Output = FixtureChunk;

    fn kind() -> &'static str {
        "fixture-breaks"
    }

    fn describe(request: &BreaksRequest) -> String {
        format!("every {}", request.every)
    }
}

/// A demand kind nothing provides.
pub(crate) struct Unprovided;

impl DemandDefinition for Unprovided {
    type Request = ();
    type Output = FixtureChunk;

    fn kind() -> &'static str {
        "unprovided"
    }
}

struct PassItems;

impl StageInstance for PassItems {}

impl Provider<DeviceSide> for PassItems {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs,
    ) -> Result<Vec<StreamItem>, StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;

        Ok(vec![item])
    }
}

/// Serves [FixtureBreaks] before the latest node every demanding path passes through.
///
/// It refuses a request for a break every 0 frames once it sees its input.
pub(super) struct Breaking;

impl StageDefinition for Breaking {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "breaking"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl ProviderDefinition for Breaking {
    type Side = DeviceSide;
    type Demand = FixtureBreaks;

    fn placement() -> Placement {
        Placement::Shared
    }

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        offered: StreamFormat,
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn check(args: &Self::Args, _input: &FrameFormat) -> Result<(), StageError> {
        let requests = args
            .iter()
            .filter_map(|served| served.demand.request::<FixtureBreaks>());

        for request in requests {
            if request.every == 0 {
                return Err(StageError::msg("a stage asks for a break every 0 frames"));
            }
        }

        Ok(())
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Provider<DeviceSide>>, StageError> {
        Ok(Box::new(PassItems))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct DemandingArgs {
    #[serde(default)]
    every: u32,
    #[serde(default)]
    after: Option<String>,
    /// Whether the demand is the stage's chunk request rather than a plain demand.
    #[serde(default)]
    chunks: bool,
    /// Whether the stage demands [Unprovided] instead.
    #[serde(default)]
    unprovided: bool,
    /// Whether the stage demands [HostFixtureCuts] instead, which a host provider serves.
    #[serde(default)]
    host_cuts: bool,
}

impl DemandingArgs {
    fn request(&self) -> BreaksRequest {
        BreaksRequest {
            every: self.every,
            after: self.after.clone(),
        }
    }

    fn demand(&self) -> AnyDemand {
        if self.unprovided {
            return AnyDemand::new::<Unprovided>(());
        }

        if self.host_cuts {
            return AnyDemand::new::<HostFixtureCuts>(());
        }

        AnyDemand::new::<FixtureBreaks>(self.request())
    }

    fn chunked(&self) -> Parallelism {
        if self.unprovided {
            return Parallelism::chunk::<Unprovided>(());
        }

        if self.host_cuts {
            return Parallelism::chunk::<HostFixtureCuts>(());
        }

        Parallelism::chunk::<FixtureBreaks>(self.request())
    }
}

struct DemandingStage;

impl StageInstance for DemandingStage {}

impl Filter for DemandingStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// A filter demanding breaks on its one input, as a plain demand or as its chunk request.
pub(super) struct Demanding;

impl StageDefinition for Demanding {
    type Args = DemandingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "demanding"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.chunks {
            true => args.chunked(),
            false => Parallelism::Frame,
        }
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        if args.chunks {
            return Vec::new();
        }

        let demand = InputDemand {
            input: "input",
            demand: args.demand(),
        };

        vec![demand]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Demanding {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: frames_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(DemandingStage))
    }
}
