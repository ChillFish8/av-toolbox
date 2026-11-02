use av_toolbox_core::testing::{FixtureChunk, FixtureCuts, HostFixtureCuts, StampingInstance};
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
    MetaInputs,
    MetaItem,
    MetaProvider,
    MetaSide,
    MetaSink,
    MetaSinkDefinition,
    Parallelism,
    PathEdge,
    PathView,
    PlaceError,
    Provider,
    ProviderDefinition,
    Provision,
    ProvisionError,
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
    LayerId,
    Offered,
    OutputFormats,
    ResolvedFormats,
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
    /// Whether [Breaking] refuses to place this request.
    #[serde(default)]
    pub(crate) refuse: bool,
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

/// Asks for marks every `every` frames, which [Marking] copies from breaks.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
pub(crate) struct MarksRequest {
    pub(crate) every: u32,
    /// Whether [Marking] demands marks of its own input rather than breaks.
    #[serde(default)]
    pub(crate) cycle: bool,
}

/// A demand kind served by [Marking], which itself demands [FixtureBreaks].
pub(crate) struct FixtureMarks;

impl DemandDefinition for FixtureMarks {
    type Request = MarksRequest;
    type Output = FixtureChunk;

    fn kind() -> &'static str {
        "fixture-marks"
    }

    fn describe(request: &MarksRequest) -> String {
        format!("marks every {}", request.every)
    }
}

/// Serves [FixtureBreaks] after the stage a request names, or after the source.
///
/// Every layer it serves takes the chunk the source stamped on each frame. It refuses to place a
/// request marked `refuse`, and refuses a request for a break every 0 frames once it sees its input.
pub(super) struct Breaking;

impl StageDefinition for Breaking {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "breaking"
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

impl ProviderDefinition for Breaking {
    type Side = DeviceSide;
    type Demand = FixtureBreaks;

    fn place(request: &BreaksRequest, path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
        if request.refuse {
            return Err(PlaceError::msg("the fixture refuses to place this request"));
        }

        match request.after.as_deref() {
            Some(name) => path.after_named(name),
            None => Ok(path.after_source()),
        }
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

    fn build(
        _context: &BuildContext,
        args: Self::Args,
        _inputs: &BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Provider<DeviceSide>>, StageError> {
        let instance = StampingInstance::new(&args);

        Ok(Box::new(instance))
    }
}

/// Serves [FixtureMarks] after the source, copying each frame's breaks into its own layers.
///
/// It demands one set of breaks on its input, every as many frames as its smallest request asks.
/// When any request is marked `cycle`, it demands marks of itself instead.
pub(super) struct Marking;

impl StageDefinition for Marking {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "marking"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        let requests: Vec<MarksRequest> = args
            .iter()
            .filter_map(|served| served.demand.request::<FixtureMarks>())
            .collect();
        let Some(demand) = marking_demand(&requests) else {
            return Vec::new();
        };

        vec![InputDemand {
            input: "input",
            demand,
        }]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl ProviderDefinition for Marking {
    type Side = DeviceSide;
    type Demand = FixtureMarks;

    fn place(_request: &MarksRequest, path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
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

    fn build(
        _context: &BuildContext,
        args: Self::Args,
        inputs: &BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Provider<DeviceSide>>, StageError> {
        let breaks = inputs
            .provision::<FixtureBreaks>("input")
            .map_err(unprovided_breaks)?;
        let layers = args.iter().map(|served| served.layer).collect();

        Ok(Box::new(MarkingInstance { breaks, layers }))
    }
}

/// What [Marking] demands to serve `requests`, or nothing when it serves none.
fn marking_demand(requests: &[MarksRequest]) -> Option<AnyDemand> {
    let cycle = requests.iter().find(|request| request.cycle);
    if let Some(request) = cycle {
        let demand = AnyDemand::new::<FixtureMarks>(request.clone());
        return Some(demand);
    }

    let every = requests.iter().map(|request| request.every).min()?;
    let breaks = BreaksRequest {
        every,
        after: None,
        refuse: false,
    };

    Some(AnyDemand::new::<FixtureBreaks>(breaks))
}

fn unprovided_breaks(error: ProvisionError) -> StageError {
    StageError::with_source("the marking fixture was given no breaks", error)
}

/// The instance [Marking] builds.
struct MarkingInstance {
    breaks: Provision<FixtureBreaks>,
    layers: Vec<LayerId>,
}

impl StageInstance for MarkingInstance {}

impl Provider<DeviceSide> for MarkingInstance {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs,
    ) -> Result<Vec<StreamItem>, StageError> {
        let mut item = inputs.take(0).ok_or_else(missing_item)?;

        let meta = item.meta_mut();
        let breaks = self.breaks.get(meta).copied();
        let chunk = breaks.unwrap_or(FixtureChunk(0));
        for layer in &self.layers {
            meta.provided.set(*layer, chunk);
        }

        Ok(vec![item])
    }
}

fn missing_item() -> StageError {
    StageError::msg("expected an item in slot 0")
}

fn missing_frame() -> StageError {
    StageError::msg("expected a frame in slot 0")
}

/// A demand kind served by [Relaying], which passes each request on to [Marking].
pub(crate) struct FixtureRelays;

impl DemandDefinition for FixtureRelays {
    type Request = u32;
    type Output = FixtureChunk;

    fn kind() -> &'static str {
        "fixture-relays"
    }
}

/// Serves [FixtureRelays] after the source by demanding marks every as many frames as each request.
pub(super) struct Relaying;

impl StageDefinition for Relaying {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "relaying"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        args.iter()
            .filter_map(|served| served.demand.request::<FixtureRelays>())
            .map(|every| {
                let request = MarksRequest { every, cycle: false };

                InputDemand {
                    input: "input",
                    demand: AnyDemand::new::<FixtureMarks>(request),
                }
            })
            .collect()
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl ProviderDefinition for Relaying {
    type Side = DeviceSide;
    type Demand = FixtureRelays;

    fn place(_request: &u32, path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
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

    fn build(
        _context: &BuildContext,
        args: Self::Args,
        _inputs: &BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Provider<DeviceSide>>, StageError> {
        let instance = StampingInstance::new(&args);

        Ok(Box::new(instance))
    }
}

/// What [Tagging] writes onto each frame.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Tag(pub(crate) bool);

/// Asks [Tagging] for tags.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
pub(crate) struct TagsRequest {
    /// Whether [Tagging] demands [HostFixtureCuts] of its own input.
    #[serde(default)]
    pub(crate) host_cuts: bool,
}

/// A demand kind served by [Tagging], which runs on metadata alone.
pub(crate) struct FixtureTags;

impl DemandDefinition for FixtureTags {
    type Request = TagsRequest;
    type Output = Tag;

    fn kind() -> &'static str {
        "tags"
    }
}

/// Serves [FixtureTags] after the source by writing `Tag(true)` onto every frame.
pub(super) struct Tagging;

impl StageDefinition for Tagging {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "tagging"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        let reads_host_cuts = args
            .iter()
            .filter_map(|served| served.demand.request::<FixtureTags>())
            .any(|request| request.host_cuts);
        if !reads_host_cuts {
            return Vec::new();
        }

        vec![InputDemand {
            input: "input",
            demand: AnyDemand::new::<HostFixtureCuts>(()),
        }]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl ProviderDefinition for Tagging {
    type Side = MetaSide;
    type Demand = FixtureTags;

    fn place(_request: &TagsRequest, path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
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

    fn build(
        _context: &BuildContext,
        args: Self::Args,
        _inputs: &BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaProvider>, StageError> {
        let layers = args.iter().map(|served| served.layer).collect();

        Ok(Box::new(TaggingInstance { layers }))
    }
}

/// The instance [Tagging] builds.
struct TaggingInstance {
    layers: Vec<LayerId>,
}

impl StageInstance for TaggingInstance {}

impl MetaProvider for TaggingInstance {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut item: MetaItem,
    ) -> Result<Vec<MetaItem>, StageError> {
        let meta = item.meta_mut();
        for layer in &self.layers {
            meta.provided.set(*layer, Tag(true));
        }

        Ok(vec![item])
    }
}

/// A metadata-only sink demanding [FixtureTags], which fails on any frame not tagged `Tag(true)`.
pub(super) struct TagReading;

impl StageDefinition for TagReading {
    type Args = TagsRequest;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "tag-reading"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        vec![InputDemand {
            input: "input",
            demand: AnyDemand::new::<FixtureTags>(args.clone()),
        }]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MetaSinkDefinition for TagReading {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaSink>, StageError> {
        let tags = inputs
            .provision::<FixtureTags>("input")
            .map_err(unprovided_tags)?;

        Ok(Box::new(TagReadingInstance { tags }))
    }
}

fn unprovided_tags(error: ProvisionError) -> StageError {
    StageError::with_source("the tag reading fixture was given no tags", error)
}

/// The instance [TagReading] builds.
struct TagReadingInstance {
    tags: Provision<FixtureTags>,
}

impl StageInstance for TagReadingInstance {}

impl MetaSink for TagReadingInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
        let item = inputs.take(0).ok_or_else(missing_item)?;

        let tag = self.tags.get(item.meta());
        if tag != Some(&Tag(true)) {
            return Err(StageError::msg("a frame arrived without Tag(true)"));
        }

        Ok(())
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
    /// Whether the request asks [Breaking] to refuse placing it.
    #[serde(default)]
    refuse: bool,
    /// Whether the stage demands [Unprovided] instead.
    #[serde(default)]
    unprovided: bool,
    /// Whether the stage demands [HostFixtureCuts] instead, which a host provider serves.
    #[serde(default)]
    host_cuts: bool,
    /// Whether the stage demands [FixtureMarks] instead.
    #[serde(default)]
    marks: bool,
    /// Whether the marks request asks [Marking] to demand marks of itself.
    #[serde(default)]
    cycle: bool,
    /// Whether the stage also demands [FixtureCuts], listed before its own demand.
    #[serde(default)]
    cuts: bool,
    /// Whether the stage demands [FixtureRelays] instead.
    #[serde(default)]
    relays: bool,
    /// Whether the stage demands [FixtureTags] instead, which reads host cuts when `host_cuts` is set.
    #[serde(default)]
    tags: bool,
}

impl DemandingArgs {
    fn request(&self) -> BreaksRequest {
        BreaksRequest {
            every: self.every,
            after: self.after.clone(),
            refuse: self.refuse,
        }
    }

    fn demand(&self) -> AnyDemand {
        if self.tags {
            let request = TagsRequest {
                host_cuts: self.host_cuts,
            };

            return AnyDemand::new::<FixtureTags>(request);
        }

        if self.unprovided {
            return AnyDemand::new::<Unprovided>(());
        }

        if self.host_cuts {
            return AnyDemand::new::<HostFixtureCuts>(());
        }

        if self.relays {
            return AnyDemand::new::<FixtureRelays>(self.every);
        }

        if self.marks {
            let request = MarksRequest {
                every: self.every,
                cycle: self.cycle,
            };

            return AnyDemand::new::<FixtureMarks>(request);
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
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;

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

    fn description() -> &'static str {
        "A stage used by tests."
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

        let mut demands = Vec::new();
        if args.cuts {
            demands.push(InputDemand {
                input: "input",
                demand: AnyDemand::new::<FixtureCuts>(()),
            });
        }

        demands.push(InputDemand {
            input: "input",
            demand: args.demand(),
        });

        demands
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

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(DemandingStage))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct DemandingPairArgs {
    #[serde(default)]
    every: u32,
    /// Whether `other` is demanded on as well as `clip`.
    #[serde(default)]
    both: bool,
}

/// A filter with a `clip` and an `other` input, demanding breaks on `clip` and optionally `other`.
pub(super) struct DemandingPair;

impl StageDefinition for DemandingPair {
    type Args = DemandingPairArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "demanding-pair"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        let request = BreaksRequest {
            every: args.every,
            after: None,
            refuse: false,
        };
        let demand = AnyDemand::new::<FixtureBreaks>(request);

        let mut demands = vec![InputDemand {
            input: "clip",
            demand: demand.clone(),
        }];
        if args.both {
            demands.push(InputDemand {
                input: "other",
                demand,
            });
        }

        demands
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("clip").accepting(StreamType::Frames),
            InputParam::required("other").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for DemandingPair {
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

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(DemandingStage))
    }
}
