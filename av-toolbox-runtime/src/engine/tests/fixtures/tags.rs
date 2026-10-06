use std::sync::Mutex;

use av_toolbox_core::testing::HostFixtureCuts;
use av_toolbox_core::{
    AnyDemand,
    BoundInputs,
    DemandDefinition,
    InputDemand,
    InputParam,
    InputSignature,
    LayerDemand,
    MetaInputs,
    MetaItem,
    MetaProvider,
    MetaSide,
    MetaSink,
    MetaSinkDefinition,
    PathEdge,
    PathView,
    PlaceError,
    ProviderDefinition,
    ProvisionError,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    FrameFormat,
    LayerId,
    Offered,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Every item a `TagRecording` sink consumed, as `(timestamp, tag)` pairs, in consumption order.
pub(crate) static TAGGED: Mutex<Vec<(u64, Option<Tag>)>> = Mutex::new(Vec::new());

/// What a `Tagging` provider writes onto each frame.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Tag(pub(crate) bool);

/// Asks `Tagging` for tags.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, JsonSchema)]
pub(crate) struct TagsRequest {
    /// Whether `Tagging` demands [HostFixtureCuts] of its own input.
    #[serde(default)]
    pub(crate) host_cuts: bool,
}

/// A demand kind served by `Tagging`, which runs on metadata alone.
pub(crate) struct FixtureTags;

impl DemandDefinition for FixtureTags {
    type Request = TagsRequest;
    type Output = Tag;

    fn kind() -> &'static str {
        "tags"
    }
}

/// Writes `Tag(true)` onto every item, into each layer it serves.
pub(crate) struct TaggingInstance {
    layers: Vec<LayerId>,
}

impl TaggingInstance {
    pub(crate) fn new(args: &[LayerDemand]) -> Self {
        let layers = args.iter().map(|served| served.layer).collect();

        Self { layers }
    }
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

/// Serves [FixtureTags] directly after the source, on metadata alone.
///
/// It demands [HostFixtureCuts] of its input when any request it serves asks it to.
pub(crate) struct Tagging;

impl StageDefinition for Tagging {
    type Args = Vec<LayerDemand>;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "tagging"
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
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
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

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: &BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaProvider>, StageError> {
        Ok(Box::new(TaggingInstance::new(&args)))
    }
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct TagRecordingArgs {
    /// Whether the tags it demands are served by reading host cuts.
    #[serde(default)]
    host_cuts: bool,
}

/// Records the timestamp and tag of every item it consumes, reading the tag from `layer`.
pub(crate) struct TagRecordingInstance {
    layer: LayerId,
}

impl TagRecordingInstance {
    pub(crate) fn new(layer: LayerId) -> Self {
        Self { layer }
    }
}

impl StageInstance for TagRecordingInstance {}

impl MetaSink for TagRecordingInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;
        let meta = item.meta();
        let timestamp = meta
            .timestamp
            .expect("every test frame carries a timestamp")
            .as_secs();
        let tag = meta.provided.get::<Tag>(self.layer).copied();

        TAGGED.lock().expect("not poisoned").push((timestamp, tag));

        Ok(())
    }
}

/// A metadata-only sink demanding [FixtureTags] and recording what each item carried.
pub(crate) struct TagRecording;

impl StageDefinition for TagRecording {
    type Args = TagRecordingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "tag-recording"
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        let request = TagsRequest {
            host_cuts: args.host_cuts,
        };

        vec![InputDemand {
            input: "input",
            demand: AnyDemand::new::<FixtureTags>(request),
        }]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl MetaSinkDefinition for TagRecording {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaSink>, StageError> {
        let tags = inputs
            .provision::<FixtureTags>("input")
            .map_err(unprovided_tags)?;

        Ok(Box::new(TagRecordingInstance::new(tags.layer())))
    }
}

fn unprovided_tags(error: ProvisionError) -> StageError {
    StageError::with_source("the tag recording fixture was given no tags", error)
}
