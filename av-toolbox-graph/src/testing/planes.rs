use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ChromaLocation,
    ColourInfo,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostMask,
    HostSide,
    Mask as GpuMask,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::shared::{NoArgs, first_bound, fixture_mask_format, frames_template, mask_template};

#[derive(Debug, Copy, Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum FixtureLayout {
    Yuv420,
    Yuv422,
    Yuv444,
    Rgb,
    Grey,
}

#[derive(Debug, Copy, Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum FixtureLocation {
    Left,
    Center,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct LayoutArgs {
    layout: FixtureLayout,
    location: FixtureLocation,
}

fn layout_of(layout: FixtureLayout) -> FrameLayout {
    match layout {
        FixtureLayout::Yuv420 => FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        FixtureLayout::Yuv422 => FrameLayout::Yuv {
            subsampling: Subsampling::Yuv422,
        },
        FixtureLayout::Yuv444 => FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        },
        FixtureLayout::Rgb => FrameLayout::Rgb,
        FixtureLayout::Grey => FrameLayout::Grey,
    }
}

fn location_of(location: FixtureLocation) -> ChromaLocation {
    match location {
        FixtureLocation::Left => ChromaLocation::Left,
        FixtureLocation::Center => ChromaLocation::Center,
    }
}

struct LayoutFileStage;

impl StageInstance for LayoutFileStage {}

impl Source for LayoutFileStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

/// A device source whose frame layout and chroma location come from its arguments.
pub(super) struct LayoutFile;

impl StageDefinition for LayoutFile {
    type Args = LayoutArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "layout-file"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for LayoutFile {
    type Side = DeviceSide;
    type Opened = FrameFormat;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        let chroma_location = location_of(args.location);
        let defaults = ColourInfo::default();
        let colour = ColourInfo {
            chroma_location,
            ..defaults
        };
        let layout = layout_of(args.layout);

        Ok(FrameFormat {
            width: 64,
            height: 64,
            layout,
            sample: SampleFormat::U8,
            colour,
            frame_rate: None,
        })
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let template = StreamTemplate::Frames(*opened);

        Ok(OutputFormats {
            template,
            samples: FormatSet::of([opened.sample]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(LayoutFileStage))
    }
}

fixture_stage!(
    Combine,
    CombineStage,
    Mask,
    MaskDefinition,
    "combine",
    NoArgs,
    &InputSignature::with_variadic_of(
        &[InputParam::required("first").accepting(StreamType::Mask)],
        StreamType::Mask,
    ),
    mask_template
);

fixture_stage!(
    Blend,
    BlendStage,
    Filter,
    FilterDefinition,
    "blend",
    NoArgs,
    &InputSignature::new(&[
        InputParam::required("base").accepting(StreamType::Frames),
        InputParam::required("overlay").accepting(StreamType::Frames),
        InputParam::required("mask")
            .accepting(StreamType::Mask)
            .planes_of("base"),
    ]),
    frames_template
);

fixture_stage!(
    PlainBlend,
    PlainBlendStage,
    Filter,
    FilterDefinition,
    "plain-blend",
    NoArgs,
    &InputSignature::new(&[
        InputParam::required("base").accepting(StreamType::Frames),
        InputParam::required("overlay").accepting(StreamType::Frames),
        InputParam::required("mask").accepting(StreamType::Mask),
    ]),
    frames_template
);

struct HostMaskerStage;

impl StageInstance for HostMaskerStage {}

impl Mask<HostSide> for HostMaskerStage {
    fn select(
        &mut self,
        _context: &mut FrameContext,
        _inputs: Inputs<HostSide>,
    ) -> Result<HostMask, StageError> {
        Err(StageError::msg("the host mask fixture never runs"))
    }
}

/// A mask that runs on the host, reading one frame stream.
pub(super) struct HostMasker;

impl StageDefinition for HostMasker {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "host-masker"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("base").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MaskDefinition for HostMasker {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let sample = offered.sample();

        Ok(FormatSet::of([sample]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;
        let template = mask_template(format);
        let sample = format.sample();

        Ok(OutputFormats {
            template,
            samples: FormatSet::of([sample]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask<HostSide>>, StageError> {
        Ok(Box::new(HostMaskerStage))
    }
}
