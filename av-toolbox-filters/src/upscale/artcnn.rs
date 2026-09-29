use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Parallelism,
    StageDefinition,
    StageError,
    StreamType,
};
use av_toolbox_cubek::artcnn::{Model, Precision};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    ElementType,
    FormatSet,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use av_toolbox_models::{Located, MissingModel, Store};
use schemars::JsonSchema;
use serde::Deserialize;

const STAGE: &str = "upscale/artcnn";

/// Arguments for `upscale/artcnn`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtcnnArgs {
    /// The network: `r8f64`, or the larger and slower `r16f96`.
    #[serde(default)]
    pub(crate) model: ModelName,

    /// The feature maps' precision: `f16`, or the slower and more exact `f32`.
    #[serde(default)]
    pub(crate) precision: PrecisionName,
}

#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ModelName {
    #[default]
    R8f64,
    R16f96,
}

#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum PrecisionName {
    #[default]
    F16,
    F32,
}

impl ModelName {
    pub(crate) fn model(self) -> Model {
        match self {
            Self::R8f64 => Model::R8F64,
            Self::R16f96 => Model::R16F96,
        }
    }
}

impl PrecisionName {
    pub(crate) fn precision(self) -> Precision {
        match self {
            Self::F16 => Precision::F16,
            Self::F32 => Precision::F32,
        }
    }
}

fn upscalable_layout(layout: FrameLayout) -> bool {
    matches!(layout, FrameLayout::Grey | FrameLayout::Yuv { .. })
}

fn upscalable_sample(sample: SampleFormat) -> bool {
    matches!(sample, SampleFormat::U8 | SampleFormat::U16 { .. })
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} upscales frames, not masks");

    StageError::msg(message)
}

fn unusable_layout(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} upscales grey or yuv frames, not {layout} frames");

    StageError::msg(message)
}

fn unusable_sample(sample: SampleFormat) -> StageError {
    let message = format!("{STAGE} upscales U8 or U16 frames, not {sample} frames");

    StageError::msg(message)
}

fn model_missing(missing: MissingModel) -> StageError {
    let message = format!("{STAGE} cannot find its model. {missing}");

    StageError::msg(message)
}

fn no_f16_arithmetic() -> StageError {
    let message = format!(
        "{STAGE} was given precision f16, but this device has no f16 arithmetic. Use precision = \"f32\"",
    );

    StageError::msg(message)
}

/// Finds `model`'s weight file in `store`, or the copy built into the binary.
pub(crate) fn locate(store: &Store, model: Model) -> Result<Located, StageError> {
    let found = store.find("artcnn", model.file_name());

    match found {
        Ok(located) => Ok(located),
        Err(missing) => Err(model_missing(missing)),
    }
}

/// The input format, once it is known to be U8 or U16 frames in a grey or YUV layout.
pub(crate) fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    if !upscalable_layout(format.layout) {
        return Err(unusable_layout(format.layout));
    }

    if !upscalable_sample(format.sample) {
        return Err(unusable_sample(format.sample));
    }

    Ok(*format)
}

/// The samples accepted from `offered`, refusing masks and layouts the stage cannot upscale.
fn accepted_samples(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    if !upscalable_layout(format.layout) {
        return Err(unusable_layout(format.layout));
    }

    if !upscalable_sample(format.sample) {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

pub(crate) fn doubled(input: &FrameFormat) -> FrameFormat {
    FrameFormat {
        width: input.width * 2,
        height: input.height * 2,
        ..*input
    }
}

/// The feature maps' precision, refusing f16 on a device without f16 arithmetic.
pub(crate) fn checked_precision(
    capabilities: &Capabilities,
    precision: PrecisionName,
) -> Result<Precision, StageError> {
    let computes_f16 = capabilities.can_compute(ElementType::F16);

    if precision == PrecisionName::F16 && !computes_f16 {
        return Err(no_f16_arithmetic());
    }

    let chosen = precision.precision();
    Ok(chosen)
}

/// Doubles a clip's size, running its luma through the ArtCNN network and its chroma through Catrom.
pub(crate) struct Artcnn;

impl StageDefinition for Artcnn {
    type Args = ArtcnnArgs;

    fn package() -> &'static str {
        "upscale"
    }

    fn identifier() -> &'static str {
        "artcnn"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Artcnn {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepted_samples(offered)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        checked_precision(capabilities, args.precision)?;

        let store = Store::from_environment();
        let model = args.model.model();
        locate(&store, model)?;

        let output = doubled(&input);
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([output.sample]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let message = format!("{STAGE} is not built yet");

        Err(StageError::msg(message))
    }
}
