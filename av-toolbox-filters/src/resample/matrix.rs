use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::matrix::{
    AffineValues,
    Coefficients,
    Direction,
    MatrixConversion,
    MatrixPlanes,
    MatrixSamples,
    SampleRange,
    convert,
    upload_affine,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Client,
    ColourInfo,
    ColourRange,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Matrix as StreamMatrix,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::bits::sample_kind;
use crate::checks::accepts_integer_or_f32_frames;
use crate::scratch::device_plane;
use crate::sizing::Sizing;

const STAGE: &str = "resample/matrix";

/// The colour family a clip converts to.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum TargetName {
    Rgb,
    Yuv,
}

/// A YUV matrix, as the stage's arguments name it.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MatrixName {
    Bt709,
    Bt601,
    Bt2020Ncl,
}

impl MatrixName {
    fn coefficients(self) -> Coefficients {
        match self {
            Self::Bt709 => Coefficients::Bt709,
            Self::Bt601 => Coefficients::Bt601,
            Self::Bt2020Ncl => Coefficients::Bt2020Ncl,
        }
    }

    fn stream_matrix(self) -> StreamMatrix {
        match self {
            Self::Bt709 => StreamMatrix::Bt709,
            Self::Bt601 => StreamMatrix::Bt601,
            Self::Bt2020Ncl => StreamMatrix::Bt2020Ncl,
        }
    }
}

/// A sample range, as the stage's arguments name it.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RangeName {
    Limited,
    Full,
}

impl RangeName {
    fn sample_range(self) -> SampleRange {
        match self {
            Self::Limited => SampleRange::Limited,
            Self::Full => SampleRange::Full,
        }
    }

    fn colour_range(self) -> ColourRange {
        match self {
            Self::Limited => ColourRange::Limited,
            Self::Full => ColourRange::Full,
        }
    }
}

/// Arguments for `resample/matrix`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct MatrixArgs {
    /// The colour family to convert to, `rgb` or `yuv`.
    pub(crate) to: TargetName,
    /// The YUV matrix. Going to RGB it overrides the stream's, and going to YUV it is required.
    pub(crate) matrix: Option<MatrixName>,
    /// The YUV range. Going to RGB it overrides the stream's, and going to YUV it defaults to `limited`.
    pub(crate) range: Option<RangeName>,
}

/// The output format and the conversion that makes it.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) struct Converted {
    pub(super) output: FrameFormat,
    pub(super) conversion: MatrixConversion,
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} converts frames, not masks");

    StageError::msg(message)
}

fn subsampled(subsampling: Subsampling) -> StageError {
    let message = format!(
        "{STAGE} converts 4:4:4 frames, not {subsampling}. Resample the chroma to 4:4:4 with resample/chroma first"
    );

    StageError::msg(message)
}

fn wrong_layout(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} converts yuv or rgb frames, not {layout} frames");

    StageError::msg(message)
}

fn already(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} was asked to convert {layout} frames to what they already are");

    StageError::msg(message)
}

fn unknown_matrix() -> StageError {
    let message =
        format!("{STAGE} cannot tell which matrix the stream uses. Set matrix to bt709, bt601 or bt2020_ncl");

    StageError::msg(message)
}

fn unknown_range() -> StageError {
    let message = format!("{STAGE} cannot tell which range the stream uses. Set range to limited or full");

    StageError::msg(message)
}

fn yuv_needs_matrix() -> StageError {
    let message = format!(
        "{STAGE} needs matrix to convert rgb frames to yuv. Set matrix to bt709, bt601 or bt2020_ncl"
    );

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn missing_channel() -> StageError {
    let message = format!("{STAGE} was handed a frame missing a channel");

    StageError::msg(message)
}

/// The input format, once it is known to be frames.
fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    Ok(*format)
}

fn stream_coefficients(matrix: StreamMatrix) -> Option<Coefficients> {
    match matrix {
        StreamMatrix::Bt709 => Some(Coefficients::Bt709),
        StreamMatrix::Bt601 => Some(Coefficients::Bt601),
        StreamMatrix::Bt2020Ncl => Some(Coefficients::Bt2020Ncl),
        StreamMatrix::Unspecified | StreamMatrix::Identity => None,
    }
}

fn stream_range(range: ColourRange) -> Option<SampleRange> {
    match range {
        ColourRange::Limited => Some(SampleRange::Limited),
        ColourRange::Full => Some(SampleRange::Full),
        ColourRange::Unspecified => None,
    }
}

/// YUV to full range float RGB, from the stream's matrix and range unless the arguments give them.
fn to_rgb(input: &FrameFormat, args: &MatrixArgs) -> Result<Converted, StageError> {
    let stream_matrix = stream_coefficients(input.colour.matrix);
    let Some(coefficients) = args.matrix.map(MatrixName::coefficients).or(stream_matrix) else {
        return Err(unknown_matrix());
    };

    let stream_sample_range = stream_range(input.colour.range);
    let given = args.range.map(RangeName::sample_range).or(stream_sample_range);
    let input_range = match (given, input.sample) {
        (Some(range), _) => range,
        (None, SampleFormat::F32) => SampleRange::Full,
        (None, _) => return Err(unknown_range()),
    };

    let colour = ColourInfo {
        matrix: StreamMatrix::Identity,
        range: ColourRange::Full,
        ..input.colour
    };
    let output = FrameFormat {
        layout: FrameLayout::Rgb,
        sample: SampleFormat::F32,
        colour,
        ..*input
    };
    let input_samples = sample_kind(input.sample);
    let conversion = MatrixConversion {
        coefficients,
        direction: Direction::ToRgb,
        input_samples,
        output_samples: MatrixSamples::F32,
        input_range,
        output_range: SampleRange::Full,
    };

    Ok(Converted { output, conversion })
}

/// RGB to float YUV, with the given matrix, labelled with the given range or limited range.
fn to_yuv(input: &FrameFormat, args: &MatrixArgs) -> Result<Converted, StageError> {
    let Some(matrix) = args.matrix else {
        return Err(yuv_needs_matrix());
    };

    let range = args.range.unwrap_or(RangeName::Limited);
    let input_range = stream_range(input.colour.range).unwrap_or(SampleRange::Full);

    let colour = ColourInfo {
        matrix: matrix.stream_matrix(),
        range: range.colour_range(),
        ..input.colour
    };
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let output = FrameFormat {
        layout,
        sample: SampleFormat::F32,
        colour,
        ..*input
    };
    let input_samples = sample_kind(input.sample);
    let conversion = MatrixConversion {
        coefficients: matrix.coefficients(),
        direction: Direction::ToYuv,
        input_samples,
        output_samples: MatrixSamples::F32,
        input_range,
        output_range: range.sample_range(),
    };

    Ok(Converted { output, conversion })
}

/// Checks the arguments against the input, giving the output format and the conversion.
pub(super) fn converted(input: &FrameFormat, args: &MatrixArgs) -> Result<Converted, StageError> {
    match (input.layout, args.to) {
        (FrameLayout::Yuv { .. }, TargetName::Yuv) | (FrameLayout::Rgb, TargetName::Rgb) => {
            Err(already(input.layout))
        },
        (FrameLayout::Yuv { subsampling }, TargetName::Rgb) if subsampling != Subsampling::Yuv444 => {
            Err(subsampled(subsampling))
        },
        (FrameLayout::Yuv { .. }, TargetName::Rgb) => to_rgb(input, args),
        (FrameLayout::Rgb, TargetName::Yuv) => to_yuv(input, args),
        (layout, _) => Err(wrong_layout(layout)),
    }
}

/// Converts a clip between 4:4:4 YUV and RGB.
pub(crate) struct Matrix;

impl StageDefinition for Matrix {
    type Args = MatrixArgs;

    fn package() -> &'static str {
        "resample"
    }

    fn identifier() -> &'static str {
        "matrix"
    }

    fn description() -> &'static str {
        "Converts a clip between RGB and YUV.\n\n\
         YUV must be 4:4:4, so run `resample/chroma` first on a clip with subsampled colour. \
         The result is always float."
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

impl FilterDefinition for Matrix {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(samples) = accepts_integer_or_f32_frames(offered) else {
            return Err(not_frames());
        };

        Ok(samples)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        let planned = converted(&input, args)?;
        let format = StreamFormat::Frames(planned.output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([SampleFormat::F32]),
        })
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs)?;
        let planned = converted(&input, &args)?;

        let client = context.client().clone();
        let affine = planned.conversion.affine();
        let values = upload_affine(&client, &affine);
        let converter = MatrixConverter {
            client,
            input,
            output: planned.output,
            conversion: planned.conversion,
            values,
        };

        Ok(Box::new(converter))
    }
}

/// Converts every frame of a clip through one affine map.
struct MatrixConverter {
    client: Client,
    input: FrameFormat,
    output: FrameFormat,
    conversion: MatrixConversion,
    values: AffineValues,
}

/// The three planes of `frame` in `format`'s channel order.
fn frame_planes<'a>(frame: &'a Frame, format: &FrameFormat) -> Result<[&'a Plane; 3], StageError> {
    let channels = format.channels();
    let mut found = Vec::with_capacity(3);

    for channel in channels {
        let Some(plane) = frame.plane(*channel) else {
            return Err(missing_channel());
        };

        found.push(plane);
    }

    let Ok(planes) = <[&Plane; 3]>::try_from(found) else {
        return Err(missing_channel());
    };

    Ok(planes)
}

/// The planes as device planes.
///
/// # Safety
///
/// Every plane must hold samples of the format the conversion reads or writes it as.
unsafe fn device_planes<'a>(planes: [&'a Plane; 3]) -> [DevicePlane<'a>; 3] {
    // Safety: the caller guarantees the element type, and a plane holds `width * height` of them.
    planes.map(|plane| unsafe { device_plane(plane) })
}

impl StageInstance for MatrixConverter {}

impl Filter for MatrixConverter {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        let meta = input.meta().clone();
        let output = context.alloc_frame(&self.output, meta);
        let sources = frame_planes(&input, &self.input)?;
        let targets = frame_planes(&output, &self.output)?;

        // Safety: both frames are in the formats the converter was built for, so their planes hold
        // the samples `self.conversion` reads and writes.
        let inputs = unsafe { device_planes(sources) };

        // Safety: as above.
        let outputs = unsafe { device_planes(targets) };
        let planes = MatrixPlanes { inputs, outputs };
        let sizing = Sizing(context);
        convert(&self.client, &sizing, planes, &self.conversion, &self.values);

        Ok(vec![output])
    }
}
