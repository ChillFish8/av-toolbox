use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputSignature,
    Parallelism,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::Prepared;
use super::args::{MatrixArg, RangeArg, Shape, check_frames, frame_format};
use super::colour::Colour;
use super::convert::{check, convert};
use super::like::{REFERENCE_SIGNATURE, accepts_reference, reference_format};
use super::repeat::{RepeatFilter, RepeatSource};
use crate::shared::{Carried, produced_by};

const FILL: &str = "solid/fill";

/// Arguments for `solid/fill`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct FillArgs {
    /// The frame width.
    pub(crate) width: u32,
    /// The frame height.
    pub(crate) height: u32,
    /// How many frames to produce.
    pub(crate) frames: u64,
    /// The colour of every pixel.
    pub(crate) colour: Colour,
    /// The pixel format, yuv420p by default.
    #[serde(default = "yuv420p")]
    pub(crate) format: Carried,
    /// The matrix for YUV and grey frames, bt709 by default.
    pub(crate) matrix: Option<MatrixArg>,
    /// The range, limited for YUV and full otherwise by default.
    pub(crate) range: Option<RangeArg>,
    /// Frames per second as a numerator and denominator, unknown by default.
    pub(crate) frame_rate: Option<[u32; 2]>,
}

fn yuv420p() -> Carried {
    Carried::Yuv420p
}

/// Produces frames of a single colour.
pub(crate) struct Fill;

impl StageDefinition for Fill {
    type Args = FillArgs;

    fn package() -> &'static str {
        "solid"
    }

    fn identifier() -> &'static str {
        "fill"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Fill {
    type Side = DeviceSide;
    type Opened = Prepared;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        check_frames(args.frames, FILL)?;

        let shape = Shape {
            width: args.width,
            height: args.height,
            format: args.format,
            matrix: args.matrix,
            range: args.range,
            frame_rate: args.frame_rate,
        };
        let format = frame_format(&shape, FILL)?;
        let picture = args.colour.picture(FILL)?;
        let converted = convert(&picture, &format, FILL)?;

        Ok(Prepared {
            converted,
            frames: args.frames,
        })
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let produced = produced_by(&opened.converted.format);

        Ok(produced)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<DeviceSide>>, StageError> {
        let source = RepeatSource::new(opened.converted, opened.frames);

        Ok(Box::new(source))
    }
}

const LIKE: &str = "solid/like";

/// Arguments for `solid/like`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct LikeArgs {
    /// The colour of every pixel.
    pub(crate) colour: Colour,
}

/// Produces a frame of a single colour for each frame of a reference clip, in its format.
pub(crate) struct SolidLike;

impl StageDefinition for SolidLike {
    type Args = LikeArgs;

    fn package() -> &'static str {
        "solid"
    }

    fn identifier() -> &'static str {
        "like"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        REFERENCE_SIGNATURE
    }
}

impl FilterDefinition for SolidLike {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_reference(offered, LIKE)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = reference_format(chosen, LIKE)?;
        let picture = args.colour.picture(LIKE)?;
        check(&picture, &format, LIKE)?;

        let produced = produced_by(&format);

        Ok(produced)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let format = reference_format(inputs, LIKE)?;
        let picture = args.colour.picture(LIKE)?;
        let converted = convert(&picture, &format, LIKE)?;
        let filter = RepeatFilter::new(converted, LIKE);

        Ok(Box::new(filter))
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, FrameFormat, FrameLayout, SampleFormat, default_accelerators};
    use serde_json::json;

    use super::*;

    fn capabilities() -> Capabilities {
        let accelerator = default_accelerators()[0];

        Capabilities::builder(accelerator)
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build()
    }

    fn grey(sample: SampleFormat) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width: 4,
            height: 2,
            layout: FrameLayout::Grey,
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        })
    }

    fn like_args(value: serde_json::Value) -> LikeArgs {
        serde_json::from_value(value).expect("parses")
    }

    #[test]
    fn solid_like_produces_the_reference_format() {
        let reference = grey(SampleFormat::U8);
        let args = like_args(json!({ "colour": "#000000" }));
        let chosen = [Some(reference)];

        let outputs = SolidLike::produces(&capabilities(), &args, &chosen).expect("valid");

        let produced = outputs.template.with_sample(SampleFormat::U8);
        assert_eq!(produced, reference);
    }

    #[test]
    fn solid_like_refuses_planes_that_do_not_fit_the_reference() {
        let reference = grey(SampleFormat::U8);
        let args = like_args(json!({ "colour": { "planes": [16, 128, 128] } }));
        let chosen = [Some(reference)];

        let refused = SolidLike::produces(&capabilities(), &args, &chosen);

        assert!(refused.is_err());
    }

    #[test]
    fn solid_like_leaves_out_float_references() {
        let args = like_args(json!({ "colour": "#000000" }));
        let inputs = [Offered::Unknown];
        let offered = grey(SampleFormat::F32);

        let accepted = SolidLike::accepts_input(&capabilities(), &args, 0, offered, &inputs).expect("frames");

        assert!(accepted.is_empty());
    }

    #[test]
    fn solid_like_runs_per_frame() {
        let args = like_args(json!({ "colour": "#000000" }));

        assert_eq!(SolidLike::parallelism(&args), Parallelism::Frame);
    }
}
