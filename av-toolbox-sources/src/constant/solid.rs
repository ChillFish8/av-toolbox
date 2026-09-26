use av_toolbox_core::{BoundInputs, InputSignature, Source, SourceDefinition, StageDefinition, StageError};
use av_toolbox_gpu::{BuildContext, Capabilities, DeviceSide, OutputFormats, ResolvedFormats, Runtime};
use schemars::JsonSchema;
use serde::Deserialize;

use super::Prepared;
use super::args::{MatrixArg, RangeArg, Shape, check_frames, frame_format};
use super::colour::Colour;
use super::convert::convert;
use super::repeat::RepeatSource;
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
