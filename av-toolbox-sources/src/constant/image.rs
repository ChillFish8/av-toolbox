use std::path::PathBuf;

use av_toolbox_core::{BoundInputs, InputSignature, Source, SourceDefinition, StageDefinition, StageError};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    OutputFormats,
    Primaries,
    ResolvedFormats,
    Runtime,
    Transfer,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::Prepared;
use super::args::{MatrixArg, RangeArg, Shape, check_frames, frame_format};
use super::convert::convert;
use super::picture::{Picture, decode};
use super::repeat::RepeatSource;
use crate::shared::{Carried, produced_by};

const FILE: &str = "image/file";

/// Arguments for `image/file`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileArgs {
    /// The PNG or JPEG image to show.
    pub(crate) path: PathBuf,
    /// How many frames to produce.
    pub(crate) frames: u64,
    /// The pixel format, the one that carries the image without loss by default.
    pub(crate) format: Option<Carried>,
    /// The matrix for YUV and grey frames, bt709 by default.
    pub(crate) matrix: Option<MatrixArg>,
    /// The range, limited for YUV and full otherwise by default.
    pub(crate) range: Option<RangeArg>,
    /// Frames per second as a numerator and denominator, unknown by default.
    pub(crate) frame_rate: Option<[u32; 2]>,
}

fn picture_size(picture: &Picture) -> (u32, u32) {
    match picture {
        Picture::Rgb(rgb) => (rgb.width, rgb.height),
        Picture::Grey(grey) => (grey.width, grey.height),
        Picture::Planes(_) => (1, 1),
    }
}

/// Produces frames showing one image.
pub(crate) struct File;

impl StageDefinition for File {
    type Args = FileArgs;

    fn package() -> &'static str {
        "image"
    }

    fn identifier() -> &'static str {
        "file"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for File {
    type Side = DeviceSide;
    type Opened = Prepared;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        check_frames(args.frames, FILE)?;

        let decoded = decode(&args.path, FILE)?;
        let (width, height) = picture_size(&decoded.picture);
        let shape = Shape {
            width,
            height,
            format: args.format.unwrap_or(decoded.natural),
            matrix: args.matrix,
            range: args.range,
            frame_rate: args.frame_rate,
        };
        let mut format = frame_format(&shape, FILE)?;
        format.colour.primaries = Primaries::Bt709;
        format.colour.transfer = Transfer::Srgb;

        let converted = convert(&decoded.picture, &format, FILE)?;

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
