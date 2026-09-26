use std::path::PathBuf;

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
    Primaries,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    Transfer,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::Prepared;
use super::args::{MatrixArg, RangeArg, Shape, check_frames, frame_format};
use super::convert::{convert, size_of_picture, wrong_size};
use super::like::{REFERENCE_SIGNATURE, accepts_reference, reference_format};
use super::picture::{decode, dimensions};
use super::repeat::{RepeatFilter, RepeatSource};
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
        let found = size_of_picture(&decoded.picture);
        let (width, height) = found.unwrap_or((1, 1));
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

const LIKE: &str = "image/like";

/// Arguments for `image/like`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct LikeArgs {
    /// The PNG or JPEG image to show, the same size as the reference.
    pub(crate) path: PathBuf,
}

/// Produces an image for each frame of a reference clip, in its format.
pub(crate) struct ImageLike;

impl StageDefinition for ImageLike {
    type Args = LikeArgs;

    fn package() -> &'static str {
        "image"
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

impl FilterDefinition for ImageLike {
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
        let found = dimensions(&args.path, LIKE)?;
        if found != (format.width, format.height) {
            return Err(wrong_size(LIKE, found, &format));
        }

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
        let decoded = decode(&args.path, LIKE)?;
        let converted = convert(&decoded.picture, &format, LIKE)?;
        let filter = RepeatFilter::new(converted, LIKE);

        Ok(Box::new(filter))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use av_toolbox_gpu::{ColourInfo, FrameFormat, FrameLayout, SampleFormat, default_accelerators};

    use super::*;

    fn capabilities() -> Capabilities {
        let accelerator = default_accelerators()[0];

        Capabilities::builder(accelerator)
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build()
    }

    fn grey_reference(width: u32, height: u32) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width,
            height,
            layout: FrameLayout::Grey,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
        })
    }

    fn like_args_for(folder: &Path) -> LikeArgs {
        let path = folder.join("small.png");
        let small = image::GrayImage::from_pixel(8, 4, image::Luma([255]));
        small.save(&path).expect("writes the image");

        LikeArgs { path }
    }

    #[test]
    fn image_like_produces_the_reference_format_for_an_image_of_its_size() {
        let folder = tempfile::tempdir().expect("a folder");
        let args = like_args_for(folder.path());
        let reference = grey_reference(8, 4);
        let chosen = [Some(reference)];

        let outputs = ImageLike::produces(&capabilities(), &args, &chosen).expect("the sizes match");

        let produced = outputs.template.with_sample(SampleFormat::U8);
        assert_eq!(produced, reference);
    }

    #[test]
    fn image_like_refuses_an_image_of_another_size() {
        let folder = tempfile::tempdir().expect("a folder");
        let args = like_args_for(folder.path());
        let reference = grey_reference(16, 8);
        let chosen = [Some(reference)];

        let refused = ImageLike::produces(&capabilities(), &args, &chosen);

        let error = refused.expect_err("8x4 is not 16x8").to_string();
        assert!(error.contains("resize/"), "{error}");
        assert!(error.contains("8x4"), "{error}");
        assert!(error.contains("16x8"), "{error}");
    }
}
