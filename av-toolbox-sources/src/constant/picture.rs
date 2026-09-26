use std::path::Path;

use av_toolbox_core::StageError;
use image::{ColorType, DynamicImage, ImageError, ImageReader};

use crate::shared::Carried;

/// An image on the host, normalised so every sample runs over `0.0..=1.0`.
///
/// A picture of one pixel stands for a solid colour of any size.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Picture {
    Rgb(RgbPicture),
    Grey(GreyPicture),
    /// One raw sample value per plane, used exactly as given.
    Planes(Vec<u16>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RgbPicture {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) red: Vec<f32>,
    pub(crate) green: Vec<f32>,
    pub(crate) blue: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GreyPicture {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) luma: Vec<f32>,
}

/// A decoded image and the format that carries it without loss.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Decoded {
    pub(crate) picture: Picture,
    pub(crate) natural: Carried,
}

fn unreadable(stage: &str, path: &Path, reason: &dyn std::fmt::Display) -> StageError {
    let message = format!("{stage} could not read {}: {reason}", path.display());

    StageError::msg(message)
}

fn not_an_image(stage: &str, path: &Path, error: &ImageError) -> StageError {
    let message = format!(
        "{stage} could not decode {} as a PNG or JPEG image: {error}",
        path.display()
    );

    StageError::msg(message)
}

fn unsupported_samples(stage: &str, path: &Path, colour: ColorType) -> StageError {
    let message = format!(
        "{stage} needs an 8 or 16 bit grey or RGB image, {} holds {colour:?}",
        path.display(),
    );

    StageError::msg(message)
}

/// The format that carries an image's samples without loss, or `None` when none does.
fn natural(colour: ColorType) -> Option<Carried> {
    match colour {
        ColorType::L8 | ColorType::La8 => Some(Carried::Gray),
        ColorType::L16 | ColorType::La16 => Some(Carried::Gray16),
        ColorType::Rgb8 | ColorType::Rgba8 => Some(Carried::Gbrp),
        ColorType::Rgb16 | ColorType::Rgba16 => Some(Carried::Gbrp16),
        _ => None,
    }
}

fn rgb_picture(image: &DynamicImage) -> RgbPicture {
    let converted = image.to_rgb32f();
    let width = converted.width();
    let height = converted.height();
    let samples = converted.into_raw();

    let mut red = Vec::with_capacity(samples.len() / 3);
    let mut green = Vec::with_capacity(samples.len() / 3);
    let mut blue = Vec::with_capacity(samples.len() / 3);
    let (pixels, _) = samples.as_chunks::<3>();
    for pixel in pixels {
        red.push(pixel[0]);
        green.push(pixel[1]);
        blue.push(pixel[2]);
    }

    RgbPicture {
        width,
        height,
        red,
        green,
        blue,
    }
}

fn grey_picture(image: &DynamicImage) -> GreyPicture {
    let converted = image.to_luma32f();
    let width = converted.width();
    let height = converted.height();
    let luma = converted.into_raw();

    GreyPicture { width, height, luma }
}

fn open(path: &Path, stage: &str) -> Result<ImageReader<std::io::BufReader<std::fs::File>>, StageError> {
    let opened = match ImageReader::open(path) {
        Ok(opened) => opened,
        Err(error) => return Err(unreadable(stage, path, &error)),
    };

    match opened.with_guessed_format() {
        Ok(guessed) => Ok(guessed),
        Err(error) => Err(unreadable(stage, path, &error)),
    }
}

/// Decodes a PNG or JPEG image, dropping any alpha.
pub(crate) fn decode(path: &Path, stage: &str) -> Result<Decoded, StageError> {
    let reader = open(path, stage)?;
    let image = match reader.decode() {
        Ok(image) => image,
        Err(error) => return Err(not_an_image(stage, path, &error)),
    };

    let colour = image.color();
    let Some(natural) = natural(colour) else {
        return Err(unsupported_samples(stage, path, colour));
    };

    let picture = if colour.has_color() {
        let rgb = rgb_picture(&image);
        Picture::Rgb(rgb)
    } else {
        let grey = grey_picture(&image);
        Picture::Grey(grey)
    };

    Ok(Decoded { picture, natural })
}

/// An image's width and height, read from its header without decoding it.
pub(crate) fn dimensions(path: &Path, stage: &str) -> Result<(u32, u32), StageError> {
    let reader = open(path, stage)?;

    match reader.into_dimensions() {
        Ok(found) => Ok(found),
        Err(error) => Err(not_an_image(stage, path, &error)),
    }
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage, Rgba, RgbaImage};

    use super::*;

    #[test]
    fn an_rgb_png_decodes_to_normalised_channels() {
        let folder = tempfile::tempdir().expect("a folder");
        let path = folder.path().join("orange.png");
        let orange = RgbImage::from_pixel(3, 2, Rgb([255, 128, 0]));
        orange.save(&path).expect("writes the image");

        let decoded = decode(&path, "image/file").expect("decodes");

        let Picture::Rgb(rgb) = decoded.picture else {
            panic!("expected an RGB picture");
        };
        assert_eq!(decoded.natural, Carried::Gbrp);
        assert_eq!((rgb.width, rgb.height), (3, 2));
        assert_eq!(rgb.red, vec![1.0; 6]);
        assert_eq!(rgb.green, vec![128.0 / 255.0; 6]);
        assert_eq!(rgb.blue, vec![0.0; 6]);
    }

    #[test]
    fn a_sixteen_bit_grey_png_is_grey16() {
        let folder = tempfile::tempdir().expect("a folder");
        let path = folder.path().join("grey.png");
        let grey: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_pixel(2, 2, Luma([65535]));
        grey.save(&path).expect("writes the image");

        let decoded = decode(&path, "image/file").expect("decodes");

        let Picture::Grey(picture) = decoded.picture else {
            panic!("expected a grey picture");
        };
        assert_eq!(decoded.natural, Carried::Gray16);
        assert_eq!(picture.luma, vec![1.0; 4]);
    }

    #[test]
    fn an_eight_bit_grey_png_is_gray() {
        let folder = tempfile::tempdir().expect("a folder");
        let path = folder.path().join("grey.png");
        let grey = GrayImage::from_pixel(2, 2, Luma([0]));
        grey.save(&path).expect("writes the image");

        let decoded = decode(&path, "image/file").expect("decodes");

        assert_eq!(decoded.natural, Carried::Gray);
    }

    #[test]
    fn an_rgba_png_drops_its_alpha() {
        let folder = tempfile::tempdir().expect("a folder");
        let path = folder.path().join("clear.png");
        let clear = RgbaImage::from_pixel(2, 1, Rgba([0, 255, 0, 0]));
        clear.save(&path).expect("writes the image");

        let decoded = decode(&path, "image/file").expect("decodes");

        let Picture::Rgb(rgb) = decoded.picture else {
            panic!("expected an RGB picture");
        };
        assert_eq!(decoded.natural, Carried::Gbrp);
        assert_eq!(rgb.green, vec![1.0; 2]);
    }

    #[test]
    fn a_missing_file_names_its_path() {
        let folder = tempfile::tempdir().expect("a folder");
        let path = folder.path().join("missing.png");

        let error = decode(&path, "image/file").expect_err("nothing to read");

        assert!(error.to_string().contains("image/file"));
        assert!(error.to_string().contains("missing.png"));
    }

    #[test]
    fn a_file_that_is_not_an_image_is_refused() {
        let folder = tempfile::tempdir().expect("a folder");
        let path = folder.path().join("notes.png");
        std::fs::write(&path, b"not an image").expect("writes the file");

        let error = decode(&path, "image/file").expect_err("not an image");

        assert!(error.to_string().contains("notes.png"));
    }

    #[test]
    fn dimensions_read_the_header_only() {
        let folder = tempfile::tempdir().expect("a folder");
        let path = folder.path().join("wide.png");
        let wide = GrayImage::from_pixel(7, 3, Luma([0]));
        wide.save(&path).expect("writes the image");

        let found = dimensions(&path, "image/like").expect("reads the header");

        assert_eq!(found, (7, 3));
    }
}
