use av_toolbox_core::StageError;
use av_toolbox_cubek::resize::{Kernel, Taps, taps};
use av_toolbox_gpu::{Channel, ColourRange, FrameFormat, FrameLayout, Matrix, SampleFormat, Subsampling};

use super::picture::{GreyPicture, Picture, RgbPicture};

/// A frame's samples, ready to copy into a frame of `format`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Converted {
    pub(crate) format: FrameFormat,
    /// One buffer per channel in plane order, little endian when a sample is wider than a byte.
    pub(crate) planes: Vec<Vec<u8>>,
}

/// One channel as a normalised signal, at luma size or as a single value.
///
/// Luma and RGB run over `0.0..=1.0` and chroma over `-0.5..=0.5`.
#[derive(Debug, Clone)]
struct Signal {
    width: u32,
    height: u32,
    values: Vec<f32>,
}

/// Whether a channel carries a level or a colour difference.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Role {
    Level,
    Difference,
}

fn float_target(stage: &str, sample: SampleFormat) -> StageError {
    let message = format!("{stage} makes U8 or U16 frames, not {sample}");

    StageError::msg(message)
}

fn wrong_plane_count(stage: &str, expected: usize, found: usize) -> StageError {
    let message = format!("{stage} needs {expected} plane values for this format, got {found}");

    StageError::msg(message)
}

fn past_depth(stage: &str, value: u16, largest: u16) -> StageError {
    let message = format!("{stage} needs plane values up to {largest} for this format, got {value}");

    StageError::msg(message)
}

fn wrong_size(stage: &str, found: (u32, u32), format: &FrameFormat) -> StageError {
    let message = format!(
        "{stage} needs a {}x{} image to match its frames, got {}x{}. Resize the image, or use image/file \
         followed by a resize/* stage",
        format.width, format.height, found.0, found.1,
    );

    StageError::msg(message)
}

fn size_of_picture(picture: &Picture) -> Option<(u32, u32)> {
    match picture {
        Picture::Rgb(rgb) => Some((rgb.width, rgb.height)),
        Picture::Grey(grey) => Some((grey.width, grey.height)),
        Picture::Planes(_) => None,
    }
}

/// Refuses a picture that cannot become a frame of `format`.
pub(crate) fn check(picture: &Picture, format: &FrameFormat, stage: &str) -> Result<(), StageError> {
    let Some(largest) = format.sample.max_value() else {
        return Err(float_target(stage, format.sample));
    };

    if let Picture::Planes(values) = picture {
        let expected = format.channels().len();
        if values.len() != expected {
            return Err(wrong_plane_count(stage, expected, values.len()));
        }

        for value in values {
            if *value > largest {
                return Err(past_depth(stage, *value, largest));
            }
        }

        return Ok(());
    }

    let Some(found) = size_of_picture(picture) else {
        return Ok(());
    };

    let is_single = found == (1, 1);
    let is_full = found == (format.width, format.height);
    if !is_single && !is_full {
        return Err(wrong_size(stage, found, format));
    }

    Ok(())
}

/// Converts a picture into the planes of a frame of `format`.
pub(crate) fn convert(picture: &Picture, format: &FrameFormat, stage: &str) -> Result<Converted, StageError> {
    check(picture, format, stage)?;

    let planes = match picture {
        Picture::Planes(values) => fill_planes(values, format),
        Picture::Rgb(_) | Picture::Grey(_) => encode_picture(picture, format),
    };

    Ok(Converted {
        format: *format,
        planes,
    })
}

fn bits_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        _ => 8,
    }
}

fn push_sample(bytes: &mut Vec<u8>, value: u16, sample: SampleFormat) {
    match sample {
        SampleFormat::U16 { .. } => bytes.extend_from_slice(&value.to_le_bytes()),
        _ => bytes.push(value as u8),
    }
}

fn fill_planes(values: &[u16], format: &FrameFormat) -> Vec<Vec<u8>> {
    let mut planes = Vec::with_capacity(values.len());

    for (channel, value) in format.channels().iter().zip(values) {
        let samples = format
            .channel_samples(*channel)
            .expect("the format carries every channel it lists");
        let mut bytes = Vec::with_capacity(samples * format.sample.bytes());
        for _ in 0..samples {
            push_sample(&mut bytes, *value, format.sample);
        }

        planes.push(bytes);
    }

    planes
}

/// The weights of red and blue in luma, green taking the rest.
///
/// An unspecified or identity matrix falls back to BT.709.
fn luma_weights(matrix: Matrix) -> (f32, f32) {
    match matrix {
        Matrix::Bt601 => (0.299, 0.114),
        Matrix::Bt2020Ncl => (0.2627, 0.0593),
        Matrix::Bt709 | Matrix::Identity | Matrix::Unspecified => (0.2126, 0.0722),
    }
}

fn is_limited(format: &FrameFormat) -> bool {
    match format.colour.range {
        ColourRange::Limited => true,
        ColourRange::Full => false,
        ColourRange::Unspecified => matches!(format.layout, FrameLayout::Yuv { .. }),
    }
}

/// Scales a normalised signal into a sample of `bits`, rounded and clamped.
fn quantise(value: f32, role: Role, limited: bool, bits: u32) -> u16 {
    let largest = ((1u32 << bits) - 1) as f32;
    let step = (1u32 << (bits - 8)) as f32;
    let middle = (1u32 << (bits - 1)) as f32;

    let scaled = match (role, limited) {
        (Role::Level, true) => (16.0 + 219.0 * value) * step,
        (Role::Level, false) => value * largest,
        (Role::Difference, true) => (128.0 + 224.0 * value) * step,
        (Role::Difference, false) => middle + value * largest,
    };

    scaled.round().clamp(0.0, largest) as u16
}

fn signal(width: u32, height: u32, values: Vec<f32>) -> Signal {
    Signal {
        width,
        height,
        values,
    }
}

/// Luma and the two colour differences of an RGB picture.
fn yuv_of(rgb: &RgbPicture, matrix: Matrix) -> [Signal; 3] {
    let (red_weight, blue_weight) = luma_weights(matrix);
    let green_weight = 1.0 - red_weight - blue_weight;
    let samples = rgb.red.len();

    let mut luma = Vec::with_capacity(samples);
    let mut blue_difference = Vec::with_capacity(samples);
    let mut red_difference = Vec::with_capacity(samples);
    for index in 0..samples {
        let red = rgb.red[index];
        let green = rgb.green[index];
        let blue = rgb.blue[index];
        let level = red_weight * red + green_weight * green + blue_weight * blue;

        luma.push(level);
        blue_difference.push((blue - level) / (2.0 * (1.0 - blue_weight)));
        red_difference.push((red - level) / (2.0 * (1.0 - red_weight)));
    }

    [
        signal(rgb.width, rgb.height, luma),
        signal(rgb.width, rgb.height, blue_difference),
        signal(rgb.width, rgb.height, red_difference),
    ]
}

fn neutral(grey: &GreyPicture) -> Signal {
    let samples = grey.luma.len();

    signal(grey.width, grey.height, vec![0.0; samples])
}

/// Every channel of `format`, in plane order, as a signal at the picture's size.
fn signals(picture: &Picture, format: &FrameFormat) -> Vec<Signal> {
    match (picture, format.layout) {
        (Picture::Rgb(rgb), FrameLayout::Rgb) => vec![
            signal(rgb.width, rgb.height, rgb.red.clone()),
            signal(rgb.width, rgb.height, rgb.green.clone()),
            signal(rgb.width, rgb.height, rgb.blue.clone()),
        ],
        (Picture::Rgb(rgb), FrameLayout::Yuv { .. }) => {
            let channels = yuv_of(rgb, format.colour.matrix);
            channels.into()
        },
        (Picture::Rgb(rgb), FrameLayout::Grey) => {
            let [luma, _, _] = yuv_of(rgb, format.colour.matrix);
            vec![luma]
        },
        (Picture::Grey(grey), FrameLayout::Rgb) => {
            let luma = signal(grey.width, grey.height, grey.luma.clone());
            vec![luma.clone(), luma.clone(), luma]
        },
        (Picture::Grey(grey), FrameLayout::Yuv { .. }) => vec![
            signal(grey.width, grey.height, grey.luma.clone()),
            neutral(grey),
            neutral(grey),
        ],
        (Picture::Grey(grey), FrameLayout::Grey) => {
            vec![signal(grey.width, grey.height, grey.luma.clone())]
        },
        (Picture::Planes(_), _) => unreachable!("raw planes are filled, never encoded"),
    }
}

/// Folds `index` back into `0..length` without repeating the edge sample.
fn reflect(index: i32, length: i32) -> usize {
    if length <= 1 {
        return 0;
    }

    let period = 2 * length - 2;
    let folded = index.abs() % period;
    let inside = if folded >= length { period - folded } else { folded };

    inside as usize
}

fn separable(signal: &Signal, across: &Taps, down: &Taps) -> Vec<f32> {
    let width = signal.width as i32;
    let height = signal.height as i32;
    let out_width = across.starts.len();
    let out_height = down.starts.len();

    let mut rows = vec![0.0f32; out_width * signal.height as usize];
    for row in 0..signal.height as usize {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..across.count as usize {
                let tap = across.starts[column] + step as i32;
                let source = reflect(tap, width);
                let weight = across.weights[column * across.count as usize + step];
                total += signal.values[row * signal.width as usize + source] * weight;
            }

            rows[row * out_width + column] = total;
        }
    }

    let mut output = Vec::with_capacity(out_width * out_height);
    for row in 0..out_height {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..down.count as usize {
                let tap = down.starts[row] + step as i32;
                let source = reflect(tap, height);
                let weight = down.weights[row * down.count as usize + step];
                total += rows[source * out_width + column] * weight;
            }

            output.push(total);
        }
    }

    output
}

fn factors(subsampling: Subsampling) -> (u32, u32) {
    match subsampling {
        Subsampling::Yuv444 => (1, 1),
        Subsampling::Yuv422 => (2, 1),
        Subsampling::Yuv420 => (2, 2),
    }
}

/// Shrinks a full-size chroma signal to its plane, sited where the format's chroma location says.
fn downsample(full: Signal, format: &FrameFormat, subsampling: Subsampling) -> Signal {
    let is_single = full.width == 1 && full.height == 1;
    if is_single || subsampling == Subsampling::Yuv444 {
        return full;
    }

    let (factor_x, factor_y) = factors(subsampling);
    let (offset_x, offset_y) = format.colour.chroma_location.offsets(factor_x, factor_y);
    let (width, height) = subsampling.chroma_dimensions(full.width, full.height);
    let across = taps(width, factor_x as f32, offset_x, &Kernel::Bilinear);
    let down = taps(height, factor_y as f32, offset_y, &Kernel::Bilinear);
    let values = separable(&full, &across, &down);

    signal(width, height, values)
}

/// Quantises a signal into a plane of `samples` samples, repeating a single value to fill it.
fn encode(signal: &Signal, samples: usize, role: Role, limited: bool, sample: SampleFormat) -> Vec<u8> {
    let bits = bits_of(sample);
    let is_single = signal.values.len() == 1;
    let mut bytes = Vec::with_capacity(samples * sample.bytes());

    for index in 0..samples {
        let value = if is_single {
            signal.values[0]
        } else {
            signal.values[index]
        };
        let quantised = quantise(value, role, limited, bits);
        push_sample(&mut bytes, quantised, sample);
    }

    bytes
}

fn encode_picture(picture: &Picture, format: &FrameFormat) -> Vec<Vec<u8>> {
    let limited = is_limited(format);
    let channel_signals = signals(picture, format);
    let mut planes = Vec::with_capacity(channel_signals.len());

    for (channel, full) in format.channels().iter().zip(channel_signals) {
        let is_chroma = matches!(channel, Channel::U | Channel::V);
        let role = if is_chroma { Role::Difference } else { Role::Level };
        let sited = match format.layout {
            FrameLayout::Yuv { subsampling } if is_chroma => downsample(full, format, subsampling),
            _ => full,
        };
        let samples = format
            .channel_samples(*channel)
            .expect("the format carries every channel it lists");

        let plane = encode(&sited, samples, role, limited, format.sample);
        planes.push(plane);
    }

    planes
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ChromaLocation, ColourInfo, Depth, Subsampling};

    use super::*;
    use crate::constant::picture::{GreyPicture, RgbPicture};

    fn format(
        width: u32,
        height: u32,
        layout: FrameLayout,
        sample: SampleFormat,
        colour: ColourInfo,
    ) -> FrameFormat {
        FrameFormat {
            width,
            height,
            layout,
            sample,
            colour,
            frame_rate: None,
        }
    }

    fn yuv444(sample: SampleFormat, range: ColourRange, matrix: Matrix) -> FrameFormat {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        };
        let colour = ColourInfo {
            range,
            matrix,
            ..ColourInfo::default()
        };

        format(2, 2, layout, sample, colour)
    }

    fn solid(red: f32, green: f32, blue: f32) -> Picture {
        Picture::Rgb(RgbPicture {
            width: 1,
            height: 1,
            red: vec![red],
            green: vec![green],
            blue: vec![blue],
        })
    }

    /// The first sample of each plane, read as U8 or little endian U16.
    fn first_samples(converted: &Converted) -> Vec<u16> {
        let wide = matches!(converted.format.sample, SampleFormat::U16 { .. });

        converted
            .planes
            .iter()
            .map(|plane| {
                if wide {
                    u16::from_le_bytes([plane[0], plane[1]])
                } else {
                    u16::from(plane[0])
                }
            })
            .collect()
    }

    fn converted_first(picture: Picture, target: FrameFormat) -> Vec<u16> {
        let converted = convert(&picture, &target, "solid/fill").expect("converts");

        first_samples(&converted)
    }

    #[test]
    fn white_and_black_at_bt709_limited() {
        let target = yuv444(SampleFormat::U8, ColourRange::Limited, Matrix::Bt709);

        let white = converted_first(solid(1.0, 1.0, 1.0), target);
        let black = converted_first(solid(0.0, 0.0, 0.0), target);

        assert_eq!(white, vec![235, 128, 128]);
        assert_eq!(black, vec![16, 128, 128]);
    }

    #[test]
    fn red_at_bt709_and_bt601_limited() {
        let bt709 = yuv444(SampleFormat::U8, ColourRange::Limited, Matrix::Bt709);
        let bt601 = yuv444(SampleFormat::U8, ColourRange::Limited, Matrix::Bt601);

        let red_709 = converted_first(solid(1.0, 0.0, 0.0), bt709);
        let red_601 = converted_first(solid(1.0, 0.0, 0.0), bt601);

        assert_eq!(red_709, vec![63, 102, 240]);
        assert_eq!(red_601, vec![81, 90, 240]);
    }

    #[test]
    fn ten_bit_red_scales_by_four() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let target = yuv444(ten_bit, ColourRange::Limited, Matrix::Bt709);

        let red = converted_first(solid(1.0, 0.0, 0.0), target);
        let white = converted_first(solid(1.0, 1.0, 1.0), target);

        assert_eq!(red, vec![250, 409, 960]);
        assert_eq!(white, vec![940, 512, 512]);
    }

    #[test]
    fn red_at_bt709_full() {
        let target = yuv444(SampleFormat::U8, ColourRange::Full, Matrix::Bt709);

        let red = converted_first(solid(1.0, 0.0, 0.0), target);

        assert_eq!(red, vec![54, 99, 255]);
    }

    #[test]
    fn an_unspecified_yuv_range_is_limited_and_an_unspecified_matrix_is_bt709() {
        let target = yuv444(SampleFormat::U8, ColourRange::Unspecified, Matrix::Unspecified);

        let red = converted_first(solid(1.0, 0.0, 0.0), target);

        assert_eq!(red, vec![63, 102, 240]);
    }

    #[test]
    fn rgb_to_rgb_keeps_the_channels_in_plane_order() {
        let target = format(2, 2, FrameLayout::Rgb, SampleFormat::U8, ColourInfo::default());

        let orange = converted_first(solid(1.0, 128.0 / 255.0, 0.0), target);

        assert_eq!(orange, vec![255, 128, 0]);
    }

    #[test]
    fn rgb_to_grey_keeps_the_luma() {
        let target = format(2, 2, FrameLayout::Grey, SampleFormat::U8, ColourInfo::default());

        let red = converted_first(solid(1.0, 0.0, 0.0), target);

        assert_eq!(red, vec![54]);
    }

    #[test]
    fn grey_to_yuv_is_neutral() {
        let picture = Picture::Grey(GreyPicture {
            width: 1,
            height: 1,
            luma: vec![1.0],
        });
        let target = yuv444(SampleFormat::U8, ColourRange::Limited, Matrix::Bt709);

        let found = converted_first(picture, target);

        assert_eq!(found, vec![235, 128, 128]);
    }

    #[test]
    fn grey_to_rgb_repeats_the_luma() {
        let picture = Picture::Grey(GreyPicture {
            width: 1,
            height: 1,
            luma: vec![0.5],
        });
        let target = format(2, 2, FrameLayout::Rgb, SampleFormat::U8, ColourInfo::default());

        let found = converted_first(picture, target);

        assert_eq!(found, vec![128, 128, 128]);
    }

    #[test]
    fn a_solid_colour_fills_every_sample_of_every_plane() {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let target = format(4, 2, layout, SampleFormat::U8, ColourInfo::default());

        let converted = convert(&solid(1.0, 0.0, 0.0), &target, "solid/fill").expect("converts");

        assert_eq!(converted.planes[0], vec![63; 8]);
        assert_eq!(converted.planes[1], vec![102; 2]);
        assert_eq!(converted.planes[2], vec![240; 2]);
    }

    /// Converts an 8 by 2 picture, red on the left half and blue on the right, into 4:2:0 at `location`, and
    /// hands back its V plane.
    fn split_red_blue(location: ChromaLocation) -> Vec<u8> {
        let red: Vec<f32> = (0..16)
            .map(|index| if index % 8 < 4 { 1.0 } else { 0.0 })
            .collect();
        let blue: Vec<f32> = red.iter().map(|value| 1.0 - value).collect();
        let picture = Picture::Rgb(RgbPicture {
            width: 8,
            height: 2,
            red,
            green: vec![0.0; 16],
            blue,
        });
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let colour = ColourInfo {
            range: ColourRange::Limited,
            matrix: Matrix::Bt709,
            chroma_location: location,
            ..ColourInfo::default()
        };
        let target = format(8, 2, layout, SampleFormat::U8, colour);

        let converted = convert(&picture, &target, "image/file").expect("converts");

        converted.planes[2].clone()
    }

    #[test]
    fn left_sited_chroma_blends_one_quarter_red_at_the_edge() {
        let v_plane = split_red_blue(ChromaLocation::Left);

        assert_eq!(v_plane, vec![240, 240, 148, 118]);
    }

    #[test]
    fn centre_sited_chroma_blends_one_eighth_red_at_the_edge() {
        let v_plane = split_red_blue(ChromaLocation::Center);

        assert_eq!(v_plane[2], 133);
    }

    #[test]
    fn planes_fill_each_plane_with_their_value() {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let target = format(2, 2, layout, ten_bit, ColourInfo::default());
        let picture = Picture::Planes(vec![64, 512, 1023]);

        let converted = convert(&picture, &target, "solid/fill").expect("converts");

        assert_eq!(converted.planes[0], vec![64, 0, 64, 0, 64, 0, 64, 0]);
        assert_eq!(converted.planes[1], vec![0, 2]);
        assert_eq!(converted.planes[2], vec![255, 3]);
    }

    #[test]
    fn planes_of_the_wrong_length_are_refused() {
        let target = format(2, 2, FrameLayout::Grey, SampleFormat::U8, ColourInfo::default());
        let picture = Picture::Planes(vec![16, 128, 128]);

        let error = check(&picture, &target, "solid/like").expect_err("grey has one plane");

        assert!(error.to_string().contains("solid/like"));
        assert!(error.to_string().contains('1'));
    }

    #[test]
    fn a_plane_value_past_the_depth_is_refused() {
        let target = format(2, 2, FrameLayout::Grey, SampleFormat::U8, ColourInfo::default());
        let picture = Picture::Planes(vec![1000]);

        let error = check(&picture, &target, "solid/fill").expect_err("1000 is past 255");

        assert!(error.to_string().contains("255"));
    }

    #[test]
    fn a_picture_of_another_size_is_refused() {
        let picture = Picture::Grey(GreyPicture {
            width: 3,
            height: 3,
            luma: vec![0.0; 9],
        });
        let target = format(4, 4, FrameLayout::Grey, SampleFormat::U8, ColourInfo::default());

        let error = check(&picture, &target, "image/like").expect_err("3x3 is not 4x4");

        assert!(error.to_string().contains("3x3"));
        assert!(error.to_string().contains("4x4"));
        assert!(error.to_string().contains("resize/"));
    }

    #[test]
    fn a_float_target_is_refused() {
        let target = format(2, 2, FrameLayout::Grey, SampleFormat::F32, ColourInfo::default());

        let refused = check(&solid(0.0, 0.0, 0.0), &target, "solid/like");

        assert!(refused.is_err());
    }
}
