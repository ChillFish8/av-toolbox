// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::{DevicePlane, Dispatch};

/// The weights a YUV matrix gives red and blue in luma.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Coefficients {
    Bt709,
    Bt601,
    Bt2020Ncl,
}

impl Coefficients {
    /// `Kr` and `Kb`. Green's weight is what is left of one.
    pub fn weights(self) -> (f64, f64) {
        match self {
            Self::Bt709 => (0.2126, 0.0722),
            Self::Bt601 => (0.299, 0.114),
            Self::Bt2020Ncl => (0.2627, 0.0593),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Direction {
    ToRgb,
    ToYuv,
}

/// Whether integer samples use their depth's full range or the narrower video range.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SampleRange {
    Limited,
    Full,
}

/// How a plane's samples are stored.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MatrixSamples {
    U8,
    U16 { bits: u32 },
    F32,
}

impl MatrixSamples {
    pub(crate) fn bits(self) -> Option<u32> {
        match self {
            Self::U8 => Some(8),
            Self::U16 { bits } => Some(bits),
            Self::F32 => None,
        }
    }
}

fn assert_supported_depth(bits: u32) {
    assert!(
        (8..=16).contains(&bits),
        "matrix samples must be 8 to 16 bits deep, not {bits}"
    );
}

/// The code value of zero and the code value span of one for a plane, as `(offset, scale)`.
///
/// Float samples are always full range, with chroma already centred on zero, so they take `(0, 1)`.
pub fn scaling(samples: MatrixSamples, range: SampleRange, chroma: bool) -> (f64, f64) {
    let Some(bits) = samples.bits() else {
        return (0.0, 1.0);
    };

    assert_supported_depth(bits);

    let step = f64::from(1u32 << (bits - 8));
    let peak = f64::from((1u32 << bits) - 1);
    let half = f64::from(1u32 << (bits - 1));

    match (range, chroma) {
        (SampleRange::Limited, false) => (16.0 * step, 219.0 * step),
        (SampleRange::Limited, true) => (128.0 * step, 224.0 * step),
        (SampleRange::Full, false) => (0.0, peak),
        (SampleRange::Full, true) => (half, peak),
    }
}

/// One conversion between 4:4:4 YUV and RGB.
///
/// The three planes read share `input_samples` and the three written share `output_samples`.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatrixConversion {
    pub coefficients: Coefficients,
    pub direction: Direction,
    pub input_samples: MatrixSamples,
    pub output_samples: MatrixSamples,
    pub input_range: SampleRange,
    pub output_range: SampleRange,
}

const YUV_CHROMA: [bool; 3] = [false, true, true];

const RGB_CHROMA: [bool; 3] = [false, false, false];

impl MatrixConversion {
    /// Which input planes, in channel order, are chroma.
    pub fn input_chroma(&self) -> [bool; 3] {
        match self.direction {
            Direction::ToRgb => YUV_CHROMA,
            Direction::ToYuv => RGB_CHROMA,
        }
    }

    /// Which output planes, in channel order, are chroma.
    pub fn output_chroma(&self) -> [bool; 3] {
        match self.direction {
            Direction::ToRgb => RGB_CHROMA,
            Direction::ToYuv => YUV_CHROMA,
        }
    }

    /// The conversion as one affine map from input code values to output code values.
    pub fn affine(&self) -> Affine {
        let normalised = normalised_matrix(self.coefficients, self.direction);
        let input_chroma = self.input_chroma();
        let output_chroma = self.output_chroma();

        let mut weights = [[0.0; 3]; 3];
        let mut offsets = [0.0; 3];
        for output in 0..3 {
            let (output_offset, output_scale) =
                scaling(self.output_samples, self.output_range, output_chroma[output]);
            let mut shift = 0.0;

            for input in 0..3 {
                let (input_offset, input_scale) =
                    scaling(self.input_samples, self.input_range, input_chroma[input]);
                let weight = output_scale * normalised[output][input] / input_scale;

                weights[output][input] = weight;
                shift += weight * input_offset;
            }

            offsets[output] = output_offset - shift;
        }

        Affine { weights, offsets }
    }
}

/// The matrix between normalised YUV, chroma centred on zero, and normalised RGB.
fn normalised_matrix(coefficients: Coefficients, direction: Direction) -> [[f64; 3]; 3] {
    let (red, blue) = coefficients.weights();
    let green = 1.0 - red - blue;

    match direction {
        Direction::ToRgb => [
            [1.0, 0.0, 2.0 * (1.0 - red)],
            [
                1.0,
                -2.0 * blue * (1.0 - blue) / green,
                -2.0 * red * (1.0 - red) / green,
            ],
            [1.0, 2.0 * (1.0 - blue), 0.0],
        ],
        Direction::ToYuv => [
            [red, green, blue],
            [-red / (2.0 * (1.0 - blue)), -green / (2.0 * (1.0 - blue)), 0.5],
            [0.5, -green / (2.0 * (1.0 - red)), -blue / (2.0 * (1.0 - red))],
        ],
    }
}

/// `output[k] = sum of weights[k][j] * input[j], plus offsets[k]`, on code values.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Affine {
    pub weights: [[f64; 3]; 3],
    pub offsets: [f64; 3],
}

/// An affine map on the device, as nine weights by row and then three offsets.
#[derive(Debug)]
pub struct AffineValues {
    values: Handle,
}

const AFFINE_VALUES: usize = 12;

impl AffineValues {
    fn values<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.values.clone();

        // Safety: the buffer was made from exactly `AFFINE_VALUES` f32 values.
        unsafe { ArrayArg::from_raw_parts(handle, AFFINE_VALUES) }
    }
}

pub fn upload_affine<R: Runtime>(client: &ComputeClient<R>, affine: &Affine) -> AffineValues {
    let weights = affine.weights.iter().flatten();
    let all = weights.chain(affine.offsets.iter());
    let values: Vec<f32> = all.map(|value| *value as f32).collect();
    let bytes = f32::as_bytes(&values);
    let handle = client.create_from_slice(bytes);

    AffineValues { values: handle }
}

/// The three planes one conversion reads and the three it writes, all the same size, in channel order.
#[derive(Debug, Clone, Copy)]
pub struct MatrixPlanes<'a> {
    pub inputs: [DevicePlane<'a>; 3],
    pub outputs: [DevicePlane<'a>; 3],
}

#[cube]
fn affine_row(affine: &Array<f32>, row: u32, first: f32, second: f32, third: f32) -> f32 {
    let start = (row * 3) as usize;
    let weighted = affine[start] * first + affine[start + 1] * second + affine[start + 2] * third;

    weighted + affine[(9 + row) as usize]
}

#[cube]
fn finished(value: f32, largest: f32, #[comptime] rounds: bool) -> f32 {
    let mut result = value;

    if rounds {
        let rounded = f32::floor(value + 0.5);
        result = f32::clamp(rounded, 0.0, largest);
    }

    result
}

#[cube(launch_unchecked)]
fn matrix_kernel<I: CubePrimitive, O: CubePrimitive>(
    first: &Array<I>,
    second: &Array<I>,
    third: &Array<I>,
    affine: &Array<f32>,
    first_out: &mut Array<O>,
    second_out: &mut Array<O>,
    third_out: &mut Array<O>,
    largest: f32,
    #[comptime] rounds: bool,
) {
    if ABSOLUTE_POS >= first_out.len() {
        terminate!();
    }

    let first_value = f32::cast_from(first[ABSOLUTE_POS]);
    let second_value = f32::cast_from(second[ABSOLUTE_POS]);
    let third_value = f32::cast_from(third[ABSOLUTE_POS]);

    let first_mapped = affine_row(affine, 0, first_value, second_value, third_value);
    let second_mapped = affine_row(affine, 1, first_value, second_value, third_value);
    let third_mapped = affine_row(affine, 2, first_value, second_value, third_value);

    first_out[ABSOLUTE_POS] = O::cast_from(finished(first_mapped, largest, rounds));
    second_out[ABSOLUTE_POS] = O::cast_from(finished(second_mapped, largest, rounds));
    third_out[ABSOLUTE_POS] = O::cast_from(finished(third_mapped, largest, rounds));
}

/// Converts `planes.inputs`, held as `conversion.input_samples`, into `planes.outputs`, held as
/// `conversion.output_samples`, through `values`.
///
/// Integer outputs are rounded to nearest and clamped to their depth. Float outputs are written as they come, so
/// values outside the gamut survive.
pub fn convert<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MatrixPlanes<'_>,
    conversion: &MatrixConversion,
    values: &AffineValues,
) {
    let width = planes.outputs[0].width();
    let height = planes.outputs[0].height();
    let every_plane = planes.inputs.iter().chain(planes.outputs.iter());
    for plane in every_plane {
        assert!(
            plane.width() == width && plane.height() == height,
            "every plane of a matrix conversion must be {width}x{height}"
        );
    }

    let output = conversion.output_samples;
    match conversion.input_samples {
        MatrixSamples::U8 => launch_into::<u8, R>(client, dispatch, planes, output, values),
        MatrixSamples::U16 { bits } => {
            assert_supported_depth(bits);

            launch_into::<u16, R>(client, dispatch, planes, output, values)
        },
        MatrixSamples::F32 => launch_into::<f32, R>(client, dispatch, planes, output, values),
    }
}

/// Launches the kernel reading `I` and writing samples held as `output`.
fn launch_into<I: CubePrimitive, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MatrixPlanes<'_>,
    output: MatrixSamples,
    values: &AffineValues,
) {
    match output {
        MatrixSamples::U8 => launch::<I, u8, R>(client, dispatch, planes, values, 255.0, true),
        MatrixSamples::U16 { bits } => {
            assert_supported_depth(bits);

            let largest = ((1u32 << bits) - 1) as f32;

            launch::<I, u16, R>(client, dispatch, planes, values, largest, true)
        },
        MatrixSamples::F32 => launch::<I, f32, R>(client, dispatch, planes, values, 0.0, false),
    }
}

fn launch<I: CubePrimitive, O: CubePrimitive, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MatrixPlanes<'_>,
    values: &AffineValues,
    largest: f32,
    rounds: bool,
) {
    let [first, second, third] = planes.inputs.map(|plane| plane.array::<R>());
    let [first_out, second_out, third_out] = planes.outputs.map(|plane| plane.array::<R>());
    let affine = values.values::<R>();
    let samples = planes.outputs[0].samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane is the same size, checked by `convert`. Each input holds `width * height` samples of
    // `I` and each output as many of `O`, which `DevicePlane::new` requires. The affine buffer holds its 12 values.
    unsafe {
        matrix_kernel::launch_unchecked::<I, O, R>(
            client, cubes, unit, first, second, third, affine, first_out, second_out, third_out, largest,
            rounds,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn applied(affine: &Affine, input: [f64; 3]) -> [f64; 3] {
        let mut output = [0.0; 3];

        for (row, value) in output.iter_mut().enumerate() {
            let weighted: f64 = (0..3)
                .map(|column| affine.weights[row][column] * input[column])
                .sum();
            *value = weighted + affine.offsets[row];
        }

        output
    }

    fn conversion(direction: Direction, samples: MatrixSamples, range: SampleRange) -> MatrixConversion {
        let (input_range, output_range) = match direction {
            Direction::ToRgb => (range, SampleRange::Full),
            Direction::ToYuv => (SampleRange::Full, range),
        };

        MatrixConversion {
            coefficients: Coefficients::Bt709,
            direction,
            input_samples: samples,
            output_samples: samples,
            input_range,
            output_range,
        }
    }

    fn assert_close(found: [f64; 3], expected: [f64; 3]) {
        for (value, wanted) in found.iter().zip(expected) {
            assert!((value - wanted).abs() < 1e-9, "{found:?} != {expected:?}");
        }
    }

    #[test]
    fn eight_bit_limited_black_and_white_map_to_full_range() {
        let samples = MatrixSamples::U8;
        let affine = conversion(Direction::ToRgb, samples, SampleRange::Limited).affine();

        let black = applied(&affine, [16.0, 128.0, 128.0]);
        let white = applied(&affine, [235.0, 128.0, 128.0]);

        assert_close(black, [0.0; 3]);
        assert_close(white, [255.0; 3]);
    }

    #[test]
    fn a_ten_bit_affine_maps_ten_bit_black_and_white() {
        let samples = MatrixSamples::U16 { bits: 10 };
        let affine = conversion(Direction::ToRgb, samples, SampleRange::Limited).affine();

        let black = applied(&affine, [64.0, 512.0, 512.0]);
        let white = applied(&affine, [940.0, 512.0, 512.0]);

        assert_close(black, [0.0; 3]);
        assert_close(white, [1023.0; 3]);
    }

    #[test]
    fn an_eight_bit_affine_maps_limited_black_and_white_onto_float() {
        let conversion = MatrixConversion {
            output_samples: MatrixSamples::F32,
            ..conversion(Direction::ToRgb, MatrixSamples::U8, SampleRange::Limited)
        };
        let affine = conversion.affine();

        let black = applied(&affine, [16.0, 128.0, 128.0]);
        let white = applied(&affine, [235.0, 128.0, 128.0]);

        assert_close(black, [0.0; 3]);
        assert_close(white, [1.0; 3]);
    }

    #[test]
    fn to_rgb_then_to_yuv_is_the_identity_in_float() {
        let samples = MatrixSamples::F32;
        let forward = conversion(Direction::ToRgb, samples, SampleRange::Full).affine();
        let backward = conversion(Direction::ToYuv, samples, SampleRange::Full).affine();
        let input = [0.4, -0.2, 0.3];

        let rgb = applied(&forward, input);
        let back = applied(&backward, rgb);

        assert_close(back, input);
    }

    #[test]
    fn full_range_chroma_is_centred_on_half_of_its_depth() {
        let scaled = scaling(MatrixSamples::U16 { bits: 16 }, SampleRange::Full, true);

        assert_eq!(scaled, (32768.0, 65535.0));
    }
}
