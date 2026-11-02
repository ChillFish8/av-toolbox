mod pattern;
#[cfg(test)]
mod tests;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

pub use self::pattern::{Shift, Thresholds, bayer_ranks, blue_noise_ranks, plane_shift};
use crate::matrix::{MatrixSamples, SampleRange, scaling};
use crate::{DevicePlane, Dispatch};

/// How a plane's samples are stored.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Element {
    U8,
    U16,
}

impl Element {
    fn for_bits(bits: u32) -> Self {
        if bits == 8 { Self::U8 } else { Self::U16 }
    }
}

/// How the samples a conversion reads are stored.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Source {
    U8,
    U16,
    F32,
}

impl From<Element> for Source {
    fn from(element: Element) -> Self {
        match element {
            Element::U8 => Self::U8,
            Element::U16 => Self::U16,
        }
    }
}

/// How the samples a conversion writes are stored.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Destination {
    U8,
    U16,
    F32,
}

impl From<Element> for Destination {
    fn from(element: Element) -> Self {
        match element {
            Element::U8 => Self::U8,
            Element::U16 => Self::U16,
        }
    }
}

/// Everything one change of depth needs besides its rounding.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Conversion {
    pub input: Source,
    pub output: Destination,
    /// What each input sample is multiplied by.
    pub multiplier: f32,
    /// What is added to each sample after multiplying.
    pub offset: f32,
    /// What each sample is divided by after the offset. It is 1 unless the output is `F32`.
    pub divisor: f32,
    /// The largest output sample.
    pub largest: f32,
}

impl Conversion {
    /// A change from `input_bits` to `output_bits` integers in `range`.
    ///
    /// Full range stretches so the largest input becomes the largest output, with `chroma` staying
    /// centred on the middle value. Limited range shifts, which keeps black and white where video
    /// expects them.
    pub fn new(input_bits: u32, output_bits: u32, range: SampleRange, chroma: bool) -> Self {
        let input = Element::for_bits(input_bits);
        let output = Element::for_bits(output_bits);

        let input_samples = integer_samples(input, input_bits);
        let output_samples = integer_samples(output, output_bits);
        let (input_offset, input_scale) = scaling(input_samples, range, chroma);
        let (output_offset, output_scale) = scaling(output_samples, range, chroma);

        let multiplier = output_scale / input_scale;
        let offset = output_offset - input_offset * multiplier;
        let output_peak = ((1u32 << output_bits) - 1) as f32;

        Self {
            input: Source::from(input),
            output: Destination::from(output),
            multiplier: multiplier as f32,
            offset: offset as f32,
            divisor: 1.0,
            largest: output_peak,
        }
    }

    /// A change from full range float samples to `output_bits` integers in `range`.
    ///
    /// Float samples run between 0 and 1, or between -0.5 and 0.5 for `chroma`, which is centred on
    /// the middle output value. A NaN sample reads as 0.
    pub fn from_float(output_bits: u32, range: SampleRange, chroma: bool) -> Self {
        let element = Element::for_bits(output_bits);
        let samples = integer_samples(element, output_bits);
        let (offset, scale) = scaling(samples, range, chroma);
        let output_peak = ((1u32 << output_bits) - 1) as f32;

        Self {
            input: Source::F32,
            output: Destination::from(element),
            multiplier: scale as f32,
            offset: offset as f32,
            divisor: 1.0,
            largest: output_peak,
        }
    }

    /// A change from `input_bits` integers in `range` to full range float samples.
    ///
    /// Floats run between 0 and 1, or between -0.5 and 0.5 for `chroma`. The result is neither
    /// rounded nor clamped. Dividing by the range's span, rather than multiplying by its
    /// inverse, keeps black and white on 0 and 1 to within f32 rounding.
    pub fn to_float(input_bits: u32, range: SampleRange, chroma: bool) -> Self {
        let element = Element::for_bits(input_bits);
        let samples = integer_samples(element, input_bits);
        let (offset, scale) = scaling(samples, range, chroma);

        Self {
            input: Source::from(element),
            output: Destination::F32,
            multiplier: 1.0,
            offset: -offset as f32,
            divisor: scale as f32,
            largest: 1.0,
        }
    }
}

fn integer_samples(element: Element, bits: u32) -> MatrixSamples {
    match element {
        Element::U8 => MatrixSamples::U8,
        Element::U16 => MatrixSamples::U16 { bits },
    }
}

/// Converts one sample on the CPU, exactly as the kernel does.
pub fn convert_sample(value: f32, conversion: &Conversion, threshold: f32) -> u16 {
    let defined = if value.is_nan() { 0.0 } else { value };
    let scaled = defined * conversion.multiplier + conversion.offset;
    let rounded = (scaled + threshold).floor();

    rounded.clamp(0.0, conversion.largest) as u16
}

/// Converts one sample to a float on the CPU, exactly as the kernel does.
pub fn convert_to_float_sample(value: f32, conversion: &Conversion) -> f32 {
    (value * conversion.multiplier + conversion.offset) / conversion.divisor
}

/// A threshold grid on the device.
#[derive(Debug)]
pub struct Pattern {
    values: Handle,
    size: u32,
}

impl Pattern {
    fn values(&self) -> BufferArg {
        let handle = self.values.clone();
        let len = (self.size * self.size) as usize;

        // Safety: the buffer was made from exactly `size * size` f32 values.
        unsafe { BufferArg::from_raw_parts(handle, len) }
    }
}

pub fn upload_pattern(client: &Client, thresholds: &Thresholds) -> Pattern {
    let bytes = f32::as_bytes(&thresholds.values);
    let values = client.create_from_slice(bytes);

    Pattern {
        values,
        size: thresholds.size,
    }
}

/// The plane one conversion reads and the plane it writes, both the same size.
#[derive(Debug, Clone, Copy)]
pub struct DepthPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn convert_depth<I: Numeric, O: Int>(
    input: &[I],
    pattern: &[f32],
    output: &mut [O],
    width: u32,
    size: u32,
    across: u32,
    down: u32,
    multiplier: f32,
    offset: f32,
    largest: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;
    let pattern_column = (column + across) % size;
    let pattern_row = (row + down) % size;
    let threshold = pattern[(pattern_row * size + pattern_column) as usize];

    let value = f32::cast_from(input[ABSOLUTE_POS]);
    let defined = select(f32::is_nan(value), 0.0, value);
    let scaled = defined * multiplier + offset;
    let rounded = f32::floor(scaled + threshold);
    output[ABSOLUTE_POS] = O::cast_from(f32::clamp(rounded, 0.0, largest));
}

#[cube(launch_unchecked)]
fn convert_to_float<I: Numeric>(input: &[I], output: &mut [f32], multiplier: f32, offset: f32, divisor: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let value = f32::cast_from(input[ABSOLUTE_POS]);
    output[ABSOLUTE_POS] = (value * multiplier + offset) / divisor;
}

/// Converts `planes.input` into `planes.output`, rounding through `pattern` moved by `shift`.
pub fn convert(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: DepthPlanes<'_>,
    conversion: &Conversion,
    pattern: &Pattern,
    shift: Shift,
) {
    match (conversion.input, conversion.output) {
        (Source::U8, Destination::U8) => {
            launch::<u8, u8>(client, dispatch, planes, conversion, pattern, shift)
        },
        (Source::U8, Destination::U16) => {
            launch::<u8, u16>(client, dispatch, planes, conversion, pattern, shift)
        },
        (Source::U16, Destination::U8) => {
            launch::<u16, u8>(client, dispatch, planes, conversion, pattern, shift)
        },
        (Source::U16, Destination::U16) => {
            launch::<u16, u16>(client, dispatch, planes, conversion, pattern, shift)
        },
        (Source::F32, Destination::U8) => {
            launch::<f32, u8>(client, dispatch, planes, conversion, pattern, shift)
        },
        (Source::F32, Destination::U16) => {
            launch::<f32, u16>(client, dispatch, planes, conversion, pattern, shift)
        },
        (Source::U8, Destination::F32) => launch_to_float::<u8>(client, dispatch, planes, conversion),
        (Source::U16, Destination::F32) => launch_to_float::<u16>(client, dispatch, planes, conversion),
        (Source::F32, Destination::F32) => launch_to_float::<f32>(client, dispatch, planes, conversion),
    }
}

fn launch_to_float<I: Numeric>(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: DepthPlanes<'_>,
    conversion: &Conversion,
) {
    let input = planes.input.array();
    let output = planes.output.array();
    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and output hold the same number of samples of `I` and `f32`.
    unsafe {
        convert_to_float::launch_unchecked::<I>(
            client,
            cubes,
            unit,
            input,
            output,
            conversion.multiplier,
            conversion.offset,
            conversion.divisor,
        );
    }
}

fn launch<I: Numeric, O: Int>(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: DepthPlanes<'_>,
    conversion: &Conversion,
    pattern: &Pattern,
    shift: Shift,
) {
    let width = planes.input.width();
    let input = planes.input.array();
    let values = pattern.values();
    let output = planes.output.array();
    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and output hold `width * height` samples of `I` and `O`, and every pattern
    // index is wrapped inside its `size * size` values.
    unsafe {
        convert_depth::launch_unchecked::<I, O>(
            client,
            cubes,
            unit,
            input,
            values,
            output,
            width,
            pattern.size,
            shift.across,
            shift.down,
            conversion.multiplier,
            conversion.offset,
            conversion.largest,
        );
    }
}
