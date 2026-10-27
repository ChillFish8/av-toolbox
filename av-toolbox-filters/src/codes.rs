use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_cubek::values::{FloatCodes, codes_to_float, float_to_codes, narrow, widen};
use av_toolbox_cubek::{DevicePlane, Dispatch};
use av_toolbox_gpu::{Channel, ColourRange, ComputeClient, FrameFormat, Runtime, SampleFormat};

use crate::bits::bits_of;

/// The depth of the code values an F32 clip maps onto.
pub(crate) const FLOAT_CODE_BITS: u32 = 16;

/// The integer range a clip labelled `range` uses. Anything not labelled full range is limited.
pub(crate) fn sample_range(range: ColourRange) -> SampleRange {
    match range {
        ColourRange::Full => SampleRange::Full,
        _ => SampleRange::Limited,
    }
}

/// How the samples of a plane are held in memory.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the runners using code values arrive in later changes")
)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Storage {
    Integer(Element),
    Float,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the runners using code values arrive in later changes")
)]
impl Storage {
    fn of(sample: SampleFormat) -> Self {
        match sample {
            SampleFormat::F32 => Self::Float,
            SampleFormat::U8 => Self::Integer(Element::U8),
            SampleFormat::U16 { .. } => Self::Integer(Element::U16),
        }
    }
}

/// Runs integer code value kernels on integer and F32 clips alike.
///
/// Integer samples widen to their codes. F32 samples map onto 16-bit codes in the range the clip is
/// labelled with, and map back when stored.
#[expect(dead_code, reason = "the runners using code values arrive in later changes")]
#[derive(Debug, Copy, Clone)]
pub(crate) struct CodeValues {
    input: Storage,
    output: Storage,
    output_sample: SampleFormat,
    bits: u32,
    range: SampleRange,
}

#[expect(dead_code, reason = "the runners using code values arrive in later changes")]
impl CodeValues {
    /// Code values for `input`, stored as `output`, which is the input's sample or F32.
    ///
    /// # Panics
    ///
    /// Panics when `output` is neither the input's sample nor F32.
    pub(crate) fn new(input: &FrameFormat, output: SampleFormat) -> Self {
        let kept = output == input.sample;
        let floated = output == SampleFormat::F32;
        assert!(
            kept || floated,
            "code values of {:?} samples are stored as the same samples or F32, not {output:?}",
            input.sample,
        );

        Self {
            input: Storage::of(input.sample),
            output: Storage::of(output),
            output_sample: output,
            bits: bits_of(input.sample),
            range: sample_range(input.colour.range),
        }
    }

    /// The depth of the code values.
    pub(crate) fn bits(&self) -> u32 {
        self.bits
    }

    /// The largest code value.
    pub(crate) fn peak(&self) -> f32 {
        ((1u32 << self.bits) - 1) as f32
    }

    /// The sample format results are stored as.
    pub(crate) fn output_sample(&self) -> SampleFormat {
        self.output_sample
    }

    /// Widens `source` into f32 code values held in `values`.
    pub(crate) fn widen<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        source: DevicePlane<'_>,
        values: DevicePlane<'_>,
        channel: Channel,
    ) {
        match self.input {
            Storage::Integer(element) => widen(client, dispatch, source, values, element),
            Storage::Float => {
                let codes = self.float_codes(channel);
                float_to_codes(client, dispatch, source, values, codes);
            },
        }
    }

    /// Stores f32 code values from `values` into `target`.
    ///
    /// Integers are rounded and clamped. F32 samples are neither.
    pub(crate) fn store<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        values: DevicePlane<'_>,
        target: DevicePlane<'_>,
        channel: Channel,
    ) {
        match self.output {
            Storage::Integer(element) => narrow(client, dispatch, values, target, element, self.peak()),
            Storage::Float => {
                let codes = self.float_codes(channel);
                codes_to_float(client, dispatch, values, target, codes);
            },
        }
    }

    fn float_codes(&self, channel: Channel) -> FloatCodes {
        let chroma = matches!(channel, Channel::U | Channel::V);

        FloatCodes::new(self.bits, self.range, chroma)
    }
}
