use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_cubek::values::{FloatCodes, float_to_whole_codes};
use av_toolbox_gpu::{
    Channel,
    ColourRange,
    FormatSet,
    Frame,
    FrameFormat,
    FrameLayout,
    FrameRate,
    MaskFormat,
    OutputFormats,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::sizing::Sizing;

/// Takes the luma of YUV and Grey frames at any sample format.
pub(crate) fn accepts_luma(stage: &str, offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        let message = format!("{stage} reads frames, not masks");
        return Err(StageError::msg(message));
    };

    if format.layout == FrameLayout::Rgb {
        let message = format!("{stage} reads YUV or Grey frames, not RGB");
        return Err(StageError::msg(message));
    }

    if let FrameLayout::Chroma { .. } = format.layout {
        let message = format!("{stage} reads YUV or Grey frames, not chroma planes");
        return Err(StageError::msg(message));
    }

    Ok(FormatSet::of([format.sample]))
}

fn bound_frames(stage: &str, format: Option<StreamFormat>) -> Result<FrameFormat, StageError> {
    let Some(format) = format else {
        let message = format!("{stage} needs its input bound");
        return Err(StageError::msg(message));
    };

    let Some(frames) = format.as_frames() else {
        let message = format!("{stage} reads frames, not masks");
        return Err(StageError::msg(message));
    };

    Ok(*frames)
}

/// A U8 mask the size of the input's luma.
pub(crate) fn produces_luma_mask(
    stage: &str,
    chosen: &[Option<StreamFormat>],
) -> Result<OutputFormats, StageError> {
    let first = chosen.first().copied().flatten();
    let format = bound_frames(stage, first)?;
    let mask = MaskFormat {
        frame_rate: format.frame_rate,
        width: format.width,
        height: format.height,
        sample: SampleFormat::U8,
        chroma: None,
    };

    Ok(OutputFormats {
        template: StreamTemplate::Mask(mask),
        samples: FormatSet::of([SampleFormat::U8]),
    })
}

/// The depth F32 luma is read at.
const FLOAT_CODE_BITS: u32 = 16;

/// How the luma a stage reads is stored.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Luma {
    pub(crate) width: u32,
    pub(crate) height: u32,
    frame_rate: Option<FrameRate>,
    /// The largest sample value, which maps to 1.
    largest: f32,
    wide: bool,
    /// How F32 samples map onto 16-bit codes, set for float clips only.
    float_codes: Option<FloatCodes>,
}

/// The integer range a clip labelled `range` uses. Anything not labelled full range is limited.
fn sample_range(range: ColourRange) -> SampleRange {
    match range {
        ColourRange::Full => SampleRange::Full,
        _ => SampleRange::Limited,
    }
}

impl Luma {
    pub(crate) fn from_formats(stage: &str, formats: &ResolvedFormats) -> Result<Self, StageError> {
        let input = formats.input(0);
        let format = bound_frames(stage, input)?;

        let (largest, wide, float_codes) = match format.sample {
            SampleFormat::U8 => (255.0, false, None),
            SampleFormat::U16 { depth } => {
                let max_value = depth.max_value();

                (f32::from(max_value), true, None)
            },
            SampleFormat::F32 => {
                let range = sample_range(format.colour.range);
                let codes = FloatCodes::new(FLOAT_CODE_BITS, range, false);

                (codes.peak, false, Some(codes))
            },
        };

        Ok(Self {
            width: format.width,
            height: format.height,
            frame_rate: format.frame_rate,
            largest,
            wide,
            float_codes,
        })
    }

    pub(crate) const fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub(crate) const fn mask_format(&self) -> MaskFormat {
        MaskFormat {
            frame_rate: self.frame_rate,
            width: self.width,
            height: self.height,
            sample: SampleFormat::U8,
            chroma: None,
        }
    }

    /// The largest code value, 255, the U16 depth's largest, or 65535 for floats.
    pub(crate) const fn peak(&self) -> f32 {
        self.largest
    }
}

#[cube(launch_unchecked)]
fn normalise<E: Int>(input: &Array<E>, output: &mut Array<f32>, scale: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = f32::cast_from(input[ABSOLUTE_POS]) * scale;
}

#[cube(launch_unchecked)]
fn scale_kernel(values: &mut Array<f32>, scale: f32) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    values[ABSOLUTE_POS] = values[ABSOLUTE_POS] * scale;
}

fn no_luma(stage: &str) -> StageError {
    let message = format!("{stage} was handed a frame without luma");

    StageError::msg(message)
}

fn scale_codes<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    values: DevicePlane<'_>,
    scale: f32,
) {
    if scale == 1.0 {
        return;
    }

    let written = values.array::<R>();
    let cubes = sizing.0.cube_count(values.samples());
    let shape = sizing.0.cube_dim();

    // Safety: `values` holds `samples` f32 values and the kernel stays inside its length.
    unsafe {
        scale_kernel::launch_unchecked::<R>(client, cubes, shape, written, scale);
    }
}

fn load_luma_scaled<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    stage: &str,
    frame: &Frame,
    luma: &Luma,
    output: DevicePlane<'_>,
    scale: f32,
) -> Result<(), StageError> {
    let plane = frame.plane(Channel::Y).ok_or_else(|| no_luma(stage))?;

    if let Some(codes) = luma.float_codes {
        let handle = plane.buffer().handle();

        // Safety: an F32 luma plane holds exactly `width * height` f32 samples.
        let source = unsafe { DevicePlane::new(handle, luma.width, luma.height) };
        float_to_whole_codes(client, sizing, source, output, codes);
        scale_codes(client, sizing, output, scale);

        return Ok(());
    }

    let written = output.array::<R>();
    let samples = luma.samples();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: the luma plane holds `samples` values of the declared width, `output` holds
    // `samples` f32 values, and the kernel writes nothing past the output's length.
    unsafe {
        if luma.wide {
            let read = plane.array::<u16, R>();
            normalise::launch_unchecked::<u16, R>(client, cubes, shape, read, written, scale);
        } else {
            let read = plane.array::<u8, R>();
            normalise::launch_unchecked::<u8, R>(client, cubes, shape, read, written, scale);
        }
    }

    Ok(())
}

/// Writes the frame's luma into `output` as f32 between 0 and 1.
pub(crate) fn load_luma<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    stage: &str,
    frame: &Frame,
    luma: &Luma,
    output: DevicePlane<'_>,
) -> Result<(), StageError> {
    let scale = 1.0 / luma.largest;

    load_luma_scaled(client, sizing, stage, frame, luma, output, scale)
}

/// Writes the frame's luma into `output` as f32 code values, unscaled.
pub(crate) fn load_luma_code_values<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    stage: &str,
    frame: &Frame,
    luma: &Luma,
    output: DevicePlane<'_>,
) -> Result<(), StageError> {
    load_luma_scaled(client, sizing, stage, frame, luma, output, 1.0)
}

#[cube(launch_unchecked)]
fn store_mask_kernel(input: &Array<f32>, output: &mut Array<u8>, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let clamped = f32::clamp(input[ABSOLUTE_POS], 0.0, peak);
    let scaled = f32::round(clamped * 255.0 / peak);

    output[ABSOLUTE_POS] = u8::cast_from(scaled);
}

/// Writes `round(clamp(v, 0, peak) * 255 / peak)` from `input` into `output` as u8.
pub(crate) fn store_mask<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    peak: f32,
) {
    let samples = output.samples();
    let read = input.array::<R>();
    let written = output.array::<R>();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: `input` holds `samples` f32 values and `output` holds `samples` u8 values.
    unsafe {
        store_mask_kernel::launch_unchecked::<R>(client, cubes, shape, read, written, peak);
    }
}

/// The code value a `fraction` of `peak` binarizes at, or `None` when `fraction` is 0 and binarizing
/// is skipped.
pub(crate) fn binarize_threshold(fraction: f32, peak: f32) -> Option<f32> {
    if fraction <= 0.0 {
        return None;
    }

    Some((fraction * peak).round())
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, FrameRate, Subsampling};

    use super::*;

    fn frames(layout: FrameLayout, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width: 9,
            height: 7,
            layout,
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        })
    }

    #[test]
    fn rgb_is_refused_naming_the_stage() {
        let offered = frames(FrameLayout::Rgb, SampleFormat::U8);

        let error = accepts_luma("lines/test", offered).expect_err("rgb has no luma");

        assert!(error.to_string().contains("lines/test reads YUV or Grey"));
    }

    #[test]
    fn a_chroma_clip_is_refused() {
        let layout = FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        };
        let offered = frames(layout, SampleFormat::U8);

        let refused = accepts_luma("luma/test", offered);

        assert!(refused.is_err());
    }

    #[test]
    fn float_samples_are_accepted() {
        let offered = frames(FrameLayout::Grey, SampleFormat::F32);

        let accepted = accepts_luma("lines/test", offered).expect("grey is readable");

        assert_eq!(accepted, FormatSet::of([SampleFormat::F32]));
    }

    #[test]
    fn a_float_clip_reads_at_sixteen_bits() {
        let resolved = ResolvedFormats::new(vec![Some(frames(FrameLayout::Grey, SampleFormat::F32))], None);

        let luma = Luma::from_formats("lines/test", &resolved).expect("floats are readable");

        assert_eq!(luma.peak(), 65535.0);
    }

    #[test]
    fn the_mask_is_the_size_of_the_luma() {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let chosen = [Some(frames(layout, SampleFormat::U8))];

        let produced = produces_luma_mask("lines/test", &chosen).expect("yuv is readable");

        let expected = MaskFormat {
            frame_rate: None,
            width: 9,
            height: 7,
            sample: SampleFormat::U8,
            chroma: None,
        };
        assert_eq!(produced.template, StreamTemplate::Mask(expected));
    }

    fn rated_frames(frame_rate: Option<FrameRate>) -> StreamFormat {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let StreamFormat::Frames(format) = frames(layout, SampleFormat::U8) else {
            unreachable!("frames builds frames");
        };

        StreamFormat::Frames(FrameFormat { frame_rate, ..format })
    }

    #[test]
    fn a_luma_mask_runs_at_its_frames_rate() {
        let rate = FrameRate::new(24000, 1001);
        let chosen = [Some(rated_frames(rate))];

        let produced = produces_luma_mask("lines/test", &chosen).expect("yuv is readable");

        let StreamTemplate::Mask(mask) = produced.template else {
            panic!("expected a mask template");
        };
        assert_eq!(mask.frame_rate, rate);
    }

    #[test]
    fn a_luma_mask_of_rateless_frames_has_no_rate() {
        let chosen = [Some(rated_frames(None))];

        let produced = produces_luma_mask("lines/test", &chosen).expect("yuv is readable");

        let StreamTemplate::Mask(mask) = produced.template else {
            panic!("expected a mask template");
        };
        assert_eq!(mask.frame_rate, None);
    }

    #[test]
    fn the_luma_mask_format_keeps_the_frames_rate() {
        let rate = FrameRate::new(24000, 1001);
        let resolved = ResolvedFormats::new(vec![Some(rated_frames(rate))], None);

        let luma = Luma::from_formats("lines/test", &resolved).expect("yuv is readable");

        assert_eq!(luma.mask_format().frame_rate, rate);
    }

    #[test]
    fn zero_brz_turns_binarize_off() {
        let off = binarize_threshold(0.0, 255.0);

        assert_eq!(off, None);
    }

    #[test]
    fn a_brz_becomes_a_rounded_code_value() {
        let eight_bit_detail = binarize_threshold(0.005, 255.0);
        let eight_bit_lines = binarize_threshold(0.08, 255.0);
        let ten_bit_detail = binarize_threshold(0.005, 1023.0);
        let ten_bit_lines = binarize_threshold(0.08, 1023.0);

        assert_eq!(eight_bit_detail, Some(1.0));
        assert_eq!(eight_bit_lines, Some(20.0));
        assert_eq!(ten_bit_detail, Some(5.0));
        assert_eq!(ten_bit_lines, Some(82.0));
    }

    #[test]
    fn a_zero_brz_skips_binarize() {
        assert_eq!(binarize_threshold(0.0, 255.0), None);
    }
}
