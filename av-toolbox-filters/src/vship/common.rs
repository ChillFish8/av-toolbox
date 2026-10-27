use av_toolbox_core::StageError;
use av_toolbox_gpu::{
    Channel,
    ChromaLocation,
    ColourRange,
    Device,
    FormatSet,
    FrameFormat,
    FrameLayout,
    HostFrame,
    Matrix,
    Offered,
    Primaries,
    SampleFormat,
    StreamFormat,
    Subsampling,
    Transfer,
};
use avtbx_vship::{
    ChromaLocation as VshipChromaLocation,
    ColorSpace,
    DeviceInfo,
    Planes,
    Range,
    SampleType,
    VshipError,
};

const REFERENCE: usize = 0;
const DISTORTED: usize = 1;

/// A colour or siting a clip left unspecified, or that Vship cannot take, replaced by a default.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Fallback {
    Range,
    Matrix,
    Transfer,
    Primaries,
    ChromaLocation,
}

/// The fallbacks a stage has logged, so each is logged once.
#[derive(Debug)]
pub(crate) struct Fallbacks {
    stage: &'static str,
    logged: Vec<Fallback>,
}

impl Fallbacks {
    pub(crate) fn new(stage: &'static str) -> Self {
        Self {
            stage,
            logged: Vec::new(),
        }
    }

    /// Logs `kind` the first time it is noted.
    ///
    /// Range and colour codes log at `info`. A moved chroma location logs at `warn`, since it shifts
    /// what is scored.
    pub(crate) fn note(&mut self, kind: Fallback) {
        if self.logged.contains(&kind) {
            return;
        }

        self.logged.push(kind);

        let stage = self.stage;
        match kind {
            Fallback::Range => {
                tracing::info!(
                    "{stage}: the clip's range is unspecified, so it is scored as limited for YUV and full for RGB"
                );
            },
            Fallback::Matrix => {
                tracing::info!(
                    "{stage}: the clip's matrix is unspecified or identity, so it is scored as BT.709"
                );
            },
            Fallback::Transfer => {
                tracing::info!("{stage}: the clip's transfer is unspecified, so it is scored as BT.709");
            },
            Fallback::Primaries => {
                tracing::info!("{stage}: the clip's primaries are unspecified, so they are scored as BT.709");
            },
            Fallback::ChromaLocation => {
                tracing::warn!(
                    "{stage}: Vship has no bottom chroma locations, so bottom is scored as centre and bottom left as left"
                );
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn noted(&self, kind: Fallback) -> bool {
        self.logged.contains(&kind)
    }
}

/// The Vship device a stage scores on.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct DeviceChoice {
    pub(crate) index: i32,
    /// The pipeline's device had no match in Vship's list, so device 0 was taken.
    pub(crate) fell_back: bool,
}

pub(crate) fn vship_sample(sample: SampleFormat) -> Option<SampleType> {
    let mapped = match sample {
        SampleFormat::U8 => SampleType::Uint8,
        SampleFormat::U16 { depth } => match depth.bits() {
            9 => SampleType::Uint9,
            10 => SampleType::Uint10,
            12 => SampleType::Uint12,
            14 => SampleType::Uint14,
            16 => SampleType::Uint16,
            _ => return None,
        },
        SampleFormat::F32 => SampleType::Float,
    };

    Some(mapped)
}

/// Vship's description of `format`, with the clip's colour mapped to H.273 codes.
///
/// Anything unspecified, or that Vship cannot take, falls back to a default noted in `fallbacks`.
pub(crate) fn color_space(format: &FrameFormat, fallbacks: &mut Fallbacks) -> Result<ColorSpace, StageError> {
    let Some(sample) = vship_sample(format.sample) else {
        return Err(unsupported_sample_error(format.sample));
    };

    let colour = format.colour;
    let (sub_sampling_w, sub_sampling_h, is_yuv) = match format.layout {
        FrameLayout::Yuv { subsampling } => {
            let (shift_w, shift_h) = subsampling_shift(subsampling);
            (shift_w, shift_h, true)
        },
        FrameLayout::Rgb => (0, 0, false),
        FrameLayout::Grey | FrameLayout::Chroma { .. } => return Err(layout_error(format.layout)),
    };

    let range = match colour.range {
        _ if format.sample == SampleFormat::F32 => Range::Full,
        ColourRange::Limited => Range::Limited,
        ColourRange::Full => Range::Full,
        ColourRange::Unspecified if is_yuv => {
            fallbacks.note(Fallback::Range);
            Range::Limited
        },
        ColourRange::Unspecified => {
            fallbacks.note(Fallback::Range);
            Range::Full
        },
    };

    let matrix = if is_yuv {
        matrix_code(colour.matrix, fallbacks)
    } else {
        0
    };
    let chroma_location = if is_yuv {
        chroma_location(colour.chroma_location, fallbacks)
    } else {
        VshipChromaLocation::Left
    };
    let transfer = transfer_code(colour.transfer, fallbacks);
    let primaries = primaries_code(colour.primaries, fallbacks);

    Ok(ColorSpace {
        sample,
        width: i64::from(format.width),
        height: i64::from(format.height),
        range,
        sub_sampling_w,
        sub_sampling_h,
        chroma_location,
        matrix,
        transfer,
        primaries,
    })
}

/// The log2 shift of the chroma planes' width and height.
fn subsampling_shift(subsampling: Subsampling) -> (i32, i32) {
    match subsampling {
        Subsampling::Yuv444 => (0, 0),
        Subsampling::Yuv422 => (1, 0),
        Subsampling::Yuv420 => (1, 1),
    }
}

fn matrix_code(matrix: Matrix, fallbacks: &mut Fallbacks) -> u32 {
    match matrix {
        Matrix::Bt709 => 1,
        Matrix::Bt601 => 6,
        Matrix::Bt2020Ncl => 9,
        Matrix::Unspecified | Matrix::Identity => {
            fallbacks.note(Fallback::Matrix);
            1
        },
    }
}

fn transfer_code(transfer: Transfer, fallbacks: &mut Fallbacks) -> u32 {
    match transfer {
        Transfer::Bt709 => 1,
        Transfer::Srgb => 13,
        Transfer::Pq => 16,
        Transfer::Hlg => 18,
        Transfer::Unspecified => {
            fallbacks.note(Fallback::Transfer);
            1
        },
    }
}

fn primaries_code(primaries: Primaries, fallbacks: &mut Fallbacks) -> i32 {
    match primaries {
        Primaries::Bt709 => 1,
        Primaries::Bt601 => 6,
        Primaries::Bt2020 => 9,
        Primaries::Unspecified => {
            fallbacks.note(Fallback::Primaries);
            1
        },
    }
}

fn chroma_location(location: ChromaLocation, fallbacks: &mut Fallbacks) -> VshipChromaLocation {
    match location {
        ChromaLocation::Left => VshipChromaLocation::Left,
        ChromaLocation::Center => VshipChromaLocation::Center,
        ChromaLocation::TopLeft => VshipChromaLocation::TopLeft,
        ChromaLocation::Top => VshipChromaLocation::Top,
        ChromaLocation::Bottom => {
            fallbacks.note(Fallback::ChromaLocation);
            VshipChromaLocation::Center
        },
        ChromaLocation::BottomLeft => {
            fallbacks.note(Fallback::ChromaLocation);
            VshipChromaLocation::Left
        },
    }
}

/// A frame's planes as Vship reads them.
///
/// Vship reads full range float chroma as centred on 0.5, so float U and V are held as copies moved
/// up by 0.5. Every other plane is borrowed from the frame.
pub(crate) struct VshipPlanes<'a> {
    data: [&'a [u8]; 3],
    strides: [usize; 3],
    centred: Option<[Vec<u8>; 2]>,
}

impl<'a> VshipPlanes<'a> {
    pub(crate) fn of(frame: &'a HostFrame) -> Result<Self, StageError> {
        let format = frame.format();
        let channels = match format.layout {
            FrameLayout::Rgb => [Channel::R, Channel::G, Channel::B],
            _ => [Channel::Y, Channel::U, Channel::V],
        };

        let mut data: [&[u8]; 3] = [&[]; 3];
        let mut strides = [0usize; 3];
        for (index, channel) in channels.into_iter().enumerate() {
            let plane = frame.plane(channel).ok_or_else(|| missing_plane_error(channel))?;
            let (width, _) = format
                .channel_dimensions(channel)
                .ok_or_else(|| missing_plane_error(channel))?;

            data[index] = plane.bytes();
            strides[index] = width as usize * format.sample.bytes();
        }

        let is_float_yuv = format.sample == SampleFormat::F32 && format.layout != FrameLayout::Rgb;
        let centred = is_float_yuv.then(|| [centre_chroma(data[1]), centre_chroma(data[2])]);

        Ok(Self {
            data,
            strides,
            centred,
        })
    }

    pub(crate) fn planes(&self) -> Planes<'_> {
        let mut data = self.data;
        if let Some([u_plane, v_plane]) = &self.centred {
            data[1] = u_plane;
            data[2] = v_plane;
        }

        Planes {
            data,
            strides: self.strides,
        }
    }
}

/// Little-endian f32 samples with 0.5 added to each.
fn centre_chroma(bytes: &[u8]) -> Vec<u8> {
    let mut shifted = Vec::with_capacity(bytes.len());
    let (samples, _) = bytes.as_chunks::<4>();
    for sample in samples {
        let value = f32::from_le_bytes(*sample);
        shifted.extend_from_slice(&(value + 0.5).to_le_bytes());
    }

    shifted
}

/// The Vship device to score on, given the `gpu` arg, the pipeline's device and Vship's devices.
///
/// The `gpu` arg indexes `devices` directly. Otherwise a discrete or integrated pipeline device
/// picks the device of the same kind and ordinal, and anything else takes device 0.
pub(crate) fn choose_device(
    gpu: Option<usize>,
    pipeline: Device,
    devices: &[DeviceInfo],
) -> Result<DeviceChoice, StageError> {
    if devices.is_empty() {
        return Err(StageError::msg("Vship found no device to score on"));
    }

    if let Some(index) = gpu {
        if index >= devices.len() {
            return Err(gpu_out_of_range_error(index, devices.len()));
        }

        let index = vship_index(index)?;
        return Ok(DeviceChoice {
            index,
            fell_back: false,
        });
    }

    let matched = match pipeline {
        Device::Discrete { index } => nth_device(devices, false, index),
        Device::Integrated { index } => nth_device(devices, true, index),
        _ => None,
    };

    let Some(position) = matched else {
        return Ok(DeviceChoice {
            index: 0,
            fell_back: true,
        });
    };

    let index = vship_index(position)?;

    Ok(DeviceChoice {
        index,
        fell_back: false,
    })
}

/// The position in `devices` of the `ordinal`th device whose `integrated` flag matches.
fn nth_device(devices: &[DeviceInfo], integrated: bool, ordinal: usize) -> Option<usize> {
    devices
        .iter()
        .enumerate()
        .filter(|(_, device)| device.integrated == integrated)
        .map(|(position, _)| position)
        .nth(ordinal)
}

fn vship_index(index: usize) -> Result<i32, StageError> {
    match i32::try_from(index) {
        Ok(converted) => Ok(converted),
        Err(_) => Err(unaddressable_device_error(index)),
    }
}

pub(crate) fn vship_devices() -> Result<Vec<DeviceInfo>, StageError> {
    let count = avtbx_vship::device_count().map_err(device_count_error)?;

    let mut devices = Vec::new();
    for gpu_id in 0..count {
        let info = match avtbx_vship::device_info(gpu_id) {
            Ok(info) => info,
            Err(error) => return Err(device_info_error(gpu_id, error)),
        };
        devices.push(info);
    }

    Ok(devices)
}

/// The Vship device `stage` scores on, logging the choice.
pub(crate) fn pick_device(stage: &str, gpu: Option<usize>, pipeline: Device) -> Result<i32, StageError> {
    let devices = vship_devices()?;
    let choice = choose_device(gpu, pipeline, &devices)?;

    if choice.fell_back {
        tracing::warn!(
            "{stage}: Vship lists no device matching the pipeline's {pipeline}, so it scores on Vship device 0"
        );
    }

    let name = devices
        .get(choice.index as usize)
        .map_or("unknown", |device| device.name.as_str());
    tracing::info!("{stage}: scoring on Vship device {} ({name})", choice.index);

    Ok(choice.index)
}

pub(crate) fn accepts_clip(offered: StreamFormat, other: Offered) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames_error());
    };

    check_layout(format.layout)?;

    let other_frames = other.format();
    let other_frames = other_frames.as_ref().and_then(StreamFormat::as_frames);
    if let Some(other_format) = other_frames {
        check_same_size(other_format, format)?;
    }

    if vship_sample(format.sample).is_none() {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

/// The reference and distorted formats, once both are bound and agree.
pub(crate) fn checked_pair(
    chosen: &[Option<StreamFormat>],
) -> Result<(FrameFormat, FrameFormat), StageError> {
    let reference = chosen.get(REFERENCE).copied().flatten();
    let distorted = chosen.get(DISTORTED).copied().flatten();

    let (Some(reference), Some(distorted)) = (reference, distorted) else {
        return Err(StageError::msg(
            "a Vship metric needs reference and distorted bound",
        ));
    };

    let reference = reference.as_frames().copied().ok_or_else(not_frames_error)?;
    let distorted = distorted.as_frames().copied().ok_or_else(not_frames_error)?;

    for format in [&reference, &distorted] {
        check_layout(format.layout)?;

        if vship_sample(format.sample).is_none() {
            return Err(unsupported_sample_error(format.sample));
        }
    }

    check_same_size(&reference, &distorted)?;

    Ok((reference, distorted))
}

fn check_layout(layout: FrameLayout) -> Result<(), StageError> {
    match layout {
        FrameLayout::Yuv { .. } | FrameLayout::Rgb => Ok(()),
        FrameLayout::Grey | FrameLayout::Chroma { .. } => Err(layout_error(layout)),
    }
}

fn check_same_size(reference: &FrameFormat, distorted: &FrameFormat) -> Result<(), StageError> {
    if reference.width == distorted.width && reference.height == distorted.height {
        return Ok(());
    }

    let message = format!(
        "a Vship metric needs both clips at one size, got {}x{} and {}x{}",
        reference.width, reference.height, distorted.width, distorted.height,
    );

    Err(StageError::msg(message))
}

fn not_frames_error() -> StageError {
    StageError::msg("a Vship metric scores frames, not masks")
}

fn layout_error(layout: FrameLayout) -> StageError {
    let message = format!("a Vship metric needs a YUV or RGB clip with three planes, got a {layout} clip");

    StageError::msg(message)
}

fn unsupported_sample_error(sample: SampleFormat) -> StageError {
    StageError::msg(format!("Vship cannot read {sample} samples"))
}

fn missing_plane_error(channel: Channel) -> StageError {
    StageError::msg(format!(
        "a Vship metric was given a frame with no {channel} plane"
    ))
}

fn gpu_out_of_range_error(index: usize, count: usize) -> StageError {
    StageError::msg(format!(
        "gpu {index} is out of range, Vship lists {count} devices"
    ))
}

fn unaddressable_device_error(index: usize) -> StageError {
    StageError::msg(format!("Vship cannot address device {index}"))
}

fn device_count_error(error: VshipError) -> StageError {
    StageError::with_source("Vship could not list its devices", error)
}

fn device_info_error(gpu_id: i32, error: VshipError) -> StageError {
    StageError::with_source(format!("Vship could not describe device {gpu_id}"), error)
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, Depth, FrameMeta, HostPool};

    use super::*;

    fn frame_format(width: u32, height: u32, layout: FrameLayout, colour: ColourInfo) -> FrameFormat {
        FrameFormat {
            width,
            height,
            layout,
            sample: SampleFormat::U8,
            colour,
            frame_rate: None,
            field_order: None,
        }
    }

    fn yuv(subsampling: Subsampling, colour: ColourInfo) -> FrameFormat {
        let layout = FrameLayout::Yuv { subsampling };

        frame_format(64, 48, layout, colour)
    }

    fn bt709_limited() -> ColourInfo {
        ColourInfo {
            range: ColourRange::Limited,
            matrix: Matrix::Bt709,
            primaries: Primaries::Bt709,
            transfer: Transfer::Bt709,
            chroma_location: ChromaLocation::Left,
        }
    }

    fn u16_at(bits: u8) -> SampleFormat {
        let depth = Depth::new(bits).expect("a valid depth");

        SampleFormat::U16 { depth }
    }

    fn device(name: &str, integrated: bool) -> DeviceInfo {
        DeviceInfo {
            name: name.to_string(),
            vram: 0,
            integrated,
        }
    }

    /// Two discrete devices and two integrated ones, interleaved.
    fn mixed_devices() -> Vec<DeviceInfo> {
        vec![
            device("integrated a", true),
            device("discrete a", false),
            device("integrated b", true),
            device("discrete b", false),
        ]
    }

    fn map_format(format: &FrameFormat) -> (ColorSpace, Fallbacks) {
        let mut fallbacks = Fallbacks::new("vship/test");
        let color_space = color_space(format, &mut fallbacks).expect("the format maps");

        (color_space, fallbacks)
    }

    #[test]
    fn eight_bit_maps_to_uint8() {
        assert_eq!(vship_sample(SampleFormat::U8), Some(SampleType::Uint8));
    }

    #[test]
    fn each_supported_u16_depth_maps_to_its_type() {
        let cases = [
            (9, SampleType::Uint9),
            (10, SampleType::Uint10),
            (12, SampleType::Uint12),
            (14, SampleType::Uint14),
            (16, SampleType::Uint16),
        ];

        for (bits, expected) in cases {
            assert_eq!(vship_sample(u16_at(bits)), Some(expected), "{bits} bits");
        }
    }

    #[test]
    fn odd_u16_depths_are_refused() {
        for bits in [11, 13, 15] {
            assert_eq!(vship_sample(u16_at(bits)), None, "{bits} bits");
        }
    }

    #[test]
    fn floats_map_to_float() {
        assert_eq!(vship_sample(SampleFormat::F32), Some(SampleType::Float));
    }

    #[test]
    fn bt709_limited_420_maps_directly() {
        let format = yuv(Subsampling::Yuv420, bt709_limited());

        let (mapped, fallbacks) = map_format(&format);

        let expected = ColorSpace {
            sample: SampleType::Uint8,
            width: 64,
            height: 48,
            range: Range::Limited,
            sub_sampling_w: 1,
            sub_sampling_h: 1,
            chroma_location: VshipChromaLocation::Left,
            matrix: 1,
            transfer: 1,
            primaries: 1,
        };
        assert_eq!(mapped, expected);
        assert!(fallbacks.logged.is_empty());
    }

    #[test]
    fn bt2020_pq_full_maps_to_its_codes() {
        let colour = ColourInfo {
            range: ColourRange::Full,
            matrix: Matrix::Bt2020Ncl,
            primaries: Primaries::Bt2020,
            transfer: Transfer::Pq,
            chroma_location: ChromaLocation::TopLeft,
        };
        let format = yuv(Subsampling::Yuv422, colour);

        let (mapped, _) = map_format(&format);

        assert_eq!(mapped.range, Range::Full);
        assert_eq!((mapped.matrix, mapped.transfer, mapped.primaries), (9, 16, 9));
        assert_eq!((mapped.sub_sampling_w, mapped.sub_sampling_h), (1, 0));
        assert_eq!(mapped.chroma_location, VshipChromaLocation::TopLeft);
    }

    #[test]
    fn the_remaining_colour_codes_map() {
        let colour = ColourInfo {
            matrix: Matrix::Bt601,
            primaries: Primaries::Bt601,
            transfer: Transfer::Hlg,
            chroma_location: ChromaLocation::Center,
            ..bt709_limited()
        };
        let format = yuv(Subsampling::Yuv444, colour);
        let srgb_colour = ColourInfo {
            transfer: Transfer::Srgb,
            chroma_location: ChromaLocation::Top,
            ..bt709_limited()
        };
        let srgb_format = yuv(Subsampling::Yuv444, srgb_colour);

        let (mapped, _) = map_format(&format);
        let (srgb, _) = map_format(&srgb_format);

        assert_eq!((mapped.matrix, mapped.transfer, mapped.primaries), (6, 18, 6));
        assert_eq!((mapped.sub_sampling_w, mapped.sub_sampling_h), (0, 0));
        assert_eq!(mapped.chroma_location, VshipChromaLocation::Center);
        assert_eq!(srgb.transfer, 13);
        assert_eq!(srgb.chroma_location, VshipChromaLocation::Top);
    }

    #[test]
    fn rgb_uses_matrix_zero_and_no_subsampling() {
        let colour = ColourInfo {
            range: ColourRange::Full,
            transfer: Transfer::Srgb,
            ..bt709_limited()
        };
        let format = frame_format(32, 16, FrameLayout::Rgb, colour);

        let (mapped, _) = map_format(&format);

        assert_eq!(mapped.matrix, 0);
        assert_eq!((mapped.sub_sampling_w, mapped.sub_sampling_h), (0, 0));
        assert_eq!((mapped.width, mapped.height), (32, 16));
        assert_eq!((mapped.range, mapped.transfer), (Range::Full, 13));
    }

    #[test]
    fn unspecified_colour_codes_fall_back_to_bt709() {
        let format = yuv(Subsampling::Yuv420, ColourInfo::default());

        let (mapped, fallbacks) = map_format(&format);

        assert_eq!((mapped.matrix, mapped.transfer, mapped.primaries), (1, 1, 1));
        assert!(fallbacks.noted(Fallback::Matrix));
        assert!(fallbacks.noted(Fallback::Transfer));
        assert!(fallbacks.noted(Fallback::Primaries));
    }

    #[test]
    fn an_identity_matrix_on_yuv_falls_back_to_bt709() {
        let colour = ColourInfo {
            matrix: Matrix::Identity,
            ..bt709_limited()
        };
        let format = yuv(Subsampling::Yuv420, colour);

        let (mapped, fallbacks) = map_format(&format);

        assert_eq!(mapped.matrix, 1);
        assert_eq!((mapped.sub_sampling_w, mapped.sub_sampling_h), (1, 1));
        assert!(fallbacks.noted(Fallback::Matrix));
    }

    #[test]
    fn an_f32_clip_is_scored_as_full_range_whatever_its_label() {
        let mut format = yuv(Subsampling::Yuv420, bt709_limited());
        format.sample = SampleFormat::F32;

        let (mapped, fallbacks) = map_format(&format);

        assert_eq!(mapped.range, Range::Full);
        assert!(fallbacks.logged.is_empty());
    }

    #[test]
    fn an_unspecified_f32_range_does_not_warn() {
        let colour = ColourInfo {
            range: ColourRange::Unspecified,
            ..bt709_limited()
        };
        let mut format = yuv(Subsampling::Yuv420, colour);
        format.sample = SampleFormat::F32;

        let (mapped, fallbacks) = map_format(&format);

        assert_eq!(mapped.range, Range::Full);
        assert!(!fallbacks.noted(Fallback::Range));
    }

    #[test]
    fn an_unspecified_range_is_limited_for_yuv() {
        let colour = ColourInfo {
            range: ColourRange::Unspecified,
            ..bt709_limited()
        };
        let format = yuv(Subsampling::Yuv420, colour);

        let (mapped, fallbacks) = map_format(&format);

        assert_eq!(mapped.range, Range::Limited);
        assert!(fallbacks.noted(Fallback::Range));
    }

    #[test]
    fn an_unspecified_range_is_full_for_rgb() {
        let colour = ColourInfo {
            range: ColourRange::Unspecified,
            ..bt709_limited()
        };
        let format = frame_format(8, 8, FrameLayout::Rgb, colour);

        let (mapped, fallbacks) = map_format(&format);

        assert_eq!(mapped.range, Range::Full);
        assert!(fallbacks.noted(Fallback::Range));
    }

    #[test]
    fn bottom_chroma_locations_keep_their_horizontal_siting() {
        let cases = [
            (ChromaLocation::Bottom, VshipChromaLocation::Center),
            (ChromaLocation::BottomLeft, VshipChromaLocation::Left),
        ];

        for (location, expected) in cases {
            let colour = ColourInfo {
                chroma_location: location,
                ..bt709_limited()
            };
            let format = yuv(Subsampling::Yuv420, colour);

            let (mapped, fallbacks) = map_format(&format);

            assert_eq!(mapped.chroma_location, expected, "{location:?}");
            assert!(fallbacks.noted(Fallback::ChromaLocation), "{location:?}");
        }
    }

    #[test]
    fn a_fallback_is_kept_once() {
        let mut fallbacks = Fallbacks::new("vship/test");

        fallbacks.note(Fallback::Range);
        fallbacks.note(Fallback::Range);

        assert_eq!(fallbacks.logged, vec![Fallback::Range]);
    }

    #[test]
    fn grey_is_refused_by_name() {
        let format = frame_format(8, 8, FrameLayout::Grey, bt709_limited());
        let mut fallbacks = Fallbacks::new("vship/test");

        let error = color_space(&format, &mut fallbacks).expect_err("grey has one plane");

        assert!(error.to_string().contains("grey"), "{error}");
    }

    #[test]
    fn an_unsupported_sample_is_refused_by_name() {
        let format = FrameFormat {
            sample: u16_at(11),
            ..yuv(Subsampling::Yuv420, bt709_limited())
        };
        let mut fallbacks = Fallbacks::new("vship/test");

        let error = color_space(&format, &mut fallbacks).expect_err("Vship has no 11-bit type");

        assert!(error.to_string().contains("u16 11-bit"), "{error}");
    }

    #[test]
    fn ten_bit_planes_are_packed_two_bytes_a_sample() {
        let format = FrameFormat {
            sample: u16_at(10),
            ..yuv(Subsampling::Yuv420, bt709_limited())
        };
        let host = HostPool::new();
        let writable = host.alloc_frame(&format, FrameMeta::at(0));
        let frame = writable.freeze();

        let vship = VshipPlanes::of(&frame).expect("a yuv frame has three planes");
        let planes = vship.planes();

        assert_eq!(planes.strides, [128, 64, 64]);
        assert_eq!(planes.data[0].len(), 128 * 48);
        assert_eq!(planes.data[1].len(), 64 * 24);
    }

    #[test]
    fn rgb_planes_are_read_in_rgb_order() {
        let format = frame_format(8, 4, FrameLayout::Rgb, bt709_limited());
        let host = HostPool::new();
        let mut frame = host.alloc_frame(&format, FrameMeta::at(0));
        for (marker, (_, plane)) in frame.planes_mut().into_iter().enumerate() {
            plane.fill(marker as u8);
        }
        let frame = frame.freeze();

        let vship = VshipPlanes::of(&frame).expect("an rgb frame has three planes");
        let planes = vship.planes();

        assert_eq!(planes.strides, [8, 8, 8]);
        assert_eq!(
            [planes.data[0][0], planes.data[1][0], planes.data[2][0]],
            [0, 1, 2]
        );
    }

    fn float_frame(u_value: f32, v_value: f32) -> HostFrame {
        let format = FrameFormat {
            sample: SampleFormat::F32,
            ..yuv(Subsampling::Yuv420, bt709_limited())
        };
        let host = HostPool::new();
        let mut frame = host.alloc_frame(&format, FrameMeta::at(0));
        for (channel, plane) in frame.planes_mut() {
            let value = match channel {
                Channel::U => u_value,
                Channel::V => v_value,
                _ => 0.25,
            };
            let (samples, _) = plane.as_chunks_mut::<4>();
            for sample in samples {
                *sample = value.to_le_bytes();
            }
        }

        frame.freeze()
    }

    fn first_sample(bytes: &[u8]) -> f32 {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[test]
    fn f32_chroma_is_moved_up_by_a_half_for_vship() {
        let frame = float_frame(0.0, -0.25);

        let vship = VshipPlanes::of(&frame).expect("a yuv frame has three planes");
        let planes = vship.planes();

        let luma = frame.plane(Channel::Y).expect("a luma plane").bytes();
        assert_eq!(planes.data[0], luma);
        assert_eq!(first_sample(planes.data[1]), 0.5);
        assert_eq!(first_sample(planes.data[2]), 0.25);
        assert_eq!(planes.data[1].len(), 128 * 24);
    }

    #[test]
    fn integer_planes_are_borrowed_unchanged() {
        let format = FrameFormat {
            sample: u16_at(10),
            ..yuv(Subsampling::Yuv420, bt709_limited())
        };
        let host = HostPool::new();
        let frame = host.alloc_frame(&format, FrameMeta::at(0)).freeze();

        let vship = VshipPlanes::of(&frame).expect("a yuv frame has three planes");
        let planes = vship.planes();

        for channel in [Channel::Y, Channel::U, Channel::V] {
            let own = frame.plane(channel).expect("a plane").bytes();
            let index = channel_index(channel);
            assert!(std::ptr::eq(planes.data[index], own), "{channel:?}");
        }
    }

    fn channel_index(channel: Channel) -> usize {
        match channel {
            Channel::U => 1,
            Channel::V => 2,
            _ => 0,
        }
    }

    #[test]
    fn the_frame_keeps_its_chroma() {
        let frame = float_frame(0.0, -0.25);

        let vship = VshipPlanes::of(&frame).expect("a yuv frame has three planes");
        drop(vship);

        let u_plane = frame.plane(Channel::U).expect("a chroma plane").bytes();
        assert_eq!(first_sample(u_plane), 0.0);
    }

    #[test]
    fn the_gpu_arg_wins() {
        let devices = mixed_devices();
        let pipeline = Device::Discrete { index: 0 };

        let choice = choose_device(Some(2), pipeline, &devices).expect("2 is in range");

        let expected = DeviceChoice {
            index: 2,
            fell_back: false,
        };
        assert_eq!(choice, expected);
    }

    #[test]
    fn an_out_of_range_gpu_arg_names_the_count() {
        let devices = mixed_devices();

        let error = choose_device(Some(4), Device::Default, &devices).expect_err("4 is out of range");

        assert!(error.to_string().contains("4 devices"), "{error}");
    }

    #[test]
    fn discrete_picks_the_nth_non_integrated_device() {
        let devices = mixed_devices();
        let pipeline = Device::Discrete { index: 1 };

        let choice = choose_device(None, pipeline, &devices).expect("a device");

        let expected = DeviceChoice {
            index: 3,
            fell_back: false,
        };
        assert_eq!(choice, expected);
    }

    #[test]
    fn integrated_picks_the_nth_integrated_device() {
        let devices = mixed_devices();
        let pipeline = Device::Integrated { index: 0 };

        let choice = choose_device(None, pipeline, &devices).expect("a device");

        let expected = DeviceChoice {
            index: 0,
            fell_back: false,
        };
        assert_eq!(choice, expected);
    }

    #[test]
    fn cpu_and_unmatched_ordinals_fall_back_to_device_zero() {
        let devices = mixed_devices();
        let pipelines = [
            Device::Cpu,
            Device::Default,
            Device::Virtual { index: 0 },
            Device::Discrete { index: 2 },
        ];

        let expected = DeviceChoice {
            index: 0,
            fell_back: true,
        };
        for pipeline in pipelines {
            let choice = choose_device(None, pipeline, &devices).expect("a device");

            assert_eq!(choice, expected, "{pipeline}");
        }
    }

    #[test]
    fn no_devices_is_an_error() {
        let refused = choose_device(None, Device::Default, &[]);

        assert!(refused.is_err());
    }

    #[test]
    fn a_grey_clip_is_refused() {
        let format = frame_format(8, 8, FrameLayout::Grey, bt709_limited());
        let offered = StreamFormat::Frames(format);

        let refused = accepts_clip(offered, Offered::Unknown);

        assert!(refused.is_err());
    }

    #[test]
    fn a_chroma_clip_is_refused() {
        let layout = FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        };
        let format = frame_format(8, 8, layout, bt709_limited());
        let offered = StreamFormat::Frames(format);

        let refused = accepts_clip(offered, Offered::Unknown);

        assert!(refused.is_err());
    }

    #[test]
    fn an_unmappable_sample_is_left_out() {
        let format = FrameFormat {
            sample: u16_at(11),
            ..yuv(Subsampling::Yuv420, bt709_limited())
        };
        let offered = StreamFormat::Frames(format);

        let accepted = accepts_clip(offered, Offered::Unknown).expect("the layout is fine");

        assert!(accepted.is_empty());
    }

    #[test]
    fn a_partner_of_another_sample_is_accepted() {
        let format = yuv(Subsampling::Yuv420, bt709_limited());
        let partner_format = FrameFormat {
            sample: u16_at(10),
            ..format
        };
        let partner = Offered::Resolved(StreamFormat::Frames(partner_format));
        let offered = StreamFormat::Frames(format);

        let accepted = accepts_clip(offered, partner).expect("samples may differ");

        assert!(accepted.accepts(SampleFormat::U8));
    }

    #[test]
    fn a_partner_of_another_size_is_refused() {
        let format = yuv(Subsampling::Yuv420, bt709_limited());
        let partner_format = FrameFormat { width: 32, ..format };
        let partner = Offered::Proposed(StreamFormat::Frames(partner_format));
        let offered = StreamFormat::Frames(format);

        let refused = accepts_clip(offered, partner);

        assert!(refused.is_err());
    }

    #[test]
    fn a_pair_is_checked_and_returned_in_order() {
        let reference = yuv(Subsampling::Yuv420, bt709_limited());
        let distorted = FrameFormat {
            sample: u16_at(10),
            ..reference
        };
        let chosen = [
            Some(StreamFormat::Frames(reference)),
            Some(StreamFormat::Frames(distorted)),
        ];

        let pair = checked_pair(&chosen).expect("the pair agrees");

        assert_eq!(pair, (reference, distorted));
    }

    #[test]
    fn a_pair_of_different_sizes_is_refused() {
        let reference = yuv(Subsampling::Yuv420, bt709_limited());
        let distorted = FrameFormat {
            height: 24,
            ..reference
        };
        let chosen = [
            Some(StreamFormat::Frames(reference)),
            Some(StreamFormat::Frames(distorted)),
        ];

        let error = checked_pair(&chosen).expect_err("the sizes differ");

        assert!(error.to_string().contains("64x48"), "{error}");
    }

    #[test]
    fn a_pair_missing_an_input_is_refused() {
        let reference = yuv(Subsampling::Yuv420, bt709_limited());
        let chosen = [Some(StreamFormat::Frames(reference)), None];

        let refused = checked_pair(&chosen);

        assert!(refused.is_err());
    }
}
