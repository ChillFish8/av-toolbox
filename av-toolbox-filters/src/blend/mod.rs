mod kernel;

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Parallelism,
    StageDefinition,
    StageError,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use self::kernel::MaskedBlend;

const BASE: usize = 0;
const OVERLAY: usize = 1;
const MASK: usize = 2;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct MaskedArgs {
    /// Which planes to blend, by index.
    ///
    /// Unset blends every plane the layout carries.
    #[serde(default)]
    pub(crate) planes: Option<Vec<u8>>,
}

/// Takes each pixel from `overlay` where the mask selects it and from `base` elsewhere, blending
/// between the two where the mask is partly selected.
pub(crate) struct Masked;

impl StageDefinition for Masked {
    type Args = MaskedArgs;

    fn package() -> &'static str {
        "blend"
    }

    fn identifier() -> &'static str {
        "masked"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base")
                .accepting(StreamType::Frames)
                .chunk_authority(),
            InputParam::required("overlay").accepting(StreamType::Frames),
            InputParam::required("mask")
                .accepting(StreamType::Mask)
                .planes_of("base"),
        ]);
        SIGNATURE
    }
}

/// The samples a clip can be blended in.
fn blendable(sample: SampleFormat) -> bool {
    matches!(sample, SampleFormat::U8 | SampleFormat::U16 { .. })
}

/// Whether two clips can be blended sample for sample, ignoring colour and rate.
fn same_shape(first: &FrameFormat, second: &FrameFormat) -> bool {
    first.width == second.width
        && first.height == second.height
        && first.layout == second.layout
        && first.sample == second.sample
}

fn not_frames() -> StageError {
    StageError::msg("blend/masked blends frames, not masks")
}

fn not_a_mask() -> StageError {
    StageError::msg("blend/masked selects with a mask, not frames")
}

fn accepts_clip(offered: StreamFormat, other: Offered) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    if !blendable(format.sample) {
        return Ok(FormatSet::new());
    }

    let other_sample = other.format().map(|format| format.sample());
    if let Some(sample) = other_sample
        && sample != format.sample
    {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

fn accepts_mask(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_mask() else {
        return Err(not_a_mask());
    };

    let accepted = match format.sample {
        SampleFormat::U8 => FormatSet::of([SampleFormat::U8]),
        _ => FormatSet::new(),
    };

    Ok(accepted)
}

/// The base format, once every input agrees with it.
fn checked_base(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let base = chosen.get(BASE).copied().flatten();
    let overlay = chosen.get(OVERLAY).copied().flatten();
    let mask = chosen.get(MASK).copied().flatten();

    let (Some(base), Some(overlay), Some(mask)) = (base, overlay, mask) else {
        return Err(StageError::msg("blend/masked needs base, overlay and mask bound"));
    };

    let base = base.as_frames().copied().ok_or_else(not_frames)?;
    let overlay = overlay.as_frames().copied().ok_or_else(not_frames)?;
    let mask = mask.as_mask().copied().ok_or_else(not_a_mask)?;

    if !same_shape(&base, &overlay) {
        return Err(StageError::msg(
            "blend/masked needs base and overlay to share their size, layout and sample",
        ));
    }

    if mask.width != base.width || mask.height != base.height {
        let message = format!(
            "blend/masked needs a {}x{} mask to match its frames, got {}x{}",
            base.width, base.height, mask.width, mask.height,
        );
        return Err(StageError::msg(message));
    }

    if mask.sample != SampleFormat::U8 {
        let message = format!("blend/masked needs a U8 mask, got {}", mask.sample);
        return Err(StageError::msg(message));
    }

    Ok(base)
}

/// The plane indices `blend/masked` blends, given the layout `base` and `overlay` agreed on.
///
/// Every plane blends when `planes` is left unset.
fn resolved_planes(planes: &Option<Vec<u8>>, layout: FrameLayout) -> Result<Vec<u8>, StageError> {
    let channels = layout.channels().len();

    let Some(planes) = planes else {
        let every = (0..channels).map(|index| index as u8).collect();
        return Ok(every);
    };

    if planes.is_empty() {
        return Err(StageError::msg(
            "blend/masked needs planes to name at least one index",
        ));
    }

    let mut checked: Vec<u8> = Vec::with_capacity(planes.len());
    for index in planes {
        let position = usize::from(*index);
        if position >= channels {
            let message = format!("blend/masked needs planes indices below {channels}, got {index}");
            return Err(StageError::msg(message));
        }

        if checked.contains(index) {
            let message = format!("blend/masked needs planes without repeats, got {index} twice");
            return Err(StageError::msg(message));
        }

        checked.push(*index);
    }

    Ok(checked)
}

impl FilterDefinition for Masked {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        match index {
            BASE => {
                let other = inputs.get(OVERLAY).copied().unwrap_or(Offered::Unknown);
                accepts_clip(offered, other)
            },
            OVERLAY => {
                let other = inputs.get(BASE).copied().unwrap_or(Offered::Unknown);
                accepts_clip(offered, other)
            },
            _ => accepts_mask(offered),
        }
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let base = checked_base(chosen)?;
        resolved_planes(&args.planes, base.layout)?;
        let format = StreamFormat::Frames(base);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([base.sample]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let base = checked_base(inputs)?;
        let planes = resolved_planes(&args.planes, base.layout)?;
        let client = context.client().clone();
        let blend = MaskedBlend::new(client, base, planes);

        Ok(Box::new(blend))
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, Depth, MaskFormat, Subsampling};

    use super::*;

    fn frames(layout: FrameLayout, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width: 8,
            height: 6,
            layout,
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
        })
    }

    fn yuv444(sample: SampleFormat) -> StreamFormat {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        };
        frames(layout, sample)
    }

    fn yuv(subsampling: Subsampling, sample: SampleFormat) -> StreamFormat {
        let layout = FrameLayout::Yuv { subsampling };
        frames(layout, sample)
    }

    fn mask(width: u32, height: u32) -> StreamFormat {
        mask_sample(width, height, SampleFormat::U8)
    }

    fn mask_sample(width: u32, height: u32, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Mask(MaskFormat {
            width,
            height,
            sample,
            chroma: None,
        })
    }

    #[test]
    fn a_float_clip_is_left_out() {
        let clip = yuv444(SampleFormat::F32);
        let accepted = accepts_clip(clip, Offered::Unknown).expect("frames are readable");

        assert!(accepted.is_empty());
    }

    #[test]
    fn a_clip_whose_partner_carries_another_sample_is_left_out() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let partner_format = yuv444(ten_bit);
        let partner = Offered::Proposed(partner_format);

        let clip = yuv444(SampleFormat::U8);
        let accepted = accepts_clip(clip, partner).expect("frames are readable");

        assert!(accepted.is_empty());
    }

    #[test]
    fn a_mask_in_a_clip_slot_is_refused() {
        let clip = mask(8, 6);
        let refused = accepts_clip(clip, Offered::Unknown);

        assert!(refused.is_err());
    }

    #[test]
    fn a_420_clip_is_accepted() {
        let clip = yuv(Subsampling::Yuv420, SampleFormat::U8);
        let accepted = accepts_clip(clip, Offered::Unknown).expect("4:2:0 is accepted");

        assert!(accepted.accepts(SampleFormat::U8));
    }

    #[test]
    fn a_422_clip_is_accepted() {
        let clip = yuv(Subsampling::Yuv422, SampleFormat::U8);
        let accepted = accepts_clip(clip, Offered::Unknown).expect("4:2:2 is accepted");

        assert!(accepted.accepts(SampleFormat::U8));
    }

    #[test]
    fn a_chroma_clip_is_accepted() {
        let layout = FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        };
        let clip = frames(layout, SampleFormat::U8);

        let accepted = accepts_clip(clip, Offered::Unknown).expect("a chroma clip is frames");

        assert!(accepted.accepts(SampleFormat::U8));
    }

    #[test]
    fn the_mask_param_asks_for_planes_of_base() {
        let signature = Masked::inputs();
        let mask = signature.params()[MASK];

        assert_eq!(mask.planes_of_input(), Some("base"));
    }

    #[test]
    fn full_size_layouts_are_accepted() {
        let yuv_clip = yuv444(SampleFormat::U8);
        let yuv = accepts_clip(yuv_clip, Offered::Unknown).expect("4:4:4 is full size");
        let rgb_clip = frames(FrameLayout::Rgb, SampleFormat::U8);
        let rgb = accepts_clip(rgb_clip, Offered::Unknown).expect("rgb is full size");
        let grey_clip = frames(FrameLayout::Grey, SampleFormat::U8);
        let grey = accepts_clip(grey_clip, Offered::Unknown).expect("grey is full size");

        assert!(yuv.accepts(SampleFormat::U8));
        assert!(rgb.accepts(SampleFormat::U8));
        assert!(grey.accepts(SampleFormat::U8));
    }

    #[test]
    fn the_output_is_the_base_format() {
        let chosen = [
            Some(yuv444(SampleFormat::U8)),
            Some(yuv444(SampleFormat::U8)),
            Some(mask(8, 6)),
        ];

        let base = checked_base(&chosen).expect("everything agrees");

        assert_eq!(StreamFormat::Frames(base), yuv444(SampleFormat::U8));
    }

    #[test]
    fn a_mask_of_the_wrong_size_is_refused() {
        let chosen = [
            Some(yuv444(SampleFormat::U8)),
            Some(yuv444(SampleFormat::U8)),
            Some(mask(4, 3)),
        ];

        let error = checked_base(&chosen).expect_err("the mask is a different size");

        assert!(error.to_string().contains("8x6 mask"));
    }

    #[test]
    fn a_u16_mask_is_refused() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let chosen = [
            Some(yuv444(SampleFormat::U8)),
            Some(yuv444(SampleFormat::U8)),
            Some(mask_sample(8, 6, ten_bit)),
        ];

        let error = checked_base(&chosen).expect_err("the mask is not U8");

        assert!(error.to_string().contains("U8 mask"));
    }

    #[test]
    fn clips_of_different_layouts_are_refused() {
        let chosen = [
            Some(yuv444(SampleFormat::U8)),
            Some(frames(FrameLayout::Grey, SampleFormat::U8)),
            Some(mask(8, 6)),
        ];

        let refused = checked_base(&chosen);

        assert!(refused.is_err());
    }

    #[test]
    fn planes_unset_parses_to_none() {
        let empty = serde_json::json!({});
        let args: MaskedArgs = serde_json::from_value(empty).expect("the defaults apply");

        assert_eq!(args.planes, None);
    }

    #[test]
    fn an_out_of_range_plane_index_is_refused() {
        let planes = Some(vec![3]);
        let error = resolved_planes(&planes, FrameLayout::Grey).expect_err("grey has one channel");

        assert!(error.to_string().contains("planes"));
    }

    #[test]
    fn a_repeated_plane_index_is_refused() {
        let planes = Some(vec![0, 0]);
        let refused = resolved_planes(&planes, FrameLayout::Rgb);

        assert!(refused.is_err());
    }

    #[test]
    fn an_empty_planes_list_is_refused() {
        let planes = Some(Vec::new());
        let refused = resolved_planes(&planes, FrameLayout::Rgb);

        assert!(refused.is_err());
    }
}
