use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::heatmap::{self, HeatmapPlanes, PaintLayout, Painting, Style};
use av_toolbox_cubek::matrix::{Coefficients, SampleRange};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ColourRange,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Matrix,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::bits::sample_kind;
use crate::checks::accepts_integer_or_f32_frames;
use crate::scratch::{Scratch, device_plane};
use crate::sizing::Sizing;

const STAGE: &str = "blend/heatmap";
const CLIP: usize = 0;
const MAP: usize = 1;

/// How a heatmap is coloured, as the stage's arguments name it.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HeatmapStyle {
    /// A greyscale picture shaded from white to deep blue.
    #[default]
    Blue,
    /// A lifted greyscale picture shaded from cyan through yellow to red.
    Heat,
}

impl From<HeatmapStyle> for Style {
    fn from(style: HeatmapStyle) -> Self {
        match style {
            HeatmapStyle::Blue => Self::Blue,
            HeatmapStyle::Heat => Self::Heat,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct HeatmapArgs {
    /// How the difference is coloured.
    #[serde(default)]
    pub(crate) style: HeatmapStyle,
    /// The map value that reaches full colour.
    pub(crate) scale: f32,
}

/// Paints a distortion map over a greyscale copy of a clip.
pub(crate) struct Heatmap;

impl StageDefinition for Heatmap {
    type Args = HeatmapArgs;

    fn package() -> &'static str {
        "blend"
    }

    fn identifier() -> &'static str {
        "heatmap"
    }

    fn description() -> &'static str {
        "Paints a distortion map over a clip as a heatmap.\n\n\
         The clip turns greyscale and the map colours it, reaching full colour where the map reaches scale."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("clip")
                .accepting(StreamType::Frames)
                .chunk_authority(),
            InputParam::required("map").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} paints frames, not masks");

    StageError::msg(message)
}

fn not_yuv_or_rgb(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} paints yuv or rgb clips, not {layout} clips");

    StageError::msg(message)
}

fn map_not_grey(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} needs a grey map, got a {layout} map");

    StageError::msg(message)
}

fn map_not_f32(sample: SampleFormat) -> StageError {
    let message = format!("{STAGE} needs an F32 map, got {sample}");

    StageError::msg(message)
}

fn map_wrong_size(clip: &FrameFormat, map: &FrameFormat) -> StageError {
    let message = format!(
        "{STAGE} needs a {}x{} map to match the clip, got {}x{}. The map needs a resize to the clip's size first",
        clip.width, clip.height, map.width, map.height,
    );

    StageError::msg(message)
}

fn bad_scale(scale: f32) -> StageError {
    let message = format!("{STAGE} needs a finite scale above 0, got {scale}");

    StageError::msg(message)
}

fn unsupported_sample(sample: SampleFormat) -> StageError {
    let message = format!("{STAGE} paints u8, u16 or f32 samples, not {sample}");

    StageError::msg(message)
}

fn missing_inputs() -> StageError {
    let message = format!("{STAGE} needs clip and map bound");

    StageError::msg(message)
}

fn missing_clip() -> StageError {
    let message = format!("{STAGE} expected the clip in slot 0");

    StageError::msg(message)
}

fn missing_map() -> StageError {
    let message = format!("{STAGE} expected the map in slot 1");

    StageError::msg(message)
}

fn missing_channel() -> StageError {
    let message = format!("{STAGE} was handed a frame missing a channel");

    StageError::msg(message)
}

fn accepts_clip(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(samples) = accepts_integer_or_f32_frames(offered) else {
        return Err(not_frames());
    };

    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    if !matches!(format.layout, FrameLayout::Yuv { .. } | FrameLayout::Rgb) {
        return Err(not_yuv_or_rgb(format.layout));
    }

    Ok(samples)
}

fn accepts_map(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    if format.layout != FrameLayout::Grey {
        return Err(map_not_grey(format.layout));
    }

    if format.sample != SampleFormat::F32 {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([SampleFormat::F32]))
}

/// The clip format, once the scale and every input agree with it.
fn checked_clip(args: &HeatmapArgs, chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    if !args.scale.is_finite() || args.scale <= 0.0 {
        return Err(bad_scale(args.scale));
    }

    let clip = chosen.get(CLIP).copied().flatten();
    let map = chosen.get(MAP).copied().flatten();
    let (Some(clip), Some(map)) = (clip, map) else {
        return Err(missing_inputs());
    };

    let clip = clip.as_frames().copied().ok_or_else(not_frames)?;
    let map = map.as_frames().copied().ok_or_else(not_frames)?;

    if !matches!(clip.layout, FrameLayout::Yuv { .. } | FrameLayout::Rgb) {
        return Err(not_yuv_or_rgb(clip.layout));
    }

    if map.layout != FrameLayout::Grey {
        return Err(map_not_grey(map.layout));
    }

    if map.sample != SampleFormat::F32 {
        return Err(map_not_f32(map.sample));
    }

    if map.width != clip.width || map.height != clip.height {
        return Err(map_wrong_size(&clip, &map));
    }

    Ok(clip)
}

/// The clip's matrix, painting with BT.709 when it has no YUV matrix.
fn coefficients(matrix: Matrix) -> Coefficients {
    match matrix {
        Matrix::Bt601 => Coefficients::Bt601,
        Matrix::Bt2020Ncl => Coefficients::Bt2020Ncl,
        Matrix::Bt709 | Matrix::Unspecified | Matrix::Identity => Coefficients::Bt709,
    }
}

/// The clip's range, taking limited for YUV and full for RGB when it is unspecified.
fn sample_range(range: ColourRange, layout: PaintLayout) -> SampleRange {
    match (range, layout) {
        (ColourRange::Limited, _) => SampleRange::Limited,
        (ColourRange::Full, _) => SampleRange::Full,
        (ColourRange::Unspecified, PaintLayout::Yuv { .. }) => SampleRange::Limited,
        (ColourRange::Unspecified, PaintLayout::Rgb) => SampleRange::Full,
    }
}

/// How `args` paints over a clip of `clip`.
fn painting(args: &HeatmapArgs, clip: &FrameFormat) -> Result<Painting, StageError> {
    let layout = match clip.layout {
        FrameLayout::Rgb => PaintLayout::Rgb,
        FrameLayout::Yuv { subsampling } => {
            let (width_factor, height_factor) = subsampling.factors();
            PaintLayout::Yuv {
                width_shift: width_factor.trailing_zeros(),
                height_shift: height_factor.trailing_zeros(),
            }
        },
        layout => return Err(not_yuv_or_rgb(layout)),
    };

    let Some(samples) = sample_kind(clip.sample) else {
        return Err(unsupported_sample(clip.sample));
    };

    let range = sample_range(clip.colour.range, layout);
    let coefficients = coefficients(clip.colour.matrix);

    Ok(Painting {
        style: args.style.into(),
        scale: args.scale,
        layout,
        samples,
        range,
        coefficients,
    })
}

impl FilterDefinition for Heatmap {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        match index {
            CLIP => accepts_clip(offered),
            _ => accepts_map(offered),
        }
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let clip = checked_clip(args, chosen)?;
        let format = StreamFormat::Frames(clip);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([clip.sample]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let clip = checked_clip(&args, inputs)?;
        let painting = painting(&args, &clip)?;

        let client = context.client().clone();
        let heatmap = heatmap::Heatmap::new(&client, painting, clip.height);
        let painter = HeatmapPainter {
            client,
            clip,
            heatmap,
        };

        Ok(Box::new(painter))
    }
}

/// Paints every map frame over its clip frame.
struct HeatmapPainter<R: Runtime> {
    client: ComputeClient<R>,
    clip: FrameFormat,
    heatmap: heatmap::Heatmap,
}

/// The three planes of `frame` in `format`'s channel order.
fn frame_planes<'a>(frame: &'a Frame, format: &FrameFormat) -> Result<[&'a Plane; 3], StageError> {
    let mut found = Vec::with_capacity(3);

    for channel in format.channels() {
        let Some(plane) = frame.plane(*channel) else {
            return Err(missing_channel());
        };

        found.push(plane);
    }

    let Ok(planes) = <[&Plane; 3]>::try_from(found) else {
        return Err(missing_channel());
    };

    Ok(planes)
}

/// The planes as device planes, read or written as the samples they hold.
///
/// # Safety
///
/// Every kernel given the returned planes must use their sample format as its element type.
unsafe fn device_planes(planes: [&Plane; 3]) -> [DevicePlane<'_>; 3] {
    // Safety: the caller guarantees the element type, and a plane holds `width * height` of them.
    planes.map(|plane| unsafe { device_plane(plane) })
}

impl<R: Runtime> StageInstance for HeatmapPainter<R> {}

impl<R: Runtime> Filter for HeatmapPainter<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let clip = inputs.take_frame(CLIP).ok_or_else(missing_clip)?;
        let map = inputs.take_frame(MAP).ok_or_else(missing_map)?;

        let meta = map.meta().clone();
        let output = context.alloc_frame(&self.clip, meta);
        let sources = frame_planes(&clip, &self.clip)?;
        let targets = frame_planes(&output, &self.clip)?;
        let map_plane = map.plane(Channel::Y).ok_or_else(missing_channel)?;

        let (width, height) = (self.clip.width, self.clip.height);
        let scratch = Scratch::new::<f32>(context, width, height);
        let smoothed = Scratch::new::<f32>(context, width, height);

        // Safety: both frames are in the clip's format, which the painting was built for, so their
        // planes hold the samples `apply` reads and writes.
        let picture = unsafe { device_planes(sources) };

        // Safety: as above.
        let output_planes = unsafe { device_planes(targets) };

        // Safety: the map is an F32 grey frame, checked when the stage was built, and `apply`
        // reads it as f32.
        let map_device = unsafe { device_plane(map_plane) };

        let planes = HeatmapPlanes {
            picture,
            map: map_device,
            scratch: scratch.plane(),
            smoothed: smoothed.plane(),
            output: output_planes,
        };
        let sizing = Sizing(context);
        self.heatmap.apply(&self.client, &sizing, planes);

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, Depth, Subsampling};

    use super::*;

    fn frames(width: u32, height: u32, layout: FrameLayout, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width,
            height,
            layout,
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        })
    }

    fn yuv420(sample: SampleFormat) -> StreamFormat {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        frames(8, 6, layout, sample)
    }

    fn grey_map(width: u32, height: u32, sample: SampleFormat) -> StreamFormat {
        frames(width, height, FrameLayout::Grey, sample)
    }

    fn args(scale: f32) -> HeatmapArgs {
        HeatmapArgs {
            style: HeatmapStyle::Blue,
            scale,
        }
    }

    #[test]
    fn style_defaults_to_blue() {
        let given = serde_json::json!({ "scale": 0.003 });
        let parsed: HeatmapArgs = serde_json::from_value(given).expect("scale is given");

        assert_eq!(parsed.style, HeatmapStyle::Blue);
        assert_eq!(Style::from(parsed.style), Style::Blue);
    }

    #[test]
    fn a_missing_scale_is_refused() {
        let given = serde_json::json!({ "style": "heat" });
        let parsed = serde_json::from_value::<HeatmapArgs>(given);

        assert!(parsed.is_err());
    }

    #[test]
    fn a_scale_of_zero_or_below_or_not_finite_is_refused() {
        let chosen = [
            Some(yuv420(SampleFormat::U8)),
            Some(grey_map(8, 6, SampleFormat::F32)),
        ];

        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let refused = checked_clip(&args(scale), &chosen);

            assert!(refused.is_err(), "a scale of {scale} was accepted");
        }
    }

    #[test]
    fn yuv_and_rgb_clips_are_accepted_in_integer_and_f32() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let samples = [SampleFormat::U8, ten_bit, SampleFormat::F32];

        for sample in samples {
            let yuv = yuv420(sample);
            let rgb = frames(8, 6, FrameLayout::Rgb, sample);
            let yuv_accepted = accepts_clip(yuv).expect("yuv is accepted");
            let rgb_accepted = accepts_clip(rgb).expect("rgb is accepted");

            assert!(yuv_accepted.accepts(sample), "yuv in {sample}");
            assert!(rgb_accepted.accepts(sample), "rgb in {sample}");
        }
    }

    #[test]
    fn a_grey_clip_is_refused() {
        let grey = frames(8, 6, FrameLayout::Grey, SampleFormat::U8);
        let error = accepts_clip(grey).expect_err("grey cannot be painted");
        let message = error.to_string();

        assert!(message.contains("yuv or rgb"), "{message}");
    }

    #[test]
    fn an_f32_grey_map_is_accepted() {
        let map = grey_map(8, 6, SampleFormat::F32);
        let accepted = accepts_map(map).expect("an f32 grey map is accepted");

        assert!(accepted.accepts(SampleFormat::F32));
    }

    #[test]
    fn a_u8_map_is_left_out() {
        let map = grey_map(8, 6, SampleFormat::U8);
        let accepted = accepts_map(map).expect("grey frames are readable");

        assert!(accepted.is_empty());
    }

    #[test]
    fn a_yuv_map_is_refused() {
        let map = yuv420(SampleFormat::F32);
        let refused = accepts_map(map);

        assert!(refused.is_err());
    }

    #[test]
    fn a_map_of_another_size_is_refused() {
        let chosen = [
            Some(yuv420(SampleFormat::U8)),
            Some(grey_map(16, 12, SampleFormat::F32)),
        ];

        let error = checked_clip(&args(0.003), &chosen).expect_err("the map is a different size");
        let message = error.to_string();

        assert!(message.contains("resize"), "{message}");
    }

    #[test]
    fn the_output_is_the_clip_format() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let chosen = [Some(yuv420(ten_bit)), Some(grey_map(8, 6, SampleFormat::F32))];

        let clip = checked_clip(&args(0.003), &chosen).expect("everything agrees");

        assert_eq!(StreamFormat::Frames(clip), yuv420(ten_bit));
    }
}
