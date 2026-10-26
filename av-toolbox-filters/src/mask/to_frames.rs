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
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ColourInfo,
    ColourRange,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

const STAGE: &str = "mask/to_frames";
const INPUT: usize = 0;

/// Which plane of a mask becomes the frames.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MaskPlane {
    /// The plane at the mask's own size.
    #[default]
    Full,
    /// The subsampled plane, for masks that carry one.
    Chroma,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToFramesArgs {
    /// Which plane of the mask becomes the frames.
    #[serde(default)]
    pub(crate) plane: MaskPlane,
}

/// Turns one plane of a mask into float grey frames.
pub(crate) struct ToFrames;

impl StageDefinition for ToFrames {
    type Args = ToFramesArgs;

    fn package() -> &'static str {
        "mask"
    }

    fn identifier() -> &'static str {
        "to_frames"
    }

    fn description() -> &'static str {
        "Turns one plane of a mask into frames, so it can be painted or viewed.\n\n\
         Each frame is the mask's value between 0 and 1, as float grey."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")
            .accepting(StreamType::Mask)
            .chunk_authority()]);
        SIGNATURE
    }
}

/// Writes each mask sample as its share of 255.
#[cube(launch_unchecked)]
fn share_of_peak(mask: &Array<u8>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = f32::cast_from(mask[ABSOLUTE_POS]) / 255.0;
}

fn not_a_mask() -> StageError {
    let message = format!("{STAGE} reads masks, not frames");

    StageError::msg(message)
}

fn not_u8(sample: SampleFormat) -> StageError {
    let message = format!("{STAGE} needs a U8 mask, got {sample}");

    StageError::msg(message)
}

fn no_chroma_plane() -> StageError {
    let message = format!("{STAGE} was asked for the chroma plane of a mask with no chroma plane");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn missing_mask() -> StageError {
    let message = format!("{STAGE} expected a mask in slot 0");

    StageError::msg(message)
}

fn missing_mask_plane() -> StageError {
    let message = format!("{STAGE} was handed a mask without its chroma plane");

    StageError::msg(message)
}

fn missing_output_plane() -> StageError {
    let message = format!("{STAGE} allocated a frame without its grey plane");

    StageError::msg(message)
}

fn accepts_mask(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_mask() else {
        return Err(not_a_mask());
    };

    if format.sample != SampleFormat::U8 {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([SampleFormat::U8]))
}

/// The size of the plane `plane` picks out of `mask`.
fn plane_size(mask: &MaskFormat, plane: MaskPlane) -> Result<(u32, u32), StageError> {
    match plane {
        MaskPlane::Full => Ok((mask.width, mask.height)),
        MaskPlane::Chroma => mask.chroma_dimensions().ok_or_else(no_chroma_plane),
    }
}

/// The F32 grey frames the chosen plane of the bound mask becomes.
fn checked_output(args: &ToFramesArgs, chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.get(INPUT).copied().flatten() else {
        return Err(missing_input());
    };

    let mask = input.as_mask().copied().ok_or_else(not_a_mask)?;
    if mask.sample != SampleFormat::U8 {
        return Err(not_u8(mask.sample));
    }

    let (width, height) = plane_size(&mask, args.plane)?;
    let colour = ColourInfo {
        range: ColourRange::Full,
        ..ColourInfo::default()
    };

    Ok(FrameFormat {
        width,
        height,
        layout: FrameLayout::Grey,
        sample: SampleFormat::F32,
        colour,
        frame_rate: None,
        field_order: None,
    })
}

impl FilterDefinition for ToFrames {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_mask(offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let output = checked_output(args, chosen)?;
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([SampleFormat::F32]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let output = checked_output(&args, inputs)?;
        let client = context.client().clone();
        let converter = MaskToFrames {
            client,
            plane: args.plane,
            output,
        };

        Ok(Box::new(converter))
    }
}

/// Converts the chosen plane of every mask into a frame.
struct MaskToFrames<R: Runtime> {
    client: ComputeClient<R>,
    plane: MaskPlane,
    output: FrameFormat,
}

impl<R: Runtime> StageInstance for MaskToFrames<R> {}

impl<R: Runtime> Filter for MaskToFrames<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mask = inputs.take_mask(INPUT).ok_or_else(missing_mask)?;
        let mask_plane = match self.plane {
            MaskPlane::Full => mask.plane(),
            MaskPlane::Chroma => mask.chroma_plane().ok_or_else(missing_mask_plane)?,
        };

        let meta = mask.meta().clone();
        let output = context.alloc_frame(&self.output, meta);
        let output_plane = output.plane(Channel::Y).ok_or_else(missing_output_plane)?;

        let samples = output_plane.samples();
        let cubes = context.cube_count(samples);
        let shape = context.cube_dim();
        let read = mask_plane.array::<u8, R>();
        let write = output_plane.array::<f32, R>();

        // Safety: both bindings name exactly as many elements as their plane holds, and the kernel
        // writes nothing past the output's length.
        unsafe {
            share_of_peak::launch_unchecked::<R>(&self.client, cubes, shape, read, write);
        }

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ChromaLocation, Depth, MaskChroma, Subsampling};

    use super::*;

    fn mask(width: u32, height: u32, sample: SampleFormat, chroma: Option<MaskChroma>) -> StreamFormat {
        StreamFormat::Mask(MaskFormat {
            width,
            height,
            sample,
            chroma,
        })
    }

    fn chroma_420() -> Option<MaskChroma> {
        Some(MaskChroma {
            subsampling: Subsampling::Yuv420,
            location: ChromaLocation::Left,
        })
    }

    fn args(plane: MaskPlane) -> ToFramesArgs {
        ToFramesArgs { plane }
    }

    #[test]
    fn a_u8_mask_is_accepted() {
        let offered = mask(8, 6, SampleFormat::U8, None);
        let accepted = accepts_mask(offered).expect("a mask is readable");

        assert!(accepted.accepts(SampleFormat::U8));
    }

    #[test]
    fn a_mask_of_another_sample_is_left_out() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let offered = mask(8, 6, ten_bit, None);
        let accepted = accepts_mask(offered).expect("a mask is readable");

        assert!(accepted.is_empty());
    }

    #[test]
    fn frames_are_refused() {
        let offered = StreamFormat::Frames(FrameFormat {
            width: 8,
            height: 6,
            layout: FrameLayout::Grey,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        });
        let refused = accepts_mask(offered);

        assert!(refused.is_err());
    }

    #[test]
    fn the_output_is_f32_grey_at_the_mask_size() {
        let chosen = [Some(mask(12, 6, SampleFormat::U8, chroma_420()))];
        let output = checked_output(&args(MaskPlane::Full), &chosen).expect("the mask is readable");

        assert_eq!(output.sample, SampleFormat::F32);
        assert_eq!(output.layout, FrameLayout::Grey);
        assert_eq!(output.colour.range, ColourRange::Full);
        assert_eq!((output.width, output.height), (12, 6));
        assert_eq!(output.frame_rate, None);
        assert_eq!(output.field_order, None);
    }

    #[test]
    fn plane_defaults_to_full() {
        let given = serde_json::json!({});
        let parsed: ToFramesArgs = serde_json::from_value(given).expect("the defaults apply");

        assert_eq!(parsed.plane, MaskPlane::Full);
    }

    #[test]
    fn the_chroma_plane_is_the_subsampled_size() {
        let given = serde_json::json!({ "plane": "chroma" });
        let parsed: ToFramesArgs = serde_json::from_value(given).expect("chroma is a plane");
        let chosen = [Some(mask(13, 7, SampleFormat::U8, chroma_420()))];

        let output = checked_output(&parsed, &chosen).expect("the mask has a chroma plane");

        assert_eq!((output.width, output.height), (7, 4));
    }

    #[test]
    fn chroma_on_a_mask_without_chroma_is_refused() {
        let chosen = [Some(mask(12, 6, SampleFormat::U8, None))];
        let error = checked_output(&args(MaskPlane::Chroma), &chosen).expect_err("there is no chroma plane");
        let message = error.to_string();

        assert!(message.contains("chroma plane"), "{message}");
    }

    #[test]
    fn unknown_arguments_are_refused() {
        let given = serde_json::json!({ "scale": 1 });
        let parsed = serde_json::from_value::<ToFramesArgs>(given);

        assert!(parsed.is_err());
    }
}
