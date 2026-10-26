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

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToFramesArgs {}

/// Turns a mask into float grey frames.
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
        "Turns a mask into frames, so it can be painted or viewed.\n\n\
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

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn missing_mask() -> StageError {
    let message = format!("{STAGE} expected a mask in slot 0");

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

/// The F32 grey frames the bound mask becomes.
fn checked_output(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.get(INPUT).copied().flatten() else {
        return Err(missing_input());
    };

    let mask = input.as_mask().copied().ok_or_else(not_a_mask)?;
    if mask.sample != SampleFormat::U8 {
        return Err(not_u8(mask.sample));
    }

    let colour = ColourInfo {
        range: ColourRange::Full,
        ..ColourInfo::default()
    };

    Ok(FrameFormat {
        width: mask.width,
        height: mask.height,
        layout: FrameLayout::Grey,
        sample: SampleFormat::F32,
        colour,
        frame_rate: mask.frame_rate,
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
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let output = checked_output(chosen)?;
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([SampleFormat::F32]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let output = checked_output(inputs)?;
        let client = context.client().clone();
        let converter = MaskToFrames { client, output };

        Ok(Box::new(converter))
    }
}

/// Converts the full plane of every mask into a frame.
struct MaskToFrames<R: Runtime> {
    client: ComputeClient<R>,
    output: FrameFormat,
}

impl<R: Runtime> StageInstance for MaskToFrames<R> {}

impl<R: Runtime> Filter for MaskToFrames<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mask = inputs.take_mask(INPUT).ok_or_else(missing_mask)?;
        let mask_plane = mask.plane();

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
    use av_toolbox_gpu::{ChromaLocation, Depth, FrameRate, MaskChroma, MaskFormat, Subsampling};

    use super::*;

    fn mask(width: u32, height: u32, sample: SampleFormat, chroma: Option<MaskChroma>) -> StreamFormat {
        StreamFormat::Mask(MaskFormat {
            frame_rate: None,
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
    fn the_frames_run_at_the_masks_rate() {
        let rate = FrameRate::new(24000, 1001);
        let StreamFormat::Mask(plain) = mask(12, 6, SampleFormat::U8, None) else {
            unreachable!("mask builds masks");
        };
        let rated = MaskFormat {
            frame_rate: rate,
            ..plain
        };
        let chosen = [Some(StreamFormat::Mask(rated))];

        let output = checked_output(&chosen).expect("the mask is readable");

        assert_eq!(output.frame_rate, rate);
    }

    #[test]
    fn the_output_is_f32_grey_at_the_mask_size() {
        let chosen = [Some(mask(12, 6, SampleFormat::U8, chroma_420()))];
        let output = checked_output(&chosen).expect("the mask is readable");

        assert_eq!(output.sample, SampleFormat::F32);
        assert_eq!(output.layout, FrameLayout::Grey);
        assert_eq!(output.colour.range, ColourRange::Full);
        assert_eq!((output.width, output.height), (12, 6));
        assert_eq!(output.frame_rate, None);
        assert_eq!(output.field_order, None);
    }

    #[test]
    fn unknown_arguments_are_refused() {
        for given in [
            serde_json::json!({ "scale": 1 }),
            serde_json::json!({ "plane": "full" }),
        ] {
            let parsed = serde_json::from_value::<ToFramesArgs>(given);

            assert!(parsed.is_err());
        }
    }
}
