use av_toolbox_core::{InputParam, InputSignature, StageError, StreamType};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::lineart::{MergePlanes, merge};
use av_toolbox_cubek::values::widen;
use av_toolbox_gpu::{
    Channel,
    ComputeClient,
    FormatSet,
    FrameContext,
    FrameFormat,
    Mask,
    MaskFormat,
    Runtime,
    SampleFormat,
    StreamFormat,
};

use crate::planewise::PlaneFilter;
use crate::scratch::Scratch;
use crate::sizing::Sizing;

/// The slot of the frames every lineart stage filters.
pub(crate) const INPUT: usize = 0;

/// The slot of the optional mask every lineart stage merges through.
pub(crate) const MASK: usize = 1;

/// The largest value a mask sample holds.
const MASK_PEAK: f32 = 255.0;

/// The inputs every lineart stage takes, a required `input` and an optional `mask` carrying
/// planes to match it.
pub(crate) const LINEART_INPUTS: &InputSignature = &InputSignature::new(&[
    InputParam::required("input").accepting(StreamType::Frames),
    InputParam::optional("mask")
        .accepting(StreamType::Mask)
        .planes_of("input"),
]);

/// The samples a lineart stage accepts on `index`, U8 and U16 frames for the input and a U8
/// mask for the mask.
pub(crate) fn accepts_lineart_input(
    stage: &str,
    index: usize,
    offered: StreamFormat,
) -> Result<FormatSet, StageError> {
    match index {
        INPUT => accepts_frames(stage, offered),
        MASK => accepts_mask(stage, offered),
        _ => Err(unexpected_input(stage, index)),
    }
}

fn accepts_frames(stage: &str, offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(input_not_frames(stage));
    };

    if !matches!(format.sample, SampleFormat::U8 | SampleFormat::U16 { .. }) {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

fn accepts_mask(stage: &str, offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_mask() else {
        return Err(mask_not_a_mask(stage));
    };

    if format.sample != SampleFormat::U8 {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([SampleFormat::U8]))
}

/// Refuses a bound mask that is not U8 or not the size of `frame`.
pub(crate) fn checked_mask(
    stage: &str,
    frame: &FrameFormat,
    mask: Option<StreamFormat>,
) -> Result<(), StageError> {
    let Some(mask) = mask else {
        return Ok(());
    };

    let Some(format) = mask.as_mask() else {
        return Err(mask_not_a_mask(stage));
    };

    if format.width != frame.width || format.height != frame.height {
        return Err(mask_of_another_size(stage, frame, format));
    }

    if format.sample != SampleFormat::U8 {
        return Err(wide_mask(stage, format.sample));
    }

    Ok(())
}

fn unexpected_input(stage: &str, index: usize) -> StageError {
    let message = format!("{stage} takes only input and mask, but was offered a stream at slot {index}");

    StageError::msg(message)
}

fn input_not_frames(stage: &str) -> StageError {
    let message = format!("{stage} needs frames on its input, not a mask");

    StageError::msg(message)
}

fn mask_not_a_mask(stage: &str) -> StageError {
    let message = format!("{stage} needs a mask on its mask input, not frames");

    StageError::msg(message)
}

fn mask_of_another_size(stage: &str, frame: &FrameFormat, mask: &MaskFormat) -> StageError {
    let message = format!(
        "{stage} needs a {}x{} mask to match its frames, got {}x{}",
        frame.width, frame.height, mask.width, mask.height,
    );

    StageError::msg(message)
}

fn wide_mask(stage: &str, sample: SampleFormat) -> StageError {
    let message = format!("{stage} needs a U8 mask, got {sample}");

    StageError::msg(message)
}

fn missing_channel(stage: &str, plane_index: usize) -> StageError {
    let message = format!("{stage} was asked to filter plane {plane_index}, which its frames lack");

    StageError::msg(message)
}

fn mask_plane_of_another_size(
    stage: &str,
    channel: Channel,
    mask: (u32, u32),
    plane: (u32, u32),
) -> StageError {
    let message = format!(
        "{stage} was handed a {}x{} mask plane for {channel}, which is {}x{}",
        mask.0, mask.1, plane.0, plane.1,
    );

    StageError::msg(message)
}

fn missing_mask_plane(stage: &str, channel: Channel) -> StageError {
    let message = format!("{stage} was handed a mask without a plane to match {channel}");

    StageError::msg(message)
}

/// Runs `inner` on a plane, then moves the input toward its output by the frame's mask.
///
/// Each plane becomes `input + (filtered - input) * mask / 255`, using the mask's full plane for a
/// full size plane and its chroma plane for a subsampled one. Without a mask the plane is the
/// inner filter's output.
///
/// The merge reads `input` after the inner filter runs, so the inner filter must leave `input`
/// unchanged.
pub(crate) struct MaskedFilter<'a, R: Runtime, F> {
    pub(crate) client: &'a ComputeClient<R>,
    pub(crate) stage: &'static str,
    /// The format of the frame being filtered.
    pub(crate) format: &'a FrameFormat,
    /// The frame's mask, `None` when no mask is bound.
    pub(crate) mask: Option<&'a Mask>,
    pub(crate) inner: &'a F,
}

impl<R: Runtime, F: PlaneFilter> PlaneFilter for MaskedFilter<'_, R, F> {
    fn filter(
        &self,
        context: &FrameContext,
        plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        self.inner.filter(context, plane_index, input, output)?;

        let Some(mask) = self.mask else {
            return Ok(());
        };

        let channels = self.format.channels();
        let Some(channel) = channels.get(plane_index).copied() else {
            return Err(missing_channel(self.stage, plane_index));
        };

        let Some(mask_plane) = mask.plane_for(channel, self.format) else {
            return Err(missing_mask_plane(self.stage, channel));
        };

        let (width, height) = mask_plane.dimensions();
        let plane_size = (input.width(), input.height());
        if (width, height) != plane_size {
            return Err(mask_plane_of_another_size(
                self.stage,
                channel,
                (width, height),
                plane_size,
            ));
        }

        let sizing = Sizing(context);
        let mask_handle = mask_plane.buffer().handle();

        // Safety: a mask's plane holds exactly `width * height` U8 samples.
        let mask_samples = unsafe { DevicePlane::new(mask_handle, width, height) };

        let weight = Scratch::new::<f32>(context, width, height);
        let weight_plane = weight.plane();
        widen(self.client, &sizing, mask_samples, weight_plane, Element::U8);

        let planes = MergePlanes {
            input,
            filtered: output,
            weight: weight_plane,
        };
        merge(self.client, &sizing, planes, MASK_PEAK);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, Depth, FrameLayout, Subsampling};

    use super::*;

    const STAGE: &str = "lineart/test";

    fn frame_format(sample: SampleFormat) -> FrameFormat {
        FrameFormat {
            width: 8,
            height: 6,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
        }
    }

    fn mask(width: u32, height: u32, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Mask(MaskFormat {
            width,
            height,
            sample,
            chroma: None,
        })
    }

    #[test]
    fn the_mask_is_optional_and_asks_for_planes_of_input() {
        let params = LINEART_INPUTS.params();
        let input = params[INPUT];
        let mask = params[MASK];

        assert_eq!(input.name(), "input");
        assert!(input.is_required());
        assert_eq!(mask.name(), "mask");
        assert!(!mask.is_required());
        assert_eq!(mask.accepts(), Some(StreamType::Mask));
        assert_eq!(mask.planes_of_input(), Some("input"));
    }

    #[test]
    fn integer_frames_and_a_u8_mask_are_accepted() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let eight_bit_frames = StreamFormat::Frames(frame_format(SampleFormat::U8));
        let ten_bit_frames = StreamFormat::Frames(frame_format(ten_bit));
        let u8_mask = mask(8, 6, SampleFormat::U8);

        let eight_bit = accepts_lineart_input(STAGE, INPUT, eight_bit_frames).expect("frames are readable");
        let ten_bit_set = accepts_lineart_input(STAGE, INPUT, ten_bit_frames).expect("frames are readable");
        let masks = accepts_lineart_input(STAGE, MASK, u8_mask).expect("a mask is readable");

        assert!(eight_bit.accepts(SampleFormat::U8));
        assert!(ten_bit_set.accepts(ten_bit));
        assert!(masks.accepts(SampleFormat::U8));
    }

    #[test]
    fn float_frames_and_a_wide_mask_are_left_out() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let float_frames = StreamFormat::Frames(frame_format(SampleFormat::F32));
        let wide_mask = mask(8, 6, ten_bit);

        let floats = accepts_lineart_input(STAGE, INPUT, float_frames).expect("frames are readable");
        let wide = accepts_lineart_input(STAGE, MASK, wide_mask).expect("a mask is readable");

        assert!(floats.is_empty());
        assert!(wide.is_empty());
    }

    #[test]
    fn a_mask_in_the_input_slot_is_refused() {
        let wrong = mask(8, 6, SampleFormat::U8);

        let error = accepts_lineart_input(STAGE, INPUT, wrong).expect_err("the input takes frames");
        let message = error.to_string();

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("its input"), "{message}");
        assert!(message.contains("not a mask"), "{message}");
    }

    #[test]
    fn frames_in_the_mask_slot_are_refused() {
        let frames = StreamFormat::Frames(frame_format(SampleFormat::U8));

        let error = accepts_lineart_input(STAGE, MASK, frames).expect_err("the mask slot takes a mask");
        let message = error.to_string();

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("mask input"), "{message}");
        assert!(message.contains("frames"), "{message}");
    }

    #[test]
    fn a_stream_past_the_mask_slot_is_refused() {
        let frames = StreamFormat::Frames(frame_format(SampleFormat::U8));

        let error = accepts_lineart_input(STAGE, 2, frames).expect_err("there are only two slots");
        let message = error.to_string();

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("slot 2"), "{message}");
    }

    #[test]
    fn a_mask_of_another_size_is_refused() {
        let frame = frame_format(SampleFormat::U8);
        let small = mask(4, 3, SampleFormat::U8);

        let error = checked_mask(STAGE, &frame, Some(small)).expect_err("the mask is smaller");
        let message = error.to_string();

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("mask"), "{message}");
        assert!(message.contains("4x3"), "{message}");
    }

    #[test]
    fn a_wide_mask_is_refused() {
        let frame = frame_format(SampleFormat::U8);
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let wide = mask(8, 6, ten_bit);

        let error = checked_mask(STAGE, &frame, Some(wide)).expect_err("the mask is not U8");
        let message = error.to_string();
        let sample = ten_bit.to_string();

        assert!(message.contains(STAGE), "{message}");
        assert!(message.contains("mask"), "{message}");
        assert!(message.contains(&sample), "{message}");
    }

    #[test]
    fn a_matching_mask_is_accepted() {
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let frame = frame_format(ten_bit);
        let matching = mask(8, 6, SampleFormat::U8);

        let checked = checked_mask(STAGE, &frame, Some(matching));

        assert!(checked.is_ok());
    }

    #[test]
    fn no_mask_is_accepted() {
        let frame = frame_format(SampleFormat::U8);

        let checked = checked_mask(STAGE, &frame, None);

        assert!(checked.is_ok());
    }
}
