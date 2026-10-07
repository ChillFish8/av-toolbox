use av_toolbox_core::StageError;
use av_toolbox_gpu::{
    Depth as BitDepth,
    FormatSet,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    SampleFormat,
    StreamFormat,
};

/// Checks that a stage accepts integer and F32 frames as they are, accepts no F16 frame, and refuses a mask
/// by name.
pub(crate) fn assert_integers_and_floats_accepted_and_mask_refused(
    stage: &str,
    layout: FrameLayout,
    frame_format: impl Fn(FrameLayout, SampleFormat) -> FrameFormat,
    accepted: impl Fn(StreamFormat) -> Result<FormatSet, StageError>,
) {
    let samples = [
        SampleFormat::U8,
        SampleFormat::U16 {
            depth: BitDepth::SIXTEEN,
        },
        SampleFormat::F32,
    ];
    let half = frame_format(layout, SampleFormat::F16);
    let mask = StreamFormat::Mask(MaskFormat {
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    for sample in samples {
        let input = frame_format(layout, sample);
        let set = accepted(StreamFormat::Frames(input)).expect("frames are readable");

        assert_eq!(set, FormatSet::of([sample]));
    }

    let half_set = accepted(StreamFormat::Frames(half)).expect("frames are readable");
    let mask_error = accepted(mask).expect_err("a mask is refused");
    let mask_message = mask_error.to_string();

    assert!(half_set.is_empty());
    assert!(mask_message.contains(stage), "{mask_message}");
    assert!(mask_message.contains("mask"), "{mask_message}");
}
