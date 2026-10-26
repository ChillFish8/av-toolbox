use av_toolbox_core::StageError;
use av_toolbox_gpu::{
    Capabilities,
    Depth as BitDepth,
    ElementType,
    FormatSet,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    SampleFormat,
    StreamFormat,
    default_accelerators,
};

/// Checks that a stage accepts integer and F32 frames as they are, and refuses a mask by name.
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
    let mask = StreamFormat::Mask(MaskFormat {
        frame_rate: None,
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

    let mask_error = accepted(mask).expect_err("a mask is refused");
    let mask_message = mask_error.to_string();

    assert!(mask_message.contains(stage), "{mask_message}");
    assert!(mask_message.contains("mask"), "{mask_message}");
}

/// Capabilities of the first default accelerator that can compute each of `elements`.
pub(crate) fn capabilities_computing(elements: &[ElementType]) -> Capabilities {
    let accelerator = default_accelerators()[0];
    let elements = elements.iter().copied();

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .computable(elements)
        .build()
}
