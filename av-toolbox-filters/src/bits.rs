use av_toolbox_gpu::SampleFormat;

/// The bits one sample of `sample` holds.
pub(crate) fn bits_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        _ => 8,
    }
}
