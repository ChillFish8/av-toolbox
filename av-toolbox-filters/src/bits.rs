use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_gpu::SampleFormat;

/// The bits one sample of `sample` holds.
pub(crate) fn bits_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        _ => 8,
    }
}

/// How a kernel that takes integers and F32 stores samples of `sample`, or `None` for F16.
pub(crate) fn sample_kind(sample: SampleFormat) -> Option<MatrixSamples> {
    match sample {
        SampleFormat::U8 => Some(MatrixSamples::U8),
        SampleFormat::U16 { depth } => Some(MatrixSamples::U16 {
            bits: u32::from(depth.bits()),
        }),
        SampleFormat::F32 => Some(MatrixSamples::F32),
        SampleFormat::F16 => None,
    }
}
