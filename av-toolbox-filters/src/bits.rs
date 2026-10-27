use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_gpu::SampleFormat;

use crate::codes::FLOAT_CODE_BITS;

/// The bits one sample of `sample` holds. A float sample counts as the code depth it maps onto.
pub(crate) fn bits_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        SampleFormat::F32 => FLOAT_CODE_BITS,
        _ => 8,
    }
}

/// How a kernel that takes integers and F32 stores samples of `sample`.
pub(crate) fn sample_kind(sample: SampleFormat) -> MatrixSamples {
    match sample {
        SampleFormat::U8 => MatrixSamples::U8,
        SampleFormat::U16 { depth } => MatrixSamples::U16 {
            bits: u32::from(depth.bits()),
        },
        SampleFormat::F32 => MatrixSamples::F32,
    }
}
