use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::values::{offset, widen};
use av_toolbox_gpu::{Channel, ComputeClient, FrameContext, FrameFormat, Plane, Runtime, SampleFormat};

use crate::bits::bits_of;
use crate::codes::CodeValues;
use crate::scratch::{Scratch, device_plane};
use crate::sizing::Sizing;

/// How a plane's samples reach the network as f32 values up to a peak, and come back as F32.
#[derive(Debug, Copy, Clone)]
pub(super) enum Staging {
    /// Integer samples, widened to f32 code values and stored back as full range F32.
    Integer {
        element: Element,
        peak: f32,
        codes: CodeValues,
    },
    /// Float samples, with chroma centred on zero.
    Float,
}

impl Staging {
    pub(super) fn of(input: &FrameFormat) -> Self {
        if input.sample == SampleFormat::F32 {
            return Self::Float;
        }

        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = ((1u32 << bits) - 1) as f32;
        let codes = CodeValues::new(input, SampleFormat::F32);

        Self::Integer { element, peak, codes }
    }

    pub(super) fn peak(self) -> f32 {
        match self {
            Self::Integer { peak, .. } => peak,
            Self::Float => 1.0,
        }
    }
}

/// What a plane's values are moved by so they sit between 0 and the peak.
///
/// Float chroma is centred on zero, so it moves up by half.
pub(super) fn centre_of(sample: SampleFormat, channel: Channel) -> f32 {
    let chroma = matches!(channel, Channel::U | Channel::V);

    if sample == SampleFormat::F32 && chroma {
        return 0.5;
    }

    0.0
}

/// Copies `source` into f32 scratch the network reads.
///
/// Float planes are moved up by `centre`. Integer planes are widened to code values and ignore it.
///
/// # Safety
///
/// `source` must hold samples of the format `staging` was made from.
pub(super) unsafe fn stage_in<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    staging: Staging,
    source: &Plane,
    centre: f32,
) -> Scratch {
    let (width, height) = source.dimensions();
    let values = Scratch::new::<f32>(context, width, height);
    let values_plane = values.plane();
    let sizing = Sizing(context);

    // Safety: the caller guarantees `staging` was made from the plane's sample format, so `widen` reads it
    // as its own element and `offset` reads a float plane as f32.
    let source_plane = unsafe { device_plane(source) };

    match staging {
        Staging::Integer { element, .. } => widen(client, &sizing, source_plane, values_plane, element),
        Staging::Float => offset(client, &sizing, source_plane, values_plane, centre),
    }

    values
}

/// Writes the network's f32 `values` for `channel` into `target` as F32.
///
/// Float planes are moved back down by their centre. Integer code values become full range F32.
///
/// # Safety
///
/// `target` must hold F32 samples.
pub(super) unsafe fn stage_out<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    staging: Staging,
    values: DevicePlane<'_>,
    target: &Plane,
    channel: Channel,
) {
    let sizing = Sizing(context);

    // Safety: the caller guarantees `target` holds F32 samples, which both kernels write.
    let target_plane = unsafe { device_plane(target) };

    match staging {
        Staging::Integer { codes, .. } => codes.store(client, &sizing, values, target_plane, channel),
        Staging::Float => {
            let centre = centre_of(SampleFormat::F32, channel);
            offset(client, &sizing, values, target_plane, -centre);
        },
    }
}
