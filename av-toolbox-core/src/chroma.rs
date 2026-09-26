use av_toolbox_cubek::resize::{Kernel, ResizePlanes, Weights, apply, taps, upload};
use av_toolbox_cubek::{DevicePlane, Dispatch};
use av_toolbox_gpu::{
    BuildContext,
    FrameContext,
    Mask as DeviceMask,
    MaskChroma,
    MaskFormat,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim, Runtime};

use crate::{BuiltMask, Inputs, Mask, Sided, StageError, StageInstance};

/// Sizes launches from the frame context a stage is handed.
struct Sizing<'a>(&'a FrameContext);

impl Dispatch for Sizing<'_> {
    fn cube_count(&self, samples: usize) -> CubeCount {
        self.0.cube_count(samples)
    }

    fn cube_dim(&self) -> CubeDim {
        self.0.cube_dim()
    }
}

/// A device mask whose output also carries a chroma plane, downscaled from its full plane.
pub(crate) struct ChromaMask<R: Runtime> {
    inner: Box<dyn Mask>,
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    chroma: MaskChroma,
    across: Weights,
    down: Weights,
}

impl<R: Runtime> ChromaMask<R> {
    /// Wraps `inner`, building the bilinear weight tables for masks of `format` once.
    ///
    /// Each axis is scaled by its subsampling factor and offset to where the chroma location places
    /// a chroma sample.
    pub(crate) fn new(
        inner: Box<dyn Mask>,
        client: ComputeClient<R>,
        format: &MaskFormat,
        chroma: MaskChroma,
    ) -> Self {
        let (factor_x, factor_y) = chroma.factors();
        let (offset_x, offset_y) = chroma.location.offsets(factor_x, factor_y);
        let (chroma_width, chroma_height) = chroma.subsampling.chroma_dimensions(format.width, format.height);

        let across_taps = taps(chroma_width, factor_x as f32, offset_x, &Kernel::Bilinear);
        let down_taps = taps(chroma_height, factor_y as f32, offset_y, &Kernel::Bilinear);
        let across = upload(&client, &across_taps);
        let down = upload(&client, &down_taps);

        Self {
            inner,
            client,
            width: format.width,
            height: format.height,
            chroma,
            across,
            down,
        }
    }
}

impl<R: Runtime> StageInstance for ChromaMask<R> {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }
}

impl<R: Runtime> Mask for ChromaMask<R> {
    fn select(&mut self, context: &mut FrameContext, inputs: Inputs) -> Result<DeviceMask, StageError> {
        let selected = self.inner.select(context, inputs)?;

        let produced = selected.format();
        let resolved = (self.width, self.height, SampleFormat::U8);
        if (produced.width, produced.height, produced.sample) != resolved {
            let message = format!(
                "the mask stage produced a {produced} where a {}x{} U8 mask was resolved",
                self.width, self.height,
            );
            return Err(StageError::msg(message));
        }

        let mask = context.add_chroma(selected, self.chroma);
        let chroma_plane = mask
            .chroma_plane()
            .expect("the mask was just given a chroma plane");
        let (chroma_width, chroma_height) = chroma_plane.dimensions();
        let scratch_bytes = chroma_width as usize * self.height as usize * size_of::<f32>();
        let scratch = context.scratch(scratch_bytes);

        let full_handle = mask.plane().buffer().handle();
        let chroma_handle = chroma_plane.buffer().handle();
        let scratch_handle = scratch.handle();

        // Safety: the full plane was checked above to hold exactly `width * height` U8 samples.
        let input = unsafe { DevicePlane::new(full_handle, self.width, self.height) };

        // Safety: the scratch was sized for exactly `chroma_width * height` f32 samples.
        let scratch_plane = unsafe { DevicePlane::new(scratch_handle, chroma_width, self.height) };

        // Safety: the chroma plane was added to a U8 mask, so it holds exactly
        // `chroma_width * chroma_height` U8 samples.
        let output = unsafe { DevicePlane::new(chroma_handle, chroma_width, chroma_height) };

        let planes = ResizePlanes {
            input,
            scratch: scratch_plane,
            output,
        };
        let sizing = Sizing(context);
        apply(&self.client, &sizing, planes, &self.across, &self.down);

        Ok(mask)
    }
}

/// Wraps a built mask in a [ChromaMask] when its resolved output carries a chroma plane.
pub(crate) fn add_chroma_plane<R: Runtime>(
    context: &BuildContext<R>,
    formats: &ResolvedFormats,
    built: BuiltMask,
) -> Result<BuiltMask, StageError> {
    let Some(StreamFormat::Mask(format)) = formats.output() else {
        return Ok(built);
    };

    let Some(chroma) = format.chroma else {
        return Ok(built);
    };

    let inner = match built {
        Sided::Device(inner) => inner,
        Sided::Host(_) => {
            return Err(StageError::msg("host masks cannot carry chroma planes"));
        },
    };

    if format.sample != SampleFormat::U8 {
        let message = format!("only U8 masks can carry chroma planes, not {}", format.sample);
        return Err(StageError::msg(message));
    }

    let client = context.client().clone();
    let wrapper = ChromaMask::new(inner, client, &format, chroma);
    let wrapped: Box<dyn Mask> = Box::new(wrapper);

    Ok(Sided::Device(wrapped))
}
