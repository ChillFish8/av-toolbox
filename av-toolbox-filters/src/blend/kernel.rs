use av_toolbox_core::{Filter, Inputs, StageError, StageInstance};
use av_toolbox_gpu::{Frame, FrameContext, FrameFormat, Plane, SampleFormat};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

/// Blends one sample as `base + (overlay - base) * m / 255`, rounded to the nearest sample value.
#[cube(launch_unchecked)]
fn blend_plane<E: Int>(base: &[E], overlay: &[E], mask: &[u8], output: &mut [E]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let weight = f32::cast_from(mask[ABSOLUTE_POS]) / 255.0;
    let low = f32::cast_from(base[ABSOLUTE_POS]);
    let high = f32::cast_from(overlay[ABSOLUTE_POS]);
    let blended = f32::round(low + (high - low) * weight);

    output[ABSOLUTE_POS] = E::cast_from(blended);
}

/// Blends one float sample as `base + (overlay - base) * m / 255`, without rounding.
#[cube(launch_unchecked)]
fn blend_float_plane(base: &[f32], overlay: &[f32], mask: &[u8], output: &mut [f32]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let weight = f32::cast_from(mask[ABSOLUTE_POS]) / 255.0;
    let low = base[ABSOLUTE_POS];
    let high = overlay[ABSOLUTE_POS];

    output[ABSOLUTE_POS] = low + (high - low) * weight;
}

/// Copies one plane unchanged, for the planes `blend/masked` was not asked to blend.
#[cube(launch_unchecked)]
fn copy_plane<E: Numeric>(base: &[E], output: &mut [E]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = base[ABSOLUTE_POS];
}

fn missing_base() -> StageError {
    StageError::msg("blend/masked expected base in slot 0")
}

fn missing_overlay() -> StageError {
    StageError::msg("blend/masked expected overlay in slot 1")
}

fn missing_mask() -> StageError {
    StageError::msg("blend/masked expected a mask in slot 2")
}

fn missing_channel() -> StageError {
    StageError::msg("blend/masked was handed a frame missing a channel")
}

fn missing_chroma() -> StageError {
    StageError::msg("blend/masked was handed a mask without the chroma plane this frame needs")
}

pub(super) struct MaskedBlend {
    client: Client,
    format: FrameFormat,
    planes: Vec<u8>,
}

impl MaskedBlend {
    pub(super) fn new(client: Client, format: FrameFormat, planes: Vec<u8>) -> Self {
        Self {
            client,
            format,
            planes,
        }
    }

    fn blend(&self, context: &FrameContext, planes: PlaneSet<'_>) {
        let samples = planes.output.samples();
        let cubes = context.cube_count(samples);
        let shape = context.cube_dim();
        let mask = planes.mask.array::<u8>();

        // Safety: every binding names exactly as many elements as its plane holds, and the kernel
        // writes nothing past the output's length.
        unsafe {
            match self.format.sample {
                SampleFormat::U8 => {
                    let base = planes.base.array::<u8>();
                    let overlay = planes.overlay.array::<u8>();
                    let output = planes.output.array::<u8>();
                    blend_plane::launch_unchecked::<u8>(
                        &self.client,
                        cubes,
                        shape,
                        base,
                        overlay,
                        mask,
                        output,
                    );
                },
                SampleFormat::F32 => {
                    let base = planes.base.array::<f32>();
                    let overlay = planes.overlay.array::<f32>();
                    let output = planes.output.array::<f32>();
                    blend_float_plane::launch_unchecked(
                        &self.client,
                        cubes,
                        shape,
                        base,
                        overlay,
                        mask,
                        output,
                    );
                },
                _ => {
                    let base = planes.base.array::<u16>();
                    let overlay = planes.overlay.array::<u16>();
                    let output = planes.output.array::<u16>();
                    blend_plane::launch_unchecked::<u16>(
                        &self.client,
                        cubes,
                        shape,
                        base,
                        overlay,
                        mask,
                        output,
                    );
                },
            }
        }
    }

    fn copy(&self, context: &FrameContext, base: &Plane, output: &Plane) {
        let samples = output.samples();
        let cubes = context.cube_count(samples);
        let shape = context.cube_dim();

        // Safety: both bindings name exactly as many elements as their plane holds, and the kernel
        // writes nothing past the output's length.
        unsafe {
            match self.format.sample {
                SampleFormat::U8 => {
                    let base = base.array::<u8>();
                    let output = output.array::<u8>();
                    copy_plane::launch_unchecked::<u8>(&self.client, cubes, shape, base, output);
                },
                SampleFormat::F32 => {
                    let base = base.array::<f32>();
                    let output = output.array::<f32>();
                    copy_plane::launch_unchecked::<f32>(&self.client, cubes, shape, base, output);
                },
                _ => {
                    let base = base.array::<u16>();
                    let output = output.array::<u16>();
                    copy_plane::launch_unchecked::<u16>(&self.client, cubes, shape, base, output);
                },
            }
        }
    }
}

/// The planes of one channel, plus the mask.
struct PlaneSet<'a> {
    base: &'a Plane,
    overlay: &'a Plane,
    mask: &'a Plane,
    output: &'a Plane,
}

impl StageInstance for MaskedBlend {}

impl Filter for MaskedBlend {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let base = inputs.take_frame(0).ok_or_else(missing_base)?;
        let overlay = inputs.take_frame(1).ok_or_else(missing_overlay)?;
        let mask = inputs.take_mask(2).ok_or_else(missing_mask)?;

        let meta = base.meta().clone();
        let output = context.alloc_frame(&self.format, meta);

        for (index, channel) in self.format.channels().iter().enumerate() {
            let base_plane = base.plane(*channel).ok_or_else(missing_channel)?;
            let output_plane = output.plane(*channel).ok_or_else(missing_channel)?;
            let plane_index = index as u8;

            if self.planes.contains(&plane_index) {
                let overlay_plane = overlay.plane(*channel).ok_or_else(missing_channel)?;
                let mask_plane = mask
                    .plane_for(*channel, &self.format)
                    .ok_or_else(missing_chroma)?;
                let planes = PlaneSet {
                    base: base_plane,
                    overlay: overlay_plane,
                    mask: mask_plane,
                    output: output_plane,
                };

                self.blend(context, planes);
            } else {
                self.copy(context, base_plane, output_plane);
            }
        }

        Ok(vec![output])
    }
}
