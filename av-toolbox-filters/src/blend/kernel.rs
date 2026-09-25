use av_toolbox_core::{Filter, Inputs, StageError, StageInstance};
use av_toolbox_gpu::{Frame, FrameContext, FrameFormat, Plane, SampleFormat};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

/// Blends one sample as `base + (overlay - base) * m / 255`, rounded to the nearest sample value.
#[cube(launch_unchecked)]
fn blend_plane<E: Int>(base: &Array<E>, overlay: &Array<E>, mask: &Array<u8>, output: &mut Array<E>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let weight = f32::cast_from(mask[ABSOLUTE_POS]) / 255.0;
    let low = f32::cast_from(base[ABSOLUTE_POS]);
    let high = f32::cast_from(overlay[ABSOLUTE_POS]);
    let blended = f32::round(low + (high - low) * weight);

    output[ABSOLUTE_POS] = E::cast_from(blended);
}

/// Copies one plane unchanged, for the planes `blend/masked` was not asked to blend.
#[cube(launch_unchecked)]
fn copy_plane<E: Int>(base: &Array<E>, output: &mut Array<E>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = base[ABSOLUTE_POS];
}

pub(super) struct MaskedBlend<R: Runtime> {
    client: ComputeClient<R>,
    format: FrameFormat,
    planes: Vec<u8>,
}

impl<R: Runtime> MaskedBlend<R> {
    pub(super) fn new(client: ComputeClient<R>, format: FrameFormat, planes: Vec<u8>) -> Self {
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
        let mask = planes.mask.array::<u8, R>();

        // Safety: every binding names exactly as many elements as its plane holds, and the kernel
        // writes nothing past the output's length.
        unsafe {
            match self.format.sample {
                SampleFormat::U8 => {
                    let base = planes.base.array::<u8, R>();
                    let overlay = planes.overlay.array::<u8, R>();
                    let output = planes.output.array::<u8, R>();
                    blend_plane::launch_unchecked::<u8, R>(
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
                    let base = planes.base.array::<u16, R>();
                    let overlay = planes.overlay.array::<u16, R>();
                    let output = planes.output.array::<u16, R>();
                    blend_plane::launch_unchecked::<u16, R>(
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
                    let base = base.array::<u8, R>();
                    let output = output.array::<u8, R>();
                    copy_plane::launch_unchecked::<u8, R>(&self.client, cubes, shape, base, output);
                },
                _ => {
                    let base = base.array::<u16, R>();
                    let output = output.array::<u16, R>();
                    copy_plane::launch_unchecked::<u16, R>(&self.client, cubes, shape, base, output);
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

impl<R: Runtime> StageInstance for MaskedBlend<R> {}

impl<R: Runtime> Filter for MaskedBlend<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let base = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("blend/masked expected base in slot 0"))?;
        let overlay = inputs
            .take_frame(1)
            .ok_or_else(|| StageError::msg("blend/masked expected overlay in slot 1"))?;
        let mask = inputs
            .take_mask(2)
            .ok_or_else(|| StageError::msg("blend/masked expected a mask in slot 2"))?;

        let meta = base.meta().clone();
        let output = context.alloc_frame(&self.format, meta);
        let missing = || StageError::msg("blend/masked was handed a frame missing a channel");

        for (index, channel) in self.format.channels().iter().enumerate() {
            let base_plane = base.plane(*channel).ok_or_else(missing)?;
            let output_plane = output.plane(*channel).ok_or_else(missing)?;
            let plane_index = index as u8;

            if self.planes.contains(&plane_index) {
                let overlay_plane = overlay.plane(*channel).ok_or_else(missing)?;
                let mask_plane = mask.plane();
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

        Ok(output)
    }
}
