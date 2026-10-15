use av_toolbox_cubek::blur::{BlurPlanes, Gaussian};
use av_toolbox_cubek::labels::{CountPlanes, KeepPlanes, LabelPlanes, count, keep, label};
use av_toolbox_cubek::{DevicePlane, reflect};
use av_toolbox_gpu::FrameContext;
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::Shadows;
use super::clahe::{EqualisePlanes, equalise};
use crate::sizing::{Scratch, Sizing};

/// Smooths noise while keeping edges, over the samples within 3 of the centre.
#[cube(launch_unchecked)]
fn bilateral(input: &Array<f32>, output: &mut Array<f32>, width: u32, height: u32, colour_sigma: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = (position % width) as i32;
    let row = (position / width) as i32;
    let centre = input[ABSOLUTE_POS];
    let colour_spread = 2.0 * colour_sigma * colour_sigma;

    let mut total = 0.0f32;
    let mut weights = 0.0f32;
    for step_y in 0..7u32 {
        for step_x in 0..7u32 {
            let offset_x = step_x as i32 - 3;
            let offset_y = step_y as i32 - 3;
            let distance = f32::cast_from(offset_x * offset_x + offset_y * offset_y);

            if distance <= 9.0 {
                let source_column = reflect(column + offset_x, width as i32);
                let source_row = reflect(row + offset_y, height as i32);
                let value = input[(source_row as u32 * width + source_column as u32) as usize];
                let change = value - centre;
                let spatial = f32::exp(-distance / 98.0);
                let colour = f32::exp(-(change * change) / colour_spread);
                let weight = spatial * colour;

                total += value * weight;
                weights += weight;
            }
        }
    }

    output[ABSOLUTE_POS] = total / weights;
}

#[cube(launch_unchecked)]
fn divide_out_light(denoised: &Array<f32>, blurred: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let evened = denoised[ABSOLUTE_POS] / (blurred[ABSOLUTE_POS] + 1.0 / 255.0) * (128.0 / 255.0);

    output[ABSOLUTE_POS] = f32::clamp(evened, 0.0, 1.0);
}

/// Evens out the lighting of a luma plane.
pub(super) struct ShadowRecovery {
    light: Gaussian,
    colour_sigma: f32,
    clip: f32,
}

impl ShadowRecovery {
    pub(super) fn new<R: Runtime>(client: &ComputeClient<R>, shadows: &Shadows) -> Self {
        let light = Gaussian::new(client, shadows.blur_sigma);

        Self {
            light,
            colour_sigma: shadows.denoise / 255.0,
            clip: shadows.clahe_clip,
        }
    }

    /// Writes `luma` denoised, divided by its own lighting and equalised into `output`.
    pub(super) fn flatten<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        context: &FrameContext,
        sizing: &Sizing<'_>,
        luma: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) {
        let width = luma.width();
        let height = luma.height();
        let samples = luma.samples();

        let denoised = Scratch::new::<f32>(context, width, height);
        let luma_read = luma.array::<R>();
        let denoised_write = denoised.plane().array::<R>();
        let cubes = sizing.0.cube_count(samples);
        let shape = sizing.0.cube_dim();

        // Safety: both planes hold `samples` f32 values and the kernel writes nothing past them.
        unsafe {
            bilateral::launch_unchecked::<R>(
                client,
                cubes,
                shape,
                luma_read,
                denoised_write,
                width,
                height,
                self.colour_sigma,
            );
        }

        let temporary = Scratch::new::<f32>(context, width, height);
        let blurred = Scratch::new::<f32>(context, width, height);
        let light_planes = BlurPlanes {
            input: denoised.plane(),
            scratch: temporary.plane(),
            output: blurred.plane(),
        };
        self.light.apply(client, sizing, light_planes);

        let evened = Scratch::new::<f32>(context, width, height);
        let denoised_read = denoised.plane().array::<R>();
        let blurred_read = blurred.plane().array::<R>();
        let evened_write = evened.plane().array::<R>();
        let cubes = sizing.0.cube_count(samples);
        let shape = sizing.0.cube_dim();

        // Safety: all three planes hold `samples` f32 values and the kernel writes nothing past them.
        unsafe {
            divide_out_light::launch_unchecked::<R>(
                client,
                cubes,
                shape,
                denoised_read,
                blurred_read,
                evened_write,
            );
        }

        let equalise_planes = EqualisePlanes {
            input: evened.plane(),
            output,
        };
        equalise(client, context, sizing, equalise_planes, self.clip);
    }
}

/// Copies `lines` into `output`, leaving out groups of fewer than `min_speck` line samples.
pub(super) fn remove_specks<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    sizing: &Sizing<'_>,
    lines: DevicePlane<'_>,
    output: DevicePlane<'_>,
    min_speck: u32,
) {
    let width = lines.width();
    let height = lines.height();

    let labels = Scratch::new::<u32>(context, width, height);
    let label_planes = LabelPlanes {
        marks: lines,
        labels: labels.plane(),
    };
    label(client, sizing, label_planes);

    let counts = Scratch::new::<u32>(context, width, height);
    let count_planes = CountPlanes {
        labels: labels.plane(),
        marks: lines,
        counts: counts.plane(),
    };
    count(client, sizing, count_planes);

    let keep_planes = KeepPlanes {
        labels: labels.plane(),
        counts: counts.plane(),
        output,
    };
    keep(client, sizing, keep_planes, min_speck);
}
