use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::blur::{BlurPlanes, Gaussian};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    Mask as GpuMask,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::luma::{Luma, accepts_luma, load_luma, produces_luma_mask};
use crate::sizing::{Scratch, Sizing, mask_plane};

const NAME: &str = "lines/adaptive";

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdaptiveArgs {
    /// The width of the neighbourhood each sample is compared against. Odd and at least 3.
    #[serde(default = "default_block_size")]
    block_size: u32,
    /// How far below its neighbourhood, in U8 units, a sample has to be to count as a line.
    #[serde(default = "default_c")]
    c: f32,
    /// Evens out lighting first, so lines in deep shadow are found too.
    #[serde(default)]
    recover_shadows: bool,
    /// The offset used in place of `c` when recovering shadows. Defaults to 9.
    shadow_c: Option<f32>,
    /// How wide a lighting change to even out. Defaults to 25.
    shadow_blur_sigma: Option<f32>,
    /// How strongly local contrast is raised after evening out. Defaults to 1.8.
    shadow_clahe_clip: Option<f32>,
    /// How strongly noise is smoothed before evening out, in U8 units. Defaults to 50.
    shadow_denoise: Option<f32>,
    /// Lines in groups smaller than this many samples are dropped. Defaults to 6.
    min_speck: Option<u32>,
}

fn default_block_size() -> u32 {
    15
}

fn default_c() -> f32 {
    7.0
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Shadows {
    pub(crate) c: f32,
    pub(crate) blur_sigma: f32,
    pub(crate) clahe_clip: f32,
    pub(crate) denoise: f32,
    pub(crate) min_speck: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Settings {
    pub(crate) block_size: u32,
    pub(crate) c: f32,
    pub(crate) shadows: Option<Shadows>,
}

fn positive(name: &str, value: f32) -> Result<f32, StageError> {
    if value <= 0.0 {
        let message = format!("{NAME} needs {name} above 0, got {value}");
        return Err(StageError::msg(message));
    }

    Ok(value)
}

impl AdaptiveArgs {
    /// The first shadow argument set, by name.
    fn first_shadow_argument(&self) -> Option<&'static str> {
        let set = [
            ("shadow_c", self.shadow_c.is_some()),
            ("shadow_blur_sigma", self.shadow_blur_sigma.is_some()),
            ("shadow_clahe_clip", self.shadow_clahe_clip.is_some()),
            ("shadow_denoise", self.shadow_denoise.is_some()),
            ("min_speck", self.min_speck.is_some()),
        ];

        set.iter().find(|(_, given)| *given).map(|(name, _)| *name)
    }

    pub(crate) fn settings(&self) -> Result<Settings, StageError> {
        if self.block_size < 3 || self.block_size.is_multiple_of(2) {
            let message = format!(
                "{NAME} needs an odd block_size of at least 3, got {}",
                self.block_size
            );
            return Err(StageError::msg(message));
        }

        if !self.recover_shadows {
            let first_shadow_argument = self.first_shadow_argument();
            if let Some(name) = first_shadow_argument {
                let message = format!("{NAME} only takes {name} with recover_shadows set to true");
                return Err(StageError::msg(message));
            }

            return Ok(Settings {
                block_size: self.block_size,
                c: self.c,
                shadows: None,
            });
        }

        let blur_sigma = self.shadow_blur_sigma.unwrap_or(25.0);
        let clahe_clip = self.shadow_clahe_clip.unwrap_or(1.8);
        let denoise = self.shadow_denoise.unwrap_or(50.0);
        let checked_blur_sigma = positive("shadow_blur_sigma", blur_sigma)?;
        let checked_clahe_clip = positive("shadow_clahe_clip", clahe_clip)?;
        let checked_denoise = positive("shadow_denoise", denoise)?;
        let shadows = Shadows {
            c: self.shadow_c.unwrap_or(9.0),
            blur_sigma: checked_blur_sigma,
            clahe_clip: checked_clahe_clip,
            denoise: checked_denoise,
            min_speck: self.min_speck.unwrap_or(6),
        };

        Ok(Settings {
            block_size: self.block_size,
            c: self.c,
            shadows: Some(shadows),
        })
    }
}

pub(crate) fn block_sigma(block_size: u32) -> f32 {
    0.3 * ((block_size as f32 - 1.0) * 0.5 - 1.0) + 0.8
}

#[cube(launch_unchecked)]
fn below_local_mean(smoothed: &Array<f32>, mean: &Array<f32>, output: &mut Array<u8>, offset: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    if smoothed[ABSOLUTE_POS] <= mean[ABSOLUTE_POS] - offset {
        output[ABSOLUTE_POS] = u8::cast_from(255u32);
    } else {
        output[ABSOLUTE_POS] = u8::cast_from(0u32);
    }
}

struct AdaptiveMask<R: Runtime> {
    client: ComputeClient<R>,
    luma: Luma,
    smooth: Gaussian,
    local: Gaussian,
    offset: f32,
}

impl<R: Runtime> AdaptiveMask<R> {
    /// Writes 255 where `plane` sits `offset` below its neighbourhood, and 0 elsewhere.
    fn threshold(
        &self,
        context: &FrameContext,
        sizing: &Sizing<'_>,
        plane: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) {
        let width = self.luma.width;
        let height = self.luma.height;
        let samples = self.luma.samples();

        let temporary = Scratch::new::<f32>(context, width, height);
        let smoothed = Scratch::new::<f32>(context, width, height);
        let mean = Scratch::new::<f32>(context, width, height);

        let smooth_planes = BlurPlanes {
            input: plane,
            scratch: temporary.plane(),
            output: smoothed.plane(),
        };
        self.smooth.apply(&self.client, sizing, smooth_planes);

        let local_planes = BlurPlanes {
            input: smoothed.plane(),
            scratch: temporary.plane(),
            output: mean.plane(),
        };
        self.local.apply(&self.client, sizing, local_planes);

        let smoothed_read = smoothed.plane().array::<R>();
        let mean_read = mean.plane().array::<R>();
        let written = output.array::<R>();
        let cubes = sizing.0.cube_count(samples);
        let shape = sizing.0.cube_dim();

        // Safety: every plane holds `samples` elements of its declared type and the kernel writes
        // nothing past them.
        unsafe {
            below_local_mean::launch_unchecked::<R>(
                &self.client,
                cubes,
                shape,
                smoothed_read,
                mean_read,
                written,
                self.offset,
            );
        }
    }
}

impl<R: Runtime> StageInstance for AdaptiveMask<R> {}

fn no_frame() -> StageError {
    StageError::msg("lines/adaptive expected a frame in slot 0")
}

impl<R: Runtime> Mask for AdaptiveMask<R> {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(no_frame)?;

        let context = &*context;
        let sizing = Sizing(context);

        let luma = Scratch::new::<f32>(context, self.luma.width, self.luma.height);
        load_luma(&self.client, &sizing, &frame, &self.luma, luma.plane())?;

        let meta = frame.meta().clone();
        let format = self.luma.mask_format();
        let output = context.alloc_mask(&format, meta);
        let luma_plane = luma.plane();
        let output_plane = mask_plane(&output);
        self.threshold(context, &sizing, luma_plane, output_plane);

        Ok(output)
    }
}

/// Finds dark strokes by comparing each sample with its neighbourhood.
pub(crate) struct Adaptive;

impl StageDefinition for Adaptive {
    type Args = AdaptiveArgs;

    fn package() -> &'static str {
        "lines"
    }

    fn identifier() -> &'static str {
        "adaptive"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MaskDefinition for Adaptive {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_luma(NAME, offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        args.settings()?;

        produces_luma_mask(NAME, chosen)
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        let settings = args.settings()?;
        if settings.shadows.is_some() {
            return Err(StageError::msg("lines/adaptive cannot recover shadows yet"));
        }

        let luma = Luma::from_formats(NAME, formats)?;
        let client = context.client().clone();
        let smooth = Gaussian::with_radius(&client, 0.8, 1);
        let sigma = block_sigma(settings.block_size);
        let local = Gaussian::with_radius(&client, sigma, settings.block_size / 2);

        Ok(Box::new(AdaptiveMask {
            client,
            luma,
            smooth,
            local,
            offset: settings.c / 255.0,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn settings(value: serde_json::Value) -> Result<Settings, StageError> {
        let args: AdaptiveArgs = serde_json::from_value(value).expect("the args parse");
        args.settings()
    }

    #[test]
    fn the_defaults_are_block_size_15_c_7_and_no_shadows() {
        let plain = settings(json!({})).expect("the defaults are valid");

        assert_eq!(plain.block_size, 15);
        assert_eq!(plain.c, 7.0);
        assert_eq!(plain.shadows, None);
    }

    #[test]
    fn recovering_shadows_fills_in_the_shadow_defaults() {
        let recovering = settings(json!({ "recover_shadows": true })).expect("the defaults are valid");

        let expected = Shadows {
            c: 9.0,
            blur_sigma: 25.0,
            clahe_clip: 1.8,
            denoise: 50.0,
            min_speck: 6,
        };
        assert_eq!(recovering.shadows, Some(expected));
    }

    #[test]
    fn an_even_block_size_is_refused() {
        let error = settings(json!({ "block_size": 14 })).expect_err("14 has no centre");

        assert!(error.to_string().contains("odd block_size of at least 3, got 14"));
    }

    #[test]
    fn a_shadow_argument_without_recovering_shadows_is_refused() {
        let error = settings(json!({ "min_speck": 4 })).expect_err("min_speck needs recover_shadows");

        assert!(
            error
                .to_string()
                .contains("only takes min_speck with recover_shadows")
        );
    }

    #[test]
    fn a_zero_blur_sigma_is_refused() {
        let refused = settings(json!({ "recover_shadows": true, "shadow_blur_sigma": 0.0 }));

        assert!(refused.is_err());
    }

    #[test]
    fn the_block_sigma_matches_opencv() {
        assert!((block_sigma(3) - 0.8).abs() < 1e-6);
        assert!((block_sigma(15) - 2.6).abs() < 1e-6);
    }
}
