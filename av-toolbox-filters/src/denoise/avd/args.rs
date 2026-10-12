use std::fmt::Display;

use av_denoise_core::{
    ChannelMode,
    DenoisingMode,
    HqParams,
    MotionCompensationMode,
    MotionSearch,
    Nl4dOptions,
    NlmTuning,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    PrefilterMode,
    PsyParams,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    parse_prefilter,
};
use av_toolbox_core::{ArtefactData, ArtefactDecl, ArtefactPiece, StageError, merge_single};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use super::grain::{ARTEFACTS, GRAIN_TABLE, merge_grain_table};
use crate::planes::ChannelName;

/// How much quality a stage trades for speed, the way av-denoise's `--preset` does.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Preset {
    Veryfast,
    Fast,
    #[default]
    Base,
    Slow,
    Veryslow,
}

impl Preset {
    pub(crate) fn core(self) -> av_denoise_core::Preset {
        match self {
            Self::Veryfast => av_denoise_core::Preset::Veryfast,
            Self::Fast => av_denoise_core::Preset::Fast,
            Self::Base => av_denoise_core::Preset::Base,
            Self::Slow => av_denoise_core::Preset::Slow,
            Self::Veryslow => av_denoise_core::Preset::Veryslow,
        }
    }
}

/// The settings one av-denoise engine is built with.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum EngineOptions {
    Nlmeans(NlmeansAlgorithm),
    Nl4d(Nl4dOptions),
}

/// What every av-denoise stage's arguments answer.
pub(crate) trait AvdArgs: DeserializeOwned + JsonSchema + Send + Sync + 'static {
    const IDENTIFIER: &'static str;
    const STAGE: &'static str;
    const DESCRIPTION: &'static str;

    fn planes(&self) -> Option<&[ChannelName]>;

    fn fused(&self) -> bool;

    /// The most frames the engines hold before emitting one.
    fn held_frames(&self) -> usize;

    /// The settings of the engine that denoises `channels`, with that plane's overrides folded in.
    fn options(&self, channels: ChannelMode) -> Result<EngineOptions, StageError>;

    /// The artefacts this stage can hand over at the end of a stream.
    fn artefacts() -> &'static [ArtefactDecl] {
        &[]
    }

    /// Merges the pieces of the artefact `name`, given in chunk order.
    fn merge_artefact(name: &'static str, pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
        merge_single(name, pieces)
    }
}

fn bad_prefilter(stage: &str, error: impl Display) -> StageError {
    let message = format!("{stage} could not read its prefilter: {error}");

    StageError::msg(message)
}

/// Picks the override of the plane `channels` covers, falling back to `shared`.
///
/// A fused engine covers every plane, so it only reads `shared`.
fn per_plane<T>(channels: ChannelMode, luma: Option<T>, chroma: Option<T>, shared: Option<T>) -> Option<T> {
    let plane = match channels {
        ChannelMode::Luma => luma,
        ChannelMode::Chroma => chroma,
        ChannelMode::Yuv => None,
    };

    plane.or(shared)
}

/// The settings `nlmeans` and `nlmeans-hq` share.
#[derive(Debug, Clone, Default, PartialEq)]
struct NlmSettings {
    preset: Preset,
    temporal_radius: Option<u32>,
    search_radius: Option<u32>,
    patch_radius: Option<u32>,
    strength: Option<f32>,
    luma_strength: Option<f32>,
    chroma_strength: Option<f32>,
    prefilter: Option<String>,
    motion_compensation: bool,
}

impl NlmSettings {
    fn temporal_radius(&self) -> u32 {
        let preset = self.preset.core();

        self.temporal_radius
            .unwrap_or_else(|| nlmeans_temporal_radius_for(preset))
    }

    fn options(&self, stage: &str, channels: ChannelMode) -> Result<NlmeansOptions, StageError> {
        let preset = self.preset.core();
        let radius = self.temporal_radius();
        let mode = match radius {
            0 => DenoisingMode::Spacial,
            radius => DenoisingMode::Temporal { radius },
        };

        let search_radius = self
            .search_radius
            .unwrap_or_else(|| nlmeans_search_radius_for(preset));
        let strength = per_plane(channels, self.luma_strength, self.chroma_strength, self.strength);
        let tuning = NlmTuning {
            search_radius: Some(search_radius),
            patch_radius: self.patch_radius,
            strength,
            self_weight: None,
        };

        let prefilter = match &self.prefilter {
            None => PrefilterMode::None,
            Some(spec) => parse_prefilter(spec).map_err(|error| bad_prefilter(stage, error))?,
        };

        let motion_compensation = match self.motion_compensation {
            true => MotionCompensationMode::from(MotionSearch::default()),
            false => MotionCompensationMode::None,
        };

        Ok(NlmeansOptions {
            prefilter,
            motion_compensation,
            tuning,
            mode,
        })
    }
}

/// Arguments for `av-denoise/nlmeans`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct NlmeansArgs {
    /// The speed and quality preset, `base` when not given.
    #[serde(default)]
    pub(crate) preset: Preset,
    /// The planes denoised, every plane when not given. U and V are denoised together.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// Denoises all three planes in one engine, which needs 4:4:4 input with every plane selected.
    #[serde(default)]
    pub(crate) fused: bool,
    /// Frames on each side of the one denoised, 0 for spatial only. The preset's when not given.
    #[serde(default)]
    pub(crate) temporal_radius: Option<u32>,
    /// How far each patch is searched for within a frame. The preset's when not given.
    #[serde(default)]
    pub(crate) search_radius: Option<u32>,
    /// The half-width of each compared patch, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) patch_radius: Option<u32>,
    /// How strongly every plane is denoised, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) strength: Option<f32>,
    /// Replaces `strength` for luma.
    #[serde(default)]
    pub(crate) luma_strength: Option<f32>,
    /// Replaces `strength` for chroma.
    #[serde(default)]
    pub(crate) chroma_strength: Option<f32>,
    /// What patches are compared on, such as `nlm` or `bilateral:2.0,0.1`. The noisy input when not given.
    #[serde(default)]
    pub(crate) prefilter: Option<String>,
    /// Follows motion between frames when denoising across them.
    #[serde(default)]
    pub(crate) motion_compensation: bool,
}

impl NlmeansArgs {
    fn settings(&self) -> NlmSettings {
        NlmSettings {
            preset: self.preset,
            temporal_radius: self.temporal_radius,
            search_radius: self.search_radius,
            patch_radius: self.patch_radius,
            strength: self.strength,
            luma_strength: self.luma_strength,
            chroma_strength: self.chroma_strength,
            prefilter: self.prefilter.clone(),
            motion_compensation: self.motion_compensation,
        }
    }
}

impl AvdArgs for NlmeansArgs {
    const IDENTIFIER: &'static str = "nlmeans";
    const STAGE: &'static str = "av-denoise/nlmeans";
    const DESCRIPTION: &'static str =
        "Removes noise by blending each patch of the picture with similar patches nearby.";

    fn planes(&self) -> Option<&[ChannelName]> {
        self.planes.as_deref()
    }

    fn fused(&self) -> bool {
        self.fused
    }

    fn held_frames(&self) -> usize {
        self.settings().temporal_radius() as usize
    }

    fn options(&self, channels: ChannelMode) -> Result<EngineOptions, StageError> {
        let options = self.settings().options(Self::STAGE, channels)?;
        let algorithm = NlmeansAlgorithm::Fast(options);

        Ok(EngineOptions::Nlmeans(algorithm))
    }
}

/// Arguments for `av-denoise/nlmeans-hq`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct NlmeansHqArgs {
    /// The speed and quality preset, `base` when not given.
    #[serde(default)]
    pub(crate) preset: Preset,
    /// The planes denoised, every plane when not given. U and V are denoised together.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// Denoises all three planes in one engine, which needs 4:4:4 input with every plane selected.
    #[serde(default)]
    pub(crate) fused: bool,
    /// Frames on each side of the one denoised, 0 for spatial only. The preset's when not given.
    #[serde(default)]
    pub(crate) temporal_radius: Option<u32>,
    /// How far each patch is searched for within a frame. The preset's when not given.
    #[serde(default)]
    pub(crate) search_radius: Option<u32>,
    /// The half-width of each compared patch, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) patch_radius: Option<u32>,
    /// A multiplier on the measured noise for every plane, av-denoise's calibrated default when not given.
    #[serde(default)]
    pub(crate) strength: Option<f32>,
    /// Replaces `strength` for luma.
    #[serde(default)]
    pub(crate) luma_strength: Option<f32>,
    /// Replaces `strength` for chroma.
    #[serde(default)]
    pub(crate) chroma_strength: Option<f32>,
    /// What patches are compared on, such as `nlm` or `bilateral:2.0,0.1`. The noisy input when not given.
    #[serde(default)]
    pub(crate) prefilter: Option<String>,
    /// Follows motion between frames when denoising across them.
    #[serde(default)]
    pub(crate) motion_compensation: bool,
    /// A fixed noise level between 0 and 1, measured from each frame when not given.
    #[serde(default)]
    pub(crate) sigma: Option<f32>,
    /// A multiplier on the measured noise level, 1 when not given.
    #[serde(default)]
    pub(crate) sigma_scale: Option<f32>,
}

impl NlmeansHqArgs {
    fn settings(&self) -> NlmSettings {
        NlmSettings {
            preset: self.preset,
            temporal_radius: self.temporal_radius,
            search_radius: self.search_radius,
            patch_radius: self.patch_radius,
            strength: self.strength,
            luma_strength: self.luma_strength,
            chroma_strength: self.chroma_strength,
            prefilter: self.prefilter.clone(),
            motion_compensation: self.motion_compensation,
        }
    }
}

impl AvdArgs for NlmeansHqArgs {
    const IDENTIFIER: &'static str = "nlmeans-hq";
    const STAGE: &'static str = "av-denoise/nlmeans-hq";
    const DESCRIPTION: &'static str =
        "Removes noise by blending similar patches, scaled to the noise measured in each frame.";

    fn planes(&self) -> Option<&[ChannelName]> {
        self.planes.as_deref()
    }

    fn fused(&self) -> bool {
        self.fused
    }

    fn held_frames(&self) -> usize {
        self.settings().temporal_radius() as usize
    }

    fn options(&self, channels: ChannelMode) -> Result<EngineOptions, StageError> {
        let nlm = self.settings().options(Self::STAGE, channels)?;
        let defaults = HqParams::default();
        let hq = HqParams {
            sigma_override: self.sigma,
            sigma_scale: self.sigma_scale.unwrap_or(defaults.sigma_scale),
            ..defaults
        };
        let options = NlmeansHqOptions { nlm, hq };
        let algorithm = NlmeansAlgorithm::Hq(options);

        Ok(EngineOptions::Nlmeans(algorithm))
    }
}

/// Arguments for `av-denoise/nl4d`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Nl4dArgs {
    /// The speed and quality preset, `base` when not given.
    #[serde(default)]
    pub(crate) preset: Preset,
    /// The planes denoised, every plane when not given. U and V are denoised together.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// Denoises all three planes in one engine, which needs 4:4:4 input with every plane selected.
    #[serde(default)]
    pub(crate) fused: bool,
    /// Frames on each side of the one denoised, at least 1. The preset's when not given.
    #[serde(default)]
    pub(crate) temporal_radius: Option<u32>,
    /// A fixed noise level between 0 and 1, measured from each frame when not given.
    #[serde(default)]
    pub(crate) sigma: Option<f32>,
    /// A multiplier on the measured noise level, 1 when not given.
    #[serde(default)]
    pub(crate) sigma_scale: Option<f32>,
    /// How many noise deviations a coefficient must clear to survive, for every plane.
    ///
    /// av-denoise's calibrated value for each plane when not given.
    #[serde(default)]
    pub(crate) lambda_ht: Option<f32>,
    /// Replaces `lambda_ht` for luma.
    #[serde(default)]
    pub(crate) luma_lambda_ht: Option<f32>,
    /// Replaces `lambda_ht` for chroma.
    #[serde(default)]
    pub(crate) chroma_lambda_ht: Option<f32>,
    /// A multiplier on the resolved `lambda_ht`, 1 when not given.
    #[serde(default)]
    pub(crate) lambda_ht_scale: Option<f32>,
    /// The half-width of the search for similar patches in the centre frame. The preset's when not given.
    #[serde(default)]
    pub(crate) spatial_radius: Option<u32>,
    /// The half-width of the search around each neighbour's predicted position.
    #[serde(default)]
    pub(crate) refine: Option<u32>,
    /// Scales the luma threshold by how noisy each brightness level is, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) noise_map: Option<bool>,
    /// Filters each area by what it looks like. The `psy_*` arguments tune it and need it on.
    #[serde(default)]
    pub(crate) enable_psy: bool,
    /// How much harder flat, grainy areas are filtered, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) psy_flat_boost: Option<f32>,
    /// The same as `psy_flat_boost` for the chroma planes, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) psy_chroma_flat_boost: Option<f32>,
    /// How much more gently textured dark areas are filtered, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) psy_shadow_soften: Option<f32>,
    /// How strongly grain may line up before a flat area counts as texture, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) psy_flat_texture_cut: Option<f32>,
    /// How far around a strong line, in pixels, `psy_shadow_soften` and `psy_flat_texture_cut` stop applying.
    ///
    /// av-denoise's default when not given.
    #[serde(default)]
    pub(crate) psy_line_ring: Option<u32>,
    /// Judges each coefficient together with its frequency neighbours, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) pooled_threshold: Option<bool>,
}

impl Nl4dArgs {
    fn temporal_radius(&self) -> u32 {
        let preset = self.preset.core();

        self.temporal_radius
            .unwrap_or_else(|| nl4d_temporal_radius_for(preset))
    }

    /// The psy tuning asked for, or `None` without `enable_psy`.
    ///
    /// A `psy_*` argument without `enable_psy` is refused, so a setting is never silently ignored.
    fn psy(&self) -> Result<Option<PsyParams>, StageError> {
        let knobs = [
            ("psy_flat_boost", self.psy_flat_boost.is_some()),
            ("psy_chroma_flat_boost", self.psy_chroma_flat_boost.is_some()),
            ("psy_shadow_soften", self.psy_shadow_soften.is_some()),
            ("psy_flat_texture_cut", self.psy_flat_texture_cut.is_some()),
            ("psy_line_ring", self.psy_line_ring.is_some()),
        ];

        if !self.enable_psy {
            let given = knobs.iter().find(|(_, given)| *given);
            if let Some((name, _)) = given {
                return Err(psy_needs_enabling(name));
            }

            return Ok(None);
        }

        let defaults = PsyParams::default();
        let psy = PsyParams {
            flat_boost: self.psy_flat_boost.unwrap_or(defaults.flat_boost),
            chroma_flat_boost: self.psy_chroma_flat_boost.unwrap_or(defaults.chroma_flat_boost),
            shadow_soften: self.psy_shadow_soften.unwrap_or(defaults.shadow_soften),
            flat_texture_cut: self.psy_flat_texture_cut.unwrap_or(defaults.flat_texture_cut),
            line_ring: self.psy_line_ring.unwrap_or(defaults.line_ring),
        };

        Ok(Some(psy))
    }
}

fn psy_needs_enabling(name: &str) -> StageError {
    let message = format!("{} was given {name} without enable_psy", Nl4dArgs::STAGE);

    StageError::msg(message)
}

impl AvdArgs for Nl4dArgs {
    const IDENTIFIER: &'static str = "nl4d";
    const STAGE: &'static str = "av-denoise/nl4d";
    const DESCRIPTION: &'static str = "Removes noise by filtering groups of matching blocks \
                                       across neighbouring frames.\n\n\
                                       It can hand over the grain it removed from luma as an AV1 film grain \
                                       table, so an encoder can add similar grain back.";

    fn artefacts() -> &'static [ArtefactDecl] {
        ARTEFACTS
    }

    fn merge_artefact(name: &'static str, pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
        if name == GRAIN_TABLE {
            return merge_grain_table(pieces);
        }

        merge_single(name, pieces)
    }

    fn planes(&self) -> Option<&[ChannelName]> {
        self.planes.as_deref()
    }

    fn fused(&self) -> bool {
        self.fused
    }

    fn held_frames(&self) -> usize {
        2 * self.temporal_radius() as usize
    }

    fn options(&self, channels: ChannelMode) -> Result<EngineOptions, StageError> {
        let preset = self.preset.core();
        let defaults = Nl4dOptions::default();
        let lambda_ht = per_plane(
            channels,
            self.luma_lambda_ht,
            self.chroma_lambda_ht,
            self.lambda_ht,
        );
        let spatial_radius = self
            .spatial_radius
            .unwrap_or_else(|| nl4d_spatial_radius_for(preset));
        let psy = self.psy()?;

        let options = Nl4dOptions {
            sigma: self.sigma,
            sigma_scale: self.sigma_scale.unwrap_or(defaults.sigma_scale),
            lambda_ht,
            lambda_ht_scale: self.lambda_ht_scale.unwrap_or(defaults.lambda_ht_scale),
            spatial_radius,
            refine: self.refine.unwrap_or(defaults.refine),
            noise_map: self.noise_map.unwrap_or(defaults.noise_map),
            psy,
            pooled_threshold: self.pooled_threshold.unwrap_or(defaults.pooled_threshold),
            temporal_radius: self.temporal_radius(),
            ..defaults
        };

        Ok(EngineOptions::Nl4d(options))
    }
}
