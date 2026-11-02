use std::fmt::Display;

use av_denoise_core::{
    DenoisingMode,
    EngineSpec,
    Geometry,
    HqParams,
    MotionCompensationMode,
    MotionSearch,
    Nl4dOptions,
    NlmTuning,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    PrefilterMode,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    parse_prefilter,
};
use av_toolbox_core::{ArtefactData, ArtefactDecl, ArtefactPiece, StageError, merge_single};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use super::grain::{ARTEFACTS, GRAIN_TABLE, merge_grain_table};
use super::planes::{PlaneGroup, PlaneGroups};
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

/// The settings one plane group is denoised with.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum EngineOptions {
    Nlmeans(NlmeansAlgorithm),
    Nl4d(Nl4dOptions),
}

impl EngineOptions {
    fn nlmeans(self) -> Option<NlmeansAlgorithm> {
        match self {
            Self::Nlmeans(algorithm) => Some(algorithm),
            Self::Nl4d(_) => None,
        }
    }

    fn nl4d(self) -> Option<Nl4dOptions> {
        match self {
            Self::Nlmeans(_) => None,
            Self::Nl4d(options) => Some(options),
        }
    }
}

/// What every av-denoise stage's arguments answer.
pub(crate) trait AvdArgs: DeserializeOwned + JsonSchema + Send + Sync + 'static {
    const IDENTIFIER: &'static str;
    const STAGE: &'static str;
    const DESCRIPTION: &'static str;

    fn planes(&self) -> Option<&[ChannelName]>;

    /// The settings `group` is denoised with, with that group's overrides folded in.
    fn options(&self, group: PlaneGroup) -> Result<EngineOptions, StageError>;

    /// The artefacts this stage can hand over at the end of a stream.
    fn artefacts() -> &'static [ArtefactDecl] {
        &[]
    }

    /// Merges the pieces of the artefact `name`, given in chunk order.
    fn merge_artefact(name: &'static str, pieces: Vec<ArtefactPiece>) -> Result<ArtefactData, StageError> {
        merge_single(name, pieces)
    }
}

fn nothing_denoised(stage: &str) -> StageError {
    let message = format!("{stage} was given no planes to denoise");

    StageError::msg(message)
}

/// The options of `group`, or `None` when it is not denoised.
fn group_options<A: AvdArgs>(
    args: &A,
    selected: bool,
    group: PlaneGroup,
) -> Result<Option<EngineOptions>, StageError> {
    if !selected {
        return Ok(None);
    }

    let options = args.options(group)?;

    Ok(Some(options))
}

/// The engine that denoises the `groups` of a `geometry` frame with `args`.
///
/// A group left out passes through the engine unchanged.
pub(crate) fn engine_spec<A: AvdArgs>(
    args: &A,
    groups: PlaneGroups,
    geometry: Geometry,
) -> Result<EngineSpec, StageError> {
    let luma = group_options(args, groups.luma, PlaneGroup::Luma)?;
    let chroma = group_options(args, groups.chroma, PlaneGroup::Chroma)?;

    let Some(first) = luma.or(chroma) else {
        return Err(nothing_denoised(A::STAGE));
    };

    let spec = match first {
        EngineOptions::Nlmeans(_) => EngineSpec::Nlmeans {
            luma: luma.and_then(EngineOptions::nlmeans),
            chroma: chroma.and_then(EngineOptions::nlmeans),
            geometry,
        },
        EngineOptions::Nl4d(_) => EngineSpec::Nl4d {
            luma: luma.and_then(EngineOptions::nl4d),
            chroma: chroma.and_then(EngineOptions::nl4d),
            geometry,
        },
    };

    Ok(spec)
}

fn bad_prefilter(stage: &str, error: impl Display) -> StageError {
    let message = format!("{stage} could not read its prefilter: {error}");

    StageError::msg(message)
}

/// Picks the override of `group`, falling back to `shared`.
pub(super) fn per_plane<T>(
    group: PlaneGroup,
    luma: Option<T>,
    chroma: Option<T>,
    shared: Option<T>,
) -> Option<T> {
    let plane = match group {
        PlaneGroup::Luma => luma,
        PlaneGroup::Chroma => chroma,
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

    fn options(&self, stage: &str, group: PlaneGroup) -> Result<NlmeansOptions, StageError> {
        let preset = self.preset.core();
        let radius = self.temporal_radius();
        let mode = match radius {
            0 => DenoisingMode::Spacial,
            radius => DenoisingMode::Temporal { radius },
        };

        let search_radius = self
            .search_radius
            .unwrap_or_else(|| nlmeans_search_radius_for(preset));
        let strength = per_plane(group, self.luma_strength, self.chroma_strength, self.strength);
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

    fn options(&self, group: PlaneGroup) -> Result<EngineOptions, StageError> {
        let options = self.settings().options(Self::STAGE, group)?;
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

    fn options(&self, group: PlaneGroup) -> Result<EngineOptions, StageError> {
        let nlm = self.settings().options(Self::STAGE, group)?;
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
    /// The planes denoised, every plane when not given. U and V are denoised together.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// Frames on each side of the one denoised, at least 1. av-denoise's default when not given.
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
    /// The half-width of the search for similar patches in the centre frame, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) spatial_radius: Option<u32>,
    /// The half-width of the search around each neighbour's predicted position.
    #[serde(default)]
    pub(crate) refine: Option<u32>,
    /// Scales the luma threshold by how noisy each brightness level is, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) noise_map: Option<bool>,
    /// Judges each coefficient together with its frequency neighbours, av-denoise's default when not given.
    #[serde(default)]
    pub(crate) pooled_threshold: Option<bool>,
}

impl Nl4dArgs {
    fn temporal_radius(&self) -> u32 {
        let defaults = Nl4dOptions::default();

        self.temporal_radius.unwrap_or(defaults.temporal_radius)
    }
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

    fn options(&self, group: PlaneGroup) -> Result<EngineOptions, StageError> {
        let defaults = Nl4dOptions::default();
        let lambda_ht = per_plane(group, self.luma_lambda_ht, self.chroma_lambda_ht, self.lambda_ht);

        let options = Nl4dOptions {
            sigma: self.sigma,
            sigma_scale: self.sigma_scale.unwrap_or(defaults.sigma_scale),
            lambda_ht,
            lambda_ht_scale: self.lambda_ht_scale.unwrap_or(defaults.lambda_ht_scale),
            spatial_radius: self.spatial_radius.unwrap_or(defaults.spatial_radius),
            refine: self.refine.unwrap_or(defaults.refine),
            noise_map: self.noise_map.unwrap_or(defaults.noise_map),
            pooled_threshold: self.pooled_threshold.unwrap_or(defaults.pooled_threshold),
            temporal_radius: self.temporal_radius(),
            ..defaults
        };

        Ok(EngineOptions::Nl4d(options))
    }
}
