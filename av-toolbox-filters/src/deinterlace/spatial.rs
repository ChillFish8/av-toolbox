use std::marker::PhantomData;

use av_toolbox_core::StageError;
use av_toolbox_cubek::aa::{Eedi2Arguments, Eedi3Arguments};
use av_toolbox_gpu::FrameFormat;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer};

use crate::aa::based::Eedi3Tuning;
use crate::aa::direction::{Direction, missing_plane};
use crate::aa::eedi2::{Eedi2Tuning, checked_eedi2_size, checked_eedi2_tuning};
use crate::aa::eedi3::{
    checked_eedi3_size,
    checked_tuning,
    default_alpha,
    default_beta,
    default_gamma,
    default_nrad,
    default_vcheck,
    default_vthresh,
    thresholds_for,
};
use crate::aa::sangnom::{SangnomTuning, checked_sangnom_tuning};

/// The stage whose name a [Spatial] is refused under while it is read.
pub(crate) trait SpatialStage {
    const STAGE: &'static str;

    /// The furthest direction an EEDI3 `spatial` searches when `mdis` is not given.
    const DEFAULT_MDIS: u32;
}

/// Where the spatial prediction of each rebuilt row comes from.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields, bound = "")]
#[schemars(rename = "Spatial", bound = "")]
pub(crate) enum Spatial<S: SpatialStage> {
    /// The stage's plain interpolation.
    ///
    /// `deinterlace/bwdif` and the bwdif fallback of `ivtc/match` run bwdif's own cubic down each column.
    /// `deinterlace/qtgmc` runs a bicubic resize of each field.
    Cubic {},
    /// EEDI2's edge directed interpolation, with `aa/eedi2`'s defaults.
    Eedi2(Eedi2Tuning),
    /// EEDI3's edge directed interpolation, with `aa/eedi3`'s defaults.
    Eedi3(Eedi3Spatial<S>),
    /// SangNom's edge following interpolation, with `aa/sangnom`'s defaults.
    Sangnom(SangnomTuning),
}

impl<S: SpatialStage> Default for Spatial<S> {
    fn default() -> Self {
        Self::Cubic {}
    }
}

/// EEDI3's settings as `spatial` gives them.
///
/// It is separate from [Eedi3Tuning](crate::aa::based::Eedi3Tuning) because that struct carries `aa/based`'s
/// defaults.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
#[schemars(rename = "Eedi3Spatial", bound = "")]
pub(crate) struct Eedi3Spatial<S: SpatialStage> {
    /// The weight of the neighbourhood cost, 0 or between 1e-30 and 1.
    #[serde(default = "default_alpha")]
    pub(crate) alpha: f64,
    /// The weight of the vertical difference cost, 0 or between 1e-30 and 1, on the 8-bit scale.
    ///
    /// `alpha + beta` may not pass 1.
    #[serde(default = "default_beta")]
    pub(crate) beta: f64,
    /// The cost of changing direction, between 0 and 1e30, on the 8-bit scale.
    #[serde(default = "default_gamma")]
    pub(crate) gamma: f64,
    /// The neighbourhood radius, between 0 and 3.
    #[serde(default = "default_nrad")]
    pub(crate) nrad: u32,
    /// The furthest direction searched, between 1 and 40.
    #[serde(default = "stage_mdis::<S>")]
    pub(crate) mdis: u32,
    /// How strongly an unreliable direction falls back, between 0 and 3.
    #[serde(default = "default_vcheck")]
    pub(crate) vcheck: u32,
    /// The three thresholds of the reliability check, each at least 1e-30.
    ///
    /// The first two are on the 8-bit scale.
    #[serde(default = "default_vthresh", deserialize_with = "three_thresholds::<S, _>")]
    pub(crate) vthresh: [f64; 3],
    #[serde(skip)]
    #[schemars(skip)]
    stage: PhantomData<S>,
}

fn stage_mdis<S: SpatialStage>() -> u32 {
    S::DEFAULT_MDIS
}

fn three_thresholds<'de, S: SpatialStage, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<[f64; 3], D::Error> {
    thresholds_for(S::STAGE, deserializer)
}

impl<S: SpatialStage> Eedi3Spatial<S> {
    /// EEDI3's defaults, with the stage's own `mdis`.
    pub(crate) fn defaults() -> Self {
        Self {
            alpha: default_alpha(),
            beta: default_beta(),
            gamma: default_gamma(),
            nrad: default_nrad(),
            mdis: S::DEFAULT_MDIS,
            vcheck: default_vcheck(),
            vthresh: default_vthresh(),
            stage: PhantomData,
        }
    }

    pub(crate) fn tuning(&self) -> Eedi3Tuning {
        Eedi3Tuning {
            alpha: self.alpha,
            beta: self.beta,
            gamma: self.gamma,
            nrad: self.nrad,
            mdis: self.mdis,
            vcheck: self.vcheck,
            vthresh: self.vthresh,
        }
    }
}

/// A spatial prediction once its settings are checked.
#[derive(Debug, Copy, Clone)]
pub(crate) enum Predictor {
    Cubic,
    Eedi2(Eedi2Arguments),
    Eedi3(Eedi3Arguments),
    Sangnom { aa: u32 },
}

/// The prediction `spatial` names, once its settings are checked, refusing them under `stage`.
pub(crate) fn checked_predictor<S: SpatialStage>(
    stage: &str,
    spatial: &Spatial<S>,
) -> Result<Predictor, StageError> {
    let predictor = match spatial {
        Spatial::Cubic {} => Predictor::Cubic,
        Spatial::Eedi2(tuning) => {
            let arguments = checked_eedi2_tuning(stage, tuning)?;
            Predictor::Eedi2(arguments)
        },
        Spatial::Eedi3(spatial) => {
            let tuning = spatial.tuning();
            let arguments = checked_tuning(stage, &tuning)?;
            Predictor::Eedi3(arguments)
        },
        Spatial::Sangnom(tuning) => {
            let aa = checked_sangnom_tuning(stage, tuning)?;
            Predictor::Sangnom { aa }
        },
    };

    Ok(predictor)
}

/// Refuses, under `stage`, a selected plane of `input` that `predictor`'s field pass cannot run on.
pub(crate) fn checked_predictor_planes(
    stage: &str,
    predictor: Predictor,
    input: &FrameFormat,
    selected: &[bool],
) -> Result<(), StageError> {
    let channels = input.channels();
    let chosen = channels
        .iter()
        .zip(selected)
        .filter(|(_, is_selected)| **is_selected);
    for (channel, _) in chosen {
        let Some((width, height)) = input.channel_dimensions(*channel) else {
            return Err(missing_plane(stage, *channel));
        };

        match predictor {
            Predictor::Eedi2(_) => {
                checked_eedi2_size(stage, *channel, Direction::Vertical, width, height)?;
            },
            Predictor::Eedi3(arguments) => {
                checked_eedi3_size(
                    stage,
                    *channel,
                    Direction::Vertical,
                    arguments.mdis,
                    width,
                    height,
                )?;
            },
            Predictor::Cubic | Predictor::Sangnom { .. } => {},
        }
    }

    Ok(())
}
