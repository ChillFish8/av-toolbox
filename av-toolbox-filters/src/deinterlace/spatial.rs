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
    default_mdis,
    default_nrad,
    default_vcheck,
    default_vthresh,
    thresholds_for,
};
use crate::aa::sangnom::{SangnomTuning, checked_sangnom_tuning};
use crate::deinterlace::bwdif::STAGE;

/// Where `deinterlace/bwdif` takes the spatial prediction of each rebuilt row from.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub(crate) enum Spatial {
    /// bwdif's own cubic down each column.
    Cubic {},
    /// EEDI2's edge directed interpolation, with `aa/eedi2`'s defaults.
    Eedi2(Eedi2Tuning),
    /// EEDI3's edge directed interpolation, with `aa/eedi3`'s defaults.
    Eedi3(Eedi3Spatial),
    /// SangNom's edge following interpolation, with `aa/sangnom`'s defaults.
    Sangnom(SangnomTuning),
}

impl Default for Spatial {
    fn default() -> Self {
        Self::Cubic {}
    }
}

/// EEDI3's settings as `spatial` gives them.
///
/// It is separate from [Eedi3Tuning](crate::aa::based::Eedi3Tuning) because that struct carries `aa/based`'s
/// defaults.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Eedi3Spatial {
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
    #[serde(default = "default_mdis")]
    pub(crate) mdis: u32,
    /// How strongly an unreliable direction falls back, between 0 and 3.
    #[serde(default = "default_vcheck")]
    pub(crate) vcheck: u32,
    /// The three thresholds of the reliability check, each at least 1e-30.
    ///
    /// The first two are on the 8-bit scale.
    #[serde(default = "default_vthresh", deserialize_with = "three_thresholds")]
    pub(crate) vthresh: [f64; 3],
}

fn three_thresholds<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f64; 3], D::Error> {
    thresholds_for(STAGE, deserializer)
}

impl Eedi3Spatial {
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

/// The prediction `spatial` names, once its settings are checked.
pub(crate) fn checked_predictor(spatial: &Spatial) -> Result<Predictor, StageError> {
    let predictor = match spatial {
        Spatial::Cubic {} => Predictor::Cubic,
        Spatial::Eedi2(tuning) => {
            let arguments = checked_eedi2_tuning(STAGE, tuning)?;
            Predictor::Eedi2(arguments)
        },
        Spatial::Eedi3(spatial) => {
            let tuning = spatial.tuning();
            let arguments = checked_tuning(STAGE, &tuning)?;
            Predictor::Eedi3(arguments)
        },
        Spatial::Sangnom(tuning) => {
            let aa = checked_sangnom_tuning(STAGE, tuning)?;
            Predictor::Sangnom { aa }
        },
    };

    Ok(predictor)
}

/// Refuses a selected plane of `input` that `predictor`'s field pass cannot run on.
pub(crate) fn checked_predictor_planes(
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
            return Err(missing_plane(STAGE, *channel));
        };

        match predictor {
            Predictor::Eedi2(_) => {
                checked_eedi2_size(STAGE, *channel, Direction::Vertical, width, height)?;
            },
            Predictor::Eedi3(arguments) => {
                checked_eedi3_size(
                    STAGE,
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
