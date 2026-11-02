use std::path::PathBuf;

use av_denoise_core::{TuningConfig, TuningMode};
use av_toolbox_gpu::Tuning;

/// The av-denoise tuning settings for `tuning`, keeping results in `store` when given.
pub(crate) fn tuning_config(tuning: Tuning, store: Option<PathBuf>) -> TuningConfig {
    let mode = match tuning {
        Tuning::Auto => TuningMode::Auto,
        Tuning::Cached => TuningMode::Cached,
        Tuning::Off => TuningMode::Off,
    };

    TuningConfig {
        mode,
        store,
        ..TuningConfig::default()
    }
}
