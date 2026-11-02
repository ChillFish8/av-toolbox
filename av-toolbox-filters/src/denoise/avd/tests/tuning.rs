use std::path::PathBuf;

use av_denoise_core::{TuningConfig, TuningMode};
use av_toolbox_gpu::Tuning;

use crate::denoise::avd::tuning::tuning_config;

#[test]
fn each_tuning_maps_onto_its_av_denoise_mode() {
    let modes = [
        (Tuning::Auto, TuningMode::Auto),
        (Tuning::Cached, TuningMode::Cached),
        (Tuning::Off, TuningMode::Off),
    ];

    for (tuning, mode) in modes {
        let config = tuning_config(tuning, None);

        assert_eq!(config.mode, mode, "{tuning}");
    }
}

#[test]
fn the_store_budget_and_determinism_are_carried_or_defaulted() {
    let store = PathBuf::from("/tmp/avd-tuning");
    let defaults = TuningConfig::default();

    let config = tuning_config(Tuning::Cached, Some(store.clone()));

    assert_eq!(config.store, Some(store));
    assert_eq!(config.budget, defaults.budget);
    assert_eq!(config.deterministic, defaults.deterministic);
    assert!(config.cache_hook.is_none());
}
