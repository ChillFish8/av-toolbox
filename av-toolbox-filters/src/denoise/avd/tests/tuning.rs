use std::path::PathBuf;

use av_denoise_core::tune::{TuneOutcome, tuning_frame_count};
use av_denoise_core::{
    ChromaLayout,
    ChromaSiting,
    DenoisingMode,
    EngineSpec,
    Geometry,
    Nl4dOptions,
    NlmeansAlgorithm,
    NlmeansOptions,
    SampleFormat,
    TuningConfig,
    TuningMode,
};
use av_toolbox_gpu::Tuning;

use crate::denoise::avd::tuning::{guarded_tuning, needs_tuning_pass, tuning_config, tuning_prefix};

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

#[test]
fn tuning_without_a_cache_keeps_results_in_memory() {
    let config = tuning_config(Tuning::Auto, None);

    assert_eq!(config.mode, TuningMode::Auto);
    assert_eq!(config.store, None);
}

#[test]
fn only_auto_tuning_asks_whether_the_spec_is_tuned() {
    for tuning in [Tuning::Cached, Tuning::Off] {
        let needed = needs_tuning_pass(tuning, || panic!("{tuning} asked whether the spec is tuned"));

        assert!(!needed, "{tuning}");
    }
}

#[test]
fn auto_tuning_runs_a_pass_only_for_an_untuned_spec() {
    assert!(needs_tuning_pass(Tuning::Auto, || false));
    assert!(!needs_tuning_pass(Tuning::Auto, || true));
}

fn geometry() -> Geometry {
    Geometry {
        width: 64,
        height: 32,
        chroma: ChromaLayout::Yuv420,
        input: SampleFormat::U8,
        output: SampleFormat::F32,
        chroma_siting: ChromaSiting::Unknown,
    }
}

#[test]
fn the_tuning_prefix_never_outgrows_the_engines_window() {
    let spec = EngineSpec::Nl4d {
        luma: Some(Nl4dOptions::default()),
        chroma: Some(Nl4dOptions::default()),
        geometry: geometry(),
    };

    assert!(tuning_frame_count(&[spec]) > spec.max_held_frames());
    assert_eq!(tuning_prefix(spec), spec.max_held_frames());
}

#[test]
fn a_spatial_engine_still_tunes_on_one_frame() {
    let options = NlmeansOptions {
        mode: DenoisingMode::Spacial,
        ..NlmeansOptions::default()
    };
    let algorithm = NlmeansAlgorithm::Fast(options);
    let spec = EngineSpec::Nlmeans {
        luma: Some(algorithm),
        chroma: Some(algorithm),
        geometry: geometry(),
    };

    assert_eq!(spec.max_held_frames(), 0);
    assert_eq!(tuning_prefix(spec), 1);
}

#[test]
fn a_panicking_tuning_pass_is_reported_as_skipped() {
    let report = guarded_tuning("av-denoise/nl4d", || panic!("the device went away"));

    let TuneOutcome::Skipped(reason) = report.outcome else {
        panic!("expected a skipped pass, got {:?}", report.outcome);
    };
    assert!(reason.contains("the device went away"), "{reason}");
}
