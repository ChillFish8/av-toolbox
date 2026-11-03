use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use av_denoise_core::tune::{
    CacheHook,
    CachePhase,
    TuneOutcome,
    TuneReport,
    TuneRequest,
    TuningFrame,
    is_tuned,
    tune_frames,
    tuning_frame_count,
};
use av_denoise_core::{EngineSpec, SearchMode, TuningConfig, TuningMode};
use av_toolbox_gpu::{
    Client,
    Tuning,
    compilation_cache_dir,
    merge_compilation_cache,
    refill_compilation_cache,
};

/// The directory av-denoise keeps its tuning results in, inside the kernel cache's build directory.
const STORE_DIR: &str = "av-denoise-tune";

/// Every device, store and spec this process has run a tuning pass for.
///
/// Holding it while tuning keeps passes one at a time, so they never time each other's kernels.
static TUNED: Mutex<Vec<String>> = Mutex::new(Vec::new());

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
        cache_hook: Some(kernel_cache_hook()),
        ..TuningConfig::default()
    }
}

/// What the kernel cache does at a point in a tuning pass.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum CacheAction {
    Refill,
    Merge,
}

/// The action the kernel cache takes at `phase`.
pub(crate) fn hook_action(phase: CachePhase) -> CacheAction {
    match phase {
        CachePhase::LockTaken => CacheAction::Refill,
        CachePhase::KernelsCompiled => CacheAction::Merge,
    }
}

/// The hook a tuning pass calls to keep this process's kernel cache in step with other processes.
///
/// A taken lock refills the private cache from the shared one, and compiled kernels are merged back.
fn kernel_cache_hook() -> CacheHook {
    Arc::new(|phase| match hook_action(phase) {
        CacheAction::Refill => refill_compilation_cache(),
        CacheAction::Merge => merge_compilation_cache(),
    })
}

/// Where tuning results are stored, or `None` to keep them for this process only.
pub(crate) fn tuning_store() -> Option<PathBuf> {
    let cache = compilation_cache_dir()?;

    Some(cache.join(STORE_DIR))
}

/// Whether a stage must tune on its first frames before it builds its engine.
///
/// Only [Tuning::Auto] tunes, so `is_tuned` is asked for nothing else.
pub(crate) fn needs_tuning_pass(tuning: Tuning, is_tuned: impl FnOnce() -> bool) -> bool {
    if tuning != Tuning::Auto {
        return false;
    }

    !is_tuned()
}

/// How many frames a stage on `spec` holds for its tuning pass before its engine sees any.
///
/// It is what av-denoise asks for, but never more than the engine itself holds, so the stage stays
/// within the frames it declares. The pass cycles through fewer frames when it gets fewer.
pub(crate) fn tuning_prefix(spec: EngineSpec) -> usize {
    let specs = [spec];
    let wanted = tuning_frame_count(&specs);
    let held = spec.max_held_frames();

    wanted.min(held).max(1)
}

/// Tunes `spec` on `frames` unless this process already has, or the store holds its result.
///
/// A key is recorded before its pass runs, so a pass that panics is never tried again. The lock
/// it leaves poisoned is still taken by later passes.
pub(crate) fn tune_once(
    device: &cubecl::Device,
    client: &Client,
    spec: EngineSpec,
    frames: &[TuningFrame],
    config: &TuningConfig,
) -> TuneReport {
    let key = format!("{device:?} {:?} {spec:?}", config.store);
    let mut tuned = TUNED.lock().unwrap_or_else(PoisonError::into_inner);

    if tuned.contains(&key) {
        return already_tuned();
    }

    let specs = [spec];
    if is_tuned(device, client, &specs, config) {
        return already_tuned();
    }

    tuned.push(key);

    let request = TuneRequest {
        force: false,
        mode: SearchMode::Families,
    };

    tune_frames(device, client, &specs, frames, config, &request)
}

/// Runs a tuning pass through `work` and logs what it did, turning a panic inside it into a skipped pass.
///
/// Tuning only ever speeds a run up, so a pass that fails or panics never fails the run.
pub(crate) fn guarded_tuning(stage: &str, work: impl FnOnce() -> TuneReport) -> TuneReport {
    let result = panic::catch_unwind(AssertUnwindSafe(work));

    let report = match result {
        Ok(report) => report,
        Err(payload) => {
            let message = panic_message(payload.as_ref());
            tracing::warn!(stage, %message, "the av-denoise tuning pass panicked, running without it");

            let reason = format!("the tuning pass panicked: {message}");
            TuneReport::skipped(reason)
        },
    };

    log_report(&report);
    report
}

/// The text a panic was raised with, when it carried any.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }

    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }

    "a panic without a message".to_string()
}

/// A report for a spec the store or this process already tuned.
fn already_tuned() -> TuneReport {
    TuneReport {
        outcome: TuneOutcome::AlreadyTuned,
        tuned_keys: 0,
        partial_keys: 0,
        compile: Duration::ZERO,
        elapsed: Duration::ZERO,
        saved: false,
        summary: None,
    }
}

fn log_report(report: &TuneReport) {
    match &report.outcome {
        TuneOutcome::Tuned => tracing::info!(
            keys = report.tuned_keys,
            partial_keys = report.partial_keys,
            seconds = report.elapsed.as_secs_f64(),
            compile_seconds = report.compile.as_secs_f64(),
            saved = report.saved,
            "tuned the av-denoise kernels for this device"
        ),
        TuneOutcome::AlreadyTuned => tracing::info!("using the stored av-denoise tuning results"),
        TuneOutcome::Skipped(reason) => tracing::debug!(%reason, "av-denoise kernel tuning skipped"),
    }
}
