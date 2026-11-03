pub(crate) mod file;
pub(crate) mod model;
mod pass;
pub(crate) mod report;
mod specs;
mod summary;

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::io::IsTerminal;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use av_toolbox::report::PipelineFiles;
use av_toolbox_core::PipelineRegistry;
use av_toolbox_graph::StageSpec;
use av_toolbox_runtime::{Profiling, RunReport, Settings};
use codespan_reporting::term::termcolor::WriteColor;

use self::file::ProfileFile;
use self::model::{NodeCost, Width};
use self::pass::{Interrupt, Pass, PassEnd, PassSetup, Warnings, run_pass};
use self::report::Summary;
use self::summary::RowSources;
use crate::cli::{Args, ProfileArgs};
use crate::execute::CANCELLED;
use crate::load;

/// What the three passes found.
struct Outcome {
    measured: Pass,
    baseline: Pass,
    counted: u64,
    costs: Vec<NodeCost>,
    tuned: Vec<Width>,
    baseline_fps: f64,
    tuned_fps: f64,
}

pub(crate) fn run(args: &Args, profile: &ProfileArgs) -> ExitCode {
    let mut stderr = args.color.stderr();

    let Some(loaded) = load::load(&profile.pipeline, &profile.vars, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(registry) = load::registry() else {
        return ExitCode::FAILURE;
    };

    let mut written = loaded.specs.clone();
    let discarded = specs::discard_sinks(&mut written, &registry);

    if !reads_only_files(&registry, &written, &loaded.files, &mut stderr) {
        return ExitCode::FAILURE;
    }

    let Some(backend) = load::backend(args) else {
        return ExitCode::FAILURE;
    };

    let interrupt = Interrupt::install();
    let setup = PassSetup {
        registry: &registry,
        backend: &backend,
        files: &loaded.files,
        interrupt: &interrupt,
    };

    let profiled = profile_passes(&setup, profile, &written, &mut stderr);
    let outcome = match profiled {
        Ok(outcome) => outcome,
        Err(code) => return code,
    };

    let kept_tuned = model::keeps_tuned(outcome.baseline_fps, outcome.tuned_fps);
    let workers = if kept_tuned {
        worker_map(&outcome.tuned)
    } else {
        summary::planned_workers(&outcome.baseline.nodes)
    };

    let accelerator = backend.accelerator();
    let opened = backend.device();
    let device = file::device_label(accelerator, opened);
    let profile_file = ProfileFile {
        pipeline: profile.pipeline.display().to_string(),
        device: device.clone(),
        frames: outcome.counted,
        workers,
    };

    let path = output_path(profile);
    let saved = file::write(&path, &profile_file);
    if let Err(error) = saved {
        tracing::error!("{error}");

        return ExitCode::FAILURE;
    }

    let sources = RowSources {
        measured: &outcome.measured,
        baseline: &outcome.baseline,
        tuned: &outcome.tuned,
        written: &written,
        discarded: &discarded,
    };
    let rows = summary::rows(&sources);
    let overhead = summary::overhead(&outcome.measured.report, outcome.counted);
    let floor = model::floor(&outcome.costs);
    let summary = Summary {
        device,
        frames: outcome.counted,
        rows,
        overhead,
        floor,
        baseline_fps: outcome.baseline_fps,
        tuned_fps: outcome.tuned_fps,
        kept_tuned,
        discarded,
        written: path,
    };

    let bars = std::io::stdout().is_terminal();
    let rendered = report::render(&summary, bars);
    print!("{rendered}");

    ExitCode::SUCCESS
}

/// Builds the pipeline once to refuse a source that reads a live stream, since every pass opens it again.
fn reads_only_files(
    registry: &PipelineRegistry,
    specs: &[StageSpec],
    files: &PipelineFiles,
    stderr: &mut dyn WriteColor,
) -> bool {
    let Some(pipeline) = load::build(registry, specs, files, stderr) else {
        return false;
    };

    let live_sources = pipeline.live_sources();
    let Some(first) = live_sources.first() else {
        return true;
    };

    tracing::error!(
        "profile runs the pipeline three times, so {} cannot be read again. Give it a file source instead",
        first.stage_type
    );

    false
}

/// Runs the measure, baseline and confirm passes, or hands back the exit code a failed one ends with.
fn profile_passes(
    setup: &PassSetup<'_>,
    profile: &ProfileArgs,
    written: &[StageSpec],
    stderr: &mut dyn WriteColor,
) -> Result<Outcome, ExitCode> {
    let sink_inputs = specs::sink_inputs(written);
    let warmup_items = profile.warmup.saturating_mul(sink_inputs);

    let profiling = Profiling {
        sync: setup.backend.device_sync(),
        warmup: profile.warmup,
    };
    tracing::info!(
        "measuring every stage over {} + {} frames (pass 1 of 3)",
        profile.warmup,
        profile.frames
    );

    let stall_deadline = av_toolbox::steps::stall_deadline_for(setup.backend.context().tuning());
    let measure_settings = pass_settings(profile, warmup_items, stall_deadline, Some(profiling));
    let undeclared = specs::undeclared(written);
    let measure_end = run_pass(setup, &undeclared, measure_settings, Warnings::Show, stderr);
    let measured = finished(measure_end, setup.interrupt)?;
    let counted = timed_frames(&measured.report, profile, warmup_items, sink_inputs)?;

    let costs = summary::costs(&measured);
    let max_workers = max_workers(profile);
    let mut tuned = model::suggest(&costs, max_workers);
    mark_declared(&mut tuned, written);

    let plain_settings = pass_settings(profile, warmup_items, stall_deadline, None);
    tracing::info!("running the pipeline as written (pass 2 of 3)");
    let baseline_end = run_pass(setup, written, plain_settings.clone(), Warnings::Hide, stderr);
    let baseline = finished(baseline_end, setup.interrupt)?;
    let baseline_fps = frame_rate(&baseline.report, warmup_items, sink_inputs);

    tracing::info!("running the suggested widths (pass 3 of 3)");
    let confirmed = confirm(setup, written, &mut tuned, &plain_settings, stderr)?;
    let tuned_fps = frame_rate(&confirmed.report, warmup_items, sink_inputs);

    Ok(Outcome {
        measured,
        baseline,
        counted,
        costs,
        tuned,
        baseline_fps,
        tuned_fps,
    })
}

/// Flags every width whose stage declares its own workers, since a profile cannot change those.
fn mark_declared(widths: &mut [Width], written: &[StageSpec]) {
    let declared: Vec<String> = written
        .iter()
        .filter(|spec| spec.workers.is_some())
        .map(file::stage_key)
        .collect();

    for width in widths {
        width.declared = declared.contains(&width.key);
    }
}

/// Runs `written` at the `tuned` widths, taking a worker away and trying again each time it fails.
fn confirm(
    setup: &PassSetup<'_>,
    written: &[StageSpec],
    tuned: &mut [Width],
    settings: &Settings,
    stderr: &mut dyn WriteColor,
) -> Result<Pass, ExitCode> {
    loop {
        let workers = worker_map(tuned);
        let mut widened = written.to_vec();
        file::apply(&workers, &mut widened);

        let end = run_pass(setup, &widened, settings.clone(), Warnings::Hide, stderr);
        let failed = matches!(end, PassEnd::Failed);
        if !failed || setup.interrupt.interrupted() {
            return finished(end, setup.interrupt);
        }

        if !model::step_down(tuned) {
            return Err(ExitCode::FAILURE);
        }

        tracing::warn!("the confirm pass failed, trying again with fewer workers");
    }
}

/// The pass that finished, or the exit code for one that failed or was interrupted.
fn finished(end: PassEnd, interrupt: &Interrupt) -> Result<Pass, ExitCode> {
    if interrupt.interrupted() {
        return Err(ExitCode::from(CANCELLED));
    }

    let pass = match end {
        PassEnd::Finished(pass) => pass,
        PassEnd::Failed => return Err(ExitCode::FAILURE),
        PassEnd::Interrupted => return Err(ExitCode::from(CANCELLED)),
    };

    Ok(pass)
}

/// How many frames the measure pass timed, failing when it timed none.
fn timed_frames(
    report: &RunReport,
    profile: &ProfileArgs,
    warmup_items: u64,
    sink_inputs: u64,
) -> Result<u64, ExitCode> {
    let counted = specs::counted_frames(report.frames, warmup_items, sink_inputs);
    let ended_after = report.frames / sink_inputs;

    let Some(counted) = counted else {
        tracing::error!(
            "the pipeline ended after {ended_after} frames, before its {} warmup frames",
            profile.warmup
        );

        return Err(ExitCode::FAILURE);
    };

    if counted == 0 {
        tracing::error!(
            "the pipeline ended after {ended_after} frames, so none were left to time after its {} warmup frames",
            profile.warmup
        );

        return Err(ExitCode::FAILURE);
    }

    if counted < profile.frames {
        tracing::warn!(
            "only {counted} frames were timed, fewer than the {} asked for",
            profile.frames
        );
    }

    Ok(counted)
}

fn pass_settings(
    profile: &ProfileArgs,
    warmup_items: u64,
    stall_deadline: Option<Duration>,
    profiling: Option<Profiling>,
) -> Settings {
    let frame_limit = profile.warmup.saturating_add(profile.frames);

    Settings {
        frame_budget: profile.frame_budget,
        stall_deadline,
        frame_limit: Some(frame_limit),
        warmup: warmup_items,
        profiling,
    }
}

/// The counted frames over the counted wall time, zero when nothing was counted.
fn frame_rate(report: &RunReport, warmup_items: u64, sink_inputs: u64) -> f64 {
    let seconds = report.counted_wall.as_secs_f64();
    if seconds == 0.0 {
        return 0.0;
    }

    let counted = specs::counted_frames(report.frames, warmup_items, sink_inputs);
    let counted = counted.unwrap_or(0);

    counted as f64 / seconds
}

/// The `--max-workers` given, or the number of CPU threads.
fn max_workers(profile: &ProfileArgs) -> usize {
    let Some(given) = profile.max_workers else {
        let threads = std::thread::available_parallelism();
        return threads.map_or(1, NonZeroUsize::get);
    };

    usize::try_from(given).unwrap_or(usize::MAX)
}

fn worker_map(widths: &[Width]) -> BTreeMap<String, usize> {
    widths
        .iter()
        .map(|width| (width.key.clone(), width.workers))
        .collect()
}

/// The `--output` given, or `<pipeline stem>.profile.toml` in the working directory.
fn output_path(profile: &ProfileArgs) -> PathBuf {
    if let Some(output) = &profile.output {
        return output.clone();
    }

    let fallback = OsStr::new("pipeline");
    let stem = profile.pipeline.file_stem();
    let stem = stem.unwrap_or(fallback);
    let mut name = OsString::from(stem);
    name.push(".profile.toml");

    PathBuf::from(name)
}
