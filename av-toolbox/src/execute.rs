use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use av_toolbox_pipeline::CompiledPipeline;
use av_toolbox_runtime::{Cancel, Progress, RunReport, Settings};
use indicatif::MultiProgress;

use crate::cli::{Args, ExecuteArgs};
use crate::progress::{Reporter, Snapshot, clock};
use crate::{load, report};

/// The exit code of a run stopped with Ctrl+C.
pub(crate) const CANCELLED: u8 = 130;

pub(crate) fn run(args: &Args, execute: &ExecuteArgs, bars: &MultiProgress) -> ExitCode {
    let mut stderr = report::stderr(args.color);

    let Some(mut loaded) = load::load(&execute.pipeline, &execute.vars, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let mut profile = None;
    if let Some(path) = &execute.profile {
        let Some(read) = load::apply_profile(path, &mut loaded) else {
            return ExitCode::FAILURE;
        };

        profile = Some(read);
    }

    let Some(registry) = load::registry() else {
        return ExitCode::FAILURE;
    };

    let Some(pipeline) = load::build(&registry, &loaded.specs, &loaded.files, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(backend) = load::backend(args) else {
        return ExitCode::FAILURE;
    };

    if let Some(profile) = &profile {
        load::warn_on_other_device(profile, &backend);
    }

    let Some(compiled) = load::compile(pipeline, &backend, &loaded.files, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let stall_deadline = load::stall_deadline_for(backend.context().tuning());
    let settings = Settings {
        frame_budget: execute.frame_budget,
        stall_deadline,
        ..Settings::default()
    };
    let built = compiled.engine(settings);
    let engine = match built {
        Ok(engine) => engine,
        Err(error) => {
            let found = error.into_diagnostics().into_inner();
            report::diagnostics(&mut stderr, &loaded.files, &found);

            return ExitCode::FAILURE;
        },
    };

    let cancel = engine.cancel_handle();
    stop_on_interrupt(cancel, bars.clone());

    let progress = engine.progress();
    let names = sink_names(&compiled, &progress);
    let reporter = Reporter::start(execute.progress, bars, progress.clone(), names.clone());

    let finished = engine.run();

    if let Some(reporter) = reporter {
        reporter.stop();
    }

    match finished {
        Ok(run) => finish(&run, &progress, &names),
        Err(error) => {
            let stage = error.node().and_then(|node| compiled.stage_of(node));
            report::run_error(&mut stderr, &loaded.files, &error, stage);

            ExitCode::FAILURE
        },
    }
}

/// Every sink's name in the order [Progress::sinks](av_toolbox_runtime::Progress::sinks) lists them.
fn sink_names(compiled: &CompiledPipeline<'_, '_>, progress: &Progress) -> Vec<String> {
    progress
        .sinks()
        .iter()
        .map(|sink| match compiled.stage_of(sink.node) {
            Some(stage) => stage.name.to_string(),
            None => "sink".to_string(),
        })
        .collect()
}

fn finish(run: &RunReport, progress: &Progress, names: &[String]) -> ExitCode {
    let snapshot = Snapshot::take(progress, names, run.wall);
    let elapsed = clock(run.wall);
    let read = snapshot.read;

    if run.cancelled {
        tracing::warn!(frames = run.frames, read, %elapsed, "the run was cancelled");
        for line in snapshot.sink_lines() {
            tracing::warn!("{line}");
        }

        return ExitCode::from(CANCELLED);
    }

    tracing::info!(frames = run.frames, read, artefacts = run.artefacts, %elapsed, "the run finished");
    for line in snapshot.sink_lines() {
        tracing::info!("{line}");
    }

    ExitCode::SUCCESS
}

/// Stops the run on the first Ctrl+C so sinks flush, and exits at once on the second after clearing `bars`.
fn stop_on_interrupt(cancel: Arc<Cancel>, bars: MultiProgress) {
    let interrupted = AtomicBool::new(false);

    let installed = ctrlc::set_handler(move || {
        let again = interrupted.swap(true, Ordering::SeqCst);
        if again {
            let _ = bars.clear();

            let code = i32::from(CANCELLED);
            std::process::exit(code);
        }

        tracing::warn!("stopping the run, press Ctrl+C again to exit at once");
        cancel.stop();
    });

    if let Err(error) = installed {
        tracing::warn!(%error, "Ctrl+C will stop the process without flushing the sinks");
    }
}
