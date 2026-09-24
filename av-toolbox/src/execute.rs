use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use av_toolbox_runtime::{Cancel, RunReport, Settings};

use crate::cli::{Args, ExecuteArgs};
use crate::{load, report};

/// The exit code of a run stopped with Ctrl+C.
const CANCELLED: u8 = 130;

pub(crate) fn run(args: &Args, execute: &ExecuteArgs) -> ExitCode {
    let mut stderr = report::stderr(args.color);

    let Some(loaded) = load::load(&execute.pipeline, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(registry) = load::registry(&mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(pipeline) = load::build(&registry, &loaded, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(backend) = load::backend(args, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(compiled) = load::compile(pipeline, &backend, &loaded, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let settings = Settings {
        frame_budget: execute.frame_budget,
        ..Settings::default()
    };
    let built = compiled.engine(settings);
    let engine = match built {
        Ok(engine) => engine,
        Err(error) => {
            let found = error.into_diagnostics().into_inner();
            report::diagnostics(&mut stderr, &loaded.file, &found);

            return ExitCode::FAILURE;
        },
    };

    let cancel = engine.cancel_handle();
    stop_on_interrupt(cancel);

    let finished = engine.run();
    match finished {
        Ok(run) => finish(&run),
        Err(error) => {
            let stage = error.node().and_then(|node| compiled.stage_of(node));
            report::run_error(&mut stderr, &loaded.file, &error, stage);

            ExitCode::FAILURE
        },
    }
}

fn finish(run: &RunReport) -> ExitCode {
    for warning in &run.warnings {
        tracing::warn!("{warning}");
    }

    if run.cancelled {
        tracing::warn!(frames = run.frames, "the run was cancelled");
        return ExitCode::from(CANCELLED);
    }

    tracing::info!(frames = run.frames, "the run finished");
    ExitCode::SUCCESS
}

/// Stops the run on the first Ctrl+C so sinks flush, and exits at once on the second.
fn stop_on_interrupt(cancel: Arc<Cancel>) {
    let interrupted = AtomicBool::new(false);

    let installed = ctrlc::set_handler(move || {
        let again = interrupted.swap(true, Ordering::SeqCst);
        if again {
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
