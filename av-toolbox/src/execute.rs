use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use av_toolbox_gpu::Backend;
use av_toolbox_runtime::{Cancel, RunReport, Settings};
use codespan_reporting::term::termcolor::WriteColor;

use crate::cli::{Args, ExecuteArgs};
use crate::load::Loaded;
use crate::profile::file::{self, ProfileFile};
use crate::{load, report};

/// The exit code of a run stopped with Ctrl+C.
pub(crate) const CANCELLED: u8 = 130;

pub(crate) fn run(args: &Args, execute: &ExecuteArgs) -> ExitCode {
    let mut stderr = report::stderr(args.color);

    let Some(mut loaded) = load::load(&execute.pipeline, &execute.vars, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let mut profile = None;
    if let Some(path) = &execute.profile {
        let Some(read) = apply_profile(path, &mut loaded, &mut stderr) else {
            return ExitCode::FAILURE;
        };

        profile = Some(read);
    }

    let Some(registry) = load::registry(&mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(pipeline) = load::build(&registry, &loaded.specs, &loaded.files, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(backend) = load::backend(args, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    if let Some(profile) = &profile {
        warn_on_other_device(profile, &backend, args, &mut stderr);
    }

    let Some(compiled) = load::compile(pipeline, &backend, &loaded.files, &mut stderr) else {
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
            report::diagnostics(&mut stderr, &loaded.files, &found);

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
            report::run_error(&mut stderr, &loaded.files, &error, stage);

            ExitCode::FAILURE
        },
    }
}

/// Reads the profile at `path` and applies its worker counts, warning about keys that name no stage.
fn apply_profile(path: &Path, loaded: &mut Loaded, stderr: &mut dyn WriteColor) -> Option<ProfileFile> {
    let read = file::read(path);
    let profile = match read {
        Ok(profile) => profile,
        Err(error) => {
            let message = error.to_string();
            report::error_line(stderr, &message);

            return None;
        },
    };

    let unknown = file::apply(&profile.workers, &mut loaded.specs);
    for key in unknown {
        let message = format!("the profile names {key}, which is not a stage of this pipeline");
        report::warning_line(stderr, &message);
    }

    Some(profile)
}

fn warn_on_other_device(profile: &ProfileFile, backend: &Backend, args: &Args, stderr: &mut dyn WriteColor) {
    let accelerator = backend.accelerator();
    let opened = file::device_label(accelerator, args.device);

    if let Some(message) = device_mismatch(&profile.device, &opened) {
        report::warning_line(stderr, &message);
    }
}

/// The warning for a profile measured on a different device than the one opened, if they differ.
fn device_mismatch(profile_device: &str, opened: &str) -> Option<String> {
    if opened == profile_device {
        return None;
    }

    Some(format!(
        "the profile was measured on {profile_device}, but this run uses {opened}"
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profile_from_the_opened_device_raises_no_warning() {
        assert_eq!(device_mismatch("vulkan discrete:1", "vulkan discrete:1"), None);
    }

    #[test]
    fn a_profile_from_another_device_names_both() {
        let message = device_mismatch("vulkan discrete:1", "vulkan integrated:0");

        let expected = "the profile was measured on vulkan discrete:1, but this run uses vulkan integrated:0";
        assert_eq!(message.as_deref(), Some(expected));
    }
}
