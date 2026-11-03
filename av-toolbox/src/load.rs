use std::path::Path;

use av_toolbox::report::{self, PipelineFiles};
use av_toolbox::{BackendOptions, steps};
use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Accelerator, Backend, Device};
use av_toolbox_graph::StageSpec;
use av_toolbox_pipeline::{CompiledPipeline, DiskFiles, Pipeline};
use codespan_reporting::term::termcolor::WriteColor;

use crate::cli::Args;
use crate::profile::file::{self, ProfileFile};

/// Every built-in stage, or `None` after logging why they could not be registered.
pub(crate) fn registry() -> Option<PipelineRegistry> {
    let registered = steps::registry();

    match registered {
        Ok(registry) => Some(registry),
        Err(error) => {
            tracing::error!("{error}");

            None
        },
    }
}

/// A pipeline's files and the stages they define.
pub(crate) struct Loaded {
    pub(crate) files: PipelineFiles,
    pub(crate) specs: Vec<StageSpec>,
}

/// Reads a pipeline and every file it calls, or prints why it cannot be.
pub(crate) fn load(path: &Path, vars: &[(String, String)], stderr: &mut dyn WriteColor) -> Option<Loaded> {
    let supplied = steps::supplied(vars);

    let loaded = av_toolbox_pipeline::load(path, &DiskFiles, &supplied);

    match loaded {
        Ok(loaded) => {
            let files = PipelineFiles::new(loaded.sources);
            Some(Loaded {
                files,
                specs: loaded.specs,
            })
        },
        Err(failure) => {
            let files = PipelineFiles::new(failure.sources);
            let found = failure.diagnostics.into_inner();
            report::diagnostics(stderr, &files, &found);

            None
        },
    }
}

/// Builds the pipeline, or prints its diagnostics.
pub(crate) fn build<'r>(
    registry: &'r PipelineRegistry,
    specs: &[StageSpec],
    files: &PipelineFiles,
    stderr: &mut dyn WriteColor,
) -> Option<Pipeline<'r>> {
    let built = steps::build(registry, specs);

    match built {
        Ok(pipeline) => Some(pipeline),
        Err(found) => {
            report::diagnostics(stderr, files, &found);

            None
        },
    }
}

/// Installs the kernel cache and opens the first backend that starts on the chosen device.
pub(crate) fn backend(args: &Args) -> Option<Backend> {
    let options = BackendOptions {
        accelerators: args.accelerators.clone(),
        device: args.device,
        tuning: args.tune,
    };

    let opened = av_toolbox::open_backend(&options);
    match opened {
        Ok(backend) => Some(backend),
        Err(error) => {
            tracing::error!("{error}");

            None
        },
    }
}

/// Compiles the pipeline and prints its warnings, or prints why it failed with those warnings.
pub(crate) fn compile<'r, 'b>(
    pipeline: Pipeline<'r>,
    backend: &'b Backend,
    files: &PipelineFiles,
    stderr: &mut dyn WriteColor,
) -> Option<CompiledPipeline<'r, 'b>> {
    compile_pipeline(pipeline, backend, files, stderr, true)
}

/// The same as [compile], but a pipeline that compiles prints no warnings.
pub(crate) fn compile_quietly<'r, 'b>(
    pipeline: Pipeline<'r>,
    backend: &'b Backend,
    files: &PipelineFiles,
    stderr: &mut dyn WriteColor,
) -> Option<CompiledPipeline<'r, 'b>> {
    compile_pipeline(pipeline, backend, files, stderr, false)
}

fn compile_pipeline<'r, 'b>(
    pipeline: Pipeline<'r>,
    backend: &'b Backend,
    files: &PipelineFiles,
    stderr: &mut dyn WriteColor,
    show_warnings: bool,
) -> Option<CompiledPipeline<'r, 'b>> {
    let compiled = steps::compile(pipeline, backend);

    match compiled {
        Ok(compiled) => {
            if show_warnings {
                let compiled_warnings = compiled.warnings();
                report::diagnostics(stderr, files, compiled_warnings);
            }

            Some(compiled)
        },
        Err(found) => {
            report::diagnostics(stderr, files, &found);

            None
        },
    }
}

/// Reads the profile at `path` and applies its worker counts, warning about keys that name no stage.
pub(crate) fn apply_profile(path: &Path, loaded: &mut Loaded) -> Option<ProfileFile> {
    let read = file::read(path);
    let profile = match read {
        Ok(profile) => profile,
        Err(error) => {
            tracing::error!("{error}");

            return None;
        },
    };

    let unknown = file::apply(&profile.workers, &mut loaded.specs);
    for key in unknown {
        tracing::warn!("the profile names {key}, which is not a stage of this pipeline");
    }

    Some(profile)
}

pub(crate) fn warn_on_other_device(profile: &ProfileFile, backend: &Backend) {
    let accelerator = backend.accelerator();
    if names_the_default(&profile.device, accelerator) {
        return;
    }

    let device = backend.device();
    let opened = file::device_label(accelerator, device);

    if let Some(message) = device_mismatch(&profile.device, &opened) {
        tracing::warn!("{message}");
    }
}

/// Whether a profile names the default device, which matches whatever device the backend picks.
fn names_the_default(profile_device: &str, accelerator: Accelerator) -> bool {
    let unresolved = file::device_label(accelerator, Device::Default);
    profile_device == unresolved
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

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::default_accelerators;

    use super::*;

    #[test]
    fn a_profile_from_the_opened_device_raises_no_warning() {
        assert_eq!(device_mismatch("vulkan discrete:1", "vulkan discrete:1"), None);
    }

    #[test]
    fn a_profile_naming_the_default_device_matches_any() {
        let accelerator = default_accelerators()[0];

        assert!(names_the_default(&format!("{accelerator} default"), accelerator));
        assert!(!names_the_default(
            &format!("{accelerator} discrete:0"),
            accelerator
        ));
    }

    #[test]
    fn a_profile_from_another_device_names_both() {
        let message = device_mismatch("vulkan discrete:1", "vulkan integrated:0");

        let expected = "the profile was measured on vulkan discrete:1, but this run uses vulkan integrated:0";
        assert_eq!(message.as_deref(), Some(expected));
    }
}
