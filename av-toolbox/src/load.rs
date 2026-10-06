use std::path::Path;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Accelerator, Backend, COMPILATION_CACHE_ENV, Device, install_compilation_cache};
use av_toolbox_graph::{Diagnostic, StageSpec};
use av_toolbox_pipeline::{CompiledPipeline, DiskFiles, Pipeline, PipelineError, Supplied};
use codespan_reporting::term::termcolor::WriteColor;

use crate::cli::Args;
use crate::profile::file::{self, ProfileFile};
use crate::report::{self, PipelineFiles};

/// Every built-in stage, or `None` after logging why they could not be registered.
pub(crate) fn registry() -> Option<PipelineRegistry> {
    let registered = av_toolbox_pipeline::registry();

    match registered {
        Ok(registry) => Some(registry),
        Err(error) => {
            tracing::error!("the built-in stages could not be registered: {error}");

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
    let mut supplied = Supplied::from_env();
    for (name, value) in vars {
        supplied.set_flag(name.clone(), value.clone());
    }

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
    let built = Pipeline::build(registry, specs);

    match built {
        Ok(pipeline) => Some(pipeline),
        Err(error) => {
            let warnings = Vec::new();
            print_error(error, warnings, files, stderr);

            None
        },
    }
}

/// Installs the kernel cache and opens the first backend that starts on the chosen device.
pub(crate) fn backend(args: &Args) -> Option<Backend> {
    let installed = install_compilation_cache();
    match installed {
        Ok(Some(path)) => tracing::info!(?path, "caching compiled kernels"),
        Ok(None) => tracing::info!(
            "kernel caching is off and autotune results go to CubeCL's default place, unset \
             {COMPILATION_CACHE_ENV} to turn it back on"
        ),
        Err(error) => {
            tracing::error!("unable to install the kernel cache: {error}");

            return None;
        },
    }

    let opened = av_toolbox_pipeline::open_backend(&args.accelerators, args.device);
    match opened {
        Ok(backend) => {
            let accelerator = backend.accelerator();
            tracing::info!(accelerator = %accelerator, device = %args.device, "opened the backend");

            Some(backend)
        },
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
    let warnings = pipeline.warnings().to_vec();
    let compiled = pipeline.compile(backend);

    match compiled {
        Ok(compiled) => {
            if show_warnings {
                let compiled_warnings = compiled.warnings();
                report::diagnostics(stderr, files, compiled_warnings);
            }

            Some(compiled)
        },
        Err(error) => {
            print_error(error, warnings, files, stderr);

            None
        },
    }
}

fn print_error(
    error: PipelineError,
    warnings: Vec<Diagnostic>,
    files: &PipelineFiles,
    stderr: &mut dyn WriteColor,
) {
    let mut found = error.into_diagnostics().into_inner();
    found.extend(warnings);

    report::diagnostics(stderr, files, &found);
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
    use super::*;

    #[test]
    fn a_profile_from_the_opened_device_raises_no_warning() {
        assert_eq!(device_mismatch("vulkan discrete:1", "vulkan discrete:1"), None);
    }

    #[test]
    fn a_profile_naming_the_default_device_matches_any() {
        assert!(names_the_default("vulkan default", Accelerator::Vulkan));
        assert!(!names_the_default("vulkan discrete:0", Accelerator::Vulkan));
    }

    #[test]
    fn a_profile_from_another_device_names_both() {
        let message = device_mismatch("vulkan discrete:1", "vulkan integrated:0");

        let expected = "the profile was measured on vulkan discrete:1, but this run uses vulkan integrated:0";
        assert_eq!(message.as_deref(), Some(expected));
    }
}
