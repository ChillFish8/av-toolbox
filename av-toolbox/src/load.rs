use std::fs;
use std::path::Path;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Backend, COMPILATION_CACHE_ENV, install_compilation_cache};
use av_toolbox_graph::{Diagnostic, StageSpec};
use av_toolbox_pipeline::{CompiledPipeline, Pipeline, PipelineError};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term::termcolor::WriteColor;

use crate::cli::Args;
use crate::report::{self, PipelineFile};

/// Every built-in stage, or `None` after saying why they could not be registered.
pub(crate) fn registry(stderr: &mut dyn WriteColor) -> Option<PipelineRegistry> {
    let registered = av_toolbox_pipeline::registry();

    match registered {
        Ok(registry) => Some(registry),
        Err(error) => {
            let message = format!("the built-in stages could not be registered: {error}");
            report::error_line(stderr, &message);

            None
        },
    }
}

/// A pipeline file and the stages it defines.
pub(crate) struct Loaded {
    pub(crate) file: PipelineFile,
    pub(crate) specs: Vec<StageSpec>,
}

/// Reads and parses a pipeline file, or prints why it cannot be.
pub(crate) fn load(path: &Path, stderr: &mut dyn WriteColor) -> Option<Loaded> {
    let display = path.display();
    let name = display.to_string();
    let read = fs::read_to_string(path);
    let text = match read {
        Ok(text) => text,
        Err(error) => {
            let message = format!("cannot read {name}: {error}");
            report::error_line(stderr, &message);

            return None;
        },
    };

    let parsed = av_toolbox_pipeline::toml::parse(&text);
    let file = SimpleFile::new(name, text);

    match parsed {
        Ok(specs) => Some(Loaded { file, specs }),
        Err(diagnostics) => {
            let found = diagnostics.into_inner();
            report::diagnostics(stderr, &file, &found);

            None
        },
    }
}

/// Builds the pipeline, or prints its diagnostics.
pub(crate) fn build<'r>(
    registry: &'r PipelineRegistry,
    loaded: &Loaded,
    stderr: &mut dyn WriteColor,
) -> Option<Pipeline<'r>> {
    let built = Pipeline::build(registry, &loaded.specs);

    match built {
        Ok(pipeline) => Some(pipeline),
        Err(error) => {
            let warnings = Vec::new();
            print_error(error, warnings, &loaded.file, stderr);

            None
        },
    }
}

/// Installs the kernel cache and opens the first backend that starts on the chosen device.
pub(crate) fn backend(args: &Args, stderr: &mut dyn WriteColor) -> Option<Backend> {
    let installed = install_compilation_cache();
    match installed {
        Ok(Some(path)) => tracing::info!(?path, "caching compiled kernels"),
        Ok(None) => tracing::info!("kernel caching is off, unset {COMPILATION_CACHE_ENV} to turn it back on"),
        Err(error) => {
            let message = format!("unable to install the kernel cache: {error}");
            report::error_line(stderr, &message);

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
            let message = error.to_string();
            report::error_line(stderr, &message);

            None
        },
    }
}

/// Compiles the pipeline and prints its warnings, or prints why it failed with those warnings.
pub(crate) fn compile<'r, 'b>(
    pipeline: Pipeline<'r>,
    backend: &'b Backend,
    loaded: &Loaded,
    stderr: &mut dyn WriteColor,
) -> Option<CompiledPipeline<'r, 'b>> {
    let warnings = pipeline.warnings().to_vec();
    let compiled = pipeline.compile(backend);

    match compiled {
        Ok(compiled) => {
            let compiled_warnings = compiled.warnings();
            report::diagnostics(stderr, &loaded.file, compiled_warnings);

            Some(compiled)
        },
        Err(error) => {
            print_error(error, warnings, &loaded.file, stderr);

            None
        },
    }
}

fn print_error(
    error: PipelineError,
    warnings: Vec<Diagnostic>,
    file: &PipelineFile,
    stderr: &mut dyn WriteColor,
) {
    let mut found = error.into_diagnostics().into_inner();
    found.extend(warnings);

    report::diagnostics(stderr, file, &found);
}
