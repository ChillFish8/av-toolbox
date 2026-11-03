use std::fs;
use std::path::Path;
use std::process::ExitCode;

use av_toolbox::report;
use av_toolbox_pipeline::{LiveSource, Pipeline};
use codespan_reporting::term::termcolor::WriteColor;

use crate::cli::{Args, ExplainArgs};
use crate::load::{self, Loaded};

const DEFAULT_HTML_PATH: &str = "plan.html";

pub(crate) fn run(args: &Args, explain: &ExplainArgs) -> ExitCode {
    let mut stderr = args.color.stderr();

    let Some(mut loaded) = load::load(&explain.pipeline, &explain.vars, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let mut profile = None;
    if let Some(path) = &explain.profile {
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

    if explain.structure_only {
        return draw_structure(&pipeline, explain, &loaded, &mut stderr);
    }

    let live_sources = pipeline.live_sources();
    if !live_sources.is_empty() {
        log_live_sources(&live_sources);
        return draw_structure(&pipeline, explain, &loaded, &mut stderr);
    }

    let Some(backend) = load::backend(args) else {
        return ExitCode::FAILURE;
    };

    if let Some(profile) = &profile {
        load::warn_on_other_device(profile, &backend);
    }

    let Some(compiled) = load::compile(pipeline, &backend, &loaded.files, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let drawn = if explain.html {
        compiled.render_html()
    } else {
        compiled.render_ascii()
    };
    output(explain, &drawn)
}

fn draw_structure(
    pipeline: &Pipeline<'_>,
    explain: &ExplainArgs,
    loaded: &Loaded,
    stderr: &mut dyn WriteColor,
) -> ExitCode {
    let warnings = pipeline.warnings();
    report::diagnostics(stderr, &loaded.files, warnings);

    let drawn = if explain.html {
        pipeline.render_html()
    } else {
        pipeline.render_ascii()
    };

    output(explain, &drawn)
}

fn log_live_sources(live_sources: &[LiveSource<'_>]) {
    for live_source in live_sources {
        let name = live_source.stage.name;
        let stage_type = live_source.stage_type;

        tracing::info!(
            "source {name} ({stage_type}) reads a live stream, so formats and transfers are not shown. \
             Pipe a stream in and run execute to see them."
        );
    }
}

fn output(explain: &ExplainArgs, drawn: &str) -> ExitCode {
    if !explain.html {
        print!("{drawn}");
        return ExitCode::SUCCESS;
    }

    let default_path = Path::new(DEFAULT_HTML_PATH);
    let path = explain.path.as_deref().unwrap_or(default_path);
    let written = fs::write(path, drawn);
    let shown = path.display();
    let Err(error) = written else {
        println!("wrote {shown}");

        return ExitCode::SUCCESS;
    };

    tracing::error!("cannot write {shown}: {error}");

    ExitCode::FAILURE
}
