use std::fs;
use std::path::Path;
use std::process::ExitCode;

use av_toolbox_pipeline::{LiveSource, Pipeline};
use codespan_reporting::term::termcolor::WriteColor;

use crate::cli::{Args, ExplainArgs};
use crate::load::Loaded;
use crate::{load, report};

const DEFAULT_HTML_PATH: &str = "plan.html";

pub(crate) fn run(args: &Args, explain: &ExplainArgs) -> ExitCode {
    let mut stderr = report::stderr(args.color);

    let Some(mut loaded) = load::load(&explain.pipeline, &explain.vars, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let mut profile = None;
    if let Some(path) = &explain.profile {
        let Some(read) = load::apply_profile(path, &mut loaded, &mut stderr) else {
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

    if explain.structure_only {
        return draw_structure(&pipeline, explain, &loaded, &mut stderr);
    }

    let live_sources = pipeline.live_sources();
    if !live_sources.is_empty() {
        log_live_sources(&live_sources);
        return draw_structure(&pipeline, explain, &loaded, &mut stderr);
    }

    let Some(backend) = load::backend(args, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    if let Some(profile) = &profile {
        load::warn_on_other_device(profile, &backend, args, &mut stderr);
    }

    let Some(compiled) = load::compile(pipeline, &backend, &loaded.files, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let drawn = if explain.html {
        compiled.render_html()
    } else {
        compiled.render_ascii()
    };
    output(explain, &drawn, &mut stderr)
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

    output(explain, &drawn, stderr)
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

fn output(explain: &ExplainArgs, drawn: &str, stderr: &mut dyn WriteColor) -> ExitCode {
    if !explain.html {
        print!("{drawn}");
        return ExitCode::SUCCESS;
    }

    let default_path = Path::new(DEFAULT_HTML_PATH);
    let path = explain.path.as_deref().unwrap_or(default_path);
    let written = fs::write(path, drawn);
    let display = path.display();
    let Err(error) = written else {
        let message = format!("wrote {display}");
        report::progress_line(stderr, &message);

        return ExitCode::SUCCESS;
    };

    let message = format!("cannot write {display}: {error}");
    report::error_line(stderr, &message);

    ExitCode::FAILURE
}
