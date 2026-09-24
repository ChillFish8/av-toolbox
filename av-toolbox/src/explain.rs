use std::fs;
use std::path::Path;
use std::process::ExitCode;

use codespan_reporting::term::termcolor::WriteColor;

use crate::cli::{Args, ExplainArgs};
use crate::{load, report};

const DEFAULT_HTML_PATH: &str = "plan.html";

pub(crate) fn run(args: &Args, explain: &ExplainArgs) -> ExitCode {
    let mut stderr = report::stderr(args.color);

    let Some(loaded) = load::load(&explain.pipeline, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(registry) = load::registry(&mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(pipeline) = load::build(&registry, &loaded, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    if explain.structure_only {
        let warnings = pipeline.warnings();
        report::diagnostics(&mut stderr, &loaded.file, warnings);

        let drawn = if explain.html {
            pipeline.render_html()
        } else {
            pipeline.render_ascii()
        };
        return output(explain, &drawn, &mut stderr);
    }

    let Some(backend) = load::backend(args, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let Some(compiled) = load::compile(pipeline, &backend, &loaded, &mut stderr) else {
        return ExitCode::FAILURE;
    };

    let drawn = if explain.html {
        compiled.render_html()
    } else {
        compiled.render_ascii()
    };
    output(explain, &drawn, &mut stderr)
}

fn output(explain: &ExplainArgs, drawn: &str, stderr: &mut dyn WriteColor) -> ExitCode {
    if !explain.html {
        print!("{drawn}");
        return ExitCode::SUCCESS;
    }

    let default_path = Path::new(DEFAULT_HTML_PATH);
    let path = explain.path.as_deref().unwrap_or(default_path);
    let written = fs::write(path, drawn);
    let Err(error) = written else {
        return ExitCode::SUCCESS;
    };

    let display = path.display();
    let message = format!("cannot write {display}: {error}");
    report::error_line(stderr, &message);

    ExitCode::FAILURE
}
