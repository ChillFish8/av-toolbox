mod cli;
mod execute;
mod explain;
mod list_devices;
mod list_stages;
mod load;
mod profile;
mod pull;
mod report;

use std::process::ExitCode;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use self::cli::{Args, Command};

fn main() -> ExitCode {
    // SAFETY: no other thread exists yet, so nothing can race the environment change.
    unsafe { av_toolbox_gpu::raise_codegen_stack_limit() };

    let args = Args::parse();
    start_logging(&args);

    match &args.command {
        Command::Execute(execute) => execute::run(&args, execute),
        Command::Profile(profile_args) => profile::run(&args, profile_args),
        Command::ListDevices => list_devices::run(&args.accelerators),
        Command::ListStages(list) => list_stages::run(list),
        Command::Explain(explain) => explain::run(&args, explain),
        Command::Pull(pull_args) => pull::run(pull_args),
    }
}

/// What every command logs by default, with only warnings from crates outside the workspace.
const QUIET_FILTER: &str = "warn";

/// What a command that does work logs by default, with the workspace's own progress on top.
const PROGRESS_FILTER: &str = "warn,av_toolbox=info";

/// Logs to stderr, showing progress only for the commands that do work.
fn start_logging(args: &Args) {
    let shows_progress = matches!(
        args.command,
        Command::Execute(_) | Command::Profile(_) | Command::Pull(_)
    );
    let default = if shows_progress {
        PROGRESS_FILTER
    } else {
        QUIET_FILTER
    };
    let from_env = EnvFilter::try_from_default_env();
    let filter = from_env.unwrap_or_else(|_| EnvFilter::new(default));
    let ansi = args.color.enabled();

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi)
        .with_writer(std::io::stderr)
        .init();
}
