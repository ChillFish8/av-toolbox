mod cli;
mod execute;
mod explain;
mod list_devices;
mod list_stages;
mod load;
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
    start_logging(&args.command);

    match &args.command {
        Command::Execute(execute) => execute::run(&args, execute),
        Command::ListDevices => list_devices::run(&args.accelerators),
        Command::ListStages(list) => list_stages::run(&args, list),
        Command::Explain(explain) => explain::run(&args, explain),
        Command::Pull(pull_args) => pull::run(&args, pull_args),
    }
}

/// Logs to stderr, quieter for the commands that only print a table.
fn start_logging(command: &Command) {
    let quiet = matches!(command, Command::ListDevices | Command::ListStages(_));
    let default = if quiet { "warn" } else { "info" };
    let from_env = EnvFilter::try_from_default_env();
    let filter = from_env.unwrap_or_else(|_| EnvFilter::new(default));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}
