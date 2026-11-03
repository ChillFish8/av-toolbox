mod cli;
mod describe;
mod execute;
mod explain;
mod inspect;
#[cfg(feature = "cuda")]
mod libdevice;
mod list_devices;
mod list_stages;
mod load;
mod profile;
mod progress;
mod pull;

use std::process::ExitCode;

use clap::Parser;
use indicatif::MultiProgress;
use mimalloc::MiMalloc;
use tracing_indicatif::writer::{IndicatifWriter, Stderr};
use tracing_subscriber::EnvFilter;

use self::cli::{Args, Command};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> ExitCode {
    // SAFETY: no other thread exists yet, so nothing can race the environment change.
    unsafe { av_toolbox_gpu::raise_codegen_stack_limit() };

    // SAFETY: still single-threaded, so nothing can race the environment change.
    #[cfg(feature = "cuda")]
    let libdevice_outcome = unsafe { self::libdevice::point_cuda_path_at_bundle() };

    let args = Args::parse();
    let bars = MultiProgress::new();
    start_logging(&args, &bars);

    #[cfg(feature = "cuda")]
    if let Err(error) = libdevice_outcome {
        tracing::warn!(%error, "cannot install the bundled libdevice");
    }

    match &args.command {
        Command::Execute(execute) => execute::run(&args, execute, &bars),
        Command::Profile(profile_args) => profile::run(&args, profile_args),
        Command::ListDevices => list_devices::run(&args.accelerators),
        Command::ListStages(list) => list_stages::run(list),
        Command::Describe(describe_args) => describe::run(describe_args),
        Command::Explain(explain) => explain::run(&args, explain),
        Command::Inspect(inspect_args) => inspect::run(&args, inspect_args),
        Command::Pull(pull_args) => pull::run(pull_args),
    }
}

/// What every command logs by default, with only warnings from crates outside the workspace.
const QUIET_FILTER: &str = "warn";

/// What a command that does work logs by default, with the workspace's own progress on top.
const PROGRESS_FILTER: &str = "warn,av_toolbox=info";

/// Logs to stderr above any progress bars, showing progress only for the commands that do work.
fn start_logging(args: &Args, bars: &MultiProgress) {
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
    let writer: IndicatifWriter<Stderr> = IndicatifWriter::new(bars.clone());

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi)
        .with_writer(writer)
        .init();
}
