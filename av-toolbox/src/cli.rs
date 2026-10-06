use std::io::IsTerminal;
use std::path::PathBuf;

use av_toolbox_gpu::{Accelerator, Device, default_accelerators};
use clap::{Parser, Subcommand, ValueEnum};
use codespan_reporting::term::termcolor::ColorChoice;

#[derive(Debug, Parser)]
#[command(
    name = "av-toolbox",
    about = "GPU accelerated video filtering pipelines",
    version
)]
pub(crate) struct Args {
    /// Which hardware backends to try, in order of preference.
    ///
    /// The first backend that starts is used. If none do, the program exits with an error.
    ///
    /// The list is comma-separated, for example `cuda,vulkan`.
    #[arg(
        short = 'A',
        long,
        env = "AVTB_ACCELERATORS",
        value_delimiter = ',',
        default_values_t = default_accelerators(),
        global = true
    )]
    pub(crate) accelerators: Vec<Accelerator>,

    /// Which device to use on the chosen backend.
    ///
    /// `default` lets the backend pick.
    ///
    /// `discrete[:N]` picks the Nth discrete GPU (default 0). Works on CUDA, ROCm, and Vulkan.
    ///
    /// `integrated[:N]` picks the Nth integrated GPU. Vulkan only.
    ///
    /// `virtual[:N]` picks the Nth virtual GPU. Vulkan only.
    ///
    /// `cpu` picks a software device where the platform offers one, such as lavapipe under Vulkan.
    /// It is for testing a pipeline, not for real encodes.
    #[arg(short, long, env = "AVTB_DEVICE", default_value = "default", global = true)]
    pub(crate) device: Device,

    /// When to colour diagnostics.
    ///
    /// `auto` colours them when stderr is a terminal and `NO_COLOR` is unset.
    #[arg(long, value_enum, default_value = "auto", global = true)]
    pub(crate) color: ColorMode,

    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run a pipeline.
    Execute(ExecuteArgs),

    /// Time every stage of a pipeline and work out how many workers each should run.
    ///
    /// The pipeline runs three times over the same frames, with its sinks replaced so nothing is written. The
    /// worker counts go to a profile file that `execute --profile` applies.
    Profile(ProfileArgs),

    /// List the devices each backend can see on this machine.
    ///
    /// Every row names a device in the spelling `--device` takes, next to the backends that offer
    /// it. `--accelerators` narrows which backends are asked.
    ///
    /// Ordinals are counted per backend, so the same row under two backends is not always the
    /// same physical card.
    ListDevices,

    /// List every stage a pipeline can use, or describe one.
    ListStages(ListStagesArgs),

    /// Draw a pipeline as a graph.
    ///
    /// By default the pipeline is compiled on the chosen device, so the drawing shows where frames
    /// move between host and device and which stages run in parallel.
    Explain(ExplainArgs),

    /// Download a pack of model weights from this project's GitHub releases.
    ///
    /// The pack comes from the release matching this build's version, or from the latest release when that one
    /// does not exist. Every file is checked against the checksum this build expects before it is installed.
    Pull(PullArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct ExecuteArgs {
    /// The pipeline file to run.
    pub(crate) pipeline: PathBuf,

    /// Sets a pipeline variable, such as `--var clip_input=in.mkv`. Can be given many times.
    ///
    /// Takes precedence over an `AVTB_VAR_<NAME>` environment variable for the same variable.
    #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
    pub(crate) vars: Vec<(String, String)>,

    /// How much memory frames in flight may occupy.
    ///
    /// Takes a size such as `8GB`, `4.5GiB` or `512MB`, or a plain byte count. `GB` is 10^9 while
    /// `GiB` is 2^30.
    ///
    /// Left unset, the budget is worked out from the pipeline.
    #[arg(long, value_name = "SIZE", value_parser = parse_frame_budget)]
    pub(crate) frame_budget: Option<usize>,

    /// Worker counts written by `av-toolbox profile`, for every stage that declares none.
    #[arg(long, value_name = "PATH")]
    pub(crate) profile: Option<PathBuf>,

    /// When to show the run's progress.
    ///
    /// `auto` draws a live view on a terminal and logs a progress line every ten seconds otherwise. `always`
    /// draws the live view everywhere and `never` shows nothing.
    #[arg(long, value_enum, default_value = "auto")]
    pub(crate) progress: ProgressMode,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ProfileArgs {
    /// The pipeline file to profile. Its sources must read files rather than a live stream.
    pub(crate) pipeline: PathBuf,

    /// Sets a pipeline variable, such as `--var clip_input=in.mkv`. Can be given many times.
    ///
    /// Takes precedence over an `AVTB_VAR_<NAME>` environment variable for the same variable.
    #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
    pub(crate) vars: Vec<(String, String)>,

    /// How many frames each pass times, after its warmup.
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) frames: u64,

    /// How many frames each pass runs first without timing them, while kernels compile and tune.
    #[arg(long, default_value_t = 30)]
    pub(crate) warmup: u64,

    /// The most workers any one stage is given, the number of CPU threads by default.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) max_workers: Option<u64>,

    /// How much memory frames in flight may occupy, as for `execute`.
    #[arg(long, value_name = "SIZE", value_parser = parse_frame_budget)]
    pub(crate) frame_budget: Option<usize>,

    /// Where to write the profile, `<pipeline>.profile.toml` in the working directory by default.
    #[arg(long, value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,
}

/// Reads `NAME=VALUE` into its name and value.
fn parse_var(raw: &str) -> Result<(String, String), String> {
    let invalid = format!("invalid variable '{raw}'. Give it as NAME=VALUE");

    let Some((name, value)) = raw.split_once('=') else {
        return Err(invalid);
    };

    if name.is_empty() {
        return Err(invalid);
    }

    Ok((name.to_string(), value.to_string()))
}

/// Reads a size such as `8GB` into a byte count.
fn parse_frame_budget(raw: &str) -> Result<usize, String> {
    let invalid = format!(
        "invalid frame budget '{raw}'. Give a size such as 8GB, 4.5GiB or 512MB, or a plain byte count"
    );

    let parsed = raw.parse::<bytesize::ByteSize>();
    let Ok(size) = parsed else {
        return Err(invalid);
    };

    let bytes = size.as_u64();
    usize::try_from(bytes).map_err(|_| invalid)
}

#[derive(Debug, clap::Args)]
pub(crate) struct ListStagesArgs {
    /// A stage type such as `y4m/pipe`, to show its inputs and arguments.
    pub(crate) stage_type: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ExplainArgs {
    /// The pipeline file to draw.
    pub(crate) pipeline: PathBuf,

    /// Sets a pipeline variable, such as `--var clip_input=in.mkv`. Can be given many times.
    ///
    /// Takes precedence over an `AVTB_VAR_<NAME>` environment variable for the same variable.
    #[arg(long = "var", value_name = "NAME=VALUE", value_parser = parse_var)]
    pub(crate) vars: Vec<(String, String)>,

    /// Write an HTML page instead of drawing in the terminal.
    #[arg(long)]
    pub(crate) html: bool,

    /// Where to write the HTML page, `plan.html` in the working directory by default.
    #[arg(long, value_name = "PATH", requires = "html")]
    pub(crate) path: Option<PathBuf>,

    /// Draw the stages and their connections without opening a device.
    #[arg(long)]
    pub(crate) structure_only: bool,

    /// Worker counts written by `av-toolbox profile`, for every stage that declares none.
    #[arg(long, value_name = "PATH")]
    pub(crate) profile: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct PullArgs {
    /// The pack to download, such as `artcnn`.
    pub(crate) identifier: String,

    /// Delete the pack's other installed versions without asking.
    #[arg(long)]
    pub(crate) prune: bool,

    /// Download the pack again even when this version is already installed.
    #[arg(long)]
    pub(crate) force: bool,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
pub(crate) enum ColorMode {
    Auto,
    Always,
    Never,
}

impl ColorMode {
    pub(crate) fn choice(self) -> ColorChoice {
        match self {
            Self::Always => ColorChoice::Always,
            Self::Never => ColorChoice::Never,
            Self::Auto if std::io::stderr().is_terminal() => ColorChoice::Auto,
            Self::Auto => ColorChoice::Never,
        }
    }

    /// Whether stderr gets coloured output.
    pub(crate) fn enabled(self) -> bool {
        self.choice() != ColorChoice::Never
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
pub(crate) enum ProgressMode {
    Auto,
    Always,
    Never,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_budget_reads_sizes_and_byte_counts() {
        assert_eq!(parse_frame_budget("4KiB"), Ok(4096));
        assert_eq!(parse_frame_budget("1000"), Ok(1000));
    }

    #[test]
    fn a_bad_frame_budget_shows_the_expected_form() {
        let error = parse_frame_budget("lots").expect_err("not a size");

        assert!(error.contains("4.5GiB"));
    }

    #[test]
    fn a_var_splits_at_the_first_equals() {
        assert_eq!(
            parse_var("path=a=b.y4m"),
            Ok(("path".to_string(), "a=b.y4m".to_string()))
        );
    }

    #[test]
    fn a_var_without_equals_is_refused() {
        let error = parse_var("sigma").expect_err("no value");

        assert!(error.contains("NAME=VALUE"));
    }

    #[test]
    fn a_var_without_a_name_is_refused() {
        assert!(parse_var("=1").is_err());
    }
}
