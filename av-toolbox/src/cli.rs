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

    /// Write an HTML page instead of drawing in the terminal.
    #[arg(long)]
    pub(crate) html: bool,

    /// Where to write the HTML page, `plan.html` in the working directory by default.
    #[arg(long, value_name = "PATH", requires = "html")]
    pub(crate) path: Option<PathBuf>,

    /// Draw the stages and their connections without opening a device.
    #[arg(long)]
    pub(crate) structure_only: bool,
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
}
