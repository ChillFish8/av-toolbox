use av_toolbox_core::StageError;
use av_toolbox_cubek::resize::Kernel;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use super::geometry::Window;

/// What sets one `resize/*` stage apart from another.
pub(crate) trait ResizeKernel: 'static {
    type Args: DeserializeOwned + JsonSchema + Send + Sync;

    fn identifier() -> &'static str;

    fn window(args: &Self::Args) -> Window;

    /// The kernel the arguments describe, or an error when they describe none.
    fn kernel(args: &Self::Args) -> Result<Kernel, StageError>;
}

pub(crate) fn stage_name<K: ResizeKernel>() -> String {
    format!("resize/{}", K::identifier())
}

/// Arguments for a resize whose kernel takes no settings.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResizeArgs {
    /// The output width.
    pub(crate) width: u32,
    /// The output height.
    pub(crate) height: u32,
    /// Where the source window starts across, in input samples.
    #[serde(default)]
    pub(crate) src_left: f32,
    /// Where the source window starts down, in input samples.
    #[serde(default)]
    pub(crate) src_top: f32,
    /// How wide the source window is, the input width by default.
    pub(crate) src_width: Option<f32>,
    /// How tall the source window is, the input height by default.
    pub(crate) src_height: Option<f32>,
}

/// Arguments for `resize/bicubic`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct BicubicArgs {
    /// The output width.
    pub(crate) width: u32,
    /// The output height.
    pub(crate) height: u32,
    /// Where the source window starts across, in input samples.
    #[serde(default)]
    pub(crate) src_left: f32,
    /// Where the source window starts down, in input samples.
    #[serde(default)]
    pub(crate) src_top: f32,
    /// How wide the source window is, the input width by default.
    pub(crate) src_width: Option<f32>,
    /// How tall the source window is, the input height by default.
    pub(crate) src_height: Option<f32>,
    /// How much the cubic blurs.
    #[serde(default = "one_third")]
    pub(crate) b: f32,
    /// How much the cubic sharpens.
    #[serde(default = "one_third")]
    pub(crate) c: f32,
}

/// Arguments for `resize/lanczos`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct LanczosArgs {
    /// The output width.
    pub(crate) width: u32,
    /// The output height.
    pub(crate) height: u32,
    /// Where the source window starts across, in input samples.
    #[serde(default)]
    pub(crate) src_left: f32,
    /// Where the source window starts down, in input samples.
    #[serde(default)]
    pub(crate) src_top: f32,
    /// How wide the source window is, the input width by default.
    pub(crate) src_width: Option<f32>,
    /// How tall the source window is, the input height by default.
    pub(crate) src_height: Option<f32>,
    /// How many input samples the kernel reaches either side.
    #[serde(default = "three")]
    pub(crate) taps: u32,
}

fn one_third() -> f32 {
    1.0 / 3.0
}

fn three() -> u32 {
    3
}

/// The most taps `resize/lanczos` accepts.
const MAX_TAPS: u32 = 32;

fn zero_taps() -> StageError {
    StageError::msg("resize/lanczos needs taps above 0")
}

fn too_many_taps(taps: u32) -> StageError {
    let message = format!("resize/lanczos needs taps of at most {MAX_TAPS}, got {taps}");

    StageError::msg(message)
}

fn cubic_not_finite() -> StageError {
    StageError::msg("resize/bicubic needs b and c to be finite numbers")
}

fn window_of(args: &ResizeArgs) -> Window {
    Window {
        width: args.width,
        height: args.height,
        src_left: args.src_left,
        src_top: args.src_top,
        src_width: args.src_width,
        src_height: args.src_height,
    }
}

pub(crate) struct Point;

impl ResizeKernel for Point {
    type Args = ResizeArgs;

    fn identifier() -> &'static str {
        "point"
    }

    fn window(args: &Self::Args) -> Window {
        window_of(args)
    }

    fn kernel(_args: &Self::Args) -> Result<Kernel, StageError> {
        Ok(Kernel::Point)
    }
}

pub(crate) struct Bilinear;

impl ResizeKernel for Bilinear {
    type Args = ResizeArgs;

    fn identifier() -> &'static str {
        "bilinear"
    }

    fn window(args: &Self::Args) -> Window {
        window_of(args)
    }

    fn kernel(_args: &Self::Args) -> Result<Kernel, StageError> {
        Ok(Kernel::Bilinear)
    }
}

pub(crate) struct Bicubic;

impl ResizeKernel for Bicubic {
    type Args = BicubicArgs;

    fn identifier() -> &'static str {
        "bicubic"
    }

    fn window(args: &Self::Args) -> Window {
        Window {
            width: args.width,
            height: args.height,
            src_left: args.src_left,
            src_top: args.src_top,
            src_width: args.src_width,
            src_height: args.src_height,
        }
    }

    fn kernel(args: &Self::Args) -> Result<Kernel, StageError> {
        if !args.b.is_finite() || !args.c.is_finite() {
            return Err(cubic_not_finite());
        }

        Ok(Kernel::Bicubic { b: args.b, c: args.c })
    }
}

pub(crate) struct Lanczos;

impl ResizeKernel for Lanczos {
    type Args = LanczosArgs;

    fn identifier() -> &'static str {
        "lanczos"
    }

    fn window(args: &Self::Args) -> Window {
        Window {
            width: args.width,
            height: args.height,
            src_left: args.src_left,
            src_top: args.src_top,
            src_width: args.src_width,
            src_height: args.src_height,
        }
    }

    fn kernel(args: &Self::Args) -> Result<Kernel, StageError> {
        if args.taps == 0 {
            return Err(zero_taps());
        }

        if args.taps > MAX_TAPS {
            return Err(too_many_taps(args.taps));
        }

        Ok(Kernel::Lanczos { taps: args.taps })
    }
}

pub(crate) struct Spline16;

impl ResizeKernel for Spline16 {
    type Args = ResizeArgs;

    fn identifier() -> &'static str {
        "spline16"
    }

    fn window(args: &Self::Args) -> Window {
        window_of(args)
    }

    fn kernel(_args: &Self::Args) -> Result<Kernel, StageError> {
        Ok(Kernel::Spline16)
    }
}

pub(crate) struct Spline36;

impl ResizeKernel for Spline36 {
    type Args = ResizeArgs;

    fn identifier() -> &'static str {
        "spline36"
    }

    fn window(args: &Self::Args) -> Window {
        window_of(args)
    }

    fn kernel(_args: &Self::Args) -> Result<Kernel, StageError> {
        Ok(Kernel::Spline36)
    }
}

pub(crate) struct Spline64;

impl ResizeKernel for Spline64 {
    type Args = ResizeArgs;

    fn identifier() -> &'static str {
        "spline64"
    }

    fn window(args: &Self::Args) -> Window {
        window_of(args)
    }

    fn kernel(_args: &Self::Args) -> Result<Kernel, StageError> {
        Ok(Kernel::Spline64)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse<A: DeserializeOwned>(value: serde_json::Value) -> Result<A, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn a_window_defaults_to_the_whole_input() {
        let value = json!({ "width": 1280, "height": 720 });
        let args: ResizeArgs = parse(value).expect("parses");

        let window = Spline36::window(&args);

        assert_eq!(
            window,
            Window {
                width: 1280,
                height: 720,
                src_left: 0.0,
                src_top: 0.0,
                src_width: None,
                src_height: None,
            }
        );
    }

    #[test]
    fn bicubic_defaults_to_mitchell() {
        let value = json!({ "width": 8, "height": 8 });
        let args: BicubicArgs = parse(value).expect("parses");

        let kernel = Bicubic::kernel(&args).expect("a valid kernel");

        assert_eq!(
            kernel,
            Kernel::Bicubic {
                b: 1.0 / 3.0,
                c: 1.0 / 3.0
            }
        );
    }

    #[test]
    fn lanczos_defaults_to_three_taps() {
        let value = json!({ "width": 8, "height": 8 });
        let args: LanczosArgs = parse(value).expect("parses");

        let kernel = Lanczos::kernel(&args).expect("a valid kernel");

        assert_eq!(kernel, Kernel::Lanczos { taps: 3 });
    }

    #[test]
    fn lanczos_refuses_zero_taps() {
        let value = json!({ "width": 8, "height": 8, "taps": 0 });
        let args: LanczosArgs = parse(value).expect("parses");

        let error = Lanczos::kernel(&args).expect_err("zero taps is refused");

        assert!(error.to_string().contains("resize/lanczos"));
        assert!(error.to_string().contains("taps"));
    }

    #[test]
    fn lanczos_refuses_more_than_32_taps() {
        let value = json!({ "width": 8, "height": 8, "taps": 33 });
        let args: LanczosArgs = parse(value).expect("parses");

        let error = Lanczos::kernel(&args).expect_err("33 taps is refused");

        assert!(error.to_string().contains("resize/lanczos"));
        assert!(error.to_string().contains("taps"));
    }

    #[test]
    fn lanczos_accepts_32_taps() {
        let value = json!({ "width": 8, "height": 8, "taps": 32 });
        let args: LanczosArgs = parse(value).expect("parses");

        let kernel = Lanczos::kernel(&args).expect("32 taps is accepted");

        assert_eq!(kernel, Kernel::Lanczos { taps: 32 });
    }

    #[test]
    fn bicubic_refuses_an_infinite_b() {
        let value = json!({ "width": 8, "height": 8, "b": 1e39 });
        let args: BicubicArgs = parse(value).expect("parses");

        let refused = Bicubic::kernel(&args);

        assert!(refused.is_err());
    }

    #[test]
    fn an_unknown_argument_is_refused() {
        let value = json!({ "width": 8, "height": 8, "taps": 3 });
        let parsed: Result<ResizeArgs, _> = parse(value);

        assert!(parsed.is_err());
    }

    #[test]
    fn each_kernel_names_its_stage() {
        assert_eq!(stage_name::<Point>(), "resize/point");
        assert_eq!(stage_name::<Bilinear>(), "resize/bilinear");
        assert_eq!(stage_name::<Bicubic>(), "resize/bicubic");
        assert_eq!(stage_name::<Lanczos>(), "resize/lanczos");
        assert_eq!(stage_name::<Spline16>(), "resize/spline16");
        assert_eq!(stage_name::<Spline36>(), "resize/spline36");
        assert_eq!(stage_name::<Spline64>(), "resize/spline64");
    }
}
