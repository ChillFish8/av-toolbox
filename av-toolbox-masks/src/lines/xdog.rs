use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::blur::{BlurPlanes, Gaussian};
use av_toolbox_cubek::histogram::{BINS, PercentilePlanes, percentile};
use av_toolbox_cubek::reduce::{extremes, sums};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    Mask as GpuMask,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

use super::arguments::{finite_positive, reachable_sigma};
use crate::luma::{Luma, accepts_luma, load_luma, produces_luma_mask};
use crate::sizing::{Reduction, Scratch, Sizing, mask_plane};

const NAME: &str = "lines/xdog";

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct XdogArgs {
    /// The deviation of the first, narrower blur.
    #[serde(default = "default_sigma")]
    sigma: f32,
    /// How much wider the second blur is than the first.
    #[serde(default = "default_k")]
    k: f32,
    /// How strongly the wider blur is taken away from the narrower one.
    #[serde(default = "default_tau")]
    tau: f32,
    /// The response at which a sample stops being a line. Found per frame when unset.
    eps: Option<f32>,
    /// How sharply samples fall from line to background. Found per frame when unset.
    phi: Option<f32>,
    /// The percentile of the response that `eps` is taken from when unset.
    #[serde(default = "default_eps_percentile")]
    eps_percentile: f32,
}

fn default_sigma() -> f32 {
    1.0
}

fn default_k() -> f32 {
    1.6
}

fn default_tau() -> f32 {
    0.95
}

fn default_eps_percentile() -> f32 {
    8.0
}

impl XdogArgs {
    fn validate(&self) -> Result<(), StageError> {
        reachable_sigma(NAME, "sigma", self.sigma)?;

        if !self.k.is_finite() {
            let message = format!("{NAME} needs a finite k, got {}", self.k);
            return Err(StageError::msg(message));
        }

        if self.k <= 1.0 {
            let message = format!("{NAME} needs k above 1, got {}", self.k);
            return Err(StageError::msg(message));
        }

        let far_sigma = self.sigma * self.k;
        reachable_sigma(NAME, "sigma * k", far_sigma)?;

        if let Some(phi) = self.phi {
            finite_positive(NAME, "phi", phi)?;
        }

        if !(0.0..=100.0).contains(&self.eps_percentile) {
            let message = format!(
                "{NAME} needs eps_percentile between 0 and 100, got {}",
                self.eps_percentile
            );
            return Err(StageError::msg(message));
        }

        Ok(())
    }
}

#[cube(launch_unchecked)]
fn difference_of_gaussians(near: &Array<f32>, far: &Array<f32>, output: &mut Array<f32>, tau: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = near[ABSOLUTE_POS] - tau * far[ABSOLUTE_POS];
}

/// Settles `eps` and `phi` into `parameters`, from the frame when asked to and from the args otherwise.
#[cube(launch_unchecked)]
fn resolve_parameters(
    found_eps: &Array<f32>,
    totals: &Array<f32>,
    parameters: &mut Array<f32>,
    fixed_eps: f32,
    fixed_phi: f32,
    automatic_eps: u32,
    automatic_phi: u32,
    samples: f32,
) {
    if automatic_eps == 1u32 {
        parameters[0] = found_eps[0];
    } else {
        parameters[0] = fixed_eps;
    }

    if automatic_phi == 1u32 {
        let mean = totals[0] / samples;
        let variance = f32::max(totals[1] / samples - mean * mean, 0.0);
        parameters[1] = 4.0 / (f32::sqrt(variance) + 0.000001);
    } else {
        parameters[1] = fixed_phi;
    }
}

#[cube(launch_unchecked)]
fn soft_threshold(difference: &Array<f32>, parameters: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let eps = parameters[0];
    let phi = parameters[1];
    let value = difference[ABSOLUTE_POS];

    let mut edge = 1.0f32;
    if value < eps {
        edge = 1.0 + f32::tanh(phi * (value - eps));
    }

    output[ABSOLUTE_POS] = f32::clamp(edge, 0.0, 1.0);
}

#[cube(launch_unchecked)]
fn stretch_into_mask(edges: &Array<f32>, range: &Array<f32>, output: &mut Array<u8>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let low = range[0];
    let high = range[1];

    let mut edge = edges[ABSOLUTE_POS];
    if high > low {
        edge = (edge - low) / (high - low);
    }

    let rounded = f32::round((1.0 - edge) * 255.0);
    output[ABSOLUTE_POS] = u8::cast_from(rounded);
}

fn no_frame() -> StageError {
    StageError::msg("lines/xdog expected a frame in slot 0")
}

struct XdogMask<R: Runtime> {
    client: ComputeClient<R>,
    luma: Luma,
    near: Gaussian,
    far: Gaussian,
    tau: f32,
    eps: Option<f32>,
    phi: Option<f32>,
    eps_percentile: f32,
}

impl<R: Runtime> XdogMask<R> {
    /// Finds `eps` and `phi` for this frame's response, leaving them on the device.
    fn parameters(
        &self,
        context: &FrameContext,
        sizing: &Sizing<'_>,
        difference: DevicePlane<'_>,
    ) -> Scratch {
        let samples = difference.samples();
        let found_eps = Scratch::line::<f32>(context, 1);
        let totals = Scratch::line::<f32>(context, 2);
        let reduction = Reduction::new(context, samples);

        if self.eps.is_none() {
            let range = Scratch::line::<f32>(context, 2);
            let bins = Scratch::line::<u32>(context, BINS as usize);

            let range_planes = reduction.planes(difference, &range);
            extremes(&self.client, sizing, range_planes);

            let percentile_planes = PercentilePlanes {
                input: difference,
                range: range.plane(),
                bins: bins.plane(),
                result: found_eps.plane(),
            };
            percentile(&self.client, sizing, percentile_planes, self.eps_percentile);
        }

        if self.phi.is_none() {
            let totals_planes = reduction.planes(difference, &totals);
            sums(&self.client, sizing, totals_planes);
        }

        let parameters = Scratch::line::<f32>(context, 2);
        let found = found_eps.plane().array::<R>();
        let summed = totals.plane().array::<R>();
        let settled = parameters.plane().array::<R>();
        let fixed_eps = self.eps.unwrap_or(0.0);
        let fixed_phi = self.phi.unwrap_or(0.0);
        let eps_is_unset = self.eps.is_none();
        let automatic_eps = u32::from(eps_is_unset);
        let phi_is_unset = self.phi.is_none();
        let automatic_phi = u32::from(phi_is_unset);
        let one_cube = CubeCount::Static(1, 1, 1);
        let one_unit = CubeDim::new_1d(1);

        // Safety: each buffer holds the one or two f32 values the kernel reads or writes.
        unsafe {
            resolve_parameters::launch_unchecked::<R>(
                &self.client,
                one_cube,
                one_unit,
                found,
                summed,
                settled,
                fixed_eps,
                fixed_phi,
                automatic_eps,
                automatic_phi,
                samples as f32,
            );
        }

        parameters
    }
}

impl<R: Runtime> StageInstance for XdogMask<R> {}

impl<R: Runtime> Mask for XdogMask<R> {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(no_frame)?;

        let context = &*context;
        let sizing = Sizing(context);
        let width = self.luma.width;
        let height = self.luma.height;
        let samples = self.luma.samples();

        let luma = Scratch::new::<f32>(context, width, height);
        let luma_plane = luma.plane();
        load_luma(&self.client, &sizing, &frame, &self.luma, luma_plane)?;

        let temporary = Scratch::new::<f32>(context, width, height);
        let near = Scratch::new::<f32>(context, width, height);
        let far = Scratch::new::<f32>(context, width, height);

        let near_planes = BlurPlanes {
            input: luma_plane,
            scratch: temporary.plane(),
            output: near.plane(),
        };
        self.near.apply(&self.client, &sizing, near_planes);

        let far_planes = BlurPlanes {
            output: far.plane(),
            ..near_planes
        };
        self.far.apply(&self.client, &sizing, far_planes);

        let difference = Scratch::new::<f32>(context, width, height);
        let near_read = near.plane().array::<R>();
        let far_read = far.plane().array::<R>();
        let difference_write = difference.plane().array::<R>();
        let cubes = sizing.0.cube_count(samples);
        let shape = sizing.0.cube_dim();

        // Safety: every plane holds `samples` f32 values and the kernel writes nothing past them.
        unsafe {
            difference_of_gaussians::launch_unchecked::<R>(
                &self.client,
                cubes,
                shape,
                near_read,
                far_read,
                difference_write,
                self.tau,
            );
        }

        let difference_plane = difference.plane();
        let parameters = self.parameters(context, &sizing, difference_plane);

        let edges = Scratch::new::<f32>(context, width, height);
        let difference_read = difference.plane().array::<R>();
        let parameters_read = parameters.plane().array::<R>();
        let edges_write = edges.plane().array::<R>();
        let cubes = sizing.0.cube_count(samples);
        let shape = sizing.0.cube_dim();

        // Safety: the difference and edges hold `samples` f32 values, the parameters two, and the
        // kernel writes nothing past the edges' length.
        unsafe {
            soft_threshold::launch_unchecked::<R>(
                &self.client,
                cubes,
                shape,
                difference_read,
                parameters_read,
                edges_write,
            );
        }

        let range = Scratch::line::<f32>(context, 2);
        let reduction = Reduction::new(context, samples);
        let edges_plane = edges.plane();
        let range_planes = reduction.planes(edges_plane, &range);
        extremes(&self.client, &sizing, range_planes);

        let meta = frame.meta().clone();
        let format = self.luma.mask_format();
        let output = context.alloc_mask(&format, meta);
        let edges_read = edges.plane().array::<R>();
        let range_read = range.plane().array::<R>();
        let mask_write = mask_plane(&output).array::<R>();
        let cubes = sizing.0.cube_count(samples);
        let shape = sizing.0.cube_dim();

        // Safety: the edges hold `samples` f32 values, the range two, and the mask `samples` bytes.
        unsafe {
            stretch_into_mask::launch_unchecked::<R>(
                &self.client,
                cubes,
                shape,
                edges_read,
                range_read,
                mask_write,
            );
        }

        Ok(output)
    }
}

/// Finds line art with an extended difference of gaussians.
pub(crate) struct Xdog;

impl StageDefinition for Xdog {
    type Args = XdogArgs;

    fn package() -> &'static str {
        "lines"
    }

    fn identifier() -> &'static str {
        "xdog"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MaskDefinition for Xdog {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_luma(NAME, offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        args.validate()?;

        produces_luma_mask(NAME, chosen)
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        args.validate()?;

        let luma = Luma::from_formats(NAME, formats)?;
        let client = context.client().clone();
        let near = Gaussian::new(&client, args.sigma);
        let far_sigma = args.sigma * args.k;
        let far = Gaussian::new(&client, far_sigma);

        Ok(Box::new(XdogMask {
            client,
            luma,
            near,
            far,
            tau: args.tau,
            eps: args.eps,
            phi: args.phi,
            eps_percentile: args.eps_percentile,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn args(value: serde_json::Value) -> XdogArgs {
        serde_json::from_value(value).expect("the args parse")
    }

    #[test]
    fn the_defaults_are_sigma_1_k_1_6_tau_0_95() {
        let defaults = args(json!({}));

        assert_eq!(defaults.sigma, 1.0);
        assert_eq!(defaults.k, 1.6);
        assert_eq!(defaults.tau, 0.95);
        assert_eq!(defaults.eps, None);
        assert_eq!(defaults.phi, None);
        assert_eq!(defaults.eps_percentile, 8.0);
        assert!(defaults.validate().is_ok());
    }

    #[test]
    fn a_zero_sigma_is_refused() {
        let error = args(json!({ "sigma": 0.0 }))
            .validate()
            .expect_err("sigma must be positive");

        assert!(error.to_string().contains("lines/xdog needs sigma above 0"));
    }

    #[test]
    fn a_k_of_one_is_refused() {
        assert!(args(json!({ "k": 1.0 })).validate().is_err());
    }

    #[test]
    fn a_zero_phi_is_refused() {
        assert!(args(json!({ "phi": 0.0 })).validate().is_err());
    }

    fn refusal(args: XdogArgs) -> String {
        let error = args.validate().expect_err("the args should be refused");

        error.to_string()
    }

    #[test]
    fn a_nan_sigma_is_refused() {
        let defaults = args(json!({}));
        let message = refusal(XdogArgs {
            sigma: f32::NAN,
            ..defaults
        });

        assert!(
            message.contains("lines/xdog needs a finite sigma, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn an_infinite_sigma_is_refused() {
        let defaults = args(json!({}));
        let message = refusal(XdogArgs {
            sigma: f32::INFINITY,
            ..defaults
        });

        assert!(
            message.contains("lines/xdog needs a finite sigma, got inf"),
            "{message}"
        );
    }

    #[test]
    fn a_sigma_too_wide_to_blur_is_refused() {
        let wide = args(json!({ "sigma": 1.0e9 }));
        let message = refusal(wide);

        assert!(
            message.contains("lines/xdog needs sigma of at most 1024"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_k_is_refused() {
        let defaults = args(json!({}));
        let message = refusal(XdogArgs {
            k: f32::NAN,
            ..defaults
        });

        assert!(
            message.contains("lines/xdog needs a finite k, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn an_infinite_k_is_refused() {
        let defaults = args(json!({}));
        let message = refusal(XdogArgs {
            k: f32::INFINITY,
            ..defaults
        });

        assert!(
            message.contains("lines/xdog needs a finite k, got inf"),
            "{message}"
        );
    }

    #[test]
    fn a_k_widening_the_blur_too_far_is_refused() {
        let widened = args(json!({ "sigma": 1000.0, "k": 1.6 }));
        let message = refusal(widened);

        assert!(
            message.contains("lines/xdog needs sigma * k of at most 1024"),
            "{message}"
        );
    }

    #[test]
    fn a_nan_phi_is_refused() {
        let defaults = args(json!({}));
        let message = refusal(XdogArgs {
            phi: Some(f32::NAN),
            ..defaults
        });

        assert!(
            message.contains("lines/xdog needs a finite phi, got NaN"),
            "{message}"
        );
    }

    #[test]
    fn an_infinite_phi_is_refused() {
        let defaults = args(json!({}));
        let message = refusal(XdogArgs {
            phi: Some(f32::INFINITY),
            ..defaults
        });

        assert!(
            message.contains("lines/xdog needs a finite phi, got inf"),
            "{message}"
        );
    }

    #[test]
    fn a_percentile_past_100_is_refused() {
        assert!(args(json!({ "eps_percentile": 101.0 })).validate().is_err());
    }
}
