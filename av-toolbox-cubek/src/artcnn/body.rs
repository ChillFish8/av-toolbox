// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::Dispatch;
use crate::artcnn::{Network, Precision};

/// What the epilogue applies after adding the bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None,
    Silu,
}

#[cube(launch_unchecked)]
fn epilogue_kernel<F: Float>(
    values: &mut Array<F>,
    bias: &Array<F>,
    residual: &Array<F>,
    features: u32,
    silu: u32,
    add_residual: u32,
) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    let channel = ABSOLUTE_POS as u32 % features;
    let mut value = f32::cast_from(values[ABSOLUTE_POS]) + f32::cast_from(bias[channel as usize]);

    if silu == 1 {
        value = value / (1.0 + f32::exp(-value));
    }

    if add_residual == 1 {
        value += f32::cast_from(residual[ABSOLUTE_POS]);
    }

    values[ABSOLUTE_POS] = F::cast_from(value);
}

/// Adds `layer`'s bias to the NHWC maps in `values`, then applies `activation`, then adds `residual` if given.
///
/// The maps hold `samples` samples of the network's features, and are updated in place.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn epilogue<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    values: &Handle,
    layer: usize,
    samples: u32,
    activation: Activation,
    residual: Option<&Handle>,
) {
    let buffers = network.layer(layer);
    let features = network.features();
    let units = (samples * features) as usize;
    let bytes = (units * network.precision().bytes()) as u64;
    let residual_handle = residual.unwrap_or(values);
    assert!(
        values.size_in_used() >= bytes,
        "the values are smaller than the samples' maps"
    );
    assert!(
        residual_handle.size_in_used() >= bytes,
        "the residual is smaller than the values"
    );

    let silu = u32::from(activation == Activation::Silu);
    let add_residual = u32::from(residual.is_some());
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the values and the residual hold `samples * F` values and the bias `F`. Without a residual, the values
    // are bound in its place and never read through it.
    unsafe {
        let values_array = ArrayArg::from_raw_parts(values.clone(), units);
        let bias = ArrayArg::from_raw_parts(buffers.bias.clone(), features as usize);
        let residual_array = ArrayArg::from_raw_parts(residual_handle.clone(), units);

        match network.precision() {
            Precision::F16 => epilogue_kernel::launch_unchecked::<half::f16, R>(
                client,
                cubes,
                unit,
                values_array,
                bias,
                residual_array,
                features,
                silu,
                add_residual,
            ),
            Precision::F32 => epilogue_kernel::launch_unchecked::<f32, R>(
                client,
                cubes,
                unit,
                values_array,
                bias,
                residual_array,
                features,
                silu,
                add_residual,
            ),
        }
    }
}
