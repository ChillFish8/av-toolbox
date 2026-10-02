mod direct;

#[cfg(test)]
mod tests;

use cubecl::prelude::{ComputeClient, Runtime};
use cubecl::server::Handle;

use self::direct::run_direct;
pub use self::direct::{DirectVariant, WeightStaging};
use super::body::{Finish, Operands, body_operands, launch_fused_convolution};
use crate::artcnn::{Activation, ArtcnnError, Network, Precision, Region};

/// One body convolution of f16 NHWC maps, with its bias and what it applies as it writes.
///
/// The weights are laid out by output channel, row, column and input channel, and the bias is in f16. The
/// output is the convolution plus the bias, through `activation`, plus `residual` if given.
#[derive(Clone, Copy)]
pub struct BodyConvolution<'a> {
    pub input: &'a Handle,
    pub weights: &'a Handle,
    pub bias: &'a Handle,
    pub residual: Option<&'a Handle>,
    pub activation: Activation,
    pub output: &'a Handle,
    pub width: u32,
    pub height: u32,
    pub features: u32,
}

impl<'a> BodyConvolution<'a> {
    /// Body layer `layer` of `network` over the maps of `region`, applying `finish`.
    ///
    /// Panics if the network's maps are not f16 or the layer does not map the features to themselves.
    pub fn of_layer(
        network: &'a Network,
        layer: usize,
        input: &'a Handle,
        output: &'a Handle,
        region: Region,
        finish: Finish<'a>,
    ) -> Self {
        assert!(
            network.precision() == Precision::F16,
            "the body convolutions run on f16 maps"
        );

        let operands = body_operands(network, layer, true);
        let bias = operands.bias.expect("the operands carry the bias");

        Self {
            input,
            weights: operands.weights,
            bias,
            residual: finish.residual,
            activation: finish.activation,
            output,
            width: region.width,
            height: region.height,
            features: network.features(),
        }
    }

    fn region(&self) -> Region {
        Region {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        }
    }

    fn finish(&self) -> Finish<'a> {
        Finish {
            activation: self.activation,
            residual: self.residual,
        }
    }

    /// Asserts that every buffer holds what the convolution reads or writes through it, and that the output
    /// is neither the input nor the residual.
    fn check(&self) {
        let half = size_of::<half::f16>() as u64;
        let features = u64::from(self.features);
        let map_bytes = u64::from(self.width) * u64::from(self.height) * features * half;
        let weight_bytes = features * 9 * features * half;
        let bias_bytes = features * half;

        assert!(
            self.features.is_multiple_of(16),
            "the features are not a multiple of 16"
        );
        assert!(
            self.input.size_in_used() >= map_bytes,
            "the input is smaller than the maps"
        );
        assert!(
            self.output.size_in_used() >= map_bytes,
            "the output is smaller than the maps"
        );
        assert!(
            self.weights.size_in_used() >= weight_bytes,
            "the weights are smaller than the layer's"
        );
        assert!(
            self.bias.size_in_used() >= bias_bytes,
            "the bias is smaller than the features"
        );
        assert!(!same_buffer(self.input, self.output), "the output is the input");

        if let Some(residual) = self.residual {
            assert!(
                residual.size_in_used() >= map_bytes,
                "the residual is smaller than the maps"
            );
            assert!(!same_buffer(residual, self.output), "the output is the residual");
        }
    }
}

/// Whether `first` and `second` are the same handle. Two handles to one buffer are not caught.
fn same_buffer(first: &Handle, second: &Handle) -> bool {
    std::ptr::eq(first, second)
}

/// A kernel that can run a body convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Candidate {
    /// The direct implicit-GEMM convolution on cooperative matrices.
    Direct(DirectVariant),
    /// `cubek-convolution` with the epilogue applied as it writes.
    FusedCubek,
}

/// Runs `convolution` with `candidate`.
///
/// Gives an error if the candidate cannot run on the device or with this convolution's shape.
pub fn run_candidate<R: Runtime>(
    client: &ComputeClient<R>,
    candidate: Candidate,
    convolution: &BodyConvolution<'_>,
) -> Result<(), ArtcnnError> {
    convolution.check();

    match candidate {
        Candidate::Direct(variant) => run_direct(client, variant, convolution),
        Candidate::FusedCubek => run_fused_cubek(client, convolution),
    }
}

fn run_fused_cubek<R: Runtime>(
    client: &ComputeClient<R>,
    convolution: &BodyConvolution<'_>,
) -> Result<(), ArtcnnError> {
    let features = convolution.features as usize;
    let operands = Operands::body(convolution.weights, convolution.bias, features, Precision::F16);
    let region = convolution.region();
    let finish = convolution.finish();
    let input = convolution.input;
    let output = convolution.output;

    launch_fused_convolution(client, &operands, input, output, region, finish, true)?;

    Ok(())
}
