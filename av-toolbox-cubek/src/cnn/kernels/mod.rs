mod add;
mod direct;
mod strided;
mod tune;
mod winograd;

#[cfg(test)]
mod tests;

use cubecl::ir::DeviceProperties;
use cubecl::prelude::{ComputeClient, Runtime};
use cubecl::server::Handle;

use self::add::add_maps;
pub use self::direct::{DirectVariant, WeightStaging};
use self::direct::{WRITE_WIDTH, check_variant, run_direct};
#[cfg(any(test, feature = "test-support"))]
pub use self::direct::{direct_variants, offered_plane_width};
pub use self::strided::{Stride, StridedLayer, StridedVariant, run_strided, strided_variant};
pub(crate) use self::tune::takes_tuner;
pub use self::tune::{Epilogue, Kernels, offered_candidates, run_tuned};
#[cfg(any(test, feature = "test-support"))]
pub use self::tune::{TunedRun, fixed_candidate};
pub use self::winograd::WinogradVariant;
pub(crate) use self::winograd::transform_values;
use self::winograd::{POSITIONS, run_winograd};
#[cfg(any(test, feature = "test-support"))]
pub use self::winograd::{transform_input, winograd_variants};
use crate::cnn::convolution::{Operands, launch_fused_convolution};
use crate::cnn::{Activation, CnnError, ConvLayer, Finish, Precision, Region};

/// One body convolution of f16 NHWC maps, with its bias and what it applies as it writes.
///
/// The weights are laid out by output channel, row, column and input channel, and the bias, if any, is in
/// f16. The output is the convolution plus the bias, through `activation`, plus `residual` and then `skip`
/// if given.
///
/// `transformed` holds the layer's Winograd weights, by position, output channel and input channel.
/// `winograd` is whether the layer is offered the Winograd candidates.
#[derive(Clone, Copy)]
pub struct BodyConvolution<'a> {
    pub input: &'a Handle,
    pub weights: &'a Handle,
    pub transformed: Option<&'a Handle>,
    pub winograd: bool,
    pub bias: Option<&'a Handle>,
    pub residual: Option<&'a Handle>,
    pub skip: Option<&'a Handle>,
    pub activation: Activation,
    pub output: &'a Handle,
    pub width: u32,
    pub height: u32,
    pub features: u32,
}

impl<'a> BodyConvolution<'a> {
    /// `layer` over the maps of `region`, applying `finish`.
    ///
    /// Panics if the layer's maps are not f16.
    pub fn of_conv(
        layer: &'a ConvLayer,
        input: &'a Handle,
        output: &'a Handle,
        region: Region,
        finish: Finish<'a>,
    ) -> Self {
        assert!(
            layer.precision == Precision::F16,
            "the body convolutions run on f16 maps"
        );

        Self {
            input,
            weights: &layer.weights,
            transformed: layer.transformed.as_ref(),
            winograd: layer.winograd,
            bias: layer.bias.as_ref(),
            residual: finish.residual,
            skip: finish.skip,
            activation: finish.activation,
            output,
            width: region.width,
            height: region.height,
            features: layer.features,
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
            skip: self.skip,
        }
    }

    /// Asserts that every buffer holds what the convolution reads or writes through it, and that the output
    /// is not the input, the residual or the skip.
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

        if let Some(bias) = self.bias {
            assert!(
                bias.size_in_used() >= bias_bytes,
                "the bias is smaller than the features"
            );
        }

        assert!(!same_buffer(self.input, self.output), "the output is the input");

        if let Some(transformed) = self.transformed {
            assert!(
                transformed.size_in_used() >= u64::from(POSITIONS) * features * features * half,
                "the transformed weights are smaller than the layer's"
            );
        }

        if let Some(residual) = self.residual {
            assert!(
                residual.size_in_used() >= map_bytes,
                "the residual is smaller than the maps"
            );
            assert!(!same_buffer(residual, self.output), "the output is the residual");
        }

        if let Some(skip) = self.skip {
            assert!(
                skip.size_in_used() >= map_bytes,
                "the skip is smaller than the maps"
            );
            assert!(!same_buffer(skip, self.output), "the output is the skip");
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
    /// Winograd F(2x2, 3x3) on cooperative matrices, for layers offered Winograd.
    Winograd(WinogradVariant),
    /// `cubek-convolution` with the epilogue applied as it writes.
    FusedCubek,
}

impl Candidate {
    /// A short name that tells every candidate apart.
    pub fn name(&self) -> String {
        let variant = match self {
            Candidate::Direct(variant) => variant,
            Candidate::Winograd(variant) => return winograd_name(variant),
            Candidate::FusedCubek => return "fused-cubek".to_string(),
        };

        let staging = match variant.weights {
            WeightStaging::Global => "global",
            WeightStaging::Single => "single",
            WeightStaging::Double => "double",
        };

        format!(
            "direct-{}x{}-groups{}-planes{}x{}-{staging}-width{}",
            variant.rows,
            variant.columns,
            variant.channel_groups,
            variant.planes_m,
            variant.planes_n,
            variant.plane_width,
        )
    }

    /// Whether a device with `properties` can run this candidate on maps of `features` channels, as far as
    /// is known before it launches.
    pub(crate) fn check(&self, properties: &DeviceProperties, features: u32) -> Result<(), String> {
        match self {
            Candidate::Direct(variant) => check_variant(properties, variant, features),
            Candidate::Winograd(variant) => winograd::check_variant(properties, variant, features),
            Candidate::FusedCubek => Ok(()),
        }
    }
}

fn winograd_name(variant: &WinogradVariant) -> String {
    format!(
        "winograd-{}x{}-planes{}x{}-width{}",
        variant.tile_rows, variant.tile_columns, variant.planes_m, variant.planes_n, variant.plane_width,
    )
}

/// Runs `convolution` with `candidate`.
///
/// Gives an error if the candidate cannot run on the device or with this convolution's shape.
#[cfg(any(test, feature = "test-support"))]
pub fn run_candidate<R: Runtime>(
    client: &ComputeClient<R>,
    candidate: Candidate,
    convolution: &BodyConvolution<'_>,
) -> Result<(), CnnError> {
    launch_candidate(client, candidate, convolution)?;

    Ok(())
}

/// Runs `convolution` with `candidate`, giving whether the forced blueprint ran.
///
/// Gives an error if the candidate cannot run on the device or with this convolution's shape.
pub(crate) fn launch_candidate<R: Runtime>(
    client: &ComputeClient<R>,
    candidate: Candidate,
    convolution: &BodyConvolution<'_>,
) -> Result<bool, CnnError> {
    convolution.check();

    match candidate {
        Candidate::Direct(variant) => {
            run_direct(client, variant, convolution)?;

            Ok(false)
        },
        Candidate::Winograd(variant) => {
            run_winograd(client, variant, convolution)?;

            Ok(false)
        },
        Candidate::FusedCubek => run_fused_cubek(client, convolution),
    }
}

fn run_fused_cubek<R: Runtime>(
    client: &ComputeClient<R>,
    convolution: &BodyConvolution<'_>,
) -> Result<bool, CnnError> {
    let features = convolution.features as usize;
    let operands = Operands::body(convolution.weights, convolution.bias, features, Precision::F16);
    let region = convolution.region();
    let finish = Finish {
        skip: None,
        ..convolution.finish()
    };
    let input = convolution.input;
    let output = convolution.output;

    let forced = launch_fused_convolution(client, &operands, input, output, region, finish, true)?;

    if let Some(skip) = convolution.skip {
        let quads = region.samples() as usize * features / WRITE_WIDTH as usize;
        add_maps(client, output, skip, quads);
    }

    Ok(forced)
}
