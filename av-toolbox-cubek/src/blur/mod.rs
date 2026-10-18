mod fixed;
mod lines;
mod plain;
mod separable;
mod staged;
#[cfg(test)]
mod tests;
mod tiled;
mod wide;

use cubecl::prelude::{ArrayArg, ComputeClient, CubeElement, Runtime};
use cubecl::server::Handle;

pub(crate) use self::fixed::{OUTPUTS_DOWN, TILE_HEIGHT, TILE_WIDTH, UNITS_ACROSS, UNITS_DOWN, fixed_cubes};
pub use self::fixed::{Smooth, smooth, vertical_121};
pub use self::separable::{SEPARABLE_MAX_RADIUS, Separable};
use crate::plane::are_distinct;
use crate::{DevicePlane, Dispatch, Edge, assert_plane_sized};

/// The widest radius a blur reaches, in samples either side of its centre.
pub const MAX_RADIUS: u32 = 4096;

/// The widest deviation a blur takes, whose radius is exactly [MAX_RADIUS].
pub const MAX_SIGMA: f32 = MAX_RADIUS as f32 / 4.0;

/// The radius a blur of `sigma` reaches, four standard deviations out.
///
/// `sigma` has to be finite and at most [MAX_SIGMA], so the radius fits
/// the kernels' arithmetic.
pub fn radius_for(sigma: f32) -> u32 {
    let reach = (4.0 * sigma).ceil() as u32;

    reach.max(1)
}

/// Gaussian weights from `-radius` to `radius`, normalised to sum to 1.
pub fn gaussian_weights(sigma: f32, radius: u32) -> Vec<f32> {
    let radius = radius as i32;
    let spread = 2.0 * sigma * sigma;

    let raw: Vec<f32> = (-radius..=radius)
        .map(|offset| {
            let distance = offset as f32;
            (-(distance * distance) / spread).exp()
        })
        .collect();
    let total: f32 = raw.iter().sum();

    raw.iter().map(|weight| weight / total).collect()
}

/// How a sample is stored on the device.
///
/// Weights and arithmetic stay f32 either way.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Storage {
    F32,
    F16,
}

impl Storage {
    /// The bytes one sample takes.
    pub const fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// The most cubes every device allows along one axis.
pub(super) const MAX_CUBES_PER_AXIS: u32 = 65535;

/// The shared memory budget, per cube, that the blur kernels are built for.
pub(super) const MAX_SHARED_BYTES: u32 = 32 * 1024;

/// The widest radius the tiled tier takes.
///
/// It is the widest whose tile and border still fit in 32 KiB of shared memory.
pub const TILED_MAX_RADIUS: u32 = 16;

/// The widest radius the staged tier takes.
pub const STAGED_MAX_RADIUS: u32 = MAX_RADIUS;

/// The narrowest radius the wide tier takes.
///
/// Narrower blurs stay exact on the tiled and staged tiers.
pub const WIDE_MIN_RADIUS: u32 = 49;

/// The kernel family a blur runs on.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum Tier {
    Plain,
    Tiled,
    Staged,
    Wide,
}

/// The tier a gaussian of `sigma` reaching `radius` runs on.
///
/// Narrow radii take the tiled tier and wide ones the wide tier, which approximates a whole gaussian
/// and so only takes kernels that reach their full four deviations. The rest take the staged tier.
pub(crate) fn tier_for(sigma: f32, radius: u32) -> Tier {
    if radius <= TILED_MAX_RADIUS {
        return Tier::Tiled;
    }

    let full_reach = radius >= radius_for(sigma);
    if radius >= WIDE_MIN_RADIUS && full_reach {
        return Tier::Wide;
    }

    if radius <= STAGED_MAX_RADIUS {
        return Tier::Staged;
    }

    Tier::Plain
}

/// One input plane and the output plane a single filter pass writes.
#[derive(Debug, Clone, Copy)]
pub struct FilterPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The planes one blur reads, passes through and writes.
///
/// All are of one size and one [Storage]. The input and output must be distinct buffers.
#[derive(Debug, Clone, Copy)]
pub struct BlurPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub scratch: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// A separable gaussian blur with its weights already on the device.
#[derive(Debug)]
pub struct Gaussian {
    weights: Handle,
    sigma: f32,
    radius: u32,
    tier: Tier,
}

impl Gaussian {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, sigma: f32) -> Self {
        let radius = radius_for(sigma);

        Self::with_radius(client, sigma, radius)
    }

    pub fn with_radius<R: Runtime>(client: &ComputeClient<R>, sigma: f32, radius: u32) -> Self {
        let mut weights = gaussian_weights(sigma, radius);
        let padded_taps = staged::padded_tap_count(radius);
        weights.resize(padded_taps as usize, 0.0);

        let bytes = f32::as_bytes(&weights);
        let weights = client.create_from_slice(bytes);
        let tier = tier_for(sigma, radius);

        Self {
            weights,
            sigma,
            radius,
            tier,
        }
    }

    pub const fn radius(&self) -> u32 {
        self.radius
    }

    pub(crate) const fn tier(&self) -> Tier {
        self.tier
    }

    fn weights<R: Runtime>(&self) -> ArrayArg<R> {
        let taps = 2 * self.radius as usize + 1;
        let handle = self.weights.clone();

        // Safety: the weights buffer was made from at least `taps` f32 values.
        unsafe { ArrayArg::from_raw_parts(handle, taps) }
    }

    /// The weights, followed by the zeros that pad them to the staged tier's tap count.
    fn padded_weights<R: Runtime>(&self) -> ArrayArg<R> {
        let taps = staged::padded_tap_count(self.radius) as usize;
        let handle = self.weights.clone();

        // Safety: the weights buffer was padded to exactly `taps` f32 values.
        unsafe { ArrayArg::from_raw_parts(handle, taps) }
    }

    pub fn apply<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: BlurPlanes<'_>,
    ) {
        self.apply_as(client, dispatch, planes, Storage::F32);
    }

    /// Blurs `planes.input` into `planes.output`, with every plane holding samples of `storage`.
    pub fn apply_as<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: BlurPlanes<'_>,
        storage: Storage,
    ) {
        let handles = [planes.input.handle(), planes.output.handle()];
        assert!(
            are_distinct(&handles),
            "the blur input and output must be distinct buffers"
        );

        let width = planes.output.width();
        let height = planes.output.height();
        let bytes = storage.bytes();
        assert_plane_sized(&planes.input, width, height, bytes, "blur input");
        assert_plane_sized(&planes.scratch, width, height, bytes, "blur scratch");
        assert_plane_sized(&planes.output, width, height, bytes, "blur output");

        let tier = self.launch_tier(width, height);
        match tier {
            Tier::Plain => plain::apply(client, dispatch, self, planes, storage),
            Tier::Tiled => {
                let weights = self.weights::<R>();
                let filter_planes = FilterPlanes {
                    input: planes.input,
                    output: planes.output,
                };
                tiled::blur_tiled(
                    client,
                    weights,
                    filter_planes,
                    self.radius,
                    Edge::Reflect,
                    storage,
                );
            },
            Tier::Staged => staged::blur_staged(client, self, planes, storage),
            Tier::Wide => wide::blur_wide(client, planes, self.sigma, storage),
        }
    }

    /// The tier a `width` by `height` plane runs on.
    ///
    /// It is the plain tier when the plane needs more cubes than a device allows.
    fn launch_tier(&self, width: u32, height: u32) -> Tier {
        let tier = self.tier();
        let fits = match tier {
            Tier::Plain => true,
            Tier::Tiled => tiled::tile_cubes(width, height).is_some(),
            Tier::Staged => staged::staged_cubes(width, height).is_some(),
            Tier::Wide => wide::wide_cubes(width, height).is_some(),
        };
        if !fits {
            return Tier::Plain;
        }

        tier
    }
}
