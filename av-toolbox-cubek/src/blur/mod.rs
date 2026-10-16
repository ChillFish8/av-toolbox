mod plain;
#[cfg(test)]
mod tests;
mod tiled;

use cubecl::prelude::{ArrayArg, ComputeClient, CubeElement, Runtime};
use cubecl::server::Handle;

use crate::{DevicePlane, Dispatch, assert_plane_sized};

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

/// The widest radius the tiled tier takes.
///
/// It is the widest whose tile and border still fit in 32 KiB of shared memory.
pub const TILED_MAX_RADIUS: u32 = 16;

/// The kernel family a blur runs on.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum Tier {
    Plain,
    Tiled,
}

/// The tier that blurs fastest for a gaussian of `sigma` reaching `radius`.
pub(crate) fn tier_for(_sigma: f32, radius: u32) -> Tier {
    if radius <= TILED_MAX_RADIUS {
        return Tier::Tiled;
    }

    Tier::Plain
}

/// One input plane and the output plane a single filter pass writes.
#[derive(Debug, Clone, Copy)]
pub struct FilterPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The planes one blur reads, passes through and writes. All are of one size and one [Storage].
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
    radius: u32,
    tier: Tier,
}

impl Gaussian {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, sigma: f32) -> Self {
        let radius = radius_for(sigma);

        Self::with_radius(client, sigma, radius)
    }

    pub fn with_radius<R: Runtime>(client: &ComputeClient<R>, sigma: f32, radius: u32) -> Self {
        let weights = gaussian_weights(sigma, radius);
        let bytes = f32::as_bytes(&weights);
        let weights = client.create_from_slice(bytes);
        let tier = tier_for(sigma, radius);

        Self {
            weights,
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

        // Safety: the weights buffer was made from exactly `taps` f32 values.
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
                tiled::blur_tiled(client, weights, planes, self.radius, storage);
            },
        }
    }

    /// The tier a `width` by `height` plane runs on.
    ///
    /// It is the plain tier when the plane needs more tiles than a device allows cubes.
    fn launch_tier(&self, width: u32, height: u32) -> Tier {
        let tier = self.tier();
        let tiles = tiled::tile_cubes(width, height);
        if tier == Tier::Tiled && tiles.is_none() {
            return Tier::Plain;
        }

        tier
    }
}
