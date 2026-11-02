use cubecl::prelude::{BufferArg, Client, CubeElement};
use cubecl::server::Handle;

use super::{FilterPlanes, Storage, plain, tiled};
use crate::plane::are_distinct;
use crate::{Dispatch, Edge, assert_plane_sized};

/// The widest radius a [Separable](crate::blur::Separable) takes.
pub const SEPARABLE_MAX_RADIUS: u32 = 3;

/// The radius and sum of `weights`, after checking they suit a [Separable](crate::blur::Separable).
///
/// It panics if the count is even, the radius is above [SEPARABLE_MAX_RADIUS], the weights are not
/// symmetric or they sum to zero.
pub(super) fn checked_radius(weights: &[f32]) -> (u32, f32) {
    assert!(
        weights.len() % 2 == 1,
        "a separable blur takes an odd count of weights, not {}",
        weights.len(),
    );

    let radius = (weights.len() / 2) as u32;
    assert!(
        radius <= SEPARABLE_MAX_RADIUS,
        "a separable blur reaches at most {SEPARABLE_MAX_RADIUS} samples, not {radius}",
    );

    let mirrored = weights.iter().rev();
    let symmetric = weights
        .iter()
        .zip(mirrored)
        .all(|(weight, opposite)| weight == opposite);
    assert!(
        symmetric,
        "a separable blur takes symmetric weights, not {weights:?}"
    );

    let total: f32 = weights.iter().sum();
    assert!(
        total != 0.0,
        "a separable blur takes weights that do not sum to zero"
    );

    (radius, total)
}

/// A blur with the same caller-given weights across and down, already on the device.
///
/// The weights are normalised to sum to 1.
#[derive(Debug)]
pub struct Separable {
    weights: Handle,
    radius: u32,
}

impl Separable {
    /// Uploads `weights`, which hold `2r + 1` symmetric values for a radius `r` of at most
    /// [SEPARABLE_MAX_RADIUS].
    pub fn new(client: &Client, weights: &[f32]) -> Self {
        let (radius, total) = checked_radius(weights);
        let normalised: Vec<f32> = weights.iter().map(|weight| weight / total).collect();

        let bytes = f32::as_bytes(&normalised);
        let weights = client.create_from_slice(bytes);

        Self { weights, radius }
    }

    fn weights(&self) -> BufferArg {
        let taps = 2 * self.radius as usize + 1;
        let handle = self.weights.clone();

        // Safety: the weights buffer was made from exactly `taps` f32 values.
        unsafe { BufferArg::from_raw_parts(handle, taps) }
    }

    /// Blurs `planes.input` into `planes.output`, reading past the edge as `edge` says.
    ///
    /// Both planes hold samples of `storage` and must be distinct buffers. The sums stay f32 either way.
    pub fn apply_as(
        &self,
        client: &Client,
        dispatch: &impl Dispatch,
        planes: FilterPlanes<'_>,
        edge: Edge,
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
        assert_plane_sized(&planes.output, width, height, bytes, "blur output");

        let weights = self.weights();
        let fits_tiles = tiled::tile_cubes(width, height).is_some();
        if fits_tiles {
            tiled::blur_tiled(client, weights, planes, self.radius, edge, storage);
        } else {
            plain::apply_in_one_pass(client, dispatch, weights, self.radius, planes, edge, storage);
        }
    }
}
