use cubecl::prelude::*;

use super::{Parity, bob_taps, fmtc_gauss_taps};
use crate::resize::{Weights, upload};

/// The bob taps of one plane height and parity, on the device.
#[derive(Debug)]
pub struct BobWeights {
    weights: Weights,
    parity: Parity,
    field_height: u32,
}

impl BobWeights {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, plane_height: u32, parity: Parity) -> Self {
        assert!(
            plane_height.is_multiple_of(2) && plane_height >= 2,
            "the plane is {plane_height} rows, expected an even count of at least 2"
        );

        let field_height = plane_height / 2;
        let taps = bob_taps(field_height, parity);

        Self {
            weights: upload(client, &taps),
            parity,
            field_height,
        }
    }

    pub const fn parity(&self) -> Parity {
        self.parity
    }

    /// How many rows the field has.
    pub const fn field_height(&self) -> u32 {
        self.field_height
    }

    pub(crate) const fn taps(&self) -> &Weights {
        &self.weights
    }
}

/// The gaussian tables of one luma plane for each sharpness QTGMC blurs with.
#[derive(Debug)]
pub(crate) struct GaussTables {
    pub(crate) across: Weights,
    pub(crate) down: Weights,
}

impl GaussTables {
    fn new<R: Runtime>(client: &ComputeClient<R>, size: (u32, u32), sharpness: f32) -> Self {
        let across = fmtc_gauss_taps(size.0, sharpness);
        let down = fmtc_gauss_taps(size.1, sharpness);

        Self {
            across: upload(client, &across),
            down: upload(client, &down),
        }
    }
}

/// Every table one format needs, built once.
#[derive(Debug)]
pub struct QtgmcTables {
    bobs: Vec<[BobWeights; 2]>,
    pub(crate) narrow: GaussTables,
    pub(crate) wide: GaussTables,
}

impl QtgmcTables {
    /// Builds the tables for planes of `planes` sizes, as `(width, height)`.
    ///
    /// The first plane is luma, which the gaussians blur.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, planes: &[(u32, u32)]) -> Self {
        assert!(!planes.is_empty(), "expected at least the luma plane");

        let bobs = planes
            .iter()
            .map(|(_, height)| {
                [
                    BobWeights::new(client, *height, Parity::Top),
                    BobWeights::new(client, *height, Parity::Bottom),
                ]
            })
            .collect();

        Self {
            bobs,
            narrow: GaussTables::new(client, planes[0], 2.0),
            wide: GaussTables::new(client, planes[0], 5.0),
        }
    }

    pub fn bob(&self, plane: usize, parity: Parity) -> &BobWeights {
        let pair = &self.bobs[plane];

        match parity {
            Parity::Top => &pair[0],
            Parity::Bottom => &pair[1],
        }
    }
}
