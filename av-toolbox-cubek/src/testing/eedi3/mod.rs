mod check;
mod fixtures;
mod path;
mod random;
#[cfg(test)]
mod tests;
mod touched;

use self::check::{PassPlanes, interpolate_row, reliability_check};
pub use self::fixtures::{EEDI3_FIXTURES, Eedi3Fixture, worked_edge_plane};
use self::path::{Grid, Mask, Weights, search_row};
pub use self::random::{
    Eedi3PlaneKind,
    SplitMix,
    random_eedi3_mask,
    random_eedi3_parameters,
    random_eedi3_plane,
};
pub use self::touched::CloseDecisionTally;

/// The arguments of one EEDI3 field pass, with the plane's bit depth.
///
/// `beta`, `gamma`, `vthresh[0]` and `vthresh[1]` are on the 8-bit scale and scale with `bits`. `alpha` and
/// `vthresh[2]` do not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eedi3Parameters {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub nrad: u32,
    pub mdis: u32,
    pub vcheck: u32,
    pub vthresh: [f32; 3],
    pub bits: u32,
}

impl Default for Eedi3Parameters {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            beta: 0.25,
            gamma: 20.0,
            nrad: 2,
            mdis: 20,
            vcheck: 2,
            vthresh: [32.0, 64.0, 4.0],
            bits: 8,
        }
    }
}

/// The optional planes a field pass reads beside its input, each at the input's size.
///
/// `sclip` replaces the vertical cubic the reliability check falls back to. `mclip` holds 8-bit mask
/// values, and a column is searched only where a non-zero value lies within `mdis` columns of it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Eedi3Clips<'a> {
    pub sclip: Option<&'a [f32]>,
    pub mclip: Option<&'a [f32]>,
}

/// One EEDI3 field pass, every plane at the input's size.
///
/// `directions` holds the direction each rebuilt sample was interpolated along, and 0 on kept rows.
/// `touched` marks the output samples and `touched_directions` the directions that a close decision can
/// change.
#[derive(Debug, Clone, PartialEq)]
pub struct Eedi3Pass {
    pub output: Vec<f32>,
    pub directions: Vec<i32>,
    pub touched: Vec<bool>,
    pub touched_directions: Vec<bool>,
}

/// EEDI3's double-rate anti-alias of one plane, with the samples a close decision can change.
#[derive(Debug, Clone, PartialEq)]
pub struct Eedi3Antialias {
    pub output: Vec<f32>,
    pub touched: Vec<bool>,
}

/// How the connection cost's multiplies, which a compiler may fuse, are rounded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Contraction {
    /// Every multiply and add rounded on its own, as the plugin does.
    #[default]
    Plain,
    /// The later product of each addition fused into it.
    FuseLater,
    /// The earlier product of each addition fused into it.
    FuseEarlier,
}

/// Which version of the row above the reliability check reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CheckOrder {
    /// The row above after its own check, as the plugin reads it.
    #[default]
    CheckedAbove,
    /// The row above before its check.
    UncheckedAbove,
}

/// Whether an inactive column 0 or 1 borrows its costs from a rebuilt row above.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Lending {
    /// Borrowed from the nearest rebuilt row above where the column is active, as the plugin does.
    #[default]
    On,
    /// Every borrowed cost is 0.
    Off,
}

/// Which direction wins a path step when two candidates cost the same.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TieOrder {
    /// The smallest direction, as the plugin's strict compare gives.
    #[default]
    SmallestDirection,
    /// The largest direction, as a compare with `<=` gives.
    LargestDirection,
}

/// The rules a field pass follows where a port could differ from the plugin.
///
/// The default is the plugin's behaviour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Eedi3Variant {
    pub contraction: Contraction,
    pub check_order: CheckOrder,
    pub lending: Lending,
    pub ties: TieOrder,
}

/// Everything one field pass reads besides its planes.
struct Settings {
    weights: Weights,
    nrad: i32,
    mdis: i32,
    vcheck: u32,
    peak: i32,
    variant: Eedi3Variant,
}

impl Settings {
    fn new(parameters: &Eedi3Parameters, variant: Eedi3Variant) -> Self {
        Self {
            weights: Weights::new(parameters),
            nrad: parameters.nrad as i32,
            mdis: parameters.mdis as i32,
            vcheck: parameters.vcheck,
            peak: (1 << parameters.bits) - 1,
            variant,
        }
    }
}

fn check_arguments(
    plane: &[f32],
    width: u32,
    height: u32,
    offset: u32,
    parameters: &Eedi3Parameters,
    clips: Eedi3Clips<'_>,
) {
    assert!(width >= 4, "an EEDI3 plane needs at least 4 columns, not {width}");
    assert!(
        height >= 4 && height.is_multiple_of(2),
        "an EEDI3 plane needs an even height of at least 4, not {height}"
    );
    assert!(offset < 2, "an EEDI3 field offset is 0 or 1, not {offset}");
    assert!(
        (8..=16).contains(&parameters.bits),
        "EEDI3 takes 8 to 16 bits per sample, not {}",
        parameters.bits
    );
    assert!(
        parameters.nrad <= 3,
        "EEDI3's nrad is at most 3, not {}",
        parameters.nrad
    );
    assert!(
        (1..=40).contains(&parameters.mdis),
        "EEDI3's mdis is between 1 and 40, not {}",
        parameters.mdis
    );
    assert!(
        parameters.vcheck <= 3,
        "EEDI3's vcheck is at most 3, not {}",
        parameters.vcheck
    );

    let samples = width as usize * height as usize;
    assert_eq!(plane.len(), samples, "the plane is not {width}x{height}");
    if let Some(sclip) = clips.sclip {
        assert_eq!(sclip.len(), samples, "the sclip plane is not {width}x{height}");
    }

    if let Some(mclip) = clips.mclip {
        assert_eq!(mclip.len(), samples, "the mclip plane is not {width}x{height}");
    }
}

/// One EEDI3 field pass on the CPU, following the rules `variant` picks.
pub fn reference_eedi3_field_variant(
    plane: &[f32],
    width: u32,
    height: u32,
    offset: u32,
    parameters: &Eedi3Parameters,
    clips: Eedi3Clips<'_>,
    variant: &Eedi3Variant,
) -> Eedi3Pass {
    check_arguments(plane, width, height, offset, parameters, clips);

    let settings = Settings::new(parameters, *variant);
    let grid = Grid::new(plane, width, height);
    let sclip_grid = clips.sclip.map(|sclip| Grid::new(sclip, width, height));
    let sclip = sclip_grid.as_ref();
    let owned_mask = clips.mclip.map(|mclip| Mask::new(mclip, width));
    let mask = owned_mask.as_ref();
    let first_rebuilt = 1 - offset as i32;
    let row_width = width as usize;

    let mut planes = PassPlanes::kept(&grid);
    for row in (first_rebuilt..grid.height).step_by(2) {
        let path = search_row(&grid, mask, row, &settings);
        let rebuilt = interpolate_row(&grid, row, &path.directions, settings.peak);

        let start = row as usize * row_width;
        let span = start..start + row_width;
        planes.output[span.clone()].copy_from_slice(&rebuilt);
        planes.directions[span.clone()].copy_from_slice(&path.directions);
        planes.touched[span.clone()].copy_from_slice(&path.touched);
        planes.touched_directions[span].copy_from_slice(&path.touched);
    }

    if settings.vcheck > 0 {
        reliability_check(&grid, sclip, &mut planes, first_rebuilt, &settings);
    }

    Eedi3Pass {
        output: planes.output.iter().map(|value| *value as f32).collect(),
        directions: planes.directions,
        touched: planes.touched,
        touched_directions: planes.touched_directions,
    }
}

/// One EEDI3 field pass on the CPU, keeping the rows of parity `offset` and rebuilding the others.
///
/// The plane holds integer code values. Every rounded operation is one f32 operation in the plugin's
/// order, so the output matches the plugin's C path exactly. Rows and columns past the plane reflect
/// without repeating the edge. The two bands shifted along a connection count only where their shifted
/// column stays inside the row, and a band that leaves takes the other one's sum.
pub fn reference_eedi3_field(
    plane: &[f32],
    width: u32,
    height: u32,
    offset: u32,
    parameters: &Eedi3Parameters,
    clips: Eedi3Clips<'_>,
) -> Eedi3Pass {
    let variant = Eedi3Variant::default();

    reference_eedi3_field_variant(plane, width, height, offset, parameters, clips, &variant)
}

/// EEDI3's double-rate anti-alias on the CPU, `floor((pass(0) + pass(1) + 1) / 2)`.
///
/// Each pass is a [reference_eedi3_field](crate::testing::reference_eedi3_field), one keeping the even
/// rows and one the odd rows. A merged sample is touched when it is touched in either pass.
pub fn reference_eedi3_antialias(
    plane: &[f32],
    width: u32,
    height: u32,
    parameters: &Eedi3Parameters,
    clips: Eedi3Clips<'_>,
) -> Eedi3Antialias {
    let even_kept = reference_eedi3_field(plane, width, height, 0, parameters, clips);
    let odd_kept = reference_eedi3_field(plane, width, height, 1, parameters, clips);

    let output = merge_eedi3_fields(&even_kept.output, &odd_kept.output);
    let touched = even_kept
        .touched
        .iter()
        .zip(&odd_kept.touched)
        .map(|(first, second)| *first || *second)
        .collect();

    Eedi3Antialias { output, touched }
}

/// Two field passes merged, `floor((first + second + 1) / 2)` per sample.
pub fn merge_eedi3_fields(first: &[f32], second: &[f32]) -> Vec<f32> {
    first
        .iter()
        .zip(second)
        .map(|(first, second)| {
            let merged = (*first as i32 + *second as i32 + 1) >> 1;
            merged as f32
        })
        .collect()
}

/// The rows the reliability check rewrites in a plane `height` rows tall, whose first rebuilt row is
/// `first_rebuilt`.
///
/// These are the rebuilt rows with a rebuilt row two above and two below.
pub fn eedi3_checked_rows(first_rebuilt: u32, height: u32) -> impl Iterator<Item = u32> {
    (first_rebuilt..height)
        .step_by(2)
        .filter(move |row| *row >= 2 && *row + 3 <= height)
}
