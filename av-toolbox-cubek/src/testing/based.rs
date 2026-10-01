use super::bilateral::clamped_index;
use super::edges::prewitt_gradients;
use super::resize::reference_resize_float;
use super::{
    Eedi3Antialias,
    Eedi3Clips,
    Eedi3Parameters,
    reference_artcnn,
    reference_bilateral,
    reference_clamped_merge,
    reference_eedi3_antialias,
    reference_extreme,
    reference_narrow,
    reference_smooth,
    transposed_plane,
};
use crate::aa::prewitt_threshold;
use crate::artcnn::Weights;
use crate::bilateral::{BilateralSettings, positions};
use crate::neighbourhood::{Extreme, Shape, Smooth, median_of_three_sample};
use crate::resize::{Kernel, centred_taps};

/// The postfilter's bilateral spatial sigma.
const SIGMA_S: f32 = 2.0;

/// The postfilter's bilateral range sigma, on the `0..1` scale.
const SIGMA_R: f32 = 1.0 / 255.0;

/// The peak of a bound mask.
pub const BOUND_MASK_PEAK: f32 = 255.0;

/// Binarizes the Prewitt gradients on the CPU in 64-bit integers, mirroring
/// [binarize_prewitt](crate::aa::binarize_prewitt).
pub fn reference_binarize_prewitt(
    input: &[f32],
    width: u32,
    height: u32,
    threshold: u32,
    peak: f32,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let (across, down) = prewitt_gradients(input, width, height, x, y);
            let gx = across as i64;
            let gy = down as i64;
            let sum = gx * gx + gy * gy;
            let value = if sum >= i64::from(threshold) { peak } else { 0.0 };

            output.push(value);
        }
    }

    output
}

/// The two samples a doubled sample at `doubled` reads on one axis, clamped to `length`, and the
/// first one's weight.
fn doubling_taps(doubled: u32, length: u32) -> (usize, usize, f32) {
    let half = doubled / 2;
    let odd = doubled % 2;
    let start = (half + odd) as i32 - 1;
    let last = length as i32 - 1;
    let first = start.clamp(0, last) as usize;
    let second = (start + 1).clamp(0, last) as usize;
    let weight = if odd == 1 { 0.75 } else { 0.25 };

    (first, second, weight)
}

/// Doubles `mask` into EEDI3's 8-bit clip on the CPU, mirroring [double_mclip](crate::aa::double_mclip).
pub fn reference_double_mclip(mask: &[f32], width: u32, height: u32, mask_peak: f32) -> Vec<f32> {
    let columns = width as usize;
    let mut output = Vec::with_capacity(4 * mask.len());

    for row in 0..2 * height {
        for column in 0..2 * width {
            let (top, bottom, top_weight) = doubling_taps(row, height);
            let (left, right, left_weight) = doubling_taps(column, width);
            let right_weight = 1.0 - left_weight;
            let bottom_weight = 1.0 - top_weight;

            let upper = left_weight * mask[top * columns + left] + right_weight * mask[top * columns + right];
            let lower =
                left_weight * mask[bottom * columns + left] + right_weight * mask[bottom * columns + right];
            let value = top_weight * upper + bottom_weight * lower;
            let clipped = if 510.0 * value > mask_peak { 255.0 } else { 0.0 };

            output.push(clipped);
        }
    }

    output
}

/// Averages each 2x2 block of a plane of even width and height on the CPU, through
/// [Kernel::Box](crate::resize::Kernel::Box) taps.
pub fn reference_box_halve(input: &[f32], width: u32, height: u32) -> Vec<f32> {
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "a halved plane is {width}x{height}, expected even sizes",
    );

    let across = centred_taps(width, width / 2, &Kernel::Box);
    let down = centred_taps(height, height / 2, &Kernel::Box);

    reference_resize_float(input, width, height, &across, &down)
}

/// Writes `aa + keep * (luma - down)` on the CPU, mirroring [pscale_blend](crate::aa::pscale_blend).
pub fn reference_pscale_blend(aa: &[f32], luma: &[f32], down: &[f32], keep: f32) -> Vec<f32> {
    aa.iter()
        .zip(luma)
        .zip(down)
        .map(|((aa_value, luma_value), down_value)| {
            let difference = luma_value - down_value;
            aa_value + keep * difference
        })
        .collect()
}

/// Takes the median of three planes on the CPU, mirroring
/// [median_of_three](crate::neighbourhood::median_of_three).
pub fn reference_median_of_three(first: &[f32], second: &[f32], third: &[f32]) -> Vec<f32> {
    first
        .iter()
        .zip(second)
        .zip(third)
        .map(|((first_value, second_value), third_value)| {
            median_of_three_sample(*first_value, *second_value, *third_value)
        })
        .collect()
}

/// The axes `aa/based` anti-aliases along.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Vertically, then horizontally on the vertical result.
    #[default]
    Both,
    Vertical,
    Horizontal,
}

impl Direction {
    fn runs_vertically(self) -> bool {
        matches!(self, Self::Both | Self::Vertical)
    }

    fn runs_horizontally(self) -> bool {
        matches!(self, Self::Both | Self::Horizontal)
    }
}

/// The arguments of `aa/based`, with the luma's bit depth.
///
/// `mask_thr` is on the 8-bit scale and scales by `2^(bits - 8)`. EEDI3 runs with `eedi3` at `bits`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasedParameters {
    pub mask_thr: u32,
    pub edge_mask: bool,
    pub pscale: f32,
    pub postfilter: bool,
    pub direction: Direction,
    pub eedi3: Eedi3Parameters,
    pub bits: u32,
}

impl Default for BasedParameters {
    fn default() -> Self {
        let eedi3 = Eedi3Parameters {
            alpha: 0.125,
            gamma: 40.0,
            vthresh: [12.0, 24.0, 4.0],
            ..Eedi3Parameters::default()
        };

        Self {
            mask_thr: 60,
            edge_mask: true,
            pscale: 0.0,
            postfilter: true,
            direction: Direction::Both,
            eedi3,
            bits: 8,
        }
    }
}

/// Every intermediate plane of `aa/based` on one luma plane.
///
/// `supersampled`, `mclip` and `antialiased` are twice the luma's size, and every other plane is the
/// luma's size. `mask` and `mclip` are empty when `edge_mask` is false. `merged` is the plane before its
/// one rounding, and `output` holds integer code values.
///
/// `touched` marks the output samples a close decision of an EEDI3 pass can change, with each pass judged
/// on the input this reference hands it. A doubled sample reaches the output sample it is halved into,
/// then, with the postfilter, every sample whose bilateral reads that one. The merge keeps a mark only
/// where the mask is non-zero.
#[derive(Debug, Clone, PartialEq)]
pub struct BasedStages {
    pub mask: Vec<f32>,
    pub mclip: Vec<f32>,
    pub supersampled: Vec<f32>,
    pub antialiased: Vec<f32>,
    pub merged: Vec<f32>,
    pub output: Vec<f32>,
    pub touched: Vec<bool>,
}

/// The plane the horizontal EEDI3 pass reads as its `sclip`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HorizontalSclip {
    /// The transposed supersampled luma.
    Supersampled,
    /// The horizontal pass's own input.
    #[cfg_attr(not(test), expect(dead_code))]
    PassInput,
}

/// The largest code value at `bits` per sample.
pub fn peak_of(bits: u32) -> f32 {
    let peak = (1u32 << bits) - 1;

    peak as f32
}

/// Runs `aa/based` on one luma plane on the CPU, from its supersampled luma.
///
/// `supersampled` holds the doubled luma in code values, `2 * width` by `2 * height`. A bound `mask`
/// holds 8-bit values and is used as it is. Without one, the mask is built from `luma` when `edge_mask`
/// is set.
pub fn reference_based_aa(
    luma: &[f32],
    width: u32,
    height: u32,
    parameters: &BasedParameters,
    mask: Option<&[f32]>,
    supersampled: &[f32],
) -> BasedStages {
    composed(
        luma,
        width,
        height,
        parameters,
        mask,
        supersampled,
        HorizontalSclip::Supersampled,
    )
}

/// [reference_based_aa](crate::testing::reference_based_aa), with the horizontal pass reading
/// `horizontal_sclip` as its `sclip`.
pub(super) fn composed(
    luma: &[f32],
    width: u32,
    height: u32,
    parameters: &BasedParameters,
    mask: Option<&[f32]>,
    supersampled: &[f32],
    horizontal_sclip: HorizontalSclip,
) -> BasedStages {
    assert!(
        width >= 2 && height >= 2,
        "aa/based needs a luma plane of at least 2 by 2, not {width}x{height}"
    );
    assert!(
        parameters.edge_mask || mask.is_none(),
        "aa/based takes no bound mask without edge_mask"
    );

    let samples = width as usize * height as usize;
    let doubled_width = 2 * width;
    let doubled_height = 2 * height;
    assert_eq!(luma.len(), samples, "the luma plane is not {width}x{height}");
    assert_eq!(
        supersampled.len(),
        4 * samples,
        "the supersampled plane is not {doubled_width}x{doubled_height}"
    );

    let peak = peak_of(parameters.bits);
    if let Some(bound) = mask {
        assert_eq!(bound.len(), samples, "the bound mask is not {width}x{height}");
    }

    let (mask, mask_peak) = match mask {
        Some(bound) => (bound.to_vec(), BOUND_MASK_PEAK),
        None if parameters.edge_mask => (reference_based_mask(luma, width, height, parameters), peak),
        None => (Vec::new(), peak),
    };

    let mclip = if parameters.edge_mask {
        reference_double_mclip(&mask, width, height, mask_peak)
    } else {
        Vec::new()
    };

    let doubled = DoubledPlanes {
        supersampled,
        mclip: parameters.edge_mask.then_some(mclip.as_slice()),
        width: doubled_width,
        height: doubled_height,
    };
    let eedi3 = Eedi3Parameters {
        bits: parameters.bits,
        ..parameters.eedi3
    };
    let antialiased = antialias(&doubled, &eedi3, parameters.direction, horizontal_sclip);

    let finish = BasedFinish {
        luma,
        supersampled,
        antialiased: &antialiased.output,
        mask: parameters.edge_mask.then_some((mask.as_slice(), mask_peak)),
    };
    let merged = reference_based_finish(&finish, width, height, parameters);

    let narrowed = reference_narrow(&merged, peak);
    let output = narrowed.iter().map(|code| f32::from(*code)).collect();
    let merge_mask = parameters.edge_mask.then_some(mask.as_slice());
    let touched = reached_output(
        &antialiased.touched,
        width,
        height,
        parameters.postfilter,
        merge_mask,
    );

    BasedStages {
        mask,
        mclip,
        supersampled: supersampled.to_vec(),
        antialiased: antialiased.output,
        merged,
        output,
        touched,
    }
}

/// Builds the mask of `aa/based` from `luma` on the CPU: a Prewitt binarize, a 3x3 maximum and a 3x3 box.
pub fn reference_based_mask(luma: &[f32], width: u32, height: u32, parameters: &BasedParameters) -> Vec<f32> {
    let peak = peak_of(parameters.bits);
    let threshold = prewitt_threshold(parameters.mask_thr, parameters.bits);

    let binarized = reference_binarize_prewitt(luma, width, height, threshold, peak);
    let widened = reference_extreme(&binarized, width, height, Extreme::Maximum, Shape::Square);

    reference_smooth(&widened, width, height, Smooth::Box)
}

/// The planes the steps of `aa/based` after EEDI3 read, each in code values.
///
/// `luma` is the luma's size, and `supersampled` and `antialiased` are twice it. `mask` holds the mask
/// and its peak, and is given exactly when `edge_mask` is set.
#[derive(Debug, Clone, Copy)]
pub struct BasedFinish<'a> {
    pub luma: &'a [f32],
    pub supersampled: &'a [f32],
    pub antialiased: &'a [f32],
    pub mask: Option<(&'a [f32], f32)>,
}

/// Runs the steps of `aa/based` after EEDI3 on the CPU, giving the merged plane before its one rounding.
///
/// The steps are the box halving, the blend, the postfilter and the merge.
pub fn reference_based_finish(
    planes: &BasedFinish<'_>,
    width: u32,
    height: u32,
    parameters: &BasedParameters,
) -> Vec<f32> {
    let peak = peak_of(parameters.bits);
    let doubled_width = 2 * width;
    let doubled_height = 2 * height;

    let halved = reference_box_halve(planes.antialiased, doubled_width, doubled_height);
    let blended = if parameters.pscale == 1.0 {
        halved
    } else {
        let down = reference_box_halve(planes.supersampled, doubled_width, doubled_height);
        let keep = 1.0 - parameters.pscale;
        reference_pscale_blend(&halved, planes.luma, &down, keep)
    };

    let filtered = if parameters.postfilter {
        let settings = BilateralSettings {
            sigma_s: SIGMA_S,
            sigma_r: SIGMA_R,
            peak,
        };
        let smoothed = reference_bilateral(&blended, planes.luma, width, height, settings);
        reference_median_of_three(&blended, planes.luma, &smoothed)
    } else {
        blended
    };

    match planes.mask {
        Some((mask, mask_peak)) => reference_clamped_merge(planes.luma, &filtered, mask, mask_peak, peak),
        None => filtered,
    }
}

/// The doubled planes EEDI3 reads.
pub(super) struct DoubledPlanes<'a> {
    pub(super) supersampled: &'a [f32],
    pub(super) mclip: Option<&'a [f32]>,
    pub(super) width: u32,
    pub(super) height: u32,
}

/// EEDI3's anti-alias of the supersampled luma along `direction`, reading it as `sclip` too.
///
/// A sample is touched when a close decision of either pass can change it.
pub(super) fn antialias(
    doubled: &DoubledPlanes<'_>,
    parameters: &Eedi3Parameters,
    direction: Direction,
    horizontal_sclip: HorizontalSclip,
) -> Eedi3Antialias {
    let width = doubled.width;
    let height = doubled.height;

    let mut output = doubled.supersampled.to_vec();
    let mut touched = vec![false; output.len()];
    if direction.runs_vertically() {
        let clips = Eedi3Clips {
            sclip: Some(doubled.supersampled),
            mclip: doubled.mclip,
        };
        let vertical = reference_eedi3_antialias(&output, width, height, parameters, clips);

        output = vertical.output;
        touched = vertical.touched;
    }

    if direction.runs_horizontally() {
        let turned = transposed_plane(&output, width, height);
        let turned_sclip = match horizontal_sclip {
            HorizontalSclip::Supersampled => transposed_plane(doubled.supersampled, width, height),
            HorizontalSclip::PassInput => turned.clone(),
        };
        let turned_mclip = doubled.mclip.map(|mclip| transposed_plane(mclip, width, height));
        let clips = Eedi3Clips {
            sclip: Some(&turned_sclip),
            mclip: turned_mclip.as_deref(),
        };
        let horizontal = reference_eedi3_antialias(&turned, height, width, parameters, clips);

        let horizontal_touched = transposed_plane(&horizontal.touched, height, width);
        output = transposed_plane(&horizontal.output, height, width);
        touched = touched
            .iter()
            .zip(&horizontal_touched)
            .map(|(first, second)| *first || *second)
            .collect();
    }

    Eedi3Antialias { output, touched }
}

/// The luma samples a touched sample of the doubled `antialiased` plane can reach.
pub(super) fn reached_output(
    antialiased: &[bool],
    width: u32,
    height: u32,
    postfilter: bool,
    mask: Option<&[f32]>,
) -> Vec<bool> {
    let doubled_width = 2 * width as usize;
    let mut halved = Vec::with_capacity(width as usize * height as usize);
    for row in 0..height as usize {
        for column in 0..width as usize {
            let top = 2 * row * doubled_width + 2 * column;
            let bottom = top + doubled_width;
            let block = [top, top + 1, bottom, bottom + 1];
            let reached = block.iter().any(|index| antialiased[*index]);

            halved.push(reached);
        }
    }

    let mut reached = if postfilter {
        bilateral_reach(&halved, width, height)
    } else {
        halved
    };

    if let Some(mask) = mask {
        for (mark, weight) in reached.iter_mut().zip(mask) {
            *mark = *mark && *weight != 0.0;
        }
    }

    reached
}

/// Marks every sample whose bilateral reads a marked sample of `marks`, itself included.
///
/// The bilateral's reads clamp to the plane, so a mark near the edge reaches further inward.
fn bilateral_reach(marks: &[bool], width: u32, height: u32) -> Vec<bool> {
    let offsets = positions(SIGMA_S);
    let mut signed = Vec::with_capacity(2 * offsets.len());
    for offset in offsets {
        signed.push(-(offset as i32));
        signed.push(offset as i32);
    }

    let mut reached = Vec::with_capacity(marks.len());
    for row in 0..height as i32 {
        for column in 0..width as i32 {
            let centre_index = clamped_index(width, height, column, row);
            let centre = marks[centre_index];
            let read = signed.iter().any(|offset_y| {
                signed.iter().any(|offset_x| {
                    let index = clamped_index(width, height, column + offset_x, row + offset_y);
                    marks[index]
                })
            });

            reached.push(centre || read);
        }
    }

    reached
}

/// ArtCNN's doubled `luma` on the CPU in f64, rounded to code values as its last layer rounds.
pub fn reference_supersample(
    luma: &[f32],
    width: u32,
    height: u32,
    weights: &Weights,
    bits: u32,
) -> Vec<f32> {
    let peak = peak_of(bits);
    let unit: Vec<f32> = luma.iter().map(|code| code / peak).collect();
    let doubled = reference_artcnn(&unit, width, height, weights);
    let wide_peak = f64::from(peak);

    doubled
        .iter()
        .map(|value| {
            let rounded = (value * wide_peak + 0.5).floor();
            rounded as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CENTRE: usize = 12;

    /// A 5 by 5 plane whose rows hold `top`, `top`, `middle`, `bottom` and `bottom`.
    fn rows_plane(top: f32, middle: f32, bottom: f32) -> Vec<f32> {
        let levels = [top, top, middle, bottom, bottom];

        levels.iter().flat_map(|level| [*level; 5]).collect()
    }

    #[test]
    fn binarize_sets_an_exact_tie_at_8_bit() {
        let plane = rows_plane(40.0, 30.0, 20.0);
        let threshold = 60 * 60;

        let at = reference_binarize_prewitt(&plane, 5, 5, threshold, 255.0);
        let above = reference_binarize_prewitt(&plane, 5, 5, threshold + 1, 255.0);

        assert_eq!(at[CENTRE], 255.0);
        assert_eq!(above[CENTRE], 0.0);
    }

    #[test]
    fn binarize_sets_an_exact_tie_at_the_16_bit_threshold() {
        let plane = rows_plane(10240.0, 7000.0, 5120.0);
        let scaled = 60 << 8;
        let threshold = scaled * scaled;
        let next = (scaled + 1) * (scaled + 1);

        let at = reference_binarize_prewitt(&plane, 5, 5, threshold, 65535.0);
        let above = reference_binarize_prewitt(&plane, 5, 5, next, 65535.0);

        assert_eq!(at[CENTRE], 65535.0);
        assert_eq!(above[CENTRE], 0.0);
    }

    #[test]
    fn binarize_sets_a_gradient_whose_square_passes_2_to_the_32() {
        let plane = [
            65535.0, 30000.0, 0.0, //
            65535.0, 30000.0, 0.0, //
            65535.0, 30000.0, 0.0,
        ];
        let scaled = 255 << 8;
        let threshold = scaled * scaled;

        let binarized = reference_binarize_prewitt(&plane, 3, 3, threshold, 65535.0);

        assert_eq!(binarized[4], 65535.0);
    }

    #[test]
    fn binarize_sets_gradients_whose_squares_sum_past_2_to_the_32() {
        let mut plane = Vec::new();
        for row in 0..3 {
            for column in 0..3 {
                let value = 65535 - 8350 * (row + column);
                plane.push(value as f32);
            }
        }

        let scaled = 255 << 8;
        let threshold = scaled * scaled;

        let binarized = reference_binarize_prewitt(&plane, 3, 3, threshold, 65535.0);

        assert_eq!(binarized[4], 65535.0);
    }

    /// The window's gradients are 20947 and 61828, whose squares sum to 7 under the threshold. An f32
    /// square root rounds that sum onto the threshold.
    #[test]
    fn binarize_leaves_a_16_bit_sum_just_under_the_threshold_unset() {
        let plane = [
            0.0, 61828.0, 0.0, //
            20947.0, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ];
        let scaled = 255 << 8;
        let threshold = scaled * scaled;

        let binarized = reference_binarize_prewitt(&plane, 3, 3, threshold, 65535.0);

        assert_eq!(binarized[4], 0.0);
    }

    /// A 5 by 5 mask holding `value` at its centre and 0 elsewhere.
    fn lone_sample(value: f32) -> Vec<f32> {
        let mut mask = vec![0.0; 25];
        mask[CENTRE] = value;

        mask
    }

    /// The doubled positions `mclip` sets.
    fn set_positions(clip: &[f32], width: usize) -> Vec<(usize, usize)> {
        clip.iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .map(|(index, _)| (index % width, index / width))
            .collect()
    }

    /// The 4 by 4 block of doubled samples that read the centre of a 5 by 5 mask.
    fn centre_block() -> Vec<(usize, usize)> {
        let mut block = Vec::new();
        for row in 3..=6 {
            for column in 3..=6 {
                block.push((column, row));
            }
        }

        block
    }

    #[test]
    fn mclip_leaves_an_exact_half_unset() {
        let mask = lone_sample(8.0);

        let clip = reference_double_mclip(&mask, 5, 5, 255.0);

        let corners = [(3, 3), (6, 3), (3, 6), (6, 6)];
        let expected: Vec<(usize, usize)> = centre_block()
            .into_iter()
            .filter(|position| !corners.contains(position))
            .collect();
        assert_eq!(set_positions(&clip, 10), expected);
    }

    #[test]
    fn mclip_sets_just_past_half() {
        let mask = lone_sample(9.0);

        let clip = reference_double_mclip(&mask, 5, 5, 255.0);

        assert_eq!(set_positions(&clip, 10), centre_block());
    }

    #[test]
    fn mclip_repeats_the_edge_sample() {
        let mut inner = vec![255.0; 25];
        let mut ring = vec![0.0; 25];
        for row in 0..5 {
            for column in 0..5 {
                let on_ring = row == 0 || row == 4 || column == 0 || column == 4;
                if on_ring {
                    inner[row * 5 + column] = 0.0;
                    ring[row * 5 + column] = 255.0;
                }
            }
        }

        let inner_clip = reference_double_mclip(&inner, 5, 5, 255.0);
        let ring_clip = reference_double_mclip(&ring, 5, 5, 255.0);

        for row in 0..10 {
            for column in 0..10 {
                let on_outer_ring = row == 0 || row == 9 || column == 0 || column == 9;
                if on_outer_ring {
                    let index = row * 10 + column;
                    assert_eq!(inner_clip[index], 0.0, "column {column}, row {row}");
                    assert_eq!(ring_clip[index], 255.0, "column {column}, row {row}");
                }
            }
        }
    }

    const AA: [f32; 3] = [10.0, 20.25, 100.5];
    const LUMA: [f32; 3] = [12.0, 30.0, 90.0];
    const DOWN: [f32; 3] = [11.0, 25.0, 95.75];

    #[test]
    fn blend_at_pscale_0_adds_the_whole_difference() {
        let blended = reference_pscale_blend(&AA, &LUMA, &DOWN, 1.0);

        assert_eq!(blended, [11.0, 25.25, 94.75]);
    }

    #[test]
    fn blend_at_pscale_half_adds_half_the_difference() {
        let blended = reference_pscale_blend(&AA, &LUMA, &DOWN, 0.5);

        assert_eq!(blended, [10.5, 22.75, 97.625]);
    }

    #[test]
    fn blend_at_pscale_1_keeps_aa() {
        let blended = reference_pscale_blend(&AA, &LUMA, &DOWN, 0.0);

        assert_eq!(blended, AA);
    }

    /// A plane holding `column + 10 * row`.
    fn ramp(width: u32, height: u32) -> Vec<f32> {
        let mut plane = Vec::new();
        for row in 0..height {
            for column in 0..width {
                let value = column + 10 * row;
                plane.push(value as f32);
            }
        }

        plane
    }

    #[test]
    fn box_halving_averages_each_block_to_an_odd_size() {
        let plane = ramp(6, 2);

        let halved = reference_box_halve(&plane, 6, 2);

        assert_eq!(halved, [5.5, 7.5, 9.5]);
    }

    #[test]
    fn box_halving_averages_each_block_to_an_even_size() {
        let plane = ramp(8, 4);

        let halved = reference_box_halve(&plane, 8, 4);

        assert_eq!(halved, [5.5, 7.5, 9.5, 11.5, 25.5, 27.5, 29.5, 31.5]);
    }

    #[test]
    fn median_of_three_picks_the_middle_value() {
        let first = [1.0, 5.0, 9.0, 4.0];
        let second = [2.0, 9.0, 1.0, 4.0];
        let third = [3.0, 1.0, 5.0, 7.0];

        let middle = reference_median_of_three(&first, &second, &third);

        assert_eq!(middle, [2.0, 5.0, 5.0, 4.0]);
    }
}
