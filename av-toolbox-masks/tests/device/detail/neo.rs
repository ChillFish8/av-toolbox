use av_toolbox_cubek::bilateral::BilateralSettings;
use av_toolbox_cubek::blur::gaussian_weights;
use av_toolbox_cubek::edges::{EdgeLimits, Operator};
use av_toolbox_cubek::neighbourhood::Spread;
use av_toolbox_cubek::rgvs::{GrainBorder, RemoveGrainMode};
use av_toolbox_cubek::testing::{
    reference_add_capped,
    reference_bilateral,
    reference_binarize,
    reference_blur,
    reference_edges,
    reference_remove_grain,
    reference_spread,
};
use serde_json::{Value, json};

use super::{
    NEAR_HALF,
    busy_frame,
    code_values,
    column_share,
    detail_scene,
    differing_count,
    jetpack_blur_radius,
    run_stage,
    store,
    store_half_margin,
    wide_code_values,
};
use crate::support::{grey, ten_bit_grey, wide_bytes};

/// `detail/neo`'s own arguments, mirroring its defaults.
struct NeoArgs {
    sigma: f32,
    detail_brz: f32,
    lines_brz: f32,
    rg_mode: u32,
}

impl Default for NeoArgs {
    fn default() -> Self {
        Self {
            sigma: 5.0,
            detail_brz: 0.005,
            lines_brz: 0.08,
            rg_mode: 17,
        }
    }
}

fn neo_args_json(args: &NeoArgs) -> Value {
    json!({
        "sigma": args.sigma,
        "detail_brz": args.detail_brz,
        "lines_brz": args.lines_brz,
        "rg_mode": args.rg_mode,
    })
}

/// The steps of the reference a test can change to check the reference notices them.
struct NeoShape {
    guide_is_luma: bool,
    inflations: usize,
}

impl Default for NeoShape {
    fn default() -> Self {
        Self {
            guide_is_luma: false,
            inflations: 4,
        }
    }
}

fn neo_mode(number: u32) -> RemoveGrainMode {
    RemoveGrainMode::from_number(number).expect("a supported RemoveGrain mode")
}

/// The code value `fraction` of `peak` binarizes at, or `None` when `fraction` is 0.
fn neo_threshold(fraction: f32, peak: f32) -> Option<f32> {
    if fraction == 0.0 {
        return None;
    }

    Some((fraction * peak).round())
}

/// Binarizes `raw` at `fraction` of `peak`, or hands it back unchanged when `fraction` is 0.
fn neo_binarized(raw: &[f32], fraction: f32, peak: f32) -> Vec<f32> {
    match neo_threshold(fraction, peak) {
        Some(threshold) => reference_binarize(raw, threshold, peak),
        None => raw.to_vec(),
    }
}

/// `max(pref - luma, 0)` of the bilateral of `luma` guided by its blur, deflated once then inflated
/// `shape.inflations` times.
fn reference_neo_detail(
    luma: &[f32],
    width: u32,
    height: u32,
    args: &NeoArgs,
    peak: f32,
    shape: &NeoShape,
) -> Vec<f32> {
    let radius = jetpack_blur_radius(args.sigma);
    let weights = gaussian_weights(args.sigma, radius);
    let blurred = reference_blur(luma, width, height, &weights);
    let guide = if shape.guide_is_luma { luma } else { &blurred };

    let settings = BilateralSettings {
        sigma_s: args.sigma,
        sigma_r: 0.02,
        peak,
    };
    let preferred = reference_bilateral(luma, guide, width, height, settings);

    let rise: Vec<f32> = preferred
        .iter()
        .zip(luma)
        .map(|(smoothed, original)| (smoothed - original).max(0.0))
        .collect();

    let mut detail = reference_spread(&rise, width, height, Spread::Deflate);
    for _ in 0..shape.inflations {
        detail = reference_spread(&detail, width, height, Spread::Inflate);
    }

    detail
}

/// Prewitt of `luma`, deflated then inflated.
fn reference_neo_lines(luma: &[f32], width: u32, height: u32, peak: f32) -> Vec<f32> {
    let open_limits = EdgeLimits::open(peak);
    let prewitt = reference_edges(luma, width, height, Operator::Prewitt, open_limits);

    let deflated = reference_spread(&prewitt, width, height, Spread::Deflate);

    reference_spread(&deflated, width, height, Spread::Inflate)
}

/// The detail and the lines binarized and added, then cleaned by RemoveGrain.
fn reference_neo_cleaned(
    luma: &[f32],
    width: u32,
    height: u32,
    args: &NeoArgs,
    peak: f32,
    shape: &NeoShape,
) -> Vec<f32> {
    let detail_raw = reference_neo_detail(luma, width, height, args, peak, shape);
    let detail = neo_binarized(&detail_raw, args.detail_brz, peak);

    let lines_raw = reference_neo_lines(luma, width, height, peak);
    let lines = neo_binarized(&lines_raw, args.lines_brz, peak);

    let joined = reference_add_capped(&detail, &lines, peak);
    let mode = neo_mode(args.rg_mode);

    reference_remove_grain(&joined, width, height, mode, GrainBorder::Reflect)
}

/// [reference_neo_cleaned] stored at `peak`.
fn reference_neo_mask_with(
    luma: &[f32],
    width: u32,
    height: u32,
    args: &NeoArgs,
    peak: f32,
    shape: &NeoShape,
) -> Vec<u8> {
    let cleaned = reference_neo_cleaned(luma, width, height, args, peak, shape);

    store(&cleaned, peak)
}

fn reference_neo_mask(luma: &[f32], width: u32, height: u32, args: &NeoArgs, peak: f32) -> Vec<u8> {
    let shape = NeoShape::default();

    reference_neo_mask_with(luma, width, height, args, peak, &shape)
}

/// The largest difference between two masks at one sample.
fn largest_difference(masks: &[u8], expected: &[u8]) -> u8 {
    masks
        .iter()
        .zip(expected)
        .map(|(got, want)| got.abs_diff(*want))
        .max()
        .unwrap_or(0)
}

/// How many of `raw` sit within `margin` below `threshold`, and how many within `margin` at or above
/// it.
fn straddling(raw: &[f32], threshold: f32, margin: f32) -> (usize, usize) {
    let below = raw
        .iter()
        .filter(|value| **value < threshold && **value >= threshold - margin)
        .count();
    let above = raw
        .iter()
        .filter(|value| **value >= threshold && **value <= threshold + margin)
        .count();

    (below, above)
}

/// The closest a reference detail sample may sit to its threshold. Float error on the device is far
/// below it, so no sample can flip.
const CLOSEST_LIMIT: f32 = 0.01;

/// The most samples a binarized `detail/neo` run may differ from the reference in. A binarize
/// threshold on a float value can flip a sample the reference computed a hair differently, and
/// RemoveGrain spreads a flip to its neighbours. The measured count is 0.
const BINARIZED_OUTLIER_LIMIT: usize = 8;

/// The most samples an unbinarized run may differ in, each by at most 1. The store rounds
/// continuous values, so a value near a half can round the other way. The measured count is 0.
const UNBINARIZED_OUTLIER_LIMIT: usize = 16;

/// Fails when more than `outlier_limit` samples of `masks` differ from `expected`, or any differs by
/// more than `largest_limit`.
fn assert_neo_matches(label: &str, masks: &[u8], expected: &[u8], outlier_limit: usize, largest_limit: u8) {
    let outliers = differing_count(masks, expected);
    let largest = largest_difference(masks, expected);
    println!("{label}: {outliers} outliers, largest difference {largest}");

    assert!(
        outliers <= outlier_limit,
        "{label}: {outliers} samples differ from the reference, allowing {outlier_limit}"
    );
    assert!(
        largest <= largest_limit,
        "{label}: a sample differs by {largest}, allowing {largest_limit}"
    );
}

#[test]
fn detail_neo_matches_the_reference() {
    let width = 256;
    let height = 32;
    let peak = 255.0;
    let scene = detail_scene(width, height);
    let luma_values = code_values(&scene);

    for rg_mode in [7, 11, 17, 22] {
        let args = NeoArgs {
            rg_mode,
            ..Default::default()
        };
        let expected = reference_neo_mask(&luma_values, width, height, &args, peak);
        assert!(
            expected.contains(&0),
            "mode {rg_mode}: the reference marks everything"
        );
        assert!(
            expected.iter().any(|sample| *sample > 0),
            "mode {rg_mode}: the reference marks nothing"
        );

        let format = grey(width, height);
        let frames = vec![vec![scene.clone()]];
        let request = neo_args_json(&args);
        let masks = run_stage("detail/neo", request, format, frames);

        let label = format!("detail/neo mode {rg_mode}");
        assert_neo_matches(&label, &masks[0], &expected, BINARIZED_OUTLIER_LIMIT, 255);
    }
}

#[test]
fn the_neo_reference_notices_each_step() {
    let width = 256;
    let height = 32;
    let peak = 255.0;
    let scene = detail_scene(width, height);
    let luma_values = code_values(&scene);
    let defaults = NeoArgs::default();
    let expected = reference_neo_mask(&luma_values, width, height, &defaults, peak);

    let luma_guide = NeoShape {
        guide_is_luma: true,
        ..Default::default()
    };
    let with_luma_guide = reference_neo_mask_with(&luma_values, width, height, &defaults, peak, &luma_guide);
    assert_ne!(expected, with_luma_guide, "guiding by the luma changes nothing");

    for inflations in [3, 5] {
        let shape = NeoShape {
            inflations,
            ..Default::default()
        };
        let changed = reference_neo_mask_with(&luma_values, width, height, &defaults, peak, &shape);
        assert_ne!(expected, changed, "{inflations} inflations change nothing");
    }

    let swapped_args = NeoArgs {
        detail_brz: defaults.lines_brz,
        lines_brz: defaults.detail_brz,
        ..Default::default()
    };
    let swapped = reference_neo_mask(&luma_values, width, height, &swapped_args, peak);
    assert_ne!(expected, swapped, "swapping the thresholds changes nothing");

    for rg_mode in [11, 22] {
        let other_mode = NeoArgs {
            rg_mode,
            ..Default::default()
        };
        let changed = reference_neo_mask(&luma_values, width, height, &other_mode, peak);
        assert_ne!(expected, changed, "mode {rg_mode} gives what mode 17 gives");
    }
}

/// With binarizing off the mask holds every level, so the four RemoveGrain modes give four masks.
#[test]
fn detail_neo_matches_the_reference_with_every_rg_mode_unbinarized() {
    let width = 256;
    let height = 32;
    let peak = 255.0;
    let scene = detail_scene(width, height);
    let luma_values = code_values(&scene);
    let mut references = Vec::new();
    let modes = [7, 11, 17, 22];

    for rg_mode in modes {
        let args = NeoArgs {
            detail_brz: 0.0,
            lines_brz: 0.0,
            rg_mode,
            ..Default::default()
        };
        let expected = reference_neo_mask(&luma_values, width, height, &args, peak);

        let format = grey(width, height);
        let frames = vec![vec![scene.clone()]];
        let request = neo_args_json(&args);
        let masks = run_stage("detail/neo", request, format, frames);

        let label = format!("unbinarized mode {rg_mode}");
        assert_neo_matches(&label, &masks[0], &expected, UNBINARIZED_OUTLIER_LIMIT, 1);

        let shape = NeoShape::default();
        let cleaned = reference_neo_cleaned(&luma_values, width, height, &args, peak, &shape);
        let (margin, near) = store_half_margin(&cleaned, peak);
        println!("{label}: closest to a store half {margin}, {near} within {NEAR_HALF}");

        references.push(expected);
    }

    for first in 0..references.len() {
        for second in first + 1..references.len() {
            assert_ne!(
                references[first], references[second],
                "modes {} and {} agree",
                modes[first], modes[second]
            );
        }
    }
}

#[test]
fn detail_neo_marks_texture_but_not_a_smooth_gradient() {
    let width = 256;
    let height = 32;
    let half = width / 2;
    let scene = detail_scene(width, height);

    let format = grey(width, height);
    let frames = vec![vec![scene]];
    let masks = run_stage("detail/neo", json!({}), format, frames);

    let texture_share = column_share(&masks[0], width, half..width, |sample| sample > 0);
    let gradient_share = column_share(&masks[0], width, 0..half, |sample| sample == 0);
    println!("neo texture marked: {texture_share}, gradient empty: {gradient_share}");

    assert!(
        texture_share >= 0.9,
        "only {texture_share} of the texture was marked"
    );
    assert!(
        gradient_share >= 0.9,
        "only {gradient_share} of the gradient was left empty"
    );
}

#[test]
fn a_flat_frame_gives_an_empty_neo_mask() {
    let width = 16;
    let height = 16;

    for value in [0u8, 128, 255] {
        let luma = vec![value; (width * height) as usize];

        let format = grey(width, height);
        let frames = vec![vec![luma]];
        let masks = run_stage("detail/neo", json!({}), format, frames);

        assert!(
            masks[0].iter().all(|sample| *sample == 0),
            "a flat frame of {value} found detail"
        );
    }
}

#[test]
fn ten_bit_neo_matches_the_reference() {
    let width = 256;
    let height = 32;
    let peak = 1023.0;
    let scene = detail_scene(width, height);
    let luma: Vec<u16> = scene.iter().map(|sample| u16::from(*sample) * 4).collect();
    let luma_values = wide_code_values(&luma);

    let args = NeoArgs::default();
    let expected = reference_neo_mask(&luma_values, width, height, &args, peak);

    let shape = NeoShape::default();
    let detail_raw = reference_neo_detail(&luma_values, width, height, &args, peak, &shape);
    let detail_threshold = neo_threshold(args.detail_brz, peak).expect("a detail threshold");
    let closest = detail_raw
        .iter()
        .map(|value| (value - detail_threshold).abs())
        .fold(f32::INFINITY, f32::min);
    println!("ten-bit closest detail sample to its threshold of {detail_threshold}: {closest}");
    assert!(
        closest >= CLOSEST_LIMIT,
        "a detail sample sits {closest} from its threshold"
    );

    let format = ten_bit_grey(width, height);
    let wide_plane = wide_bytes(&luma);
    let frames = vec![vec![wide_plane]];
    let request = neo_args_json(&args);
    let masks = run_stage("detail/neo", request, format, frames);

    assert_neo_matches(
        "ten-bit detail/neo",
        &masks[0],
        &expected,
        BINARIZED_OUTLIER_LIMIT,
        255,
    );
}

#[test]
fn a_zero_brz_leaves_its_step_unbinarized() {
    let width = 256;
    let height = 32;
    let peak = 255.0;
    let scene = detail_scene(width, height);
    let luma_values = code_values(&scene);
    let defaults = NeoArgs::default();
    let binarized = reference_neo_mask(&luma_values, width, height, &defaults, peak);

    let no_detail_brz = NeoArgs {
        detail_brz: 0.0,
        ..Default::default()
    };
    let no_lines_brz = NeoArgs {
        lines_brz: 0.0,
        ..Default::default()
    };

    for (label, args) in [("detail_brz", no_detail_brz), ("lines_brz", no_lines_brz)] {
        let expected = reference_neo_mask(&luma_values, width, height, &args, peak);
        assert_ne!(binarized, expected, "{label} 0 changes nothing");

        let format = grey(width, height);
        let frames = vec![vec![scene.clone()]];
        let request = neo_args_json(&args);
        let masks = run_stage("detail/neo", request, format, frames);

        let zeroed = format!("{label} 0");
        assert_neo_matches(&zeroed, &masks[0], &expected, UNBINARIZED_OUTLIER_LIMIT, 1);

        let shape = NeoShape::default();
        let cleaned = reference_neo_cleaned(&luma_values, width, height, &args, peak, &shape);
        let (margin, near) = store_half_margin(&cleaned, peak);
        println!("{zeroed}: closest to a store half {margin}, {near} within {NEAR_HALF}");
    }
}

/// A vertical step of `step` codes at the middle of a 24x16 frame, from `background` up.
fn step_scene(step: u16, background: u16) -> (u32, u32, Vec<u16>) {
    let width = 24;
    let rows = 16;
    let luma = (0..rows)
        .flat_map(|_row| {
            (0..width).map(move |column| {
                if column < width / 2 {
                    background
                } else {
                    background + step
                }
            })
        })
        .collect();

    (width, rows, luma)
}

/// A step of `n` codes gives Prewitt `3 * n` on the two columns beside it, and 0 elsewhere. Deflating
/// drops both columns to `5 * 3 * n / 8`, and inflating leaves them, so the lines reach
/// `15 * n / 8`. Every number is a whole multiple of 1/8, so the float maths is exact. The threshold
/// of 20 (8-bit) sits between steps of 10, which give 18.75, and 11, which give 20.625. At 10 bits the
/// threshold of 82 sits between steps of 43, which give 80.625, and 44, which give 82.5. The detail is
/// switched off by `detail_brz` 1, since a step this small never reaches the peak.
#[test]
fn neo_lines_thresholds_land_between_exact_step_strengths() {
    let lines_only = NeoArgs {
        detail_brz: 1.0,
        rg_mode: 11,
        ..Default::default()
    };
    let request = neo_args_json(&lines_only);

    let cases = [(255.0, 10u16, 11u16, 100u16), (1023.0, 43, 44, 400)];
    for (peak, below_step, at_step, background) in cases {
        let mut marked_counts = Vec::new();
        for step in [below_step, at_step] {
            let (width, height, luma) = step_scene(step, background);
            let luma_values = wide_code_values(&luma);
            let expected = reference_neo_mask(&luma_values, width, height, &lines_only, peak);

            let masks = run_neo_step_scene(peak, width, height, &luma, request.clone());

            let differing = differing_count(&masks, &expected);
            assert_eq!(
                differing, 0,
                "peak {peak}, step {step}: {differing} samples differ"
            );

            let marked = masks.iter().filter(|sample| **sample > 0).count();
            println!("peak {peak}, step {step}: {marked} marked samples");
            marked_counts.push(marked);
        }

        assert_eq!(
            marked_counts[0], 0,
            "peak {peak}: a step of {below_step} was marked"
        );
        assert!(
            marked_counts[1] > 0,
            "peak {peak}: a step of {at_step} was not marked"
        );
    }
}

/// Runs `detail/neo` over `luma` as 8-bit samples when `peak` is 255, and as 10-bit otherwise.
fn run_neo_step_scene(peak: f32, width: u32, height: u32, luma: &[u16], request: Value) -> Vec<u8> {
    if peak == 255.0 {
        let bytes: Vec<u8> = luma.iter().map(|sample| *sample as u8).collect();
        let frames = vec![vec![bytes]];
        let format = grey(width, height);
        let masks = run_stage("detail/neo", request, format, frames);

        return masks[0].clone();
    }

    let wide_plane = wide_bytes(luma);
    let frames = vec![vec![wide_plane]];
    let format = ten_bit_grey(width, height);
    let masks = run_stage("detail/neo", request, format, frames);

    masks[0].clone()
}

#[test]
fn neo_thresholds_have_samples_on_both_sides() {
    let width = 256;
    let height = 32;
    let peak = 255.0;
    let scene = detail_scene(width, height);
    let luma_values = code_values(&scene);
    let args = NeoArgs::default();
    let shape = NeoShape::default();

    let detail_raw = reference_neo_detail(&luma_values, width, height, &args, peak, &shape);
    let detail_threshold = neo_threshold(args.detail_brz, peak).expect("a detail threshold");
    let (detail_below, detail_above) = straddling(&detail_raw, detail_threshold, 0.05);
    println!("detail threshold {detail_threshold}: {detail_below} within 0.05 below, {detail_above} above");
    assert!(detail_below > 0, "no detail sample sits just below its threshold");
    assert!(detail_above > 0, "no detail sample sits just above its threshold");

    let closest = detail_raw
        .iter()
        .map(|value| (value - detail_threshold).abs())
        .fold(f32::INFINITY, f32::min);
    println!("closest detail sample to its threshold: {closest}");
    assert!(
        closest >= CLOSEST_LIMIT,
        "a detail sample sits {closest} from its threshold"
    );

    let ten_bit_peak = 1023.0;
    let ten_bit_luma: Vec<f32> = luma_values.iter().map(|value| value * 4.0).collect();
    let ten_bit_raw = reference_neo_detail(&ten_bit_luma, width, height, &args, ten_bit_peak, &shape);
    let ten_bit_threshold = neo_threshold(args.detail_brz, ten_bit_peak).expect("a detail threshold");
    let (ten_bit_below, ten_bit_above) = straddling(&ten_bit_raw, ten_bit_threshold, 0.5);
    println!(
        "ten-bit detail threshold {ten_bit_threshold}: {ten_bit_below} within 0.5 below, {ten_bit_above} above"
    );
    assert!(
        ten_bit_below > 0,
        "no 10-bit detail sample sits just below its threshold"
    );
    assert!(
        ten_bit_above > 0,
        "no 10-bit detail sample sits just above its threshold"
    );

    let ten_bit_closest = ten_bit_raw
        .iter()
        .map(|value| (value - ten_bit_threshold).abs())
        .fold(f32::INFINITY, f32::min);
    println!("closest 10-bit detail sample to its threshold: {ten_bit_closest}");
    assert!(
        ten_bit_closest >= CLOSEST_LIMIT,
        "a 10-bit detail sample sits {ten_bit_closest} from its threshold"
    );
}

#[test]
fn neo_on_a_plane_smaller_than_its_reach_matches_the_reference() {
    let width = 5;
    let height = 5;
    let peak = 255.0;
    let scene = busy_frame(width, height);
    let luma_values = code_values(&scene);
    let args = NeoArgs {
        sigma: 16.0,
        ..Default::default()
    };
    let expected = reference_neo_mask(&luma_values, width, height, &args, peak);
    assert!(
        expected.iter().any(|sample| *sample > 0),
        "the reference marks nothing"
    );

    let format = grey(width, height);
    let frames = vec![vec![scene]];
    let request = neo_args_json(&args);
    let masks = run_stage("detail/neo", request, format, frames);

    assert_neo_matches("5x5 sigma 16", &masks[0], &expected, BINARIZED_OUTLIER_LIMIT, 255);
}
