use av_toolbox_cubek::blur::gaussian_weights;
use av_toolbox_cubek::edges::{EdgeLimits, Operator};
use av_toolbox_cubek::rgvs::{GrainBorder, RemoveGrainMode};
use av_toolbox_cubek::testing::{
    reference_binarize,
    reference_blur,
    reference_edges,
    reference_remove_grain,
};
use serde_json::{Value, json};

use super::{
    NEAR_HALF,
    code_values,
    column_share,
    detail_scene,
    differing_count,
    jetpack_blur_radius,
    reference_range_values,
    run_stage,
    store,
    store_half_distance,
    wide_code_values,
};
use crate::support::{grey, ten_bit_grey, wide_bytes};

/// `detail/mask`'s own arguments, mirroring its defaults.
struct DetailArgs {
    brz_mm: f32,
    brz_ed: f32,
    radius: u32,
    sigma: Option<f32>,
}

impl Default for DetailArgs {
    fn default() -> Self {
        Self {
            brz_mm: 0.025,
            brz_ed: 0.045,
            radius: 3,
            sigma: None,
        }
    }
}

fn detail_args_json(args: &DetailArgs) -> Value {
    match args.sigma {
        Some(sigma) => json!({
            "brz_mm": args.brz_mm,
            "brz_ed": args.brz_ed,
            "radius": args.radius,
            "sigma": sigma,
        }),
        None => json!({
            "brz_mm": args.brz_mm,
            "brz_ed": args.brz_ed,
            "radius": args.radius,
        }),
    }
}

/// The blurred luma when `sigma` is set, the range and Kirsch masks binarized and joined by
/// `max`, then RemoveGrain 22 and 11 with the reflecting border, stored at `peak`.
fn reference_detail_mask(luma: &[f32], width: u32, height: u32, args: &DetailArgs, peak: f32) -> Vec<u8> {
    reference_detail_mask_scaled(luma, width, height, args, peak, peak)
}

/// The same as [reference_detail_mask], with the thresholds scaled at `threshold_peak` instead of
/// `peak`.
fn reference_detail_mask_scaled(
    luma: &[f32],
    width: u32,
    height: u32,
    args: &DetailArgs,
    peak: f32,
    threshold_peak: f32,
) -> Vec<u8> {
    let cleaned = reference_detail_cleaned(luma, width, height, args, peak, threshold_peak);

    store(&cleaned, peak)
}

/// The joined and cleaned mask, before [reference_detail_mask_scaled] stores it.
fn reference_detail_cleaned(
    luma: &[f32],
    width: u32,
    height: u32,
    args: &DetailArgs,
    peak: f32,
    threshold_peak: f32,
) -> Vec<f32> {
    let base = match args.sigma {
        Some(sigma) => {
            let radius = jetpack_blur_radius(sigma);
            let weights = gaussian_weights(sigma, radius);

            reference_blur(luma, width, height, &weights)
        },
        None => luma.to_vec(),
    };

    let range = reference_range_values(&base, width, height, args.radius);
    let range_threshold = (args.brz_mm * threshold_peak).round();
    let range_binary = reference_binarize(&range, range_threshold, peak);

    let open_limits = EdgeLimits::open(peak);
    let kirsch = reference_edges(&base, width, height, Operator::Kirsch, open_limits);
    let edge_threshold = (args.brz_ed * threshold_peak).round();
    let edge_binary = reference_binarize(&kirsch, edge_threshold, peak);

    let joined: Vec<f32> = range_binary
        .iter()
        .zip(&edge_binary)
        .map(|(range_sample, edge_sample)| range_sample.max(*edge_sample))
        .collect();

    let mode_22 = RemoveGrainMode::TwentyTwo;
    let mode_11 = RemoveGrainMode::Eleven;
    let smoothed = reference_remove_grain(&joined, width, height, mode_22, GrainBorder::Reflect);

    reference_remove_grain(&smoothed, width, height, mode_11, GrainBorder::Reflect)
}

/// Fails when a sample of `masks` differs from `expected`, allowing a difference of 1 only where the
/// reference's `cleaned` value sits within `NEAR_HALF` of a store half, since the device may round
/// that half either way. Hands back how many samples sit that near.
fn assert_exact_off_store_halves(
    label: &str,
    masks: &[u8],
    expected: &[u8],
    cleaned: &[f32],
    peak: f32,
) -> usize {
    let mut near_count = 0;

    for index in 0..expected.len() {
        let distance = store_half_distance(cleaned[index], peak);
        let near = distance < NEAR_HALF;
        let allowed = u8::from(near);
        let difference = masks[index].abs_diff(expected[index]);
        assert!(
            difference <= allowed,
            "{label}: sample {index} is {}, expected {} within {allowed}",
            masks[index],
            expected[index]
        );

        if near {
            near_count += 1;
        }
    }

    near_count
}

/// How many samples of the bump scenes' reference sit on a store half. Each holds half the peak,
/// which stores at exactly 127.5.
const BUMP_STORE_HALVES: usize = 8;

/// The most samples the blurred mask may differ in, since a threshold on float blurred values can
/// flip a sample the reference blurred a hair differently. The measured count is 0.
const OUTLIER_LIMIT: usize = 16;

#[test]
fn detail_mask_matches_the_reference() {
    let width = 256;
    let height = 32;
    let peak = 255.0;
    let scene = detail_scene(width, height);
    let luma_values = code_values(&scene);

    let exact_args = DetailArgs::default();
    let exact_expected = reference_detail_mask(&luma_values, width, height, &exact_args, peak);
    assert!(exact_expected.contains(&0), "the reference marks everything");
    assert!(exact_expected.contains(&255), "the reference marks nothing");
    let exact_format = grey(width, height);
    let exact_frames = vec![vec![scene.clone()]];
    let exact_request = detail_args_json(&exact_args);
    let exact_masks = run_stage("detail/mask", exact_request, exact_format, exact_frames);
    assert_eq!(
        exact_masks[0], exact_expected,
        "the mask without sigma did not match the reference exactly"
    );

    let blurred_args = DetailArgs {
        sigma: Some(1.0),
        ..Default::default()
    };
    let blurred_expected = reference_detail_mask(&luma_values, width, height, &blurred_args, peak);
    assert!(
        blurred_expected.contains(&0),
        "the blurred reference marks everything"
    );
    assert!(
        blurred_expected.contains(&255),
        "the blurred reference marks nothing"
    );
    let blurred_format = grey(width, height);
    let blurred_frames = vec![vec![scene]];
    let blurred_request = detail_args_json(&blurred_args);
    let blurred_masks = run_stage("detail/mask", blurred_request, blurred_format, blurred_frames);

    let outliers = differing_count(&blurred_masks[0], &blurred_expected);
    println!("detail/mask sigma 1 outlier count: {outliers}");
    assert!(
        outliers <= OUTLIER_LIMIT,
        "{outliers} samples of the blurred mask differ from the reference, allowing {OUTLIER_LIMIT}"
    );

    // Blurred, the unit step's Kirsch stays under the threshold, so its columns stay empty. Kirsch
    // on the unblurred luma would mark them.
    let step_column = width / 4;
    let step_columns = step_column - 4..step_column + 4;
    let step_marks = column_share(&blurred_masks[0], width, step_columns, |sample| sample == 0);
    assert_eq!(step_marks, 1.0, "the blurred step was marked");
}

#[test]
fn detail_mask_marks_texture_but_not_a_smooth_gradient() {
    // The gradient has a single unit step. With the default arguments, any 8-bit unit step's Kirsch
    // of 15 clears the threshold of 11, so a gradient cannot be entirely empty.
    let width = 256;
    let height = 32;
    let half = width / 2;
    let scene = detail_scene(width, height);

    let format = grey(width, height);
    let frames = vec![vec![scene]];
    let masks = run_stage("detail/mask", json!({}), format, frames);

    let texture_share = column_share(&masks[0], width, half..width, |sample| sample == 255);
    let gradient_share = column_share(&masks[0], width, 0..half, |sample| sample == 0);
    println!("texture at 255: {texture_share}, gradient at 0: {gradient_share}");

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
fn a_flat_frame_gives_an_empty_detail_mask() {
    let width = 16;
    let height = 16;
    let luma = vec![128u8; (width * height) as usize];

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run_stage("detail/mask", json!({}), format, frames);

    assert!(
        masks[0].iter().all(|sample| *sample == 0),
        "a flat frame found detail"
    );
}

#[test]
fn ten_bit_detail_mask_matches_the_reference() {
    let width = 256;
    let height = 32;
    let peak = 1023.0;
    let scene = detail_scene(width, height);
    let luma: Vec<u16> = scene.iter().map(|sample| u16::from(*sample) * 4).collect();
    let luma_values: Vec<f32> = luma.iter().map(|sample| f32::from(*sample)).collect();

    let args = DetailArgs::default();
    let expected = reference_detail_mask(&luma_values, width, height, &args, peak);

    let format = ten_bit_grey(width, height);
    let wide_plane = wide_bytes(&luma);
    let frames = vec![vec![wide_plane]];
    let request = detail_args_json(&args);
    let masks = run_stage("detail/mask", request, format, frames);

    assert_eq!(
        masks[0], expected,
        "the 10-bit mask did not match the reference exactly"
    );
}

/// One isolated bump per height on a flat background, spaced 12 columns apart on row 8.
///
/// A bump of `h` gives its 8 neighbours a Kirsch of `5 * h`, and gives every sample within 3 of it a
/// range of `h`. The samples two columns away see the range alone, since their 3x3 ring misses the
/// bump.
fn bump_scene(heights: &[u16], background: u16) -> (u32, u32, Vec<u16>) {
    let width = 12 * heights.len() as u32;
    let height = 16;
    let mut luma = vec![background; (width * height) as usize];

    for (index, bump) in heights.iter().enumerate() {
        let column = 6 + 12 * index as u32;
        luma[(8 * width + column) as usize] = background + bump;
    }

    (width, height, luma)
}

/// The share of the 12 columns around bump `index` that are 0.
fn empty_share_around(masks: &[u8], width: u32, index: u32) -> f32 {
    let start = 12 * index;

    column_share(masks, width, start..start + 12, |sample| sample == 0)
}

/// The mask sample two columns right of bump `index`, where only the range sees the bump.
fn range_only_sample(masks: &[u8], width: u32, index: u32) -> u8 {
    masks[(8 * width + 6 + 12 * index + 2) as usize]
}

/// Bumps of 2, 3, 5 and 6 at 8 bits. The default thresholds are a range of 6 and a Kirsch of 11.
/// The bumps give a Kirsch of 10, 15, 25 and 30, and a range of 2, 3, 5 and 6.
#[test]
fn eight_bit_thresholds_land_between_neighbouring_values() {
    let peak = 255.0;
    let (width, height, wide_luma) = bump_scene(&[2, 3, 5, 6], 100);
    let luma: Vec<u8> = wide_luma.iter().map(|sample| *sample as u8).collect();
    let luma_values = code_values(&luma);

    let args = DetailArgs::default();
    let cleaned = reference_detail_cleaned(&luma_values, width, height, &args, peak, peak);
    let expected = store(&cleaned, peak);
    let swapped_args = DetailArgs {
        brz_mm: args.brz_ed,
        brz_ed: args.brz_mm,
        ..Default::default()
    };
    let swapped = reference_detail_mask(&luma_values, width, height, &swapped_args, peak);
    assert_ne!(expected, swapped, "swapping the thresholds changes nothing");

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let request = detail_args_json(&args);
    let masks = run_stage("detail/mask", request, format, frames);

    let near_halves = assert_exact_off_store_halves("8-bit bumps", &masks[0], &expected, &cleaned, peak);
    let differing = differing_count(&masks[0], &expected);
    println!("8-bit bumps: {near_halves} samples on a store half, {differing} samples differ");
    assert_eq!(near_halves, BUMP_STORE_HALVES, "the store halves moved");

    let below_range = range_only_sample(&masks[0], width, 2);
    let at_range = range_only_sample(&masks[0], width, 3);
    println!("range 5 gives {below_range}, range 6 gives {at_range}");
    assert!(
        below_range < at_range,
        "a range of 5 marked as much as a range of 6"
    );

    let below_kirsch = empty_share_around(&masks[0], width, 0);
    assert_eq!(below_kirsch, 1.0, "a Kirsch of 10 was marked");
}

/// Bumps of 9, 10, 25 and 26 at 10 bits. The thresholds are a range of 26 and a Kirsch of 46. The
/// bumps give a Kirsch of 45, 50, 125 and 130, and a range of 9, 10, 25 and 26. Scaled at 255 the
/// thresholds would be 6 and 11, which mark every bump.
#[test]
fn ten_bit_thresholds_land_between_neighbouring_values() {
    let peak = 1023.0;
    let (width, height, luma) = bump_scene(&[9, 10, 25, 26], 400);
    let luma_values = wide_code_values(&luma);

    let args = DetailArgs::default();
    let cleaned = reference_detail_cleaned(&luma_values, width, height, &args, peak, peak);
    let expected = store(&cleaned, peak);

    let eight_bit_scaled = reference_detail_mask_scaled(&luma_values, width, height, &args, peak, 255.0);
    assert_ne!(expected, eight_bit_scaled, "scaling at 255 changes nothing");

    let swapped_args = DetailArgs {
        brz_mm: args.brz_ed,
        brz_ed: args.brz_mm,
        ..Default::default()
    };
    let swapped = reference_detail_mask(&luma_values, width, height, &swapped_args, peak);
    assert_ne!(expected, swapped, "swapping the thresholds changes nothing");

    let format = ten_bit_grey(width, height);
    let wide_plane = wide_bytes(&luma);
    let frames = vec![vec![wide_plane]];
    let request = detail_args_json(&args);
    let masks = run_stage("detail/mask", request, format, frames);

    let near_halves = assert_exact_off_store_halves("10-bit bumps", &masks[0], &expected, &cleaned, peak);
    let differing = differing_count(&masks[0], &expected);
    println!("10-bit bumps: {near_halves} samples on a store half, {differing} samples differ");
    assert_eq!(near_halves, BUMP_STORE_HALVES, "the store halves moved");

    let below_range = range_only_sample(&masks[0], width, 2);
    let at_range = range_only_sample(&masks[0], width, 3);
    println!("range 25 gives {below_range}, range 26 gives {at_range}");
    assert!(
        below_range < at_range,
        "a range of 25 marked as much as a range of 26"
    );

    let below_kirsch = empty_share_around(&masks[0], width, 0);
    assert_eq!(below_kirsch, 1.0, "a Kirsch of 45 was marked");
}

/// A 64x64 frame with a gradient banded into 8 steps of 4 columns on the left half and the busy
/// texture on the right half, as the detail example's test clip holds.
fn example_clip_scene() -> (u32, Vec<u8>) {
    let size = 64u32;
    let half = size / 2;
    let luma = (0..size)
        .flat_map(|row| {
            (0..size).map(move |column| {
                if column < half {
                    return (100 + column / 4) as u8;
                }

                ((row * 37 + column * 17 + row * column) % 251) as u8
            })
        })
        .collect();

    (size, luma)
}

#[test]
fn detail_mask_marks_the_example_clips_texture_and_not_its_gradient() {
    let (size, scene) = example_clip_scene();
    let half = size / 2;

    let format = grey(size, size);
    let frames = vec![vec![scene]];
    let masks = run_stage("detail/mask", json!({ "sigma": 1.0 }), format, frames);

    let texture_share = column_share(&masks[0], size, half..size, |sample| sample == 255);
    let gradient_share = column_share(&masks[0], size, 0..half, |sample| sample == 0);
    println!("example clip texture at 255: {texture_share}, gradient at 0: {gradient_share}");

    // The mask leaves 27 of the gradient's 32 columns empty, a share of 0.84, and marks the 5
    // beside the texture.
    assert_eq!(texture_share, 1.0, "part of the texture was not marked");
    assert!(
        gradient_share >= 0.8,
        "only {gradient_share} of the gradient was left empty"
    );
}
