use av_toolbox_cubek::blur::gaussian_weights;
use av_toolbox_cubek::edges::{EdgeLimits, Operator};
use av_toolbox_cubek::neighbourhood::{Expansion, Extreme, Shape, expansion_passes};
use av_toolbox_cubek::rgvs::{GrainBorder, RemoveGrainMode};
use av_toolbox_cubek::testing::{
    reference_binarize,
    reference_blur,
    reference_edges,
    reference_extreme,
    reference_remove_grain,
};
use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::reference::square;
use crate::support::{SelectFrames, backend, built_mask, grey, mask_format, ten_bit_grey, wide_bytes};

fn run(args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    run_stage("detail/range", args, format, frames)
}

fn run_stage(stage: &str, args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    let backend = backend();
    let output = mask_format(format.width, format.height);
    let mask = built_mask(&backend, stage, args, &[StreamFormat::Frames(format)], output);

    backend.build(SelectFrames { mask, format, frames })
}

fn code_values(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample)).collect()
}

/// Mirrors `store_mask`, writing `round(clamp(v, 0, peak) * 255 / peak)` as u8.
fn store(values: &[f32], peak: f32) -> Vec<u8> {
    values
        .iter()
        .map(|value| {
            let clamped = value.clamp(0.0, peak);

            (clamped * 255.0 / peak).round() as u8
        })
        .collect()
}

/// Runs `passes` of `extreme` over the CPU, feeding each pass into the next.
fn expanded(input: &[f32], width: u32, height: u32, passes: &[Shape], extreme: Extreme) -> Vec<f32> {
    let mut plane = input.to_vec();
    for shape in passes {
        plane = reference_extreme(&plane, width, height, extreme, *shape);
    }

    plane
}

/// The local maximum minus the local minimum over `radius`.
fn reference_range_values(luma: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let passes = expansion_passes(radius, radius, Expansion::Ellipse);

    let maximum = expanded(luma, width, height, &passes, Extreme::Maximum);
    let minimum = expanded(luma, width, height, &passes, Extreme::Minimum);

    maximum
        .iter()
        .zip(&minimum)
        .map(|(largest, smallest)| largest - smallest)
        .collect()
}

/// The local maximum minus the local minimum over `radius`, times `multi`, stored at `peak`.
fn reference_range_mask(
    luma: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    multi: f32,
    peak: f32,
) -> Vec<u8> {
    let range = reference_range_values(luma, width, height, radius);

    let scaled: Vec<f32> = range.iter().map(|value| value * multi).collect();

    store(&scaled, peak)
}

/// A busy pattern that never repeats within a row or a column, so its local maximum and minimum
/// differ almost everywhere.
fn busy_frame(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|row| {
            (0..width).map(move |column| {
                let value = (row * 37 + column * 17 + row * column) % 251;

                value as u8
            })
        })
        .collect()
}

#[test]
fn detail_range_matches_the_reference() {
    let width = 24;
    let height = 24;
    let peak = 255.0;
    let square_luma = square(width, height);
    let busy_luma = busy_frame(width, height);

    for radius in [1u32, 2, 5] {
        for luma in [&square_luma, &busy_luma] {
            let luma_values = code_values(luma);
            let expected = reference_range_mask(&luma_values, width, height, radius, 1.0, peak);

            let format = grey(width, height);
            let frames = vec![vec![luma.to_vec()]];
            let request = json!({ "radius": radius });
            let masks = run(request, format, frames);

            assert_eq!(
                masks[0], expected,
                "radius {radius} did not match the reference exactly"
            );
        }
    }
}

#[test]
fn a_flat_frame_gives_an_empty_range_mask() {
    let width = 16;
    let height = 16;
    let luma = vec![128u8; (width * height) as usize];

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run(json!({}), format, frames);

    assert!(
        masks[0].iter().all(|sample| *sample == 0),
        "a flat frame found a range"
    );
}

/// A 10-bit frame split into a `500` half and a `600` half, a single 100-code step.
fn ten_bit_step_frame(width: u32, height: u32) -> Vec<u16> {
    let half = width / 2;

    (0..height)
        .flat_map(|_row| (0..width).map(move |column| if column < half { 500u16 } else { 600u16 }))
        .collect()
}

/// At radius 1, the samples beside the step hold `100`, which stores as
/// `round(100 * 255 / 1023) = round(24.93) = 25`.
#[test]
fn ten_bit_range_stores_scaled() {
    let width = 10;
    let height = 4;
    let luma = ten_bit_step_frame(width, height);

    let format = ten_bit_grey(width, height);
    let wide_plane = wide_bytes(&luma);
    let frames = vec![vec![wide_plane]];
    let request = json!({ "radius": 1 });
    let masks = run(request, format, frames);

    for row in 0..height {
        let left_of_step = (row * width + 4) as usize;
        let right_of_step = (row * width + 5) as usize;

        assert_eq!(
            masks[0][left_of_step], 25,
            "row {row} left of the step did not scale to 25"
        );
        assert_eq!(
            masks[0][right_of_step], 25,
            "row {row} right of the step did not scale to 25"
        );
    }
}

#[test]
fn multi_scales_and_saturates() {
    let width = 24;
    let height = 24;
    let luma = square(width, height);

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run(json!({ "multi": 50.0 }), format, frames);

    let edge_samples: Vec<u8> = masks[0].iter().copied().filter(|sample| *sample != 0).collect();
    assert!(!edge_samples.is_empty(), "no edge was found");
    assert!(
        edge_samples.iter().all(|sample| *sample == 255),
        "an edge sample did not saturate to 255"
    );
}

/// At `multi = 0.5` the square's edge range of 160 becomes 80, neither 0 nor saturated.
#[test]
fn a_fractional_multi_scales_without_saturating() {
    let width = 24;
    let height = 24;
    let peak = 255.0;
    let luma = square(width, height);
    let luma_values = code_values(&luma);
    let expected = reference_range_mask(&luma_values, width, height, 2, 0.5, peak);
    assert!(expected.contains(&80), "the reference holds no scaled edge of 80");

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run(json!({ "multi": 0.5 }), format, frames);

    assert_eq!(
        masks[0], expected,
        "multi 0.5 did not match the reference exactly"
    );
}

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

/// The blur radius vs-jetpack gives `sigma`, `ceil(6 * sigma + 1) / 2`.
fn jetpack_blur_radius(sigma: f32) -> u32 {
    let reach = (6.0 * sigma + 1.0).ceil();

    reach as u32 / 2
}

/// The blurred luma when `sigma` is set, the range and Kirsch masks binarized and joined by
/// `max`, then RemoveGrain 22 and 11 with the reflecting border, stored at `peak`.
fn reference_detail_mask(luma: &[f32], width: u32, height: u32, args: &DetailArgs, peak: f32) -> Vec<u8> {
    reference_detail_mask_scaled(luma, width, height, args, peak, peak)
}

/// The same as `reference_detail_mask`, with the thresholds scaled at `threshold_peak` instead of
/// `peak`.
fn reference_detail_mask_scaled(
    luma: &[f32],
    width: u32,
    height: u32,
    args: &DetailArgs,
    peak: f32,
    threshold_peak: f32,
) -> Vec<u8> {
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
    let cleaned = reference_remove_grain(&smoothed, width, height, mode_11, GrainBorder::Reflect);

    store(&cleaned, peak)
}

/// A smooth gradient on the left half and a busy texture on the right half.
fn detail_scene(width: u32, height: u32) -> Vec<u8> {
    let half = width / 2;

    (0..height)
        .flat_map(|row| {
            (0..width).map(move |column| {
                if column < half {
                    return (100 + column / 64) as u8;
                }

                ((row * 37 + column * 17 + row * column) % 251) as u8
            })
        })
        .collect()
}

/// The number of `masks` samples that differ from `expected`.
fn differing_count(masks: &[u8], expected: &[u8]) -> usize {
    masks
        .iter()
        .zip(expected)
        .filter(|(got, want)| got != want)
        .count()
}

/// The share of the samples in `columns` of every row that satisfy `matches`.
fn column_share(
    masks: &[u8],
    width: u32,
    columns: std::ops::Range<u32>,
    matches: impl Fn(u8) -> bool,
) -> f32 {
    let selected: Vec<u8> = masks
        .chunks(width as usize)
        .flat_map(|row| row[columns.start as usize..columns.end as usize].iter().copied())
        .collect();
    let matching = selected.iter().filter(|sample| matches(**sample)).count();

    matching as f32 / selected.len() as f32
}

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

fn wide_code_values(samples: &[u16]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample)).collect()
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
    let expected = reference_detail_mask(&luma_values, width, height, &args, peak);
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

    assert_eq!(masks[0], expected, "the mask did not match the reference exactly");

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
    let expected = reference_detail_mask(&luma_values, width, height, &args, peak);

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

    assert_eq!(masks[0], expected, "the mask did not match the reference exactly");

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
