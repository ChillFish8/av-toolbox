use av_toolbox_cubek::blur::gaussian_weights;
use av_toolbox_cubek::edges::{EdgeLimits, Operator};
use av_toolbox_cubek::neighbourhood::{Extreme, Shape};
use av_toolbox_cubek::retinex::RetinexSettings;
use av_toolbox_cubek::testing::{
    reference_binarize,
    reference_blur,
    reference_edges,
    reference_extreme,
    reference_retinex,
};
use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::support::{SelectFrames, backend, built_mask, grey, mask_format, ten_bit_grey, wide_bytes};

fn run(args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    let backend = backend();
    let output = mask_format(format.width, format.height);
    let mask = built_mask(
        &backend,
        "edges/retinex",
        args,
        &[StreamFormat::Frames(format)],
        output,
    );

    backend.build(SelectFrames { mask, format, frames })
}

fn code_values(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample)).collect()
}

fn wide_code_values(samples: &[u16]) -> Vec<f32> {
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

/// The blur radius for `sigma`, `ceil(6 * sigma + 1) / 2`, as vs-jetpack rounds it.
fn retinex_blur_radius(sigma: f32) -> u32 {
    let reach = (6.0 * sigma + 1.0).ceil();

    reach as u32 / 2
}

fn open_limits(peak: f32) -> EdgeLimits {
    EdgeLimits {
        multi: 1.0,
        low: 0.0,
        high: peak,
        peak,
    }
}

/// `edges/retinex`'s own arguments, mirroring its defaults.
struct RetinexArgs {
    sigma: f32,
    sigmas: Vec<f32>,
    lower_thr: f32,
    upper_thr: f32,
    brz: f32,
}

impl Default for RetinexArgs {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            sigmas: vec![50.0, 200.0, 350.0],
            lower_thr: 0.001,
            upper_thr: 0.005,
            brz: 0.122,
        }
    }
}

/// Runs the stage's whole pipeline on the CPU, mirroring its kernels in order.
///
/// `retinex` finds a normalised response of the luma, an optional blur softens it before its edges
/// are found, and a join of Prewitt over the response with Kirsch over the luma gives the final
/// magnitude, cut at `peak` and optionally binarized.
fn reference_retinex_mask(luma: &[f32], width: u32, height: u32, args: &RetinexArgs, peak: f32) -> Vec<u8> {
    let sigmas = args.sigmas.clone();
    let settings = RetinexSettings::new(sigmas, args.lower_thr, args.upper_thr, peak);
    let retinex_response = reference_retinex(luma, width, height, &settings);

    let lines_source = if args.sigma > 0.0 {
        let radius = retinex_blur_radius(args.sigma);
        let weights = gaussian_weights(args.sigma, radius);
        reference_blur(&retinex_response, width, height, &weights)
    } else {
        retinex_response
    };

    let limits = open_limits(peak);
    let prewitt = reference_edges(&lines_source, width, height, Operator::Prewitt, limits);
    let lines = reference_extreme(&prewitt, width, height, Extreme::Minimum, Shape::Corners);

    let kirsch = reference_edges(luma, width, height, Operator::Kirsch, limits);

    let joined: Vec<f32> = lines
        .iter()
        .zip(&kirsch)
        .map(|(line, corner)| (line + corner).min(peak))
        .collect();

    let joined = if args.brz > 0.0 {
        let threshold = (args.brz * peak).round();
        reference_binarize(&joined, threshold, peak)
    } else {
        joined
    };

    store(&joined, peak)
}

/// A dark left half and a bright right half, each with a couple of faint lines.
fn retinex_scene(width: u32, height: u32) -> Vec<u8> {
    let half = width / 2;

    (0..height)
        .flat_map(|_row| {
            (0..width).map(move |column| {
                let bright = column >= half;
                let dark_line = (8..10).contains(&column) || (16..18).contains(&column);
                let bright_line = (half + 8..half + 10).contains(&column);
                let line = dark_line || bright_line;

                match (bright, line) {
                    (false, false) => 20,
                    (false, true) => 30,
                    (true, false) => 200,
                    (true, true) => 170,
                }
            })
        })
        .collect()
}

fn args_json(args: &RetinexArgs) -> Value {
    json!({
        "sigma": args.sigma,
        "sigmas": args.sigmas,
        "lower_thr": args.lower_thr,
        "upper_thr": args.upper_thr,
        "brz": args.brz,
    })
}

#[test]
fn the_retinex_mask_matches_the_reference() {
    let width = 48;
    let height = 40;
    let peak = 255.0;
    let luma = retinex_scene(width, height);
    let luma_values = code_values(&luma);
    let args = RetinexArgs {
        brz: 0.0,
        ..Default::default()
    };
    let expected = reference_retinex_mask(&luma_values, width, height, &args, peak);

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let request = args_json(&args);
    let masks = run(request, format, frames);

    let total = expected.len();
    let close_differing = masks[0]
        .iter()
        .zip(&expected)
        .filter(|(got, want)| got.abs_diff(**want) > 1)
        .count();
    let far_differing = masks[0]
        .iter()
        .zip(&expected)
        .filter(|(got, want)| got.abs_diff(**want) > 3)
        .count();
    let allowed = (total as f32 * 0.01).floor() as usize;

    assert!(
        close_differing <= allowed,
        "{close_differing} of {total} samples differ by more than 1 level, allowing {allowed}"
    );
    assert_eq!(
        far_differing, 0,
        "a sample differed from the reference by more than 3 levels"
    );
}

#[test]
fn binarizing_matches_the_unbinarized_mask() {
    let width = 48;
    let height = 40;
    let peak = 255.0;
    let luma = retinex_scene(width, height);
    let luma_values = code_values(&luma);
    let threshold = (0.122f32 * 255.0).round();

    let unbinarized_args = RetinexArgs {
        brz: 0.0,
        ..Default::default()
    };
    let reference_unbinarized = reference_retinex_mask(&luma_values, width, height, &unbinarized_args, peak);

    let above_count = reference_unbinarized
        .iter()
        .filter(|value| f32::from(**value) > threshold + 1.0)
        .count();
    let below_count = reference_unbinarized
        .iter()
        .filter(|value| f32::from(**value) < threshold - 1.0)
        .count();
    let above_share = above_count as f32 / reference_unbinarized.len() as f32;
    let below_share = below_count as f32 / reference_unbinarized.len() as f32;
    assert!(
        above_share >= 0.05,
        "too few samples clearly above the threshold: {above_share}"
    );
    assert!(
        below_share >= 0.05,
        "too few samples clearly below the threshold: {below_share}"
    );

    let binarized_args = RetinexArgs::default();
    let unbinarized_format = grey(width, height);
    let binarized_format = grey(width, height);
    let unbinarized_frames = vec![vec![luma.clone()]];
    let binarized_frames = vec![vec![luma]];
    let unbinarized_request = args_json(&unbinarized_args);
    let binarized_request = args_json(&binarized_args);
    let unbinarized_masks = run(unbinarized_request, unbinarized_format, unbinarized_frames);
    let binarized_masks = run(binarized_request, binarized_format, binarized_frames);

    for (index, binarized_sample) in binarized_masks[0].iter().enumerate() {
        assert!(
            *binarized_sample == 0 || *binarized_sample == 255,
            "sample {index} was not binarized, got {binarized_sample}"
        );
    }

    for (index, unbinarized_sample) in unbinarized_masks[0].iter().enumerate() {
        let distance = f32::from(*unbinarized_sample) - threshold;
        if distance.abs() <= 1.0 {
            continue;
        }

        let expected_binarized: u8 = if distance > 0.0 { 255 } else { 0 };
        assert_eq!(
            binarized_masks[0][index], expected_binarized,
            "sample {index} at {unbinarized_sample} did not binarize against {threshold}"
        );
    }
}

#[test]
fn retinex_finds_a_faint_line_in_the_dark_that_plain_edges_miss() {
    let width = 32;
    let height = 32;
    let peak = 255.0;

    let mut luma = vec![12u8; (width * height) as usize];
    for row in 0..height {
        for column in 15..18u32 {
            luma[(row * width + column) as usize] = 13;
        }
    }

    let luma_values = code_values(&luma);
    let limits = EdgeLimits {
        multi: 1.0,
        low: (0.122f32 * 255.0).round(),
        high: (0.122f32 * 255.0).round(),
        peak,
    };
    let plain_prewitt = reference_edges(&luma_values, width, height, Operator::Prewitt, limits);
    let plain_kirsch = reference_edges(&luma_values, width, height, Operator::Kirsch, limits);

    let threshold = (0.122f32 * 255.0).round();
    let plain_prewitt_max = plain_prewitt.iter().copied().fold(0.0f32, f32::max);
    let plain_kirsch_max = plain_kirsch.iter().copied().fold(0.0f32, f32::max);
    println!("plain prewitt threshold {threshold}, max magnitude {plain_prewitt_max}");
    println!("plain kirsch threshold {threshold}, max magnitude {plain_kirsch_max}");

    assert!(
        plain_prewitt.iter().all(|value| *value == 0.0),
        "plain prewitt marked something at the 1-code step"
    );
    assert!(
        plain_kirsch.iter().all(|value| *value == 0.0),
        "plain kirsch marked something at the 1-code step"
    );

    let args = RetinexArgs::default();
    let reference_mask = reference_retinex_mask(&luma_values, width, height, &args, peak);

    for row in 8..24u32 {
        let mut found = false;
        for column in 13..20u32 {
            let index = (row * width + column) as usize;
            let value = reference_mask[index];
            println!("row {row} column {column} value {value}");
            if value == 255 {
                found = true;
            }
        }

        assert!(found, "row {row} had no 255 sample within 2 columns of the line");
    }

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let request = args_json(&args);
    let masks = run(request, format, frames);

    for row in 8..24u32 {
        let found = (13..20u32).any(|column| {
            let index = (row * width + column) as usize;
            masks[0][index] == 255
        });

        assert!(
            found,
            "device row {row} had no 255 sample within 2 columns of the line"
        );
    }
}

#[test]
fn a_flat_frame_gives_an_empty_mask() {
    let width = 16;
    let height = 16;
    let luma = vec![128u8; (width * height) as usize];

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let default_args = RetinexArgs::default();
    let request = args_json(&default_args);
    let masks = run(request, format, frames);

    assert!(
        masks[0].iter().all(|sample| *sample == 0),
        "a flat frame found an edge"
    );
}

#[test]
fn ten_bit_input_matches_the_reference() {
    let width = 48;
    let height = 40;
    let peak = 1023.0;
    let scaled_ratio = 4u16;
    let luma_8bit = retinex_scene(width, height);
    let luma: Vec<u16> = luma_8bit
        .iter()
        .map(|sample| u16::from(*sample) * scaled_ratio)
        .collect();
    let luma_values = wide_code_values(&luma);

    let args = RetinexArgs {
        brz: 0.0,
        ..Default::default()
    };
    let expected = reference_retinex_mask(&luma_values, width, height, &args, peak);

    let format = ten_bit_grey(width, height);
    let wide_plane = wide_bytes(&luma);
    let frames = vec![vec![wide_plane]];
    let request = args_json(&args);
    let masks = run(request, format, frames);

    let total = expected.len();
    let close_differing = masks[0]
        .iter()
        .zip(&expected)
        .filter(|(got, want)| got.abs_diff(**want) > 1)
        .count();
    let far_differing = masks[0]
        .iter()
        .zip(&expected)
        .filter(|(got, want)| got.abs_diff(**want) > 3)
        .count();
    let allowed = (total as f32 * 0.01).floor() as usize;

    assert!(
        close_differing <= allowed,
        "{close_differing} of {total} samples differ by more than 1 level, allowing {allowed}"
    );
    assert_eq!(
        far_differing, 0,
        "a sample differed from the reference by more than 3 levels"
    );
}
