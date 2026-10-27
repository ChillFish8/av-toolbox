use av_toolbox_cubek::lineart::ThinSettings;
use av_toolbox_cubek::testing::{reference_padded_warp, reference_thin};
use av_toolbox_cubek::warp::{BlurKind, WarpParams};
use av_toolbox_gpu::{Channel, FrameFormat, SampleFormat};
use serde_json::json;

use super::{
    HEIGHT,
    WIDTH,
    assert_f32_matches_sixteen_bit,
    assert_matches,
    assert_unchanged,
    changed,
    flat_values,
    lined_scene,
    moved,
    peak_of,
    run_codes,
    run_masked_codes,
    sixteen_bit,
    yuv420,
};
use crate::support::{
    ROUND_TRIP,
    code_bits,
    element_of,
    encoded,
    shared_planes,
    ten_bit,
    uniform_mask,
    widened,
};

const THIN: &str = "lineart/thin";

/// The mask threshold the default `mthr` of 0.08 gives, `round(0.08 * peak)`, worked by hand as
/// `20.4` at 8 bits and `81.84` at 10 bits.
fn default_threshold(sample: SampleFormat) -> f32 {
    match sample {
        SampleFormat::U8 => 20.0,
        _ => 82.0,
    }
}

/// The thin settings for `sample`, with the warp's `passes` and `depth` and the mask's `radius`
/// and `threshold` given outright.
///
/// Written out here so the device is checked against values the stage did not produce.
fn thin_settings_for(
    sample: SampleFormat,
    radius: u32,
    passes: u32,
    depth: i32,
    threshold: f32,
) -> ThinSettings {
    let warp = WarpParams {
        element: element_of(sample),
        bits: code_bits(sample),
        thresh: 128,
        blur: BlurKind::R6,
        passes,
        depth,
    };

    ThinSettings {
        warp,
        threshold,
        radius,
        peak: peak_of(sample) as f32,
    }
}

/// The reference's answer for one plane as codes.
fn thin_reference(values: &[u16], width: u32, height: u32, settings: &ThinSettings) -> Vec<f32> {
    let input = widened(values);

    reference_thin(&input, width, height, settings)
}

/// The reference warp alone on one plane, as codes.
fn warp_reference(values: &[u16], width: u32, height: u32, settings: &ThinSettings) -> Vec<f32> {
    let input = widened(values);

    reference_padded_warp(&input, width, height, &settings.warp)
}

/// Checks every plane of `found` matches the reference, and that the luma thinned a sample.
fn assert_matches_thin(
    format: &FrameFormat,
    values: &[Vec<u16>],
    found: &[Vec<f32>],
    settings: &ThinSettings,
    label: &str,
) {
    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let expected = thin_reference(&values[plane], width, height, settings);
        let warped = warp_reference(&values[plane], width, height, settings);
        let thinned = changed(&found[plane], &values[plane]);
        let warp_moved = changed(&warped, &values[plane]);
        let plane_label = format!("{label} {channel}");

        println!("{plane_label}: {thinned} moved, {warp_moved} moved by the warp");
        if *channel == Channel::Y {
            assert!(thinned > 0, "{plane_label} never thinned a sample");
        }

        assert_matches(&found[plane], &expected, &plane_label);
    }
}

#[test]
fn thin_matches_the_reference() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for radius in [0u32, 2] {
            for (passes, depth) in [(2u32, 16i32), (1, 24)] {
                let format = yuv420(sample);
                let values = lined_scene(&format);
                let planes = encoded(&format, &values);
                let threshold = default_threshold(sample);
                let settings = thin_settings_for(sample, radius, passes, depth, threshold);
                let args = json!({
                    "mrad": radius,
                    "blur": passes,
                    "depth": depth,
                    "planes": ["y", "u", "v"],
                });

                let found = run_codes(THIN, format, args, &planes);

                let label = format!("{sample} mrad {radius} blur {passes} depth {depth}");
                assert_matches_thin(&format, &values, &found, &settings, &label);
            }
        }
    }
}

/// How wide the lines of `four_sample_lines` are.
const LINE_WIDTH: usize = 4;

/// The least the half-contrast width of an anti-aliased line must shrink by, in samples. The
/// smallest measured shrink is 0.17.
const NARROWED_BY: f64 = 0.1;

/// A `yuv420` frame whose luma is `background` with a `LINE_WIDTH` wide vertical line of each value in
/// `lines` at its first column, and whose chroma is flat.
fn four_sample_lines(format: &FrameFormat, background: u16, lines: &[(usize, u16)]) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let column = (index % width) as usize;
                    let line = lines
                        .iter()
                        .find(|(start, _)| column >= *start && column < *start + LINE_WIDTH);
                    match line {
                        Some((_, value)) if *channel == Channel::Y => *value,
                        _ => background,
                    }
                })
                .collect()
        })
        .collect()
}

/// A luma plane of `background` with a line of `line` down its middle, centred `shift` samples right of
/// column 32. The line is 3 samples wide at its darkest, with raised cosine shoulders 1.5 samples wide.
fn soft_line_luma(background: u16, line: u16, shift: f64) -> Vec<u16> {
    let background = f64::from(background);
    let depth = background - f64::from(line);
    let centre = 32.0 + shift;

    (0..WIDTH * HEIGHT)
        .map(|index| {
            let distance = (f64::from(index % WIDTH) - centre).abs();
            let fade = ((distance - 1.5) / 1.5).clamp(0.0, 1.0);
            let share = 0.5 * (1.0 + (std::f64::consts::PI * fade).cos());

            (background - depth * share).round() as u16
        })
        .collect()
}

/// The `yuv420` planes of `luma` over flat chroma of `flat`.
fn luma_over_flat_chroma(format: &FrameFormat, luma: &[u16], flat: u16) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            if *channel == Channel::Y {
                luma.to_vec()
            } else {
                vec![flat; samples]
            }
        })
        .collect()
}

/// The width of the single dip in the middle row of `codes` below `level`, with its two edges
/// interpolated between the samples either side of the crossing. A row with any other number of
/// crossings has no single dip and gives `None`.
fn dip_width(codes: &[f32], level: f64) -> Option<f64> {
    let row = (HEIGHT / 2 * WIDTH) as usize;
    let values: Vec<f64> = codes[row..row + WIDTH as usize]
        .iter()
        .map(|code| f64::from(*code))
        .collect();
    let mut falling = Vec::new();
    let mut rising = Vec::new();

    for column in 1..values.len() {
        let before = values[column - 1];
        let after = values[column];
        if before >= level && after < level {
            falling.push(column as f64 - 1.0 + (before - level) / (before - after));
        }

        if before < level && after >= level {
            rising.push(column as f64 - 1.0 + (level - before) / (after - before));
        }
    }

    if falling.len() != 1 || rising.len() != 1 {
        return None;
    }

    Some(rising[0] - falling[0])
}

/// The lowest sample of the middle row of `codes`.
fn row_minimum(codes: &[f32]) -> f32 {
    let row = (HEIGHT / 2 * WIDTH) as usize;

    codes[row..row + WIDTH as usize]
        .iter()
        .copied()
        .fold(f32::MAX, f32::min)
}

/// An anti-aliased line of 40 in 200 has a narrower half-contrast width after the default warp, with
/// its darkest sample unchanged.
#[test]
fn thin_narrows_an_anti_aliased_line() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for shift in [0.0f64, 0.5] {
            let scale = peak_of(sample) as u16 / 255;
            let format = yuv420(sample);
            let luma = soft_line_luma(200 * scale, 40 * scale, shift);
            let values = luma_over_flat_chroma(&format, &luma, 200 * scale);
            let planes = encoded(&format, &values);

            let found = run_codes(THIN, format, json!({}), &planes);

            let codes = widened(&luma);
            let level = f64::from(120 * scale);
            let before = dip_width(&codes, level).expect("the input has a single dip");
            let after = dip_width(&found[0], level).expect("the output has a single dip");
            let darkest_before = row_minimum(&codes);
            let darkest_after = row_minimum(&found[0]);
            println!(
                "{sample} shift {shift}: half-contrast width {before:.2} before and {after:.2} after, \
                 darkest {darkest_before} before and {darkest_after} after"
            );
            assert!(
                after < before - NARROWED_BY,
                "{sample} shift {shift} did not narrow"
            );
            assert!(
                (darkest_after - darkest_before).abs() <= ROUND_TRIP,
                "{sample} shift {shift} moved the darkest sample"
            );

            let threshold = default_threshold(sample);
            let settings = thin_settings_for(sample, 1, 2, 16, threshold);
            let label = format!("{sample} shift {shift}");
            assert_matches_thin(&format, &values, &found, &settings, &label);
        }
    }
}

/// The columns the strong line at 10 to 13 puts under its mask. Prewitt fires on 9, 10, 13 and 14,
/// `mrad = 1` widens that to 8 to 15 and the inflate to 7 to 16.
const MASK_FIRST: usize = 7;

const MASK_LAST: usize = 16;

/// At `mthr = 1` only the strong line is masked, so the weak line keeps its codes.
#[test]
fn thin_leaves_samples_outside_its_mask_untouched() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let scale = peak_of(sample) as u16 / 255;
        let format = yuv420(sample);
        let lines = [(10, 40 * scale), (40, 130 * scale)];
        let values = four_sample_lines(&format, 200 * scale, &lines);
        let planes = encoded(&format, &values);
        let args = json!({ "blur": 1, "depth": 24, "mthr": 1.0 });

        let filtered = run_codes(THIN, format, args, &planes);

        let found = &filtered[0];
        let peak = peak_of(sample) as f32;
        let settings = thin_settings_for(sample, 1, 1, 24, peak);
        let warped = warp_reference(&values[0], WIDTH, HEIGHT, &settings);
        let outside_indices: Vec<usize> = (0..found.len())
            .filter(|index| {
                let column = index % WIDTH as usize;
                !(MASK_FIRST..=MASK_LAST).contains(&column)
            })
            .collect();
        let stray: Vec<usize> = outside_indices
            .iter()
            .copied()
            .filter(|index| moved(found[*index], values[0][*index]))
            .collect();
        let warp_would_move = outside_indices
            .iter()
            .filter(|index| moved(warped[**index], values[0][**index]))
            .count();
        let thinned = changed(found, &values[0]);
        println!(
            "{sample}: {} samples outside the mask, {} differ, {warp_would_move} of them the warp \
             alone would move, {thinned} moved in all",
            outside_indices.len(),
            stray.len(),
        );
        assert!(stray.is_empty(), "{sample} moved {stray:?} outside the mask");
        assert!(
            warp_would_move > 0,
            "{sample} has nothing for the mask to hold back"
        );
        assert!(thinned > 0, "{sample} never moved a sample inside the mask");
    }
}

/// A mask of 0 hands back the input and a mask of 255 the warped plane, weak line included.
#[test]
fn a_bound_mask_replaces_the_built_in_one() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let scale = peak_of(sample) as u16 / 255;
        let format = yuv420(sample);
        let lines = [(10, 40 * scale), (40, 130 * scale)];
        let values = four_sample_lines(&format, 200 * scale, &lines);
        let planes = encoded(&format, &values);
        let scene = lined_scene(&format);
        let scene_planes = encoded(&format, &scene);
        let args = json!({ "blur": 1, "depth": 24, "mthr": 1.0, "planes": ["y", "u", "v"] });

        let zero_mask = uniform_mask(&format, 0);
        let line_zero = run_masked_codes(THIN, format, args.clone(), &planes, zero_mask.clone());
        let scene_zero = run_masked_codes(THIN, format, args.clone(), &scene_planes, zero_mask);

        let lines_label = format!("{sample} lines under a mask of 0");
        let scene_label = format!("{sample} scene under a mask of 0");
        assert_unchanged(&line_zero, &values, &lines_label);
        assert_unchanged(&scene_zero, &scene, &scene_label);

        let full_mask = uniform_mask(&format, 255);
        let line_full = run_masked_codes(THIN, format, args, &planes, full_mask);

        let peak = peak_of(sample) as f32;
        let settings = thin_settings_for(sample, 1, 1, 24, peak);
        let found = &line_full[0];
        let warped = warp_reference(&values[0], WIDTH, HEIGHT, &settings);
        let built_in = thin_reference(&values[0], WIDTH, HEIGHT, &settings);
        let held_back = (0..found.len())
            .filter(|index| {
                let code = values[0][*index];
                moved(warped[*index], code) && !moved(built_in[*index], code)
            })
            .count();
        println!("{sample}: {held_back} samples the built-in mask would have held back");
        assert!(
            held_back > 0,
            "{sample} has nothing the built-in mask would have held back"
        );
        let full_label = format!("{sample} under a mask of 255");
        assert_matches(found, &warped, &full_label);
    }
}

#[test]
fn a_flat_thin_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for radius in [0, 1] {
            let format = yuv420(sample);
            let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
            let values = flat_values(&format, flat);
            let planes = encoded(&format, &values);
            let args = json!({ "mrad": radius, "planes": ["y", "u", "v"] });

            let found = run_codes(THIN, format, args, &planes);

            let label = format!("{sample} mrad {radius}");
            assert_unchanged(&found, &values, &label);
        }
    }
}

/// A plane of 200 that falls 10 a column down to 40 in the middle and rises back, scaled to the
/// sample's depth. Its Prewitt magnitude is `3 * 20 = 60` on every column of the slopes (240 at 10
/// bits) and less at the corners.
fn ramp_luma(sample: SampleFormat) -> Vec<u16> {
    let scale = peak_of(sample) as u16 / 255;

    (0..WIDTH * HEIGHT)
        .map(|index| {
            let column = (index % WIDTH) as u16;
            let distance = column.abs_diff(32);

            (40 + 10 * distance).min(200) * scale
        })
        .collect()
}

/// Slopes of magnitude 60 (240 at 10 bits) are lines at a threshold of 59 or 60, and not at 61.
#[test]
fn thin_switches_at_the_mask_threshold_on_the_device() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let peak = peak_of(sample) as f32;
        let magnitude = if sample == SampleFormat::U8 {
            60.0f32
        } else {
            240.0
        };
        let format = yuv420(sample);
        let luma = ramp_luma(sample);
        let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
        let values: Vec<Vec<u16>> = format
            .channels()
            .iter()
            .map(|channel| {
                let samples = format.channel_samples(*channel).expect("a listed channel");
                if *channel == Channel::Y {
                    luma.clone()
                } else {
                    vec![flat; samples]
                }
            })
            .collect();
        let planes = encoded(&format, &values);

        for offset in [-1.0f32, 0.0, 1.0] {
            let threshold = magnitude + offset;
            let mthr = threshold / peak;
            let args = json!({ "blur": 1, "depth": 24, "mthr": mthr });
            let settings = thin_settings_for(sample, 1, 1, 24, threshold);

            let found = run_codes(THIN, format, args, &planes);

            let thinned = changed(&found[0], &values[0]);
            let label = format!("{sample} threshold {threshold}");
            println!("{label}: {thinned} samples moved");
            if offset > 0.0 {
                assert_unchanged(&found, &values, &label);
            } else {
                assert_matches_thin(&format, &values, &found, &settings, &label);
            }
        }
    }
}

#[test]
fn thin_leaves_unselected_planes_as_the_input_buffers() {
    let format = yuv420(SampleFormat::F32);

    let shared = shared_planes(THIN, format, json!({}));

    assert_eq!(shared, vec![false, true, true]);
}

/// The split line `40 140 140 40` that a warp can make of a plain line has two dips, not one.
#[test]
fn a_split_line_has_no_single_dip() {
    let row = [200.0f32, 200.0, 40.0, 140.0, 140.0, 40.0, 200.0, 200.0];
    let mut codes = vec![200.0f32; (WIDTH * HEIGHT) as usize];
    let start = (HEIGHT / 2 * WIDTH) as usize + 28;
    codes[start..start + row.len()].copy_from_slice(&row);

    let width = dip_width(&codes, 120.0);

    assert_eq!(width, None);
}

#[test]
fn thin_f32_matches_sixteen_bit() {
    let format = yuv420(sixteen_bit());
    let values = lined_scene(&format);
    let args = json!({ "planes": ["y", "u", "v"] });

    assert_f32_matches_sixteen_bit(THIN, args, &values, 1.0, 0.0);
}
