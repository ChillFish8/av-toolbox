use av_toolbox_cubek::lineart::ThinSettings;
use av_toolbox_cubek::testing::{reference_narrow, reference_padded_warp, reference_thin};
use av_toolbox_cubek::warp::{BlurKind, WarpParams};
use av_toolbox_gpu::{Channel, FrameFormat, SampleFormat};
use serde_json::json;

use super::{HEIGHT, Rounding, WIDTH, agreement, changed, lined_scene, luma_codes, rounding_of, yuv420};
use crate::support::{
    element_of,
    encoded,
    from_bytes,
    peak_of,
    run_masked,
    run_once,
    shared_planes,
    ten_bit,
    to_bytes,
    uniform_mask,
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
    let bits = if sample == SampleFormat::U8 { 8 } else { 10 };
    let warp = WarpParams {
        element: element_of(sample),
        bits,
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

fn widened(values: &[u16]) -> Vec<f32> {
    values.iter().map(|value| f32::from(*value)).collect()
}

/// The reference's answer for one plane as codes, and how close it sits to a rounding boundary.
fn thin_reference(values: &[u16], width: u32, height: u32, settings: &ThinSettings) -> (Vec<u16>, Rounding) {
    let input = widened(values);
    let filtered = reference_thin(&input, width, height, settings);
    let rounding = rounding_of(&filtered);
    let codes = reference_narrow(&filtered, settings.peak);

    (codes, rounding)
}

/// The samples of a plane that the warp alone changes, so the merge has something to select.
fn warp_moved_samples(values: &[u16], width: u32, height: u32, settings: &ThinSettings) -> usize {
    let input = widened(values);
    let warped = reference_padded_warp(&input, width, height, &settings.warp);
    let codes = reference_narrow(&warped, settings.peak);

    changed(&codes, values)
}

/// Checks every plane of `actual` is within a code of the reference.
///
/// The merge divides by 255 and the inflate by 8, so only the samples within `RISK` of a rounding
/// boundary may sit a code away, and every other sample is exact. Returns the count of samples a
/// code away across the frame.
fn assert_matches_thin(
    format: &FrameFormat,
    values: &[Vec<u16>],
    actual: &[Vec<u8>],
    settings: &ThinSettings,
    label: &str,
) -> usize {
    let element = element_of(format.sample);
    let mut total_off_by_one = 0usize;

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = from_bytes(&actual[plane], element);
        let (expected, rounding) = thin_reference(&values[plane], width, height, settings);
        let result = agreement(&found, &expected);
        let moved = changed(&found, &values[plane]);
        let warped = warp_moved_samples(&values[plane], width, height, settings);
        let stray: Vec<usize> = (0..found.len())
            .filter(|index| found[*index] != expected[*index] && !rounding.at_risk.contains(index))
            .collect();
        total_off_by_one += result.off_by_one;

        println!(
            "{label} {channel}: largest {}, {} of {} off, {} at risk, {moved} moved, {warped} \
             moved by the warp, nearest margin {}",
            result.largest,
            result.off_by_one,
            found.len(),
            rounding.at_risk.len(),
            rounding.nearest,
        );
        if *channel == Channel::Y {
            assert!(moved > 0, "{label} {channel} never thinned a sample");
        }

        assert!(
            result.largest <= 1,
            "{label} {channel} differs by {}",
            result.largest
        );
        assert!(
            stray.is_empty(),
            "{label} {channel} differs at {stray:?}, which do not sit near a boundary"
        );
    }

    total_off_by_one
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

                let filtered = run_once(THIN, format, format, args, &planes);

                let label = format!("{sample} mrad {radius} blur {passes} depth {depth}");
                let off_by_one = assert_matches_thin(&format, &values, &filtered, &settings, &label);
                println!("{label}: {off_by_one} samples a code away");
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
fn dip_width(codes: &[u16], level: f64) -> Option<f64> {
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
fn row_minimum(codes: &[u16]) -> u16 {
    let row = (HEIGHT / 2 * WIDTH) as usize;

    codes[row..row + WIDTH as usize]
        .iter()
        .copied()
        .min()
        .unwrap_or(0)
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

            let filtered = run_once(THIN, format, format, json!({}), &planes);

            let found = luma_codes(&filtered, sample);
            let level = f64::from(120 * scale);
            let before = dip_width(&luma, level).expect("the input has a single dip");
            let after = dip_width(&found, level).expect("the output has a single dip");
            let darkest_before = row_minimum(&luma);
            let darkest_after = row_minimum(&found);
            println!(
                "{sample} shift {shift}: half-contrast width {before:.2} before and {after:.2} after, \
                 darkest {darkest_before} before and {darkest_after} after"
            );
            assert!(
                after < before - NARROWED_BY,
                "{sample} shift {shift} did not narrow"
            );
            assert_eq!(
                darkest_after, darkest_before,
                "{sample} shift {shift} moved the darkest sample"
            );

            let threshold = default_threshold(sample);
            let settings = thin_settings_for(sample, 1, 2, 16, threshold);
            let label = format!("{sample} shift {shift}");
            assert_matches_thin(&format, &values, &filtered, &settings, &label);
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

        let filtered = run_once(THIN, format, format, args, &planes);

        let found = luma_codes(&filtered, sample);
        let peak = peak_of(sample) as f32;
        let settings = thin_settings_for(sample, 1, 1, 24, peak);
        let input = widened(&values[0]);
        let warped = reference_padded_warp(&input, WIDTH, HEIGHT, &settings.warp);
        let warped_codes = reference_narrow(&warped, settings.peak);
        let outside_indices: Vec<usize> = (0..found.len())
            .filter(|index| {
                let column = index % WIDTH as usize;
                !(MASK_FIRST..=MASK_LAST).contains(&column)
            })
            .collect();
        let stray: Vec<usize> = outside_indices
            .iter()
            .copied()
            .filter(|index| found[*index] != values[0][*index])
            .collect();
        let warp_would_move = outside_indices
            .iter()
            .filter(|index| warped_codes[**index] != values[0][**index])
            .count();
        let moved = changed(&found, &values[0]);
        println!(
            "{sample}: {} samples outside the mask, {} differ, {warp_would_move} of them the warp \
             alone would move, {moved} moved in all",
            outside_indices.len(),
            stray.len(),
        );
        assert!(stray.is_empty(), "{sample} moved {stray:?} outside the mask");
        assert!(
            warp_would_move > 0,
            "{sample} has nothing for the mask to hold back"
        );
        assert!(moved > 0, "{sample} never moved a sample inside the mask");
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

        let (zero_mask, zero_chroma) = uniform_mask(&format, 0);
        let line_zero = run_masked(
            THIN,
            format,
            args.clone(),
            &planes,
            zero_mask.clone(),
            zero_chroma.clone(),
        );
        let scene_zero = run_masked(THIN, format, args.clone(), &scene_planes, zero_mask, zero_chroma);

        assert_eq!(line_zero, planes, "{sample} lines under a mask of 0");
        assert_eq!(scene_zero, scene_planes, "{sample} scene under a mask of 0");

        let (full_mask, full_chroma) = uniform_mask(&format, 255);
        let line_full = run_masked(THIN, format, args, &planes, full_mask, full_chroma);

        let peak = peak_of(sample) as f32;
        let settings = thin_settings_for(sample, 1, 1, 24, peak);
        let found = luma_codes(&line_full, sample);
        let input = widened(&values[0]);
        let warped = reference_padded_warp(&input, WIDTH, HEIGHT, &settings.warp);
        let warped_codes = reference_narrow(&warped, settings.peak);
        let (built_in, _) = thin_reference(&values[0], WIDTH, HEIGHT, &settings);
        let held_back = (0..found.len())
            .filter(|index| {
                warped_codes[*index] != values[0][*index] && built_in[*index] == values[0][*index]
            })
            .count();
        println!("{sample}: {held_back} samples the built-in mask would have held back");
        assert!(
            held_back > 0,
            "{sample} has nothing the built-in mask would have held back"
        );
        assert_eq!(found, warped_codes, "{sample} under a mask of 255");
    }
}

#[test]
fn a_flat_thin_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for radius in [0, 1] {
            let format = yuv420(sample);
            let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
            let element = element_of(sample);
            let planes: Vec<Vec<u8>> = format
                .channels()
                .iter()
                .map(|channel| {
                    let samples = format.channel_samples(*channel).expect("a listed channel");
                    to_bytes(&vec![flat; samples], element)
                })
                .collect();
            let args = json!({ "mrad": radius, "planes": ["y", "u", "v"] });

            let filtered = run_once(THIN, format, format, args, &planes);

            assert_eq!(filtered, planes, "{sample} mrad {radius}");
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

            let filtered = run_once(THIN, format, format, args, &planes);

            let found = luma_codes(&filtered, sample);
            let moved = changed(&found, &values[0]);
            println!("{sample} threshold {threshold}: {moved} samples moved");
            if offset > 0.0 {
                assert_eq!(filtered, planes, "{sample} threshold {threshold}");
            } else {
                assert_matches_thin(
                    &format,
                    &values,
                    &filtered,
                    &settings,
                    &format!("{sample} {threshold}"),
                );
            }
        }
    }
}

#[test]
fn thin_leaves_unselected_planes_as_the_input_buffers() {
    let format = yuv420(SampleFormat::U8);

    let shared = shared_planes(THIN, format, json!({}));

    assert_eq!(shared, vec![false, true, true]);
}

/// The split line `40 140 140 40` that a warp can make of a plain line has two dips, not one.
#[test]
fn a_split_line_has_no_single_dip() {
    let row = [200u16, 200, 40, 140, 140, 40, 200, 200];
    let mut codes = vec![200u16; (WIDTH * HEIGHT) as usize];
    let start = (HEIGHT / 2 * WIDTH) as usize + 28;
    codes[start..start + row.len()].copy_from_slice(&row);

    let width = dip_width(&codes, 120.0);

    assert_eq!(width, None);
}
