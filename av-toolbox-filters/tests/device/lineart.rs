use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::lineart::DarkenSettings;
use av_toolbox_cubek::testing::{reference_darken, reference_narrow, reference_vinverse};
use av_toolbox_gpu::{Channel, ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{
    frame,
    from_bytes,
    run_masked,
    run_once,
    shared_planes,
    split_mask,
    to_bytes,
    uniform_mask,
};

const STAGE: &str = "lineart/vinverse";

const CONTRA_STR: f32 = 2.7;

const SCL: f32 = 0.25;

const WIDTH: u32 = 64;

const HEIGHT: u32 = 48;

/// The most a sample may move in `amnt_is_scaled_with_the_bit_depth`, on the 8-bit scale.
const AMNT: f32 = 20.0;

/// The samples of one plane that sit a code from the reference. None is safe, as the closest
/// reference value sits 0.00625 codes from a rounding boundary at 10 bits (0.0406 at 8 bits), far
/// beyond the 2.4e-5 of f32 error.
const OFF_BY_ONE_SAMPLES: usize = 0;

fn ten_bit() -> SampleFormat {
    SampleFormat::U16 { depth: Depth::TEN }
}

fn yuv420(sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(WIDTH, HEIGHT, layout, sample, ChromaLocation::Left)
}

fn element_of(sample: SampleFormat) -> Element {
    match sample {
        SampleFormat::U8 => Element::U8,
        _ => Element::U16,
    }
}

fn peak_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U8 => 255,
        _ => 1023,
    }
}

/// A scene with a comb on every row, a ramp down the plane, detail across it and some noise, shifted
/// per plane and scaled to the sample's depth.
fn combed_scene(format: &FrameFormat) -> Vec<Vec<u16>> {
    let peak = peak_of(format.sample);

    format
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let column = index % width;
                    let row = index / width;
                    let comb = (row % 2) * 40;
                    let ramp = row * 2;
                    let detail = (column * 5) % 24;
                    let noise = (index * 37 + plane as u32 * 11) % 9;
                    let value = (30 + comb + ramp + detail + noise) * (peak + 1) / 256;

                    value.min(peak) as u16
                })
                .collect()
        })
        .collect()
}

fn encoded(format: &FrameFormat, values: &[Vec<u16>]) -> Vec<Vec<u8>> {
    let element = element_of(format.sample);

    values.iter().map(|plane| to_bytes(plane, element)).collect()
}

/// The reference's answer for one plane, rounded to codes.
fn reference_codes(values: &[u16], width: u32, height: u32, peak: u32, amnt: Option<f32>) -> Vec<u16> {
    let widened: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();
    let filtered = reference_vinverse(&widened, width, height, CONTRA_STR, SCL, amnt);

    reference_narrow(&filtered, peak as f32)
}

/// How far a plane is from the reference.
struct Agreement {
    largest: u16,
    off_by_one: usize,
}

fn agreement(actual: &[u16], expected: &[u16]) -> Agreement {
    assert_eq!(actual.len(), expected.len());

    let differences: Vec<u16> = actual
        .iter()
        .zip(expected)
        .map(|(found, wanted)| found.abs_diff(*wanted))
        .collect();
    let largest = differences.iter().copied().max().unwrap_or(0);
    let off_by_one = differences.iter().filter(|difference| **difference == 1).count();

    Agreement { largest, off_by_one }
}

/// Checks every plane of `actual` is within a code of the reference, with `OFF_BY_ONE_SAMPLES`
/// samples a code away.
fn assert_matches_reference(
    format: &FrameFormat,
    values: &[Vec<u16>],
    actual: &[Vec<u8>],
    amnt: Option<f32>,
    label: &str,
) {
    let element = element_of(format.sample);
    let peak = peak_of(format.sample);

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = from_bytes(&actual[plane], element);
        let expected = reference_codes(&values[plane], width, height, peak, amnt);
        let result = agreement(&found, &expected);

        println!(
            "{label} {channel}: largest {}, {} of {} a code away",
            result.largest,
            result.off_by_one,
            found.len(),
        );
        assert!(
            result.largest <= 1,
            "{label} {channel} differs by {}",
            result.largest
        );
        assert_eq!(
            result.off_by_one, OFF_BY_ONE_SAMPLES,
            "{label} {channel} has samples a code away"
        );
    }
}

/// The samples of `plane` that differ from `other`.
fn changed(plane: &[u16], other: &[u16]) -> usize {
    plane
        .iter()
        .zip(other)
        .filter(|(left, right)| left != right)
        .count()
}

#[test]
fn vinverse_matches_the_reference() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);

        let filtered = run_once(STAGE, format, format, json!({}), &planes);

        let label = format!("{sample}");
        assert_matches_reference(&format, &values, &filtered, None, &label);
    }
}

#[test]
fn amnt_is_scaled_with_the_bit_depth() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let scale = (peak_of(sample) + 1) as f32 / 256.0;
        let limit = AMNT * scale;
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);

        let filtered = run_once(STAGE, format, format, json!({ "amnt": AMNT }), &planes);

        let label = format!("{sample} amnt");
        assert_matches_reference(&format, &values, &filtered, Some(limit), &label);

        let mut clamped = 0usize;
        let mut unclamped = 0usize;
        for (channel, source) in format.channels().iter().zip(&values) {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            let widened: Vec<f32> = source.iter().map(|value| f32::from(*value)).collect();
            let free = reference_vinverse(&widened, width, height, CONTRA_STR, SCL, None);

            for (result, value) in free.iter().zip(&widened) {
                if (result - value).abs() > limit {
                    clamped += 1;
                } else {
                    unclamped += 1;
                }
            }
        }

        println!("{label}: {clamped} clamped, {unclamped} unclamped");
        assert!(clamped > 0, "{label} never reaches its limit");
        assert!(unclamped > 0, "{label} is limited everywhere");
    }
}

#[test]
fn vinverse_removes_a_comb_on_the_device() {
    let format = yuv420(SampleFormat::U8);
    let planes: Vec<Vec<u8>> = format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| if (index / width) % 2 == 0 { 100 } else { 140 })
                .collect()
        })
        .collect();

    let filtered = run_once(STAGE, format, format, json!({}), &planes);

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let samples = &filtered[plane];
        let mut largest = 0u8;

        for row in 1..height - 2 {
            let this = samples[(row * width) as usize];
            let next = samples[((row + 1) * width) as usize];
            largest = largest.max(this.abs_diff(next));
        }

        println!("{channel}: largest row to row step {largest}");
        assert!(largest < 2, "{channel} still combs by {largest}");
    }
}

#[test]
fn a_flat_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
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

        let filtered = run_once(STAGE, format, format, json!({}), &planes);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn a_zero_mask_hands_back_the_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);
        let (mask, mask_chroma) = uniform_mask(&format, 0);

        let filtered = run_masked(STAGE, format, json!({}), &planes, mask, mask_chroma);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn a_full_mask_hands_back_the_filtered_frame() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);
        let (mask, mask_chroma) = uniform_mask(&format, 255);

        let filtered = run_masked(STAGE, format, json!({}), &planes, mask, mask_chroma);

        let label = format!("{sample} full mask");
        assert_matches_reference(&format, &values, &filtered, None, &label);
    }
}

#[test]
fn a_mask_merges_chroma_through_its_chroma_plane() {
    let format = yuv420(SampleFormat::U8);
    let values = combed_scene(&format);
    let planes = encoded(&format, &values);
    let (mask, mask_chroma) = split_mask(&format);

    let filtered = run_masked(STAGE, format, json!({}), &planes, mask, mask_chroma);

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = from_bytes(&filtered[plane], Element::U8);
        let source = &values[plane];
        let mut left_changed = 0usize;

        for row in 0..height {
            for column in 0..width {
                let index = (row * width + column) as usize;
                if column < width / 2 {
                    left_changed += usize::from(found[index] != source[index]);
                } else {
                    assert_eq!(found[index], source[index], "{channel} at {column},{row}");
                }
            }
        }

        let left_samples = (width / 2 * height) as usize;
        println!("{channel}: {left_changed} of {left_samples} left samples changed");
        assert!(left_changed > 0, "{channel} left half is untouched");

        let total_changed = changed(&found, source);
        assert_eq!(
            total_changed, left_changed,
            "{channel} changed samples on the right"
        );
    }
}

#[test]
fn unselected_planes_share_the_input_buffers() {
    let format = yuv420(SampleFormat::U8);

    let shared = shared_planes(STAGE, format, json!({ "planes": ["y"] }));

    assert_eq!(shared, vec![false, true, true]);
}

const DARKEN: &str = "lineart/darken";

/// The darken settings for `sample` at the defaults, with `thinning` given on its own scale.
///
/// Written out here so the device is checked against values the stage did not produce.
fn darken_settings_for(sample: SampleFormat, thinning: f32) -> DarkenSettings {
    let peak = peak_of(sample) as f32;
    let scale = (peak + 1.0) / 256.0;

    DarkenSettings {
        s: 48.0 / 128.0,
        limit: peak / 6.0,
        cap: 191.0 * scale,
        thr: 4.0 * scale,
        t: thinning / 16.0,
        n: (peak + 1.0) / 2.0,
        peak,
    }
}

/// A scene of three bands of rows, scaled to the sample's depth.
///
/// - The top band is a flat 200 with 1-wide lines. Three of them sit 1 code below, on and 1 code
///   above `cap - thr`, so their closing straddles `x + thr`. The others are deeper.
/// - The middle band is hashed noise, which puts lines of every depth in every direction.
/// - The bottom band is a flat 230, above the cap, with dark 1-wide and 2-wide lines.
fn lined_scene(format: &FrameFormat) -> Vec<Vec<u16>> {
    let sample = format.sample;
    let peak = peak_of(sample);
    let scale = (peak + 1) / 256;
    let cap = 191 * scale;
    let thr = 4 * scale;
    let depths = [
        cap - thr - 1,
        cap - thr,
        cap - thr + 1,
        150 * scale,
        100 * scale,
        20 * scale,
    ];

    format
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let column = index % width;
                    let row = index / width;
                    let band = row * 3 / height;
                    let value = match band {
                        0 if column % 4 == 2 => depths[(column / 4) as usize % depths.len()],
                        0 => 200 * scale,
                        1 => {
                            let hash = (index + plane as u32 * 7).wrapping_mul(2_654_435_761) >> 20;
                            (60 + hash % 140) * scale
                        },
                        _ if column % 8 == 3 => 150 * scale,
                        _ if column % 8 == 6 || column % 8 == 7 => 100 * scale,
                        _ => 230 * scale,
                    };

                    value.min(peak) as u16
                })
                .collect()
        })
        .collect()
}

/// The distance from a rounding boundary within which an f32 result may round the other way.
///
/// It is about 16 f32 steps at 1023, generous for the dozen operations a sample goes through.
const RISK: f32 = 1e-3;

/// How close the reference's samples sit to a rounding boundary.
struct Rounding {
    /// The indices of the samples within `RISK` of one, boundary included.
    at_risk: Vec<usize>,
    /// The distance of the closest sample that is not exactly on one.
    nearest: f32,
}

/// The reference's answer for one plane as codes, and how close it sits to a rounding boundary.
fn darken_reference(
    values: &[u16],
    width: u32,
    height: u32,
    settings: &DarkenSettings,
) -> (Vec<u16>, Rounding) {
    let widened: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();
    let filtered = reference_darken(&widened, width, height, settings);
    let margins: Vec<f32> = filtered
        .iter()
        .map(|value| ((value - value.floor()) - 0.5).abs())
        .collect();
    let at_risk: Vec<usize> = margins
        .iter()
        .enumerate()
        .filter(|(_, margin)| **margin < RISK)
        .map(|(index, _)| index)
        .collect();
    let nearest = margins
        .iter()
        .copied()
        .filter(|margin| *margin > 0.0)
        .fold(f32::MAX, f32::min);
    let codes = reference_narrow(&filtered, settings.peak);

    (codes, Rounding { at_risk, nearest })
}

/// Checks every plane of `actual` is within a code of the reference.
///
/// Without thinning every value is a multiple of 1/8 and f32 holds it exactly, so no sample may sit a
/// code away. With thinning the merge divides by 255 or 1023 and the box by 9, so only the samples
/// at indices within `RISK` of a rounding boundary may, and every other sample is exact. At 8 bits
/// the closest is 0.0019 away and at 10 bits 0.0003, so 5 samples of the 10-bit luma plane are at
/// risk. Returns the count of samples a code away across the frame.
fn assert_matches_darken(
    format: &FrameFormat,
    values: &[Vec<u16>],
    actual: &[Vec<u8>],
    settings: &DarkenSettings,
    label: &str,
) -> usize {
    let element = element_of(format.sample);
    let mut total_off_by_one = 0usize;

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = from_bytes(&actual[plane], element);
        let (expected, rounding) = darken_reference(&values[plane], width, height, settings);
        let result = agreement(&found, &expected);
        let moved = changed(&found, &values[plane]);
        let allowed: &[usize] = if settings.t == 0.0 { &[] } else { &rounding.at_risk };
        let stray: Vec<usize> = (0..found.len())
            .filter(|index| found[*index] != expected[*index] && !allowed.contains(index))
            .collect();
        total_off_by_one += result.off_by_one;

        println!(
            "{label} {channel}: largest {}, {} of {} off, {} allowed, {moved} moved, nearest margin {}",
            result.largest,
            result.off_by_one,
            found.len(),
            allowed.len(),
            rounding.nearest,
        );
        assert!(moved > 0, "{label} {channel} never darkened a sample");
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
fn darken_matches_the_reference() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for thinning in [0.0f32, 64.0] {
            let format = yuv420(sample);
            let values = lined_scene(&format);
            let planes = encoded(&format, &values);
            let settings = darken_settings_for(sample, thinning);
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let filtered = run_once(DARKEN, format, format, args, &planes);

            let label = format!("{sample} thinning {thinning}");
            let off_by_one = assert_matches_darken(&format, &values, &filtered, &settings, &label);
            println!("{label}: {off_by_one} samples a code away");
        }
    }
}

/// A `yuv420` frame whose planes are `background`, with a 1-wide vertical line of `line` down the
/// middle column of the luma plane.
fn faded_line_planes(format: &FrameFormat, background: u16, line: u16) -> Vec<Vec<u8>> {
    let element = element_of(format.sample);

    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            let values: Vec<u16> = (0..width * height)
                .map(|index| {
                    let on_line = *channel == Channel::Y && index % width == width / 2;
                    if on_line { line } else { background }
                })
                .collect();

            to_bytes(&values, element)
        })
        .collect()
}

/// The luma samples of a 64x48 frame, as a plane of codes.
fn luma_codes(planes: &[Vec<u8>], sample: SampleFormat) -> Vec<u16> {
    let element = element_of(sample);

    from_bytes(&planes[0], element)
}

/// A 150 line in 200 becomes 135, a 600 line in 800 becomes 539, and with `protection = 50` 148.
#[test]
fn darken_deepens_a_faded_line_on_the_device() {
    let cases = [
        (SampleFormat::U8, 200u16, 150u16, json!({}), 135u16),
        (ten_bit(), 800, 600, json!({}), 539),
        (SampleFormat::U8, 200, 150, json!({ "protection": 50 }), 148),
    ];

    for (sample, background, line, args, expected) in cases {
        let format = yuv420(sample);
        let planes = faded_line_planes(&format, background, line);

        let filtered = run_once(DARKEN, format, format, args.clone(), &planes);

        let luma = luma_codes(&filtered, sample);
        let column = (WIDTH / 2) as usize;
        for (index, value) in luma.iter().enumerate() {
            let want = if index % WIDTH as usize == column {
                expected
            } else {
                background
            };
            assert_eq!(*value, want, "{sample} {args} at {index}");
        }
    }
}

/// Lines 1 code beyond `thr` below `cap` darken and lines on it or nearer do not.
#[test]
fn darken_switches_branches_at_the_threshold_on_the_device() {
    let cases = [
        (
            SampleFormat::U8,
            200u16,
            [(186u16, 184u16), (187, 187), (188, 188)],
        ),
        (ten_bit(), 800, [(747, 741), (748, 748), (749, 749)]),
    ];

    for (sample, background, lines) in cases {
        for (line, expected) in lines {
            let format = yuv420(sample);
            let planes = faded_line_planes(&format, background, line);

            let filtered = run_once(DARKEN, format, format, json!({}), &planes);

            let luma = luma_codes(&filtered, sample);
            let column = (WIDTH / 2) as usize;
            let found = luma[column];
            assert_eq!(found, expected, "{sample} line {line}");
        }
    }
}

#[test]
fn a_flat_darken_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for thinning in [0, 64] {
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
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let filtered = run_once(DARKEN, format, format, args, &planes);

            assert_eq!(filtered, planes, "{sample} thinning {thinning}");
        }
    }
}

#[test]
fn a_zero_mask_hands_back_the_darken_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for thinning in [0, 64] {
            let format = yuv420(sample);
            let values = lined_scene(&format);
            let planes = encoded(&format, &values);
            let (mask, mask_chroma) = uniform_mask(&format, 0);
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let filtered = run_masked(DARKEN, format, args, &planes, mask, mask_chroma);

            assert_eq!(filtered, planes, "{sample} thinning {thinning}");
        }
    }
}

#[test]
fn darken_leaves_unselected_planes_as_the_input_buffers() {
    let format = yuv420(SampleFormat::U8);

    let shared = shared_planes(DARKEN, format, json!({}));

    assert_eq!(shared, vec![false, true, true]);
}

#[test]
fn a_strength_of_zero_hands_back_the_plane() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = lined_scene(&format);
        let planes = encoded(&format, &values);

        let filtered = run_once(DARKEN, format, format, json!({ "strength": 0 }), &planes);

        assert_eq!(filtered, planes, "{sample}");
    }
}
