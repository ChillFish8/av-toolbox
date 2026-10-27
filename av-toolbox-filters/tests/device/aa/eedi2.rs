use av_toolbox_cubek::testing::{
    Eedi2Parameters,
    assert_same_plane,
    reference_eedi2_antialias,
    reference_narrow,
    transposed_plane,
};
use av_toolbox_gpu::{Channel, Depth, FrameFormat, FrameLayout, SampleFormat};
use serde_json::json;

use super::{
    DIRECTIONS,
    assert_f32_matches_sixteen_bit,
    bits_of,
    decoded,
    grey,
    largest_vertical_step,
    noisy_diagonals,
    one_sample_diagonal,
    sixteen_bit,
    yuv420,
    yuv422,
};
use crate::support::{
    ROUND_TRIP,
    assert_codes_within,
    encoded,
    run_masked_into_codes,
    run_once,
    shared_planes,
    ten_bit,
    uniform_mask,
    widened,
};

const STAGE: &str = "aa/eedi2";

const WIDTH: u32 = 48;

const HEIGHT: u32 = 32;

/// The subsampling shift `ssw` of each pass over one plane.
#[derive(Debug, Clone, Copy)]
struct Shifts {
    /// The plane's log2 horizontal subsampling.
    vertical: u32,
    /// The plane's log2 vertical subsampling, which runs across the rows of the transposed plane.
    horizontal: u32,
}

const UNSHIFTED: Shifts = Shifts {
    vertical: 0,
    horizontal: 0,
};

/// The background texture under [aliased_lines], `(7 * column + 3 * row) % modulus` scaled by `amplitude`.
#[derive(Debug, Clone, Copy)]
struct Texture {
    modulus: u32,
    amplitude: u32,
}

/// A texture faint enough that the lines alone decide the edge mask.
const FAINT: Texture = Texture {
    modulus: 11,
    amplitude: 1,
};

/// A texture busy enough that every threshold of the edge mask decides some pixels.
const BUSY: Texture = Texture {
    modulus: 7,
    amplitude: 3,
};

/// A shallow and a steep aliased line, each two samples thick, over a faint texture, scaled to `sample`'s
/// depth.
///
/// The shallow line falls one row every 2 columns and the steep line moves one column every 2 rows, so
/// each tests the reach of the pass that runs across it.
fn aliased_lines(width: u32, height: u32, sample: SampleFormat) -> Vec<u16> {
    textured_lines(width, height, sample, FAINT)
}

/// The lines of [aliased_lines] over `texture`.
fn textured_lines(width: u32, height: u32, sample: SampleFormat, texture: Texture) -> Vec<u16> {
    let bits = bits_of(sample);
    let scale = 1 << (bits - 8);

    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            let shallow_row = height / 4 + column / 2;
            let steep_column = width / 4 + row / 2;
            let on_shallow = row >= shallow_row && row < shallow_row + 2;
            let on_steep = column >= steep_column && column < steep_column + 2;
            let pattern = (column * 7 + row * 3) % texture.modulus;
            let value = match (on_shallow, on_steep) {
                (true, _) => 220,
                (false, true) => 180,
                (false, false) => 40 + pattern * texture.amplitude,
            };

            (value * scale) as u16
        })
        .collect()
}

/// The shifts of plane `index` of `format`.
fn shifts_of(format: &FrameFormat, index: usize) -> Shifts {
    let channels = format.channels();
    let Some(channel) = channels.get(index) else {
        return UNSHIFTED;
    };

    match (format.layout, channel) {
        (FrameLayout::Yuv { subsampling }, Channel::U | Channel::V) => {
            let (across, down) = subsampling.factors();

            Shifts {
                vertical: across.ilog2(),
                horizontal: down.ilog2(),
            }
        },
        _ => UNSHIFTED,
    }
}

/// The default arguments at `sample`'s depth.
fn default_parameters(sample: SampleFormat) -> Eedi2Parameters {
    Eedi2Parameters {
        bits: bits_of(sample),
        ..Eedi2Parameters::default()
    }
}

/// The arguments of `parameters` as `aa/eedi2` takes them, along `direction`.
fn stage_arguments(parameters: &Eedi2Parameters, direction: &str) -> serde_json::Value {
    json!({
        "mthresh": parameters.mthresh,
        "lthresh": parameters.lthresh,
        "vthresh": parameters.vthresh,
        "estr": parameters.estr,
        "dstr": parameters.dstr,
        "maxd": parameters.maxd,
        "nt": parameters.nt,
        "pp": parameters.pp,
        "direction": direction,
    })
}

/// Every argument moved off its default, with `pp` 0, as `aa/eedi2` takes them.
fn custom_arguments(direction: &str) -> serde_json::Value {
    let parameters = custom_parameters(SampleFormat::U8);

    stage_arguments(&parameters, direction)
}

/// Every argument moved off its default, with `pp` 0.
fn custom_parameters(sample: SampleFormat) -> Eedi2Parameters {
    Eedi2Parameters {
        mthresh: 20,
        lthresh: 15,
        vthresh: 30,
        estr: 1,
        dstr: 3,
        maxd: 12,
        nt: 10,
        pp: 0,
        bits: bits_of(sample),
        ssw: 0,
    }
}

fn vertical_reference(values: &[f32], width: u32, height: u32, base: &Eedi2Parameters, ssw: u32) -> Vec<f32> {
    let parameters = Eedi2Parameters { ssw, ..*base };

    reference_eedi2_antialias(values, width, height, &parameters)
}

fn horizontal_reference(
    values: &[f32],
    width: u32,
    height: u32,
    base: &Eedi2Parameters,
    ssw: u32,
) -> Vec<f32> {
    let swapped = transposed_plane(values, width, height);
    let filtered = vertical_reference(&swapped, height, width, base, ssw);

    transposed_plane(&filtered, height, width)
}

/// The reference's answer along `direction` with the arguments of `base`, before narrowing.
fn reference_values(
    values: &[u16],
    width: u32,
    height: u32,
    base: &Eedi2Parameters,
    shifts: Shifts,
    direction: &str,
) -> Vec<f32> {
    let widened: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();

    match direction {
        "vertical" => vertical_reference(&widened, width, height, base, shifts.vertical),
        "horizontal" => horizontal_reference(&widened, width, height, base, shifts.horizontal),
        _ => {
            let vertical = vertical_reference(&widened, width, height, base, shifts.vertical);

            horizontal_reference(&vertical, width, height, base, shifts.horizontal)
        },
    }
}

/// The reference's answer along `direction` with the arguments of `base`, narrowed to its depth.
fn expected_plane(
    values: &[u16],
    width: u32,
    height: u32,
    base: &Eedi2Parameters,
    shifts: Shifts,
    direction: &str,
) -> Vec<u16> {
    let reference = reference_values(values, width, height, base, shifts, direction);
    let peak = ((1u32 << base.bits) - 1) as f32;

    reference_narrow(&reference, peak)
}

/// Runs a `Grey` clip of `values` along `direction` and checks it against the reference exactly.
fn assert_grey_matches(values: &[u16], sample: SampleFormat, direction: &str) {
    let format = grey(WIDTH, HEIGHT, sample);
    let plane_values = [values.to_vec()];
    let planes = encoded(&format, &plane_values);
    let args = json!({ "direction": direction });

    let filtered = run_once(STAGE, format, format, args, &planes);

    let found = decoded(&format, &filtered);
    let defaults = default_parameters(sample);
    let expected = expected_plane(values, WIDTH, HEIGHT, &defaults, UNSHIFTED, direction);
    let changed = expected
        .iter()
        .zip(values)
        .filter(|(result, value)| result != value)
        .count();

    let label = format!("{sample} {direction}");
    assert!(changed > 0, "{label} leaves the clip unchanged");
    assert_same_plane(&found[0], &expected, WIDTH, &label);
}

/// Every plane of `format`, each holding aliased lines at its own size.
fn line_planes(format: &FrameFormat) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            aliased_lines(width, height, format.sample)
        })
        .collect()
}

/// Runs every plane of `format` along `direction` and checks each against the reference at its own size
/// and shifts.
///
/// `wrong` gives shifts a chroma plane must not be run with, and the reference must tell them apart.
fn assert_every_plane_matches(format: FrameFormat, direction: &str, wrong: Shifts) {
    let defaults = default_parameters(format.sample);
    let values = line_planes(&format);
    let planes = encoded(&format, &values);
    let args = json!({ "planes": ["y", "u", "v"], "direction": direction });

    let filtered = run_once(STAGE, format, format, args, &planes);

    let found = decoded(&format, &filtered);
    for (index, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let shifts = shifts_of(&format, index);
        let expected = expected_plane(&values[index], width, height, &defaults, shifts, direction);

        let label = format!("{} {channel:?} at {width}x{height} {direction}", format.layout);
        assert_ne!(expected, values[index], "{label} leaves the plane unchanged");

        if index > 0 {
            let misread = expected_plane(&values[index], width, height, &defaults, wrong, direction);
            assert_ne!(expected, misread, "{label} reads the same with the wrong shifts");
        }

        assert_same_plane(&found[index], &expected, width, &label);
    }
}

#[test]
fn vertical_matches_the_reference() {
    let values = aliased_lines(WIDTH, HEIGHT, SampleFormat::U8);

    assert_grey_matches(&values, SampleFormat::U8, "vertical");
}

#[test]
fn horizontal_matches_the_transposed_reference() {
    let values = aliased_lines(WIDTH, HEIGHT, SampleFormat::U8);

    assert_grey_matches(&values, SampleFormat::U8, "horizontal");
}

#[test]
fn both_runs_vertical_then_horizontal() {
    let values = aliased_lines(WIDTH, HEIGHT, SampleFormat::U8);

    assert_grey_matches(&values, SampleFormat::U8, "both");
}

#[test]
fn ten_bit_matches_the_reference() {
    let sample = ten_bit();
    let values = aliased_lines(WIDTH, HEIGHT, sample);

    for direction in DIRECTIONS {
        assert_grey_matches(&values, sample, direction);
    }
}

#[test]
fn sixteen_bit_matches_the_reference() {
    let sample = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let values = aliased_lines(WIDTH, HEIGHT, sample);

    for direction in DIRECTIONS {
        assert_grey_matches(&values, sample, direction);
    }
}

#[test]
fn every_argument_reaches_the_device() {
    let sample = SampleFormat::U8;
    let format = grey(WIDTH, HEIGHT, sample);
    let values = textured_lines(WIDTH, HEIGHT, sample, BUSY);
    let plane_values = [values.clone()];
    let planes = encoded(&format, &plane_values);
    let defaults = default_parameters(sample);
    let custom = custom_parameters(sample);

    for direction in DIRECTIONS {
        let args = custom_arguments(direction);

        let filtered = run_once(STAGE, format, format, args, &planes);

        let found = decoded(&format, &filtered);
        let expected = expected_plane(&values, WIDTH, HEIGHT, &custom, UNSHIFTED, direction);
        let with_defaults = expected_plane(&values, WIDTH, HEIGHT, &defaults, UNSHIFTED, direction);

        let label = format!("custom arguments {direction}");
        assert_ne!(expected, with_defaults, "{label} read the same as the defaults");
        assert_same_plane(&found[0], &expected, WIDTH, &label);
    }
}

/// The arguments the sensitivity test moves off one at a time.
///
/// `mthresh` is above 10, so the Laplacian test decides some 8-bit pixels.
const SENSITIVITY_BASE: Eedi2Parameters = Eedi2Parameters {
    mthresh: 16,
    lthresh: 12,
    vthresh: 30,
    estr: 2,
    dstr: 4,
    maxd: 12,
    nt: 40,
    pp: 1,
    bits: 8,
    ssw: 0,
};

/// Each argument moved alone off [SENSITIVITY_BASE](crate::aa::eedi2::SENSITIVITY_BASE), by name.
fn moved_arguments() -> [(&'static str, Eedi2Parameters); 8] {
    let base = SENSITIVITY_BASE;

    [
        ("mthresh", Eedi2Parameters { mthresh: 24, ..base }),
        ("lthresh", Eedi2Parameters { lthresh: 24, ..base }),
        ("vthresh", Eedi2Parameters { vthresh: 60, ..base }),
        ("estr", Eedi2Parameters { estr: 1, ..base }),
        ("dstr", Eedi2Parameters { dstr: 2, ..base }),
        ("maxd", Eedi2Parameters { maxd: 4, ..base }),
        ("nt", Eedi2Parameters { nt: 10, ..base }),
        ("pp", Eedi2Parameters { pp: 0, ..base }),
    ]
}

/// Faint noisy stripes above shallow noisy diagonals, 8-bit.
///
/// The stripes leave the edge mask to the thresholds, and the diagonals need a long reach.
fn sensitive_plane(width: u32, height: u32) -> Vec<u16> {
    let mut plane = noisy_diagonals(width, height / 2, 9, 70, 130, 20);
    let diagonals = noisy_diagonals(width, height / 2, 6, 60, 140, 24);
    plane.extend(diagonals);

    plane
}

/// The plane of [sensitive_plane](crate::aa::eedi2::sensitive_plane), transposed for a horizontal pass so
/// that it meets the same rows a vertical pass does.
fn sensitive_plane_along(width: u32, height: u32, direction: &str) -> Vec<u16> {
    if direction != "horizontal" {
        return sensitive_plane(width, height);
    }

    let turned = sensitive_plane(height, width);
    let widened: Vec<f32> = turned.iter().map(|value| f32::from(*value)).collect();
    let swapped = transposed_plane(&widened, height, width);

    swapped.iter().map(|value| *value as u16).collect()
}

/// Runs a `Grey` clip of `values` through `aa/eedi2` with `parameters` along `direction`.
fn run_grey(values: &[u16], format: FrameFormat, parameters: &Eedi2Parameters, direction: &str) -> Vec<u16> {
    let plane_values = [values.to_vec()];
    let planes = encoded(&format, &plane_values);
    let args = stage_arguments(parameters, direction);

    let filtered = run_once(STAGE, format, format, args, &planes);

    let mut found = decoded(&format, &filtered);
    found.remove(0)
}

#[test]
fn each_argument_alone_reaches_the_device() {
    let width = 64;
    let height = 64;
    let format = grey(width, height, SampleFormat::U8);

    for direction in DIRECTIONS {
        let values = sensitive_plane_along(width, height, direction);
        let base = expected_plane(&values, width, height, &SENSITIVITY_BASE, UNSHIFTED, direction);
        let found = run_grey(&values, format, &SENSITIVITY_BASE, direction);
        let label = format!("the base {direction}");
        assert_same_plane(&found, &base, width, &label);

        for (name, parameters) in moved_arguments() {
            let expected = expected_plane(&values, width, height, &parameters, UNSHIFTED, direction);
            let found = run_grey(&values, format, &parameters, direction);

            let label = format!("{name} moved {direction}");
            assert_ne!(expected, base, "{label} reads the same as the base");
            assert_same_plane(&found, &expected, width, &label);
        }
    }
}

#[test]
fn every_plane_of_a_420_clip_matches_at_its_own_size() {
    let format = yuv420(WIDTH, HEIGHT, SampleFormat::U8);

    for direction in DIRECTIONS {
        assert_every_plane_matches(format, direction, UNSHIFTED);
    }
}

/// 4:2:2 chroma is subsampled across but not down, so only its vertical pass shrinks its reach.
#[test]
fn every_plane_of_a_422_clip_matches_at_its_own_size() {
    let format = yuv422(WIDTH, HEIGHT, SampleFormat::U8);
    let swapped = Shifts {
        vertical: 0,
        horizontal: 1,
    };

    for direction in DIRECTIONS {
        assert_every_plane_matches(format, direction, swapped);
    }
}

#[test]
fn a_one_sample_diagonal_comes_back_with_smaller_steps() {
    let width = 32;
    let height = 24;
    let format = grey(width, height, SampleFormat::U8);
    let values = one_sample_diagonal(width, height);
    let plane_values = [values.clone()];
    let planes = encoded(&format, &plane_values);
    let before = largest_vertical_step(&values, width);

    for direction in ["vertical", "both"] {
        let args = json!({ "direction": direction });

        let filtered = run_once(STAGE, format, format, args, &planes);

        let found = decoded(&format, &filtered);
        let after = largest_vertical_step(&found[0], width);
        assert!(
            after < before,
            "{direction} leaves a step of {after}, the input's was {before}"
        );
    }
}

#[test]
fn only_the_selected_planes_change() {
    let format = yuv420(WIDTH, HEIGHT, SampleFormat::U8);

    let shared = shared_planes(STAGE, format, json!({ "planes": ["y"] }));

    assert_eq!(shared, vec![false, true, true]);
}

#[test]
fn a_zero_mask_hands_back_the_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(WIDTH, HEIGHT, sample);
        let values = line_planes(&format);
        let planes = encoded(&format, &values);
        let mask = uniform_mask(&format, 0);
        let args = json!({ "planes": ["y", "u", "v"] });
        let bits = bits_of(sample);

        let found = run_masked_into_codes(STAGE, format, args, &planes, mask, bits);

        for (plane, codes) in found.iter().enumerate() {
            let input = widened(&values[plane]);
            let label = format!("{sample} plane {plane}");
            assert_codes_within(codes, &input, ROUND_TRIP, &label);
        }
    }
}

#[test]
fn a_flat_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(WIDTH, HEIGHT, sample);
        let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
        let values: Vec<Vec<u16>> = format
            .channels()
            .iter()
            .map(|channel| {
                let samples = format.channel_samples(*channel).expect("a listed channel");
                vec![flat; samples]
            })
            .collect();
        let planes = encoded(&format, &values);
        let args = json!({ "planes": ["y", "u", "v"] });

        let filtered = run_once(STAGE, format, format, args, &planes);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn eedi2_f32_matches_sixteen_bit() {
    let input = yuv420(WIDTH, HEIGHT, sixteen_bit());
    let values = line_planes(&input);
    let args = json!({ "planes": ["y", "u", "v"] });

    assert_f32_matches_sixteen_bit(STAGE, input, args, &values, 0.01, 0.0);
}
