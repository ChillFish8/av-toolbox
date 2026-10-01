use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::artcnn::{Model, Weights};
use av_toolbox_cubek::testing::{
    BasedFinish,
    BasedParameters,
    CloseDecisionTally,
    Direction,
    Eedi3Antialias,
    Eedi3Clips,
    Eedi3Parameters,
    assert_close_roundings,
    assert_plane_within,
    assert_same_plane,
    reference_based_aa,
    reference_based_finish,
    reference_based_mask,
    reference_double_mclip,
    reference_eedi3_antialias,
    reference_narrow,
    reference_supersample,
    transposed_plane,
};
use av_toolbox_filters::testing::{BasedPlanes, EdgePlanes, based_stages};
use av_toolbox_gpu::{BuildContext, BuildTask, Depth, SampleFormat};
use cubecl::prelude::{CubeElement, Runtime};
use cubecl::server::Handle;
use serde_json::json;

use super::{bits_of, decoded, grey, largest_vertical_step, one_sample_diagonal, yuv420};
use crate::support::{backend, encoded, run_masked, run_once, shared_planes, ten_bit, uniform_mask, widened};

const STAGE: &str = "aa/based";

const WIDTH: u32 = 192;

const HEIGHT: u32 = 128;

/// A 192 by 128 crop of Big Buck Bunny's luma, two little-endian bytes per 10-bit sample.
///
/// It holds a fence post's long edge and grass stalks over a blurred background. The fixtures' README gives
/// its source and licence.
const FENCE: &[u8] = include_bytes!("fixtures/based_fence.luma");

/// A small frame for the stage's simple cases.
const SMALL_WIDTH: u32 = 64;

const SMALL_HEIGHT: u32 = 48;

const DIRECTIONS: [Direction; 3] = [Direction::Vertical, Direction::Horizontal, Direction::Both];

/// How far the mask's box and the merged plane may sit from the reference, on the 8-bit scale.
///
/// The bound is 1e-4 on the 8-bit scale rather than 1e-4 codes. It scales with the bit depth, as f32's
/// rounding error does with the size of the codes.
const CODE_TOLERANCE: f32 = 1e-4;

/// The largest share of a plane's `aa2` samples EEDI3's reference may mark as touched, as a fraction.
///
/// It is 3 % because natural 8-bit footage with blurred backgrounds makes more exact path ties than
/// `aa/eedi3`'s tests do. The Big Buck Bunny crop measured at most 2.44 %, with none differing.
const TOUCHED_BOUND: (usize, usize) = (3, 100);

/// The largest share of output samples that may differ from the composed reference at f32.
const END_TO_END_SHARE: f64 = 0.001;

/// The largest share of output samples that may differ from the composed reference at f16.
///
/// It holds on natural footage. Hard synthetic lines of 220 on 40 measured 0.83 % at 10 bit.
const F16_SHARE: f64 = 0.005;

/// The largest mean difference from the composed reference at f16, on the 8-bit scale.
///
/// It holds on natural footage. Hard synthetic lines of 220 on 40 measured 0.0027 at 8 bit.
const F16_MEAN: f64 = 0.002;

/// The largest share of 16-bit output samples that may differ from the composed reference at f32.
///
/// It measured 0.256 % at most, along "both", on an AMD R9700 through RADV, and keeps about twice that.
const SIXTEEN_BIT_SHARE: f64 = 0.005;

fn sixteen_bit() -> SampleFormat {
    SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    }
}

fn direction_name(direction: Direction) -> &'static str {
    match direction {
        Direction::Vertical => "vertical",
        Direction::Horizontal => "horizontal",
        Direction::Both => "both",
    }
}

/// The defaults of `aa/based` along `direction` at `bits` per sample.
fn default_parameters(bits: u32, direction: Direction) -> BasedParameters {
    BasedParameters {
        bits,
        direction,
        ..BasedParameters::default()
    }
}

/// The arguments of `parameters` as `aa/based` takes them, at `precision`.
fn stage_arguments(parameters: &BasedParameters, precision: &str) -> serde_json::Value {
    let eedi3 = parameters.eedi3;
    let direction = direction_name(parameters.direction);

    json!({
        "mask_thr": parameters.mask_thr,
        "edge_mask": parameters.edge_mask,
        "pscale": parameters.pscale,
        "postfilter": parameters.postfilter,
        "direction": direction,
        "precision": precision,
        "eedi3": {
            "alpha": eedi3.alpha,
            "beta": eedi3.beta,
            "gamma": eedi3.gamma,
            "nrad": eedi3.nrad,
            "mdis": eedi3.mdis,
            "vcheck": eedi3.vcheck,
            "vthresh": eedi3.vthresh,
        },
    })
}

/// EEDI3's settings inside `parameters`, at its bit depth.
fn eedi3_parameters(parameters: &BasedParameters) -> Eedi3Parameters {
    Eedi3Parameters {
        bits: parameters.bits,
        ..parameters.eedi3
    }
}

/// The 10-bit crop in `bytes` at `bits` per sample, rounded to the nearest code.
fn fixture_plane(bytes: &[u8], bits: u32) -> Vec<u16> {
    let peak = f64::from((1u32 << bits) - 1);
    let (pairs, _) = bytes.as_chunks::<2>();

    pairs
        .iter()
        .map(|pair| {
            let ten_bit = u16::from_le_bytes(*pair);
            let scaled = f64::from(ten_bit) * peak / 1023.0;
            let rounded = (scaled + 0.5).floor();

            rounded as u16
        })
        .collect()
}

/// The `width` by `height` block of a `plane_width` wide plane whose top left sample is at `left`, `top`.
fn cropped(plane: &[u16], plane_width: u32, left: u32, top: u32, width: u32, height: u32) -> Vec<u16> {
    let mut block = Vec::with_capacity((width * height) as usize);
    for row in top..top + height {
        let start = (row * plane_width + left) as usize;
        let end = start + width as usize;
        block.extend_from_slice(&plane[start..end]);
    }

    block
}

fn peak_of(bits: u32) -> f32 {
    let peak = (1u32 << bits) - 1;

    peak as f32
}

/// The planes one run of the stage's chain wrote, read back from the device.
///
/// `mask` and `mclip` are empty when `edge_mask` is false.
struct DeviceStages {
    mask: Vec<f32>,
    mclip: Vec<f32>,
    supersampled: Vec<f32>,
    antialiased: Vec<f32>,
    merged: Vec<f32>,
}

/// Runs the stage's chain on one luma plane through the test-support entry, reading every step back.
struct ChainRun {
    luma: Vec<f32>,
    width: u32,
    height: u32,
    bits: u32,
    edge_mask: bool,
    args: serde_json::Value,
}

/// A plane of f32 samples over `handle`, asserting `handle` holds `width * height` of them.
fn plane(handle: &Handle, width: u32, height: u32) -> DevicePlane<'_> {
    let needed = width as u64 * height as u64 * size_of::<f32>() as u64;
    let held = handle.size_in_used();
    assert!(
        held >= needed,
        "the handle holds {held} bytes, expected at least {needed} for {width}x{height}"
    );

    // Safety: the assert above checks `handle` holds `width * height` f32 samples.
    unsafe { DevicePlane::new(handle, width, height) }
}

fn read_f32<R: Runtime>(build: &BuildContext<R>, handle: &Handle) -> Vec<f32> {
    let owned = handle.clone();
    let bytes = build
        .client()
        .read_one(owned)
        .expect("the device should hand back the plane");

    f32::from_bytes(&bytes).to_vec()
}

impl BuildTask for ChainRun {
    type Output = DeviceStages;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let width = self.width;
        let height = self.height;
        let bytes = (width * height) as usize * size_of::<f32>();
        let doubled_bytes = 4 * bytes;

        let luma_bytes = f32::as_bytes(&self.luma);
        let luma = client.create_from_slice(luma_bytes);
        let supersampled = client.empty(doubled_bytes);
        let antialiased = client.empty(doubled_bytes);
        let merged = client.empty(bytes);
        let mask = client.empty(bytes);
        let mclip = client.empty(doubled_bytes);

        let edges = self.edge_mask.then(|| EdgePlanes {
            mask: plane(&mask, width, height),
            mclip: plane(&mclip, 2 * width, 2 * height),
            bound: false,
        });
        let planes = BasedPlanes {
            luma: plane(&luma, width, height),
            supersampled: plane(&supersampled, 2 * width, 2 * height),
            antialiased: plane(&antialiased, 2 * width, 2 * height),
            merged: plane(&merged, width, height),
            edges,
        };

        based_stages(build, self.args, self.bits, planes).expect("the chain runs");

        let (mask_values, mclip_values) = if self.edge_mask {
            let mask_values = read_f32(build, &mask);
            let mclip_values = read_f32(build, &mclip);
            (mask_values, mclip_values)
        } else {
            (Vec::new(), Vec::new())
        };

        let supersampled_values = read_f32(build, &supersampled);
        let antialiased_values = read_f32(build, &antialiased);
        let merged_values = read_f32(build, &merged);

        DeviceStages {
            mask: mask_values,
            mclip: mclip_values,
            supersampled: supersampled_values,
            antialiased: antialiased_values,
            merged: merged_values,
        }
    }
}

/// Runs the chain with `parameters` at f32 on a `width` by `height` `luma` plane.
fn run_chain(luma: &[f32], width: u32, height: u32, parameters: &BasedParameters) -> DeviceStages {
    let args = stage_arguments(parameters, "f32");
    let task = ChainRun {
        luma: luma.to_vec(),
        width,
        height,
        bits: parameters.bits,
        edge_mask: parameters.edge_mask,
        args,
    };

    backend().build(task)
}

/// Runs the whole stage with `parameters` at `precision` on a grey frame of `luma`, giving its output.
fn run_stage(
    luma: &[u16],
    width: u32,
    height: u32,
    sample: SampleFormat,
    parameters: &BasedParameters,
    precision: &str,
) -> Vec<f32> {
    let format = grey(width, height, sample);
    let values = [luma.to_vec()];
    let planes = encoded(&format, &values);
    let args = stage_arguments(parameters, precision);

    let filtered = run_once(STAGE, format, format, args, &planes);

    let found = decoded(&format, &filtered);
    widened(&found[0])
}

/// EEDI3's reference on the transposed `input`, reading the transposed `supersampled` and `mclip` as its
/// clips, with its output and touched set transposed back.
fn horizontal_reference(
    input: &[f32],
    supersampled: &[f32],
    mclip: Option<&[f32]>,
    width: u32,
    height: u32,
    parameters: &Eedi3Parameters,
) -> Eedi3Antialias {
    let turned_input = transposed_plane(input, width, height);
    let turned_supersampled = transposed_plane(supersampled, width, height);
    let turned_mclip = mclip.map(|values| transposed_plane(values, width, height));
    let clips = Eedi3Clips {
        sclip: Some(&turned_supersampled),
        mclip: turned_mclip.as_deref(),
    };

    let pass = reference_eedi3_antialias(&turned_input, height, width, parameters, clips);

    let output = transposed_plane(&pass.output, height, width);
    let touched = transposed_plane(&pass.touched, height, width);
    Eedi3Antialias { output, touched }
}

/// EEDI3's reference on the device's own supersample and `mclip` of `found`, along `parameters`'s
/// direction.
///
/// For "both", the horizontal pass runs on `vertical`, the device's own vertical result.
fn expected_antialiased(
    found: &DeviceStages,
    width: u32,
    height: u32,
    parameters: &BasedParameters,
    vertical: Option<&[f32]>,
) -> Eedi3Antialias {
    let eedi3 = eedi3_parameters(parameters);
    let doubled_width = 2 * width;
    let doubled_height = 2 * height;
    let supersampled = found.supersampled.as_slice();
    let mclip = parameters.edge_mask.then_some(found.mclip.as_slice());

    match parameters.direction {
        Direction::Vertical => {
            let clips = Eedi3Clips {
                sclip: Some(supersampled),
                mclip,
            };
            reference_eedi3_antialias(supersampled, doubled_width, doubled_height, &eedi3, clips)
        },
        Direction::Horizontal => horizontal_reference(
            supersampled,
            supersampled,
            mclip,
            doubled_width,
            doubled_height,
            &eedi3,
        ),
        Direction::Both => {
            let input = vertical.expect("both needs the device's vertical result");
            horizontal_reference(input, supersampled, mclip, doubled_width, doubled_height, &eedi3)
        },
    }
}

/// The code tolerance in codes at `bits` per sample.
fn code_tolerance(bits: u32) -> f32 {
    let scale = (1u32 << (bits - 8)) as f32;

    CODE_TOLERANCE * scale
}

/// `values` rounded to codes as the stage's output is.
fn rounded(values: &[f32], bits: u32) -> Vec<f32> {
    let peak = peak_of(bits);
    let codes = reference_narrow(values, peak);

    codes.iter().map(|code| f32::from(*code)).collect()
}

/// The largest absolute difference between `found` and `expected`.
fn largest_difference(found: &[f32], expected: &[f32]) -> f32 {
    found
        .iter()
        .zip(expected)
        .map(|(actual, wanted)| (actual - wanted).abs())
        .fold(0.0, f32::max)
}

/// Checks each step of `found` against the references fed the device's own intermediates.
///
/// The mask lies within the code tolerance, which leaves no room for a binarize that differs, and `mclip`
/// matches exactly. `aa2` meets EEDI3's reference through the tally on this plane alone, with the touched
/// bound. The merged plane from the device's `aa2` lies within the code tolerance, and its rounded output
/// differs only by close roundings.
fn assert_steps_match(
    found: &DeviceStages,
    luma: &[f32],
    width: u32,
    height: u32,
    parameters: &BasedParameters,
    vertical: Option<&[f32]>,
    label: &str,
) {
    let peak = peak_of(parameters.bits);
    let tolerance = code_tolerance(parameters.bits);

    let mask = if parameters.edge_mask {
        let expected_mask = reference_based_mask(luma, width, height, parameters);
        let mask_label = format!("{label} mask");
        assert_plane_within(&found.mask, &expected_mask, tolerance, width, &mask_label);

        let expected_mclip = reference_double_mclip(&found.mask, width, height, peak);
        let mclip_label = format!("{label} mclip");
        assert_same_plane(&found.mclip, &expected_mclip, 2 * width, &mclip_label);

        Some((found.mask.as_slice(), peak))
    } else {
        None
    };

    let expected = expected_antialiased(found, width, height, parameters, vertical);
    let mut tally = CloseDecisionTally::new();
    let aa2_label = format!("{label} aa2");
    tally.compare(
        &aa2_label,
        &found.antialiased,
        &expected.output,
        &expected.touched,
        2 * width,
    );
    println!("{aa2_label}: {tally:?}");
    tally.assert_touched_within(&aa2_label, TOUCHED_BOUND);

    let finish = BasedFinish {
        luma,
        supersampled: &found.supersampled,
        antialiased: &found.antialiased,
        mask,
    };
    let expected_merged = reference_based_finish(&finish, width, height, parameters);
    let largest = largest_difference(&found.merged, &expected_merged);
    println!("{label}: the merged plane is at most {largest} codes from the reference");

    let merged_label = format!("{label} merged");
    assert_plane_within(&found.merged, &expected_merged, tolerance, width, &merged_label);

    let found_output = rounded(&found.merged, parameters.bits);
    let expected_output = rounded(&expected_merged, parameters.bits);
    let output_label = format!("{label} output");
    let close_roundings = assert_close_roundings(
        &found_output,
        &expected_output,
        &expected_merged,
        tolerance,
        width,
        &output_label,
    );
    println!("{label}: {close_roundings} close final roundings differ");
}

/// Runs the chain with `parameters` along every direction on `luma`, checking each step.
///
/// "both" is checked against the horizontal reference on the vertical run's `aa2`.
fn assert_directions_match(luma: &[u16], width: u32, height: u32, parameters: &BasedParameters, label: &str) {
    let values = widened(luma);

    let vertical_parameters = BasedParameters {
        direction: Direction::Vertical,
        ..*parameters
    };
    let vertical = run_chain(&values, width, height, &vertical_parameters);
    let vertical_label = format!("{label} vertical");
    assert_steps_match(
        &vertical,
        &values,
        width,
        height,
        &vertical_parameters,
        None,
        &vertical_label,
    );

    let horizontal_parameters = BasedParameters {
        direction: Direction::Horizontal,
        ..*parameters
    };
    let horizontal = run_chain(&values, width, height, &horizontal_parameters);
    let horizontal_label = format!("{label} horizontal");
    assert_steps_match(
        &horizontal,
        &values,
        width,
        height,
        &horizontal_parameters,
        None,
        &horizontal_label,
    );

    let both_parameters = BasedParameters {
        direction: Direction::Both,
        ..*parameters
    };
    let both = run_chain(&values, width, height, &both_parameters);
    let both_label = format!("{label} both");
    let supersample_label = format!("{both_label} supersample against the vertical run's");
    assert_same_plane(
        &both.supersampled,
        &vertical.supersampled,
        2 * width,
        &supersample_label,
    );
    assert_steps_match(
        &both,
        &values,
        width,
        height,
        &both_parameters,
        Some(&vertical.antialiased),
        &both_label,
    );
}

fn assert_steps_match_at(sample: SampleFormat) {
    let bits = bits_of(sample);
    let luma = fixture_plane(FENCE, bits);
    let parameters = default_parameters(bits, Direction::Both);
    let label = format!("{bits}-bit");

    assert_directions_match(&luma, WIDTH, HEIGHT, &parameters, &label);
}

#[test]
fn each_step_matches_at_8_bit() {
    assert_steps_match_at(SampleFormat::U8);
}

#[test]
fn each_step_matches_at_10_bit() {
    let sample = ten_bit();

    assert_steps_match_at(sample);
}

#[test]
fn each_step_matches_at_16_bit() {
    let sample = sixteen_bit();

    assert_steps_match_at(sample);
}

/// A square plane gives the horizontal pass clips of the right size even when they are not transposed,
/// so only the values can show which clips it read.
#[test]
fn each_step_matches_on_a_square_plane() {
    let size = 128;
    let fence = fixture_plane(FENCE, 8);
    let luma = cropped(&fence, WIDTH, 0, 0, size, size);
    let parameters = default_parameters(8, Direction::Both);

    assert_directions_match(&luma, size, size, &parameters, "square");
}

/// The builtin R8F64 network's weights.
fn builtin_weights() -> Weights {
    let model = Model::R8F64;
    let bytes = av_toolbox_models::builtin("artcnn", model.file_name()).expect("R8F64 is built in");

    Weights::parse(bytes, model.features(), model.blocks()).expect("the builtin weights parse")
}

/// How far the stage's output sits from the composed reference.
#[derive(Debug)]
struct OutputDifference {
    /// The share of samples that differ.
    share: f64,
    /// The mean absolute difference, on the 8-bit scale.
    mean: f64,
}

/// Compares the stage's output with the composed reference's, panicking at a difference where the
/// reference's mask is zero.
fn output_difference(
    found: &[f32],
    expected: &[f32],
    mask: &[f32],
    bits: u32,
    label: &str,
) -> OutputDifference {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong sample count");

    let scale = f64::from(1u32 << (bits - 8));
    let mut differing = 0usize;
    let mut total = 0.0f64;
    for (index, (found_value, expected_value)) in found.iter().zip(expected).enumerate() {
        if found_value == expected_value {
            continue;
        }

        assert!(
            mask[index] != 0.0,
            "{label} differs at sample {index}, where the mask is zero: found {found_value}, expected \
             {expected_value}"
        );

        let difference = f64::from(found_value - expected_value).abs();
        differing += 1;
        total += difference / scale;
    }

    let samples = found.len() as f64;
    OutputDifference {
        share: differing as f64 / samples,
        mean: total / samples,
    }
}

/// Runs the stage at `precision` along every direction on `luma`, against the composed reference on
/// ArtCNN's f64 supersample, giving the difference in each direction.
fn end_to_end(
    luma: &[u16],
    width: u32,
    height: u32,
    sample: SampleFormat,
    precision: &str,
) -> Vec<OutputDifference> {
    let bits = bits_of(sample);
    let values = widened(luma);
    let weights = builtin_weights();
    let supersampled = reference_supersample(&values, width, height, &weights, bits);

    DIRECTIONS
        .iter()
        .map(|direction| {
            let parameters = default_parameters(bits, *direction);
            let expected = reference_based_aa(&values, width, height, &parameters, None, &supersampled);
            let found = run_stage(luma, width, height, sample, &parameters, precision);

            let name = direction_name(*direction);
            let label = format!("{width}x{height} {bits}-bit {precision} {name}");
            let difference = output_difference(&found, &expected.output, &expected.mask, bits, &label);
            println!("{label}: {difference:?}");

            difference
        })
        .collect()
}

fn assert_end_to_end_matches(sample: SampleFormat) {
    let bits = bits_of(sample);
    let luma = fixture_plane(FENCE, bits);

    let differences = end_to_end(&luma, WIDTH, HEIGHT, sample, "f32");

    for (direction, difference) in DIRECTIONS.iter().zip(&differences) {
        assert!(
            difference.share <= END_TO_END_SHARE,
            "{bits}-bit {direction:?}: {} of output samples differ",
            difference.share
        );
    }
}

#[test]
fn end_to_end_matches_the_composed_reference_at_8_bit() {
    assert_end_to_end_matches(SampleFormat::U8);
}

#[test]
fn end_to_end_matches_the_composed_reference_at_10_bit() {
    let sample = ten_bit();

    assert_end_to_end_matches(sample);
}

#[test]
fn end_to_end_at_16_bit_stays_within_its_measured_share() {
    let sample = sixteen_bit();
    let luma = fixture_plane(FENCE, 16);

    let differences = end_to_end(&luma, WIDTH, HEIGHT, sample, "f32");

    for (direction, difference) in DIRECTIONS.iter().zip(&differences) {
        assert!(
            difference.share <= SIXTEEN_BIT_SHARE,
            "16-bit {direction:?}: {} of output samples differ",
            difference.share
        );
    }
}

fn assert_f16_matches(sample: SampleFormat) {
    let bits = bits_of(sample);
    let luma = fixture_plane(FENCE, bits);

    let differences = end_to_end(&luma, WIDTH, HEIGHT, sample, "f16");

    for (direction, difference) in DIRECTIONS.iter().zip(&differences) {
        assert!(
            difference.share <= F16_SHARE,
            "{bits}-bit f16 {direction:?}: {} of output samples differ",
            difference.share
        );
        assert!(
            difference.mean <= F16_MEAN,
            "{bits}-bit f16 {direction:?}: the mean difference is {}",
            difference.mean
        );
    }
}

#[test]
fn f16_matches_the_composed_reference_at_8_bit() {
    assert_f16_matches(SampleFormat::U8);
}

#[test]
fn f16_matches_the_composed_reference_at_10_bit() {
    let sample = ten_bit();

    assert_f16_matches(sample);
}

#[test]
fn an_odd_luma_plane_matches_the_composed_reference() {
    let width = 63;
    let height = 45;
    let fence = fixture_plane(FENCE, 8);
    let luma = cropped(&fence, WIDTH, 0, 0, width, height);

    let differences = end_to_end(&luma, width, height, SampleFormat::U8, "f32");

    for (direction, difference) in DIRECTIONS.iter().zip(&differences) {
        assert!(
            difference.share <= END_TO_END_SHARE,
            "63x45 {direction:?}: {} of output samples differ",
            difference.share
        );
    }
}

/// Runs the chain with `parameters` on the 8-bit fence, checking each step, after checking the reference's
/// output differs from the defaults' on the device's supersample.
fn assert_variant_matches(parameters: &BasedParameters, label: &str) {
    let luma = fixture_plane(FENCE, 8);
    let values = widened(&luma);

    let vertical = if parameters.direction == Direction::Both {
        let vertical_parameters = BasedParameters {
            direction: Direction::Vertical,
            ..*parameters
        };
        let run = run_chain(&values, WIDTH, HEIGHT, &vertical_parameters);
        Some(run.antialiased)
    } else {
        None
    };

    let found = run_chain(&values, WIDTH, HEIGHT, parameters);

    let defaults = default_parameters(8, parameters.direction);
    let default_reference = reference_based_aa(&values, WIDTH, HEIGHT, &defaults, None, &found.supersampled);
    let variant_reference = reference_based_aa(&values, WIDTH, HEIGHT, parameters, None, &found.supersampled);
    assert_ne!(
        variant_reference.output, default_reference.output,
        "{label} gives the defaults' output"
    );

    assert_steps_match(
        &found,
        &values,
        WIDTH,
        HEIGHT,
        parameters,
        vertical.as_deref(),
        label,
    );
}

#[test]
fn pscale_half_matches_the_reference() {
    let parameters = BasedParameters {
        pscale: 0.5,
        ..default_parameters(8, Direction::Both)
    };

    assert_variant_matches(&parameters, "pscale 0.5");
}

#[test]
fn pscale_1_matches_the_reference() {
    let parameters = BasedParameters {
        pscale: 1.0,
        ..default_parameters(8, Direction::Vertical)
    };

    assert_variant_matches(&parameters, "pscale 1");
}

#[test]
fn no_postfilter_matches_the_reference() {
    let parameters = BasedParameters {
        postfilter: false,
        ..default_parameters(8, Direction::Horizontal)
    };

    assert_variant_matches(&parameters, "no postfilter");
}

#[test]
fn no_edge_mask_matches_the_reference() {
    let parameters = BasedParameters {
        edge_mask: false,
        ..default_parameters(8, Direction::Both)
    };

    assert_variant_matches(&parameters, "no edge mask");
}

#[test]
fn a_flat_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(SMALL_WIDTH, SMALL_HEIGHT, sample);
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
        let args = json!({ "precision": "f32" });

        let filtered = run_once(STAGE, format, format, args, &planes);

        assert_eq!(filtered, planes, "{sample}");
    }
}

/// The block of a small plane at least 8 samples from each side, clear of the line's first and last runs.
fn interior(plane: &[u16]) -> Vec<u16> {
    let width = SMALL_WIDTH - 16;
    let height = SMALL_HEIGHT - 16;

    cropped(plane, SMALL_WIDTH, 8, 8, width, height)
}

/// Only the interior is measured, since the line's first and last runs keep their full step at the plane's
/// sides.
#[test]
fn a_one_sample_diagonal_comes_back_with_smaller_steps() {
    let format = grey(SMALL_WIDTH, SMALL_HEIGHT, SampleFormat::U8);
    let values = one_sample_diagonal(SMALL_WIDTH, SMALL_HEIGHT);
    let plane_values = [values.clone()];
    let planes = encoded(&format, &plane_values);
    let inner_width = SMALL_WIDTH - 16;
    let inner_values = interior(&values);
    let before = largest_vertical_step(&inner_values, inner_width);

    for direction in ["vertical", "both"] {
        let args = json!({ "direction": direction, "precision": "f32" });

        let filtered = run_once(STAGE, format, format, args, &planes);

        let found = decoded(&format, &filtered);
        let inner_found = interior(&found[0]);
        let after = largest_vertical_step(&inner_found, inner_width);
        assert!(
            after < before,
            "{direction} leaves a step of {after}, the input's was {before}"
        );
    }
}

#[test]
fn a_zero_mask_hands_back_the_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(SMALL_WIDTH, SMALL_HEIGHT, sample);
        let bits = bits_of(sample);
        let fence = fixture_plane(FENCE, bits);
        let values: Vec<Vec<u16>> = format
            .channels()
            .iter()
            .map(|channel| {
                let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
                cropped(&fence, WIDTH, 0, 0, width, height)
            })
            .collect();
        let planes = encoded(&format, &values);
        let (mask, mask_chroma) = uniform_mask(&format, 0);
        let args = json!({ "precision": "f32" });

        let filtered = run_masked(STAGE, format, args, &planes, mask, mask_chroma);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn chroma_is_handed_through() {
    let format = yuv420(SMALL_WIDTH, SMALL_HEIGHT, SampleFormat::U8);
    let args = json!({ "precision": "f32" });

    let shared = shared_planes(STAGE, format, args);

    assert_eq!(shared, vec![false, true, true]);
}
