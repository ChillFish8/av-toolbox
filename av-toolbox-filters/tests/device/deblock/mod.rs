mod pp7;

use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::cnn::{BodyPath, Precision};
use av_toolbox_cubek::deblock::h264::{padded_frame, thresholds};
use av_toolbox_cubek::dpir::{Dpir, DpirMaps, DpirPlanes, PACK, Variant, Weights, deblock, scratch_sizes};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_cubek::testing::{
    FixtureLayout,
    H264Plane,
    QedCase,
    QedChroma,
    QedReference,
    assert_h264_plane,
    blocky_plane,
    decode_samples,
    encode_samples,
    reference_h264,
    reference_qed,
};
use av_toolbox_gpu::{
    BuildContext,
    BuildTask,
    ChromaLocation,
    Depth,
    ElementType,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    Subsampling,
};
use av_toolbox_models::Store;
use cubecl::prelude::CubeElement;
use serde_json::json;

use crate::support::{
    Sizing,
    assert_codes_within,
    float_samples,
    frame,
    planes_as_codes,
    run_into_codes,
    run_once,
};

const STAGE: &str = "deblock/dpir";

/// How far an f16 output code may sit from the f32 run of the stage.
///
/// It measured 1 code out, with 7 of 1920 codes differing, so one code of headroom is kept.
const F16_CODE_BOUND: f32 = 2.0;

/// How far a float output, read as codes, may sit from the unrounded network.
const CODE_TOLERANCE: f32 = 0.01;

const BODY_PATHS: [BodyPath; 3] = [
    BodyPath::Convolution,
    BodyPath::UnfusedConvolution,
    BodyPath::Im2col,
];

/// Whether `variant`'s weights are installed, printing why the test does not run when they are not.
fn installed(variant: Variant) -> bool {
    let store = Store::from_environment();
    let found = store.find(PACK, variant.file_name());

    match found {
        Ok(_) => true,
        Err(missing) => {
            println!(
                "{} is not installed, so the stage is not run: {missing}",
                variant.file_name()
            );
            false
        },
    }
}

/// A run of the network straight through `dpir::deblock` at f32 without the stage, on the first body path
/// the device runs.
struct Direct {
    variant: Variant,
    planes: Vec<Vec<f32>>,
    width: u32,
    height: u32,
    strength: f32,
    peak: f32,
}

impl BuildTask for Direct {
    type Output = Vec<Vec<f32>>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let store = Store::from_environment();
        let located = store.find(PACK, self.variant.file_name()).expect("installed");
        let bytes = located.read().expect("the weights read");
        let weights = Weights::parse(&bytes, self.variant).expect("the weights parse");
        let network = Dpir::upload(client, &weights, Precision::F32);

        let inputs: Vec<_> = self
            .planes
            .iter()
            .map(|plane| {
                let bytes = f32::as_bytes(plane);

                client.create_from_slice(bytes)
            })
            .collect();
        let samples = (self.width * self.height) as usize;
        let outputs: Vec<_> = self.planes.iter().map(|_| client.empty(samples * 4)).collect();

        // Safety: each buffer holds `width * height` f32 values.
        let input_planes: Vec<DevicePlane<'_>> = inputs
            .iter()
            .map(|handle| unsafe { DevicePlane::new(handle, self.width, self.height) })
            .collect();

        // Safety: each buffer holds `width * height` f32 values.
        let output_planes: Vec<DevicePlane<'_>> = outputs
            .iter()
            .map(|handle| unsafe { DevicePlane::new(handle, self.width, self.height) })
            .collect();

        let ran = BODY_PATHS.iter().any(|path| {
            let scratch = scratch_sizes(Precision::F32, *path, self.width, self.height, usize::MAX);
            let levels = scratch
                .level_bytes
                .map(|bytes| [(); 4].map(|_| client.empty(bytes)));
            let columns = match scratch.column_bytes {
                0 => None,
                bytes => Some(client.empty(bytes)),
            };

            let planes = DpirPlanes {
                inputs: &input_planes,
                outputs: &output_planes,
            };
            let level_handles = levels.each_ref().map(|level| level.each_ref());
            let maps = DpirMaps {
                levels: level_handles,
                columns: columns.as_ref(),
            };

            let outcome = deblock(
                client,
                &sizing,
                &network,
                *path,
                planes,
                self.strength,
                self.peak,
                scratch.tile,
                maps,
            );

            outcome.is_ok()
        });
        assert!(ran, "the network ran on no body path");

        outputs
            .iter()
            .map(|handle| {
                let bytes = client
                    .read_one(handle.clone())
                    .expect("the device hands back the plane");

                f32::from_bytes(&bytes).to_vec()
            })
            .collect()
    }
}

fn direct(
    variant: Variant,
    planes: Vec<Vec<f32>>,
    width: u32,
    height: u32,
    strength: f32,
    peak: f32,
) -> Vec<Vec<f32>> {
    let task = Direct {
        variant,
        planes,
        width,
        height,
        strength,
        peak,
    };

    crate::support::backend().build(task)
}

fn peak_of(samples: MatrixSamples) -> f64 {
    match samples {
        MatrixSamples::U8 => 255.0,
        MatrixSamples::U16 { bits } => f64::from((1u32 << bits) - 1),
        MatrixSamples::F32 => 1.0,
    }
}

/// A blocky test plane of flat 8 by 8 blocks with a gentle ramp, up to the peak of `samples`.
///
/// Integer samples are rounded to whole code values.
fn blocky(width: u32, height: u32, samples: MatrixSamples, seed: u32) -> Vec<f64> {
    let peak = peak_of(samples);
    let whole = samples != MatrixSamples::F32;

    (0..height)
        .flat_map(|row| (0..width).map(move |column| (row, column)))
        .map(|(row, column)| {
            let block = (row / 8) * 31 + (column / 8) * 17 + seed;
            let level = f64::from(block % 23) / 22.0;
            let ramp = f64::from(column % 8) / 64.0;
            let value = (level * 0.8 + ramp) * peak;

            if whole {
                return value.round().min(peak);
            }

            value.min(peak)
        })
        .collect()
}

fn as_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

fn assert_within(found: &[f64], expected: &[f64], bound: f64, label: &str) {
    assert_eq!(found.len(), expected.len(), "{label}: lengths differ");

    for (index, (value, wanted)) in found.iter().zip(expected).enumerate() {
        let difference = (value - wanted).abs();

        assert!(
            difference <= bound,
            "{label}: sample {index} is {value}, expected {wanted}"
        );
    }
}

fn plane_samples(format: &FrameFormat, index: usize) -> (u32, u32) {
    let channel = format.channels()[index];

    format
        .layout
        .channel_dimensions(channel, format.width, format.height)
        .expect("a carried channel")
}

#[test]
fn the_gray_model_on_a_grey_u8_frame_matches_the_network() {
    if !installed(Variant::Gray) {
        return;
    }

    let input = frame(48, 40, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let values = blocky(48, 40, MatrixSamples::U8, 0);
    let encoded = encode_samples(&values, MatrixSamples::U8);
    let planes = vec![encoded];
    let args = json!({ "precision": "f32" });

    let found = run_into_codes(STAGE, input, args, &planes, 8);

    let network_values = as_f32(&values);
    let expected = direct(Variant::Gray, vec![network_values.clone()], 48, 40, 10.0, 255.0);
    assert_codes_within(&found[0], &expected[0], CODE_TOLERANCE, "grey u8");
    assert_ne!(found[0], network_values, "the stage changed nothing");
}

#[test]
fn the_gray_model_runs_only_the_selected_planes_each_at_its_own_strength() {
    if !installed(Variant::Gray) {
        return;
    }

    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = frame(48, 40, layout, SampleFormat::U8, ChromaLocation::Left);
    let values: Vec<Vec<f64>> = (0..3)
        .map(|seed| blocky(48, 40, MatrixSamples::U8, seed))
        .collect();
    let planes: Vec<Vec<u8>> = values
        .iter()
        .map(|plane| encode_samples(plane, MatrixSamples::U8))
        .collect();
    let args = json!({ "planes": ["u", "y"], "strength": [50, 5], "precision": "f32" });

    let found = run_into_codes(STAGE, input, args, &planes, 8);

    let luma_values = as_f32(&values[0]);
    let blue_values = as_f32(&values[1]);
    let red_values = as_f32(&values[2]);
    let luma = direct(Variant::Gray, vec![luma_values], 48, 40, 5.0, 255.0);
    let blue = direct(Variant::Gray, vec![blue_values], 48, 40, 50.0, 255.0);
    assert_codes_within(&found[0], &luma[0], CODE_TOLERANCE, "luma at 5");
    assert_codes_within(&found[1], &blue[0], CODE_TOLERANCE, "blue at 50");
    assert_codes_within(
        &found[2],
        &red_values,
        CODE_TOLERANCE,
        "the unselected red difference plane",
    );
}

#[test]
fn gray_mode_runs_luma_and_half_size_chroma_in_one_frame() {
    if !installed(Variant::Gray) {
        return;
    }

    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(48, 40, layout, SampleFormat::U8, ChromaLocation::Left);
    let sizes: Vec<(u32, u32)> = (0..3).map(|index| plane_samples(&input, index)).collect();
    let values: Vec<Vec<f64>> = sizes
        .iter()
        .zip(0u32..)
        .map(|((width, height), seed)| blocky(*width, *height, MatrixSamples::U8, seed))
        .collect();
    let planes: Vec<Vec<u8>> = values
        .iter()
        .map(|plane| encode_samples(plane, MatrixSamples::U8))
        .collect();
    let args = json!({ "precision": "f32" });

    let found = run_into_codes(STAGE, input, args, &planes, 8);

    for index in 0..3 {
        let (width, height) = sizes[index];
        let network_values = as_f32(&values[index]);
        let expected = direct(Variant::Gray, vec![network_values], width, height, 10.0, 255.0);
        let label = format!("plane {index}");

        assert_codes_within(&found[index], &expected[0], CODE_TOLERANCE, &label);
    }
}

#[test]
fn a_frame_whose_sides_are_not_multiples_of_eight_comes_back_its_own_size() {
    if !installed(Variant::Gray) {
        return;
    }

    let input = frame(50, 38, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let values = blocky(50, 38, MatrixSamples::U8, 3);
    let encoded = encode_samples(&values, MatrixSamples::U8);
    let planes = vec![encoded];
    let args = json!({ "precision": "f32" });

    let found = run_into_codes(STAGE, input, args, &planes, 8);

    let network_values = as_f32(&values);
    let expected = direct(Variant::Gray, vec![network_values], 50, 38, 10.0, 255.0);
    assert_codes_within(&found[0], &expected[0], CODE_TOLERANCE, "50x38");
}

#[test]
fn a_ten_bit_frame_comes_back_as_f32() {
    if !installed(Variant::Gray) {
        return;
    }

    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let input = frame(48, 40, FrameLayout::Grey, ten_bit, ChromaLocation::Left);
    let samples = MatrixSamples::U16 { bits: 10 };
    let values = blocky(48, 40, samples, 1);
    let encoded = encode_samples(&values, samples);
    let planes = vec![encoded];
    let args = json!({ "precision": "f32" });

    let found = run_into_codes(STAGE, input, args, &planes, 10);

    let network_values = as_f32(&values);
    let expected = direct(Variant::Gray, vec![network_values], 48, 40, 10.0, 1023.0);
    assert_codes_within(&found[0], &expected[0], CODE_TOLERANCE, "ten bit");
}

#[test]
fn an_unrun_chroma_plane_becomes_f32_centred_on_zero() {
    if !installed(Variant::Gray) {
        return;
    }

    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = frame(48, 40, layout, SampleFormat::U8, ChromaLocation::Left);
    let output = FrameFormat {
        sample: SampleFormat::F32,
        ..input
    };
    let luma = blocky(48, 40, MatrixSamples::U8, 0);
    let neutral = vec![128.0; 48 * 40];
    let chroma = blocky(48, 40, MatrixSamples::U8, 1);
    let planes = vec![
        encode_samples(&luma, MatrixSamples::U8),
        encode_samples(&neutral, MatrixSamples::U8),
        encode_samples(&chroma, MatrixSamples::U8),
    ];
    let args = json!({ "planes": ["y"], "precision": "f32" });

    let found = run_once(STAGE, input, output, args, &planes);

    let neutral_samples = float_samples(&found[1]);
    assert!(
        neutral_samples.iter().all(|sample| *sample == 0.0),
        "neutral chroma is not centred on zero"
    );

    let found_codes = planes_as_codes(&output, &found, 8);
    let chroma_values = as_f32(&chroma);
    assert_codes_within(
        &found_codes[2],
        &chroma_values,
        CODE_TOLERANCE,
        "the unrun red difference plane",
    );
}

#[test]
fn float_chroma_is_centred_for_the_network_and_back() {
    if !installed(Variant::Gray) {
        return;
    }

    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = frame(48, 40, layout, SampleFormat::F32, ChromaLocation::Left);
    let luma = blocky(48, 40, MatrixSamples::F32, 0);
    let unshifted = blocky(48, 40, MatrixSamples::F32, 1);
    let chroma: Vec<f64> = unshifted.iter().map(|value| value - 0.5).collect();
    let encoded_luma = encode_samples(&luma, MatrixSamples::F32);
    let encoded_chroma = encode_samples(&chroma, MatrixSamples::F32);
    let planes = vec![encoded_luma, encoded_chroma.clone(), encoded_chroma];
    let args = json!({ "planes": ["u"], "precision": "f32" });

    let found = run_once(STAGE, input, input, args, &planes);

    let shifted: Vec<f32> = chroma.iter().map(|value| (value + 0.5) as f32).collect();
    let expected = direct(Variant::Gray, vec![shifted], 48, 40, 10.0, 1.0);
    let expected_values: Vec<f64> = expected[0].iter().map(|value| f64::from(*value) - 0.5).collect();
    let found_values = decode_samples(&found[1], MatrixSamples::F32);
    assert!(
        expected_values.iter().any(|value| *value < -0.1),
        "the expected chroma is not centred on zero"
    );
    assert_within(&found_values, &expected_values, 1e-4, "float chroma");
    assert_eq!(found[0], planes[0], "the unselected luma plane changed");
    assert_eq!(found[2], planes[2], "the unselected red difference plane changed");
}

#[test]
fn the_colour_model_on_rgb_matches_the_network() {
    if !installed(Variant::Colour) {
        return;
    }

    let input = frame(40, 48, FrameLayout::Rgb, SampleFormat::U8, ChromaLocation::Left);
    let values: Vec<Vec<f64>> = (0..3)
        .map(|seed| blocky(40, 48, MatrixSamples::U8, seed))
        .collect();
    let planes: Vec<Vec<u8>> = values
        .iter()
        .map(|plane| encode_samples(plane, MatrixSamples::U8))
        .collect();
    let args = json!({ "strength": 30, "precision": "f32" });

    let found = run_into_codes(STAGE, input, args, &planes, 8);

    let network_planes: Vec<Vec<f32>> = values.iter().map(|plane| as_f32(plane)).collect();
    let expected = direct(Variant::Colour, network_planes, 40, 48, 30.0, 255.0);
    for index in 0..3 {
        let label = format!("rgb plane {index}");

        assert_codes_within(&found[index], &expected[index], CODE_TOLERANCE, &label);
    }
}

#[test]
fn the_default_f16_precision_matches_the_f32_run() {
    if !installed(Variant::Gray) {
        return;
    }

    let backend = crate::support::backend();
    if !backend.capabilities().can_compute(ElementType::F16) {
        println!("the device has no f16 arithmetic, so the f16 stage is not run");
        return;
    }

    let input = frame(48, 40, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let values = blocky(48, 40, MatrixSamples::U8, 2);
    let encoded = encode_samples(&values, MatrixSamples::U8);
    let planes = vec![encoded];
    let full_args = json!({ "precision": "f32" });

    let half = run_into_codes(STAGE, input, json!({}), &planes, 8);
    let full = run_into_codes(STAGE, input, full_args, &planes, 8);

    let half_values = &half[0];
    let full_values = &full[0];
    let differences: Vec<f32> = half_values
        .iter()
        .zip(full_values)
        .map(|(half_value, full_value)| (half_value - full_value).abs())
        .collect();
    let largest = differences.iter().copied().fold(0.0, f32::max);
    let differing = differences
        .iter()
        .filter(|difference| **difference > CODE_TOLERANCE)
        .count();
    println!("f16 against f32: largest difference {largest}, {differing} codes differ");

    let input_values = as_f32(&values);
    assert_codes_within(half_values, full_values, F16_CODE_BOUND, "f16 against f32");
    assert_ne!(*half_values, input_values, "the stage changed nothing");
}

const H264_STAGE: &str = "deblock/h264";

/// Blocky planes for every channel of `format`, as code values of `samples`.
fn h264_planes(format: &FrameFormat, samples: MatrixSamples) -> Vec<Vec<f64>> {
    let channels = format.channels().len();

    (0..channels)
        .map(|index| {
            let (width, height) = plane_samples(format, index);

            blocky_plane(width, height, samples, index as u32)
        })
        .collect()
}

/// Every plane of `format` deblocked on the CPU at `quant`, padded the way the stage pads it.
fn h264_reference(
    format: &FrameFormat,
    planes: &[Vec<f64>],
    samples: MatrixSamples,
    quant: i32,
) -> Vec<Vec<f64>> {
    let (padded_width, padded_height) = padded_frame(format.width, format.height);
    let frame_thresholds = thresholds(quant, 0, 0, samples);
    let channels = format.channels();

    planes
        .iter()
        .zip(channels)
        .enumerate()
        .map(|(index, (values, channel))| {
            let (width, height) = plane_samples(format, index);
            let padded = format
                .layout
                .channel_dimensions(*channel, padded_width, padded_height);
            let (plane_padded_width, plane_padded_height) = padded.expect("a carried channel");
            let plane = H264Plane {
                values,
                width,
                height,
                padded_width: plane_padded_width,
                padded_height: plane_padded_height,
            };

            reference_h264(plane, samples, &frame_thresholds)
        })
        .collect()
}

/// Runs `stage` with `args` on `planes` and reads every plane back as code values.
fn run_decoded(
    stage: &str,
    format: FrameFormat,
    samples: MatrixSamples,
    args: serde_json::Value,
    planes: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let encoded: Vec<Vec<u8>> = planes
        .iter()
        .map(|plane| encode_samples(plane, samples))
        .collect();

    let found = run_once(stage, format, format, args, &encoded);

    found.iter().map(|bytes| decode_samples(bytes, samples)).collect()
}

/// Runs `deblock/h264` with `args` on `planes` and reads every plane back as code values.
fn run_h264(
    format: FrameFormat,
    samples: MatrixSamples,
    args: serde_json::Value,
    planes: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    run_decoded(H264_STAGE, format, samples, args, planes)
}

#[test]
fn h264_on_a_420_frame_matches_the_reference() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(64, 48, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_h264(input, samples, json!({ "quant": 51 }), &planes);

    let expected = h264_reference(&input, &planes, samples, 51);

    for index in 0..3 {
        let (width, _) = plane_samples(&input, index);
        let label = format!("plane {index}");

        assert_h264_plane(&found[index], &expected[index], width, samples, &label);
        assert_ne!(found[index], planes[index], "plane {index} came back unchanged");
    }
}

#[test]
fn h264_leaves_planes_not_selected_unchanged() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(64, 48, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_h264(input, samples, json!({ "quant": 51, "planes": ["y"] }), &planes);

    let expected = h264_reference(&input, &planes, samples, 51);
    assert_h264_plane(&found[0], &expected[0], 64, samples, "luma");
    assert_eq!(found[1], planes[1], "the unselected U plane changed");
    assert_eq!(found[2], planes[2], "the unselected V plane changed");
}

#[test]
fn h264_returns_a_frame_whose_sides_are_not_multiples_of_eight_at_its_own_size() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let input = frame(70, 46, layout, ten_bit, ChromaLocation::Left);
    let samples = MatrixSamples::U16 { bits: 10 };
    let planes = h264_planes(&input, samples);

    let found = run_h264(input, samples, json!({ "quant": 40 }), &planes);

    let expected = h264_reference(&input, &planes, samples, 40);
    assert_eq!(found.len(), 3);

    for (index, plane) in found.iter().enumerate() {
        let (width, height) = plane_samples(&input, index);
        let label = format!("plane {index}");

        assert_eq!(
            plane.len(),
            (width * height) as usize,
            "{label} is {width}x{height}"
        );
        assert_h264_plane(plane, &expected[index], width, samples, &label);
    }
}

#[test]
fn h264_on_a_float_frame_matches_the_reference_within_tolerance() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = frame(70, 45, layout, SampleFormat::F32, ChromaLocation::Left);
    let samples = MatrixSamples::F32;
    let unshifted = h264_planes(&input, samples);
    let planes: Vec<Vec<f64>> = unshifted
        .iter()
        .enumerate()
        .map(|(index, plane)| {
            let centre = if index == 0 { 0.0 } else { 0.5 };

            plane.iter().map(|value| value - centre).collect()
        })
        .collect();

    let found = run_h264(input, samples, json!({ "quant": 60 }), &planes);

    let expected = h264_reference(&input, &planes, samples, 60);
    for index in 0..3 {
        let label = format!("plane {index}");

        assert_h264_plane(&found[index], &expected[index], 70, samples, &label);
    }
}

#[test]
fn h264_on_a_frame_smaller_than_its_pad_comes_back_its_own_size() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(5, 3, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_h264(input, samples, json!({ "quant": 51 }), &planes);

    let expected = h264_reference(&input, &planes, samples, 51);

    for index in 0..3 {
        let (width, height) = plane_samples(&input, index);
        let label = format!("plane {index}");

        assert_eq!(
            found[index].len(),
            (width * height) as usize,
            "{label} is {width}x{height}"
        );
        assert_h264_plane(&found[index], &expected[index], width, samples, &label);
    }
}

#[test]
fn h264_on_a_nine_bit_grey_frame_matches_the_reference() {
    let nine_bit = SampleFormat::U16 {
        depth: Depth::new(9).expect("nine is a depth"),
    };
    let input = frame(64, 48, FrameLayout::Grey, nine_bit, ChromaLocation::Left);
    let samples = MatrixSamples::U16 { bits: 9 };
    let planes = h264_planes(&input, samples);

    let found = run_h264(input, samples, json!({ "quant": 51 }), &planes);

    let expected = h264_reference(&input, &planes, samples, 51);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].len(), 64 * 48);
    assert_h264_plane(&found[0], &expected[0], 64, samples, "grey");
    assert_ne!(found[0], planes[0], "the grey plane came back unchanged");
}

#[test]
fn h264_on_an_rgb_frame_filters_all_three_planes() {
    let input = frame(64, 48, FrameLayout::Rgb, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_h264(input, samples, json!({ "quant": 51 }), &planes);

    let expected = h264_reference(&input, &planes, samples, 51);
    assert_eq!(found.len(), 3);

    for index in 0..3 {
        let label = format!("rgb plane {index}");

        assert_eq!(found[index].len(), 64 * 48, "{label} changed size");
        assert_h264_plane(&found[index], &expected[index], 64, samples, &label);
        assert_ne!(found[index], planes[index], "{label} came back unchanged");
    }
}

const QED_STAGE: &str = "deblock/qed";

/// The [FixtureLayout] of the frames these tests build.
fn qed_layout(layout: FrameLayout) -> FixtureLayout {
    match layout {
        FrameLayout::Grey => FixtureLayout::Grey,
        FrameLayout::Rgb => FixtureLayout::Rgb,
        FrameLayout::Yuv { subsampling } => {
            let chroma_shift = match subsampling {
                Subsampling::Yuv444 => (0, 0),
                Subsampling::Yuv422 => (1, 0),
                Subsampling::Yuv420 => (1, 1),
            };

            FixtureLayout::Yuv { chroma_shift }
        },
        FrameLayout::Chroma { .. } => panic!("no test here builds a chroma-only frame"),
    }
}

/// Every plane of a `format` frame deblocked on the CPU with `deblock/qed`'s default arguments and `chroma`.
fn qed_reference(
    format: &FrameFormat,
    planes: &[Vec<f64>],
    samples: MatrixSamples,
    chroma: QedChroma,
) -> QedReference {
    let layout = qed_layout(format.layout);
    let case = QedCase {
        width: format.width,
        height: format.height,
        layout,
        samples,
        quant: [24, 26],
        aoffset: [1, 1],
        boffset: [2, 2],
        chroma,
    };

    reference_qed(&case, planes)
}

/// `planes` with every chroma plane moved down by 0.5, so float chroma is centred on 0.
fn centred_chroma(planes: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    planes
        .into_iter()
        .enumerate()
        .map(|(index, plane)| {
            let centre = if index == 0 { 0.0 } else { 0.5 };

            plane.iter().map(|value| value - centre).collect()
        })
        .collect()
}

/// `plane` as the device sees it, rounded to the stored sample type.
fn stored_as_f32(plane: &[f64], samples: MatrixSamples) -> Vec<f64> {
    let encoded = encode_samples(plane, samples);

    decode_samples(&encoded, samples)
}

#[test]
fn qed_on_a_420_frame_matches_the_reference_under_each_chroma() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(64, 48, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);
    let settings = [
        ("qed", QedChroma::Qed),
        ("normal", QedChroma::Normal),
        ("strong", QedChroma::Strong),
    ];

    for (name, chroma) in settings {
        let found = run_decoded(QED_STAGE, input, samples, json!({ "chroma": name }), &planes);

        let expected = qed_reference(&input, &planes, samples, chroma);
        for index in 1..3 {
            let passes_differ = expected.normal[index] != expected.strong[index];

            assert!(
                passes_differ,
                "chroma {name}: plane {index} takes the same either pass"
            );
        }

        assert_eq!(found, expected.output, "chroma {name}");
        assert_ne!(found[0], planes[0], "chroma {name}: luma came back unchanged");
    }
}

#[test]
fn qed_leaves_planes_not_selected_unchanged() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(64, 48, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_decoded(QED_STAGE, input, samples, json!({ "planes": ["y"] }), &planes);

    let expected = qed_reference(&input, &planes, samples, QedChroma::Qed);
    assert_eq!(found[0], expected.output[0], "luma");
    assert_eq!(found[1], planes[1], "the unselected U plane changed");
    assert_eq!(found[2], planes[2], "the unselected V plane changed");
}

#[test]
fn qed_on_selected_chroma_under_normal_takes_the_normal_pass() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(64, 48, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);
    let args = json!({ "chroma": "normal", "planes": ["u", "v"] });

    let found = run_decoded(QED_STAGE, input, samples, args, &planes);

    let expected = qed_reference(&input, &planes, samples, QedChroma::Normal);
    assert_ne!(
        expected.normal[1], expected.strong[1],
        "U takes the same either pass"
    );
    assert_ne!(
        expected.normal[2], expected.strong[2],
        "V takes the same either pass"
    );
    assert_eq!(found[0], planes[0], "the unselected luma changed");
    assert_eq!(found[1], expected.normal[1], "U against its normal pass");
    assert_eq!(found[2], expected.normal[2], "V against its normal pass");
    assert_ne!(found[1], planes[1], "U came back unchanged");
}

#[test]
fn qed_returns_a_70x46_frame_at_its_own_size() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let input = frame(70, 46, layout, ten_bit, ChromaLocation::Left);
    let samples = MatrixSamples::U16 { bits: 10 };
    let planes = h264_planes(&input, samples);

    let found = run_decoded(QED_STAGE, input, samples, json!({}), &planes);

    let expected = qed_reference(&input, &planes, samples, QedChroma::Qed);
    assert_eq!(found.len(), 3);
    for (index, plane) in found.iter().enumerate() {
        let (width, height) = plane_samples(&input, index);
        let label = format!("plane {index}");

        assert_eq!(
            plane.len(),
            (width * height) as usize,
            "{label} is {width}x{height}"
        );
        assert_eq!(*plane, expected.output[index], "{label}");
    }
}

#[test]
fn qed_on_an_odd_420_frame_matches_the_reference() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(71, 45, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_decoded(QED_STAGE, input, samples, json!({}), &planes);

    let expected = qed_reference(&input, &planes, samples, QedChroma::Qed);
    assert_eq!(plane_samples(&input, 1), (36, 23));
    assert_eq!(found, expected.output);
}

#[test]
fn qed_on_an_rgb_frame_treats_every_plane_like_luma() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = frame(64, 48, FrameLayout::Rgb, sixteen_bit, ChromaLocation::Left);
    let samples = MatrixSamples::U16 { bits: 16 };
    let planes = h264_planes(&input, samples);

    let found = run_decoded(QED_STAGE, input, samples, json!({}), &planes);

    let expected = qed_reference(&input, &planes, samples, QedChroma::Qed);
    assert_eq!(expected.changes.len(), 3, "every RGB plane takes the QED output");
    for index in 0..3 {
        let label = format!("rgb plane {index}");

        assert_eq!(found[index], expected.output[index], "{label}");
        assert_ne!(found[index], planes[index], "{label} came back unchanged");
    }
}

#[test]
fn qed_on_a_float_frame_matches_the_reference_exactly() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let input = frame(70, 46, layout, SampleFormat::F32, ChromaLocation::Left);
    let samples = MatrixSamples::F32;
    let unshifted = h264_planes(&input, samples);
    let centred = centred_chroma(unshifted);
    let planes: Vec<Vec<f64>> = centred
        .iter()
        .map(|plane| stored_as_f32(plane, samples))
        .collect();

    let found = run_decoded(QED_STAGE, input, samples, json!({}), &planes);

    let expected = qed_reference(&input, &planes, samples, QedChroma::Qed);
    assert_eq!(found, expected.output);
}

#[test]
fn qed_on_a_frame_smaller_than_one_block_comes_back_its_own_size() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(6, 4, layout, SampleFormat::U8, ChromaLocation::Left);
    let samples = MatrixSamples::U8;
    let planes = h264_planes(&input, samples);

    let found = run_decoded(QED_STAGE, input, samples, json!({}), &planes);

    let expected = qed_reference(&input, &planes, samples, QedChroma::Qed);
    for (index, plane) in found.iter().enumerate() {
        let (width, height) = plane_samples(&input, index);

        assert_eq!(
            plane.len(),
            (width * height) as usize,
            "plane {index} is {width}x{height}"
        );
    }

    assert_eq!(found, expected.output);
}
