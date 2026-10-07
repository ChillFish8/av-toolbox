use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::cnn::{BodyPath, Precision};
use av_toolbox_cubek::dpir::{Dpir, DpirMaps, DpirPlanes, PACK, Variant, Weights, deblock, scratch_sizes};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_cubek::testing::{decode_samples, encode_samples};
use av_toolbox_gpu::{
    BuildContext,
    BuildTask,
    ChromaLocation,
    Depth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    Subsampling,
};
use av_toolbox_models::Store;
use cubecl::prelude::{CubeElement, Runtime};
use serde_json::json;

use crate::support::{Sizing, frame, run_once};

const STAGE: &str = "deblock/dpir";

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

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
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

fn rounded(values: &[f32], peak: f32) -> Vec<f64> {
    values
        .iter()
        .map(|value| f64::from((value + 0.5).floor().clamp(0.0, peak)))
        .collect()
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

    let found = run_once(STAGE, input, input, args, &planes);

    let network_values = as_f32(&values);
    let expected = direct(Variant::Gray, vec![network_values], 48, 40, 10.0, 255.0);
    let found_values = decode_samples(&found[0], MatrixSamples::U8);
    let expected_values = rounded(&expected[0], 255.0);
    assert_within(&found_values, &expected_values, 1.0, "grey u8");
    assert_ne!(found_values, values, "the stage changed nothing");
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

    let found = run_once(STAGE, input, input, args, &planes);

    let luma_values = as_f32(&values[0]);
    let blue_values = as_f32(&values[1]);
    let luma = direct(Variant::Gray, vec![luma_values], 48, 40, 5.0, 255.0);
    let blue = direct(Variant::Gray, vec![blue_values], 48, 40, 50.0, 255.0);
    let found_luma = decode_samples(&found[0], MatrixSamples::U8);
    let found_blue = decode_samples(&found[1], MatrixSamples::U8);
    let expected_luma = rounded(&luma[0], 255.0);
    let expected_blue = rounded(&blue[0], 255.0);
    assert_within(&found_luma, &expected_luma, 1.0, "luma at 5");
    assert_within(&found_blue, &expected_blue, 1.0, "blue at 50");
    assert_eq!(found[2], planes[2], "the unselected red difference plane changed");
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

    let found = run_once(STAGE, input, input, args, &planes);

    for index in 0..3 {
        let (width, height) = sizes[index];
        let network_values = as_f32(&values[index]);
        let expected = direct(Variant::Gray, vec![network_values], width, height, 10.0, 255.0);
        let found_values = decode_samples(&found[index], MatrixSamples::U8);
        let expected_values = rounded(&expected[0], 255.0);
        let label = format!("plane {index}");

        assert_within(&found_values, &expected_values, 1.0, &label);
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

    let found = run_once(STAGE, input, input, args, &planes);

    let network_values = as_f32(&values);
    let expected = direct(Variant::Gray, vec![network_values], 50, 38, 10.0, 255.0);
    let found_values = decode_samples(&found[0], MatrixSamples::U8);
    let expected_values = rounded(&expected[0], 255.0);
    assert_within(&found_values, &expected_values, 1.0, "50x38");
}

#[test]
fn a_ten_bit_frame_comes_back_at_ten_bits() {
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

    let found = run_once(STAGE, input, input, args, &planes);

    let network_values = as_f32(&values);
    let expected = direct(Variant::Gray, vec![network_values], 48, 40, 10.0, 1023.0);
    let found_values = decode_samples(&found[0], samples);
    let expected_values = rounded(&expected[0], 1023.0);
    assert_within(&found_values, &expected_values, 1.0, "ten bit");
    assert!(found_values.iter().all(|value| *value <= 1023.0));
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

    let found = run_once(STAGE, input, input, args, &planes);

    let network_planes: Vec<Vec<f32>> = values.iter().map(|plane| as_f32(plane)).collect();
    let expected = direct(Variant::Colour, network_planes, 40, 48, 30.0, 255.0);
    for index in 0..3 {
        let found_values = decode_samples(&found[index], MatrixSamples::U8);
        let expected_values = rounded(&expected[index], 255.0);
        let label = format!("rgb plane {index}");

        assert_within(&found_values, &expected_values, 1.0, &label);
    }
}
