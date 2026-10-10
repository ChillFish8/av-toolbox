use av_toolbox_cubek::cnn::{Precision, Stride, StridedLayer, StridedVariant, run_strided, strided_variant};
use av_toolbox_cubek::dpir::{LayerKind, WeightLayer};
use av_toolbox_cubek::testing::dpir::{reference_down, reference_up};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::layers::{Down, Up, assert_matches, bound, stored_layer, weight_layer};
use crate::artcnn::{as_stored, f16_step, read_maps, signed_values, synced, upload_maps};
use crate::support::{backend, empty};

/// The sizes each layer runs at. The first gives an output 17 samples wide, past one 16-sample tile, and the
/// second the smallest map.
const DOWN_SIZES: [(u32, u32); 3] = [(34, 6), (2, 2), (36, 10)];
const UP_SIZES: [(u32, u32); 3] = [(17, 3), (1, 1), (9, 5)];

/// A down or up layer run by its strided kernel, or `None` where the device offers none.
struct Strided {
    stride: Stride,
    layer: WeightLayer,
    maps: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Strided {
    type Output = Option<Vec<f32>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let properties = client.properties();
        let channels = self.layer.in_channels;
        let variant = strided_variant(properties, self.stride, channels)?;
        let output_values = match self.stride {
            Stride::Down => self.maps.len() / 2,
            Stride::Up => self.maps.len() * 2,
        };

        let input = upload_maps(client, &self.maps, Precision::F16);
        let weights = upload_maps(client, &self.layer.values, Precision::F16);
        let output = empty(client, output_values * Precision::F16.bytes());
        let layer = StridedLayer {
            input: &input,
            weights: &weights,
            output: &output,
            width: self.width,
            height: self.height,
            channels,
            stride: self.stride,
        };

        let ran = run_strided(client, variant, &layer);
        synced(client);
        ran.expect("the strided kernel runs");

        let values = read_maps(client, &output, Precision::F16);

        Some(values)
    }
}

fn check_down(channels: u32, width: u32, height: u32) {
    let layer = weight_layer(LayerKind::Down, channels, 2 * channels, 71, 0.25);
    let maps = signed_values((width * height * channels) as usize, 72, 1.0);
    let stored_maps = as_stored(&maps, Precision::F16);
    let stored = stored_layer(&layer, Precision::F16);
    let expected = reference_down(&stored_maps, width, height, &stored);
    let task = Strided {
        stride: Stride::Down,
        layer: layer.clone(),
        maps: maps.clone(),
        width,
        height,
    };

    let Some(actual) = backend().build(task) else {
        println!("down from {channels}: no strided kernel on this device");
        return;
    };

    let name = format!("strided down from {channels} at {width}x{height}");
    let allowed = bound(Precision::F16, 4 * channels);
    assert_matches(&name, Precision::F16, allowed, &actual, &expected);

    let fallback = Down {
        layer,
        maps,
        width,
        height,
        precision: Precision::F16,
    };
    let gathered = backend().build(fallback);
    assert_within_one_step(&name, &actual, &gathered);
}

fn check_up(channels: u32, width: u32, height: u32) {
    let layer = weight_layer(LayerKind::Up, channels, channels / 2, 81, 0.25);
    let maps = signed_values((width * height * channels) as usize, 82, 1.0);
    let stored_maps = as_stored(&maps, Precision::F16);
    let stored = stored_layer(&layer, Precision::F16);
    let expected = reference_up(&stored_maps, width, height, &stored);
    let task = Strided {
        stride: Stride::Up,
        layer: layer.clone(),
        maps: maps.clone(),
        width,
        height,
    };

    let Some(actual) = backend().build(task) else {
        println!("up from {channels}: no strided kernel on this device");
        return;
    };

    let name = format!("strided up from {channels} at {width}x{height}");
    let allowed = bound(Precision::F16, channels);
    assert_matches(&name, Precision::F16, allowed, &actual, &expected);

    let fallback = Up {
        layer,
        maps,
        width,
        height,
        precision: Precision::F16,
    };
    let scattered = backend().build(fallback);
    assert_within_one_step(&name, &actual, &scattered);
}

/// Asserts each value lies within one f16 step of the gather or scatter path's.
fn assert_within_one_step(name: &str, actual: &[f32], fallback: &[f32]) {
    assert_eq!(actual.len(), fallback.len(), "{name}");

    for (index, (got, want)) in actual.iter().zip(fallback).enumerate() {
        let step = f16_step(want.abs());
        let error = (got - want).abs();

        assert!(
            error <= step,
            "{name}: value {index} is {got}, the fallback's {want}"
        );
    }
}

#[test]
fn the_strided_down_kernel_matches_the_reference_and_the_gather() {
    for channels in [64, 128, 256] {
        for (width, height) in DOWN_SIZES {
            check_down(channels, width, height);
        }
    }
}

#[test]
fn the_strided_up_kernel_matches_the_reference_and_the_scatter() {
    for channels in [128, 256, 512] {
        for (width, height) in UP_SIZES {
            check_up(channels, width, height);
        }
    }
}

struct InputIsOutput;

impl BuildTask for InputIsOutput {
    type Output = ();

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let channels = 64;
        // The buffers are checked before the device, so any variant reaches the check.
        let variant = StridedVariant {
            pixel_tiles: 1,
            channel_tiles: 8,
            planes_m: 1,
            planes_n: 1,
            plane_width: 64,
        };
        let maps = upload_maps(client, &vec![0.0; 4 * 4 * channels as usize], Precision::F16);
        let weights = upload_maps(
            client,
            &vec![0.0; 16 * channels as usize * 2 * channels as usize],
            Precision::F16,
        );
        let layer = StridedLayer {
            input: &maps,
            weights: &weights,
            output: &maps,
            width: 4,
            height: 4,
            channels,
            stride: Stride::Down,
        };

        let _ = run_strided(client, variant, &layer);
    }
}

#[test]
#[should_panic(expected = "the output is the input")]
fn a_strided_layer_whose_output_is_its_input_is_refused() {
    backend().build(InputIsOutput);
}
