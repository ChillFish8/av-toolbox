use av_toolbox_cubek::artcnn::{
    ArtcnnError,
    BodyPath,
    Maps,
    Network,
    Precision,
    Weights,
    artcnn,
    scratch_sizes,
    warm_up,
};
use av_toolbox_cubek::testing::{reference_artcnn, synthetic_weights};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::body::{PEAK, UNTOUCHED, largest_error, r8f64, supported_runs};
use super::{synced, textured};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// Runs the network over a plane in `0..=1` one tile at a time, and gives the doubled output divided by the peak.
struct Tiled {
    weights: Weights,
    plane: Vec<f32>,
    width: u32,
    height: u32,
    path: BodyPath,
    precision: Precision,
    tile: u32,
}

impl BuildTask for Tiled {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let network = Network::upload(client, &self.weights, self.precision);
        let sizes = scratch_sizes(&network, self.path, self.tile);
        let output_samples = (self.width * self.height * 4) as usize;
        let untouched = vec![UNTOUCHED; output_samples];
        let scaled: Vec<f32> = self.plane.iter().map(|value| value * PEAK).collect();

        let input = upload_f32(client, &scaled);
        let output = upload_f32(client, &untouched);
        let conv0 = empty(client, sizes.map_bytes);
        let first = empty(client, sizes.map_bytes);
        let second = empty(client, sizes.map_bytes);
        let third = empty(client, sizes.map_bytes);
        let columns = match self.path {
            BodyPath::Convolution => None,
            BodyPath::Im2col => Some(empty(client, sizes.column_bytes)),
        };

        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width * 2, self.height * 2);
        let maps = Maps {
            conv0: &conv0,
            first: &first,
            second: &second,
            third: &third,
            columns: columns.as_ref(),
        };

        let ran = artcnn(
            client,
            &sizing,
            &network,
            self.path,
            input_plane,
            PEAK,
            self.tile,
            maps,
            output_plane,
        );
        synced(client);
        ran.expect("a supported path runs");

        let codes = read_f32(client, &output);

        codes.iter().map(|code| code / PEAK).collect()
    }
}

struct WarmUp {
    weights: Weights,
    path: BodyPath,
    precision: Precision,
}

impl BuildTask for WarmUp {
    type Output = Result<(), ArtcnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let network = Network::upload(client, &self.weights, self.precision);

        warm_up(client, &sizing, &network, self.path)
    }
}

fn run_tiled(
    weights: &Weights,
    plane: &[f32],
    width: u32,
    height: u32,
    path: BodyPath,
    precision: Precision,
    tile: u32,
) -> Vec<f32> {
    let task = Tiled {
        weights: weights.clone(),
        plane: plane.to_vec(),
        width,
        height,
        path,
        precision,
        tile,
    };

    backend().build(task)
}

#[test]
fn a_tiled_run_matches_an_untiled_run_exactly() {
    let weights = synthetic_weights(8, 2, 6);
    let plane = textured(40, 33);
    let runs = supported_runs();

    for (path, precision) in runs {
        let tiled = run_tiled(&weights, &plane, 40, 33, path, precision, 8);
        let untiled = run_tiled(&weights, &plane, 40, 33, path, precision, 64);

        let difference = tiled
            .iter()
            .zip(&untiled)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        let differing = tiled
            .iter()
            .zip(&untiled)
            .filter(|(left, right)| left != right)
            .count();
        println!("tiled {path:?} at {precision:?}: largest difference from untiled {difference:e}");

        assert_eq!(tiled.len(), untiled.len());
        assert!(
            tiled.iter().all(|value| *value >= 0.0),
            "a sample was never written"
        );
        assert_eq!(
            differing, 0,
            "{path:?} at {precision:?}: {differing} samples differ"
        );
    }
}

#[test]
fn r8f64_tiled_matches_the_reference() {
    let weights = r8f64();
    let plane = textured(70, 45);
    let expected = reference_artcnn(&plane, 70, 45, &weights);
    let runs = supported_runs();

    for (path, precision) in runs {
        let actual = run_tiled(&weights, &plane, 70, 45, path, precision, 16);

        let error = largest_error(&actual, &expected);
        let codes = error * 255.0;
        println!("R8F64 tiled {path:?} at {precision:?}: largest error {error:e}, {codes:.3} 8-bit codes");

        let bound = match precision {
            // Measured 8.08e-6 on im2col, of which up to 7.63e-6 is rounding to 16-bit codes, so 2.5x
            // headroom.
            Precision::F32 => 2e-5,
            // Measured 2.11e-4 on the convolution path and 2.42e-4 on im2col, which is 0.062 of an 8-bit
            // code, so 2x headroom.
            Precision::F16 => 5e-4,
        };
        assert!(error <= bound, "{path:?} at {precision:?}: {error:e}");
    }
}

#[test]
fn the_warm_up_succeeds_on_every_supported_path() {
    let weights = r8f64();
    let runs = supported_runs();

    for (path, precision) in runs {
        let task = WarmUp {
            weights: weights.clone(),
            path,
            precision,
        };

        let warmed = backend().build(task);

        println!("warm-up {path:?} at {precision:?}: {warmed:?}");
        assert_eq!(warmed, Ok(()), "{path:?} at {precision:?}");
    }
}
