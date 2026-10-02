use av_toolbox_cubek::artcnn::{
    ArtcnnError,
    BodyPath,
    Maps,
    Network,
    Precision,
    TILE_BUDGET,
    Weights,
    artcnn,
    fits_whole,
    scratch_sizes,
    takes_forced_blueprint,
    tiles,
    warm_up,
};
use av_toolbox_cubek::testing::{reference_artcnn, synthetic_weights};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::body::{PEAK, UNTOUCHED, largest_error, r8f64, supported_runs, supports};
use super::{forced_eligible, synced, textured};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// Runs the network over a plane in `0..=1` one tile at a time.
///
/// It gives the doubled output divided by the peak.
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
        let network = Network::upload(client, &self.weights, self.precision).expect("the weights are valid");
        let sizes = scratch_sizes(&network, self.path, self.tile, self.width, self.height);
        let output_samples = (self.width * self.height * 4) as usize;
        let untouched = vec![UNTOUCHED; output_samples];
        let scaled: Vec<f32> = self.plane.iter().map(|value| value * PEAK).collect();

        let input = upload_f32(client, &scaled);
        let output = upload_f32(client, &untouched);
        let conv0 = empty(client, sizes.conv0_bytes);
        let first = empty(client, sizes.map_bytes);
        let second = empty(client, sizes.map_bytes);
        let third = empty(client, sizes.map_bytes);
        let columns = match self.path {
            BodyPath::Convolution | BodyPath::UnfusedConvolution => None,
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
    width: u32,
    height: u32,
    tile: u32,
}

impl BuildTask for WarmUp {
    type Output = Result<(), ArtcnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let network = Network::upload(client, &self.weights, self.precision).expect("the weights are valid");
        let sizes = scratch_sizes(&network, self.path, self.tile, self.width, self.height);

        let conv0 = empty(client, sizes.conv0_bytes);
        let first = empty(client, sizes.map_bytes);
        let second = empty(client, sizes.map_bytes);
        let third = empty(client, sizes.map_bytes);
        let columns = match self.path {
            BodyPath::Convolution | BodyPath::UnfusedConvolution => None,
            BodyPath::Im2col => Some(empty(client, sizes.column_bytes)),
        };
        let maps = Maps {
            conv0: &conv0,
            first: &first,
            second: &second,
            third: &third,
            columns: columns.as_ref(),
        };

        warm_up(
            client,
            &sizing,
            &network,
            self.path,
            self.width,
            self.height,
            self.tile,
            maps,
        )
    }
}

/// How much of the scratch sizes' `conv0` a [SmallFrame] gives it.
enum Conv0Size {
    /// What the scratch sizes ask for.
    Scratch,
    /// One feature map, which is too small on a narrow convolution path.
    OneMap,
}

/// Runs the network over a 6x5 plane of zeros at a tile of 8, into an output `output_width` by
/// `output_height`.
struct SmallFrame {
    weights: Weights,
    path: BodyPath,
    precision: Precision,
    output_width: u32,
    output_height: u32,
    conv0: Conv0Size,
    with_columns: bool,
}

impl BuildTask for SmallFrame {
    type Output = Result<(), ArtcnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let network = Network::upload(client, &self.weights, self.precision).expect("the weights are valid");
        let sizes = scratch_sizes(&network, self.path, 8, 6, 5);
        let zeros = vec![0.0f32; 30];
        let output_bytes = (self.output_width * self.output_height) as usize * size_of::<f32>();
        let conv0_bytes = match self.conv0 {
            Conv0Size::Scratch => sizes.conv0_bytes,
            Conv0Size::OneMap => sizes.map_bytes,
        };

        let input = upload_f32(client, &zeros);
        let output = empty(client, output_bytes);
        let conv0 = empty(client, conv0_bytes);
        let first = empty(client, sizes.map_bytes);
        let second = empty(client, sizes.map_bytes);
        let third = empty(client, sizes.map_bytes);
        let columns = self.with_columns.then(|| empty(client, sizes.column_bytes));

        let input_plane = plane(&input, 6, 5);
        let output_plane = plane(&output, self.output_width, self.output_height);
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
            8,
            maps,
            output_plane,
        );
        synced(client);

        ran
    }
}

/// Whether each region a tiled run of a `width` by `height` frame at `tile` convolves takes the forced
/// blueprint, and whether the device is eligible for it.
struct ForcedRegions {
    weights: Weights,
    precision: Precision,
    width: u32,
    height: u32,
    tile: u32,
}

impl BuildTask for ForcedRegions {
    type Output = (Vec<bool>, bool);

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let network = Network::upload(client, &self.weights, self.precision).expect("the weights are valid");
        let halo = network.halo();
        let eligible = forced_eligible(client);

        let regions = tiles(self.width, self.height, self.tile, halo);
        let forced = regions
            .iter()
            .map(|(region, _)| takes_forced_blueprint(client, &network, *region))
            .collect();

        (forced, eligible)
    }
}

/// Asserts every region of a 200x120 run at `tile` takes the forced blueprint if `large` is set and the run
/// is at f16 on an eligible device, and that none does otherwise.
fn assert_forced_regions(weights: &Weights, precision: Precision, tile: u32, large: bool) {
    let task = ForcedRegions {
        weights: weights.clone(),
        precision,
        width: 200,
        height: 120,
        tile,
    };

    let (regions, eligible) = backend().build(task);

    let forced = large && eligible && precision == Precision::F16;
    println!("tile {tile} at {precision:?}: eligible {eligible}, regions forced {regions:?}");
    assert!(
        regions.iter().all(|region| *region == forced),
        "tile {tile} at {precision:?}: expected every region forced {forced}"
    );
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

/// Asserts a tiled run wrote every sample and matches `untiled` exactly.
fn assert_identical(tiled: &[f32], untiled: &[f32], path: BodyPath, precision: Precision) {
    let difference = tiled
        .iter()
        .zip(untiled)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let differing = tiled
        .iter()
        .zip(untiled)
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

#[test]
fn a_tiled_run_matches_an_untiled_run_exactly() {
    let weights = synthetic_weights(8, 2, 6);
    let plane = textured(40, 33);
    let runs = supported_runs();

    for (path, precision) in runs {
        let tiled = run_tiled(&weights, &plane, 40, 33, path, precision, 8);
        let untiled = run_tiled(&weights, &plane, 40, 33, path, precision, 64);

        assert_identical(&tiled, &untiled, path, precision);
    }
}

#[test]
fn a_whole_frame_run_matches_a_small_tile_run_exactly_on_the_convolution_path() {
    // At 64 features the whole 200x120 frame is one region of 24,000 samples, which at f16 takes the forced
    // blueprint on an eligible device. Tiles of 24 with a halo of 6 are regions of at most 36x36, 1,296
    // samples, which take the inferred one.
    let weights = synthetic_weights(64, 1, 9);
    let plane = textured(200, 120);
    let runs = supported_runs()
        .into_iter()
        .filter(|(path, _)| *path == BodyPath::Convolution);

    for (path, precision) in runs {
        let fits = fits_whole(64, precision, path, 200, 120, TILE_BUDGET, usize::MAX);
        assert!(fits, "a 200x120 frame at {precision:?} runs whole");

        assert_forced_regions(&weights, precision, 200, true);
        assert_forced_regions(&weights, precision, 24, false);

        let small = run_tiled(&weights, &plane, 200, 120, path, precision, 24);
        let whole = run_tiled(&weights, &plane, 200, 120, path, precision, 200);

        assert_identical(&small, &whole, path, precision);
    }
}

#[test]
fn r8f64_tiled_matches_the_reference() {
    let weights = r8f64();
    let plane = textured(60, 40);
    let expected = reference_artcnn(&plane, 60, 40, &weights);
    let runs = supported_runs();

    for (path, precision) in runs {
        // With a halo of 27, the tiles at 12 and 36 across have unclipped right and left halos, and those
        // at 0 and 36 down have unclipped bottom and top halos.
        let actual = run_tiled(&weights, &plane, 60, 40, path, precision, 12);

        let error = largest_error(&actual, &expected);
        let codes = error * 255.0;
        println!("R8F64 tiled {path:?} at {precision:?}: largest error {error:e}, {codes:.3} 8-bit codes");

        let bound = match precision {
            // Measured 8.06e-6 on im2col, of which up to 7.63e-6 is rounding to 16-bit codes, so 2.5x
            // headroom.
            Precision::F32 => 2e-5,
            // Measured 2.20e-4 on the convolution path and 3.27e-4 on im2col, which is 0.083 of an 8-bit
            // code, so 3x headroom.
            Precision::F16 => 1e-3,
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
            width: 40,
            height: 30,
            tile: 16,
        };

        let warmed = backend().build(task);

        println!("warm-up {path:?} at {precision:?}: {warmed:?}");
        assert_eq!(warmed, Ok(()), "{path:?} at {precision:?}");
    }
}

#[test]
fn a_production_sized_plane_matches_the_reference() {
    let weights = synthetic_weights(8, 1, 7);
    let plane = textured(512, 512);
    let expected = reference_artcnn(&plane, 512, 512, &weights);
    let runs = supported_runs();

    for (path, precision) in runs {
        let actual = run_tiled(&weights, &plane, 512, 512, path, precision, 512);

        let error = largest_error(&actual, &expected);
        let codes = error * 255.0;
        println!("512x512 {path:?} at {precision:?}: largest error {error:e}, {codes:.3} 8-bit codes");

        let bound = match precision {
            // Measured 7.83e-6 on im2col, of which up to 7.63e-6 is rounding to 16-bit codes, so 2.6x
            // headroom.
            Precision::F32 => 2e-5,
            // Measured 6.36e-4 on the convolution path and 5.80e-4 on im2col, which is 0.162 of an 8-bit
            // code, so 3.1x headroom.
            Precision::F16 => 2e-3,
        };
        assert!(error <= bound, "{path:?} at {precision:?}: {error:e}");
    }
}

#[test]
fn the_warm_up_refuses_an_unsupported_convolution() {
    let weights = synthetic_weights(8, 1, 6);
    let path = BodyPath::Convolution;
    let precision = Precision::F32;
    let supported = supports(path, precision);
    let task = WarmUp {
        weights,
        path,
        precision,
        width: 40,
        height: 30,
        tile: 16,
    };

    let warmed = backend().build(task);

    println!("warm-up {path:?} at {precision:?}: {warmed:?}");
    if supported {
        assert_eq!(warmed, Ok(()));
        return;
    }

    assert!(matches!(warmed, Err(ArtcnnError::Convolution(_))), "{warmed:?}");
}

#[test]
#[should_panic(expected = "the output is not twice the input on each side")]
fn a_tiled_run_refuses_an_output_of_another_size() {
    let weights = synthetic_weights(8, 1, 8);
    // The output is two samples wider than the doubled input.
    let task = SmallFrame {
        weights,
        path: BodyPath::Im2col,
        precision: Precision::F32,
        output_width: 14,
        output_height: 10,
        conv0: Conv0Size::Scratch,
        with_columns: true,
    };

    let ran = backend().build(task);

    println!("wider output: {ran:?}");
}

#[test]
#[should_panic(expected = "conv0 is smaller than the tile's scratch")]
fn a_tiled_run_refuses_a_conv0_of_one_map_on_a_narrow_convolution_path() {
    let weights = synthetic_weights(8, 1, 8);
    let task = SmallFrame {
        weights,
        path: BodyPath::Convolution,
        precision: Precision::F16,
        output_width: 12,
        output_height: 10,
        conv0: Conv0Size::OneMap,
        with_columns: false,
    };

    let ran = backend().build(task);

    println!("one map conv0: {ran:?}");
}
