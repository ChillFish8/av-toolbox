use av_toolbox_cubek::cnn::{BodyPath, CnnError, LayerRun, Precision, tiles};
use av_toolbox_cubek::dpir::{
    Dpir,
    DpirMaps,
    DpirPlanes,
    OVERLAP,
    Padded,
    ScratchSizes,
    Variant,
    Weights,
    deblock,
    scratch_sizes,
    tile_sizes,
    warm_up,
};
use av_toolbox_cubek::testing::dpir::{reference_dpir, synthetic_weights};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::{ComputeClient, Runtime};
use cubecl::server::Handle;

use super::network::{body_path, widened};
use crate::artcnn::{largest_error, synced, textured};
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// The largest absolute error allowed against the reference in f32.
const F32_BOUND: f64 = 1e-4;

/// The largest absolute difference allowed between a peak of 255 and a peak of 1 scaled by 255.
const PEAK_BOUND: f64 = 1e-3;

/// The largest absolute difference allowed between a tiled run and a whole one, rounded up from twice the
/// 6.83e-5 measured.
const TILED_BOUND: f64 = 2e-4;

/// The strength most tests run at.
const STRENGTH: f32 = 10.0;

/// The value the output planes are filled with before the pass.
const UNTOUCHED: f32 = -1.0;

/// How many ResBlock convolutions one region's pass runs.
const RESBLOCK_CONVOLUTIONS: usize = 56;

/// The scratch maps one region's pass works in.
struct Scratch {
    levels: Vec<Vec<Handle>>,
    columns: Option<Handle>,
}

impl Scratch {
    fn allocate<R: Runtime>(client: &ComputeClient<R>, sizes: ScratchSizes) -> Self {
        let levels = sizes
            .level_bytes
            .iter()
            .map(|bytes| (0..4).map(|_| empty(client, *bytes)).collect())
            .collect();
        let columns = (sizes.column_bytes > 0).then(|| empty(client, sizes.column_bytes));

        Self { levels, columns }
    }

    fn maps(&self) -> DpirMaps<'_> {
        let levels = std::array::from_fn(|level| {
            let handles = &self.levels[level];
            [&handles[0], &handles[1], &handles[2], &handles[3]]
        });

        DpirMaps {
            levels,
            columns: self.columns.as_ref(),
        }
    }
}

/// The scratch a frame split into tiles of `tile` needs, or the tile `scratch_sizes` chooses when `None`.
fn sizes_for(
    precision: Precision,
    path: BodyPath,
    width: u32,
    height: u32,
    tile: Option<u32>,
) -> ScratchSizes {
    match tile {
        Some(tile) => tile_sizes(precision, path, width, height, tile),
        None => scratch_sizes(precision, path, width, height, 0),
    }
}

/// Deblocks `planes`, each `width` by `height`, into output planes `output_width` across filled with
/// `UNTOUCHED`. Gives the output planes and how each ResBlock convolution ran.
pub(super) struct Deblocked<'a> {
    pub(super) weights: &'a Weights,
    pub(super) planes: &'a [Vec<f32>],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) output_width: u32,
    pub(super) strength: f32,
    pub(super) peak: f32,
    pub(super) path: BodyPath,
    pub(super) precision: Precision,
    pub(super) tile: Option<u32>,
}

impl BuildTask for Deblocked<'_> {
    type Output = Result<(Vec<Vec<f32>>, Vec<LayerRun>), CnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let dpir = Dpir::upload(client, self.weights, self.precision);
        let sizes = sizes_for(self.precision, self.path, self.width, self.height, self.tile);
        let scratch = Scratch::allocate(client, sizes);
        let output_samples = (self.output_width * self.height) as usize;
        let untouched = vec![UNTOUCHED; output_samples];

        let input_handles: Vec<_> = self
            .planes
            .iter()
            .map(|values| upload_f32(client, values))
            .collect();
        let output_handles: Vec<_> = self
            .planes
            .iter()
            .map(|_| upload_f32(client, &untouched))
            .collect();

        let inputs: Vec<_> = input_handles
            .iter()
            .map(|handle| plane(handle, self.width, self.height))
            .collect();
        let outputs: Vec<_> = output_handles
            .iter()
            .map(|handle| plane(handle, self.output_width, self.height))
            .collect();
        let planes = DpirPlanes {
            inputs: &inputs,
            outputs: &outputs,
        };

        let ran = deblock(
            client,
            &sizing,
            &dpir,
            self.path,
            planes,
            self.strength,
            self.peak,
            sizes.tile,
            scratch.maps(),
        );
        synced(client);
        let runs = ran?;

        let values = output_handles
            .iter()
            .map(|handle| read_f32(client, handle))
            .collect();

        Ok((values, runs))
    }
}

/// Deblocks `planes`, each `width` by `height`, at a peak of 1.0 in f32 on the device's path, in one tile
/// or tiles of `tile`.
fn deblocked_f32(
    weights: &Weights,
    planes: &[Vec<f32>],
    width: u32,
    height: u32,
    tile: Option<u32>,
) -> (Vec<Vec<f32>>, Vec<LayerRun>) {
    let precision = Precision::F32;
    let task = Deblocked {
        weights,
        planes,
        width,
        height,
        output_width: width,
        strength: STRENGTH,
        peak: 1.0,
        path: body_path(precision),
        precision,
        tile,
    };

    backend().build(task).expect("the frame deblocks")
}

/// The largest absolute error of any plane of `actual` against the same plane of `expected`.
pub(super) fn largest_plane_error(actual: &[Vec<f32>], expected: &[Vec<f64>]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "the plane counts differ");

    actual
        .iter()
        .zip(expected)
        .map(|(actual_plane, expected_plane)| largest_error(actual_plane, expected_plane))
        .fold(0.0, f64::max)
}

/// Runs `warm_up` over a `width` by `height` frame in tiles of `tile`.
struct WarmUp<'a> {
    weights: &'a Weights,
    width: u32,
    height: u32,
    tile: u32,
    path: BodyPath,
    precision: Precision,
}

impl BuildTask for WarmUp<'_> {
    type Output = Result<Vec<LayerRun>, CnnError>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let dpir = Dpir::upload(client, self.weights, self.precision);
        let sizes = tile_sizes(self.precision, self.path, self.width, self.height, self.tile);
        let scratch = Scratch::allocate(client, sizes);

        warm_up(
            client,
            &sizing,
            &dpir,
            self.path,
            self.width,
            self.height,
            self.tile,
            scratch.maps(),
        )
    }
}

#[test]
fn a_plane_narrower_than_its_padding_matches_the_reference() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(3, 5)];
    let inputs = widened(&planes);
    let expected = reference_dpir(&weights, &inputs, 3, 5, 1.0, f64::from(STRENGTH));

    let (actual, _) = deblocked_f32(&weights, &planes, 3, 5, None);

    let error = largest_plane_error(&actual, &expected);
    println!("3x5 gray F32: largest error {error:e}, bound {F32_BOUND:e}");
    assert!(error <= F32_BOUND, "error {error:e} passes {F32_BOUND:e}");
}

#[test]
fn code_values_scale_by_the_peak() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let unit_plane = textured(24, 16);
    let code_plane: Vec<f32> = unit_plane.iter().map(|value| value * 255.0).collect();
    let precision = Precision::F32;
    let unit_planes = vec![unit_plane];
    let code_planes = vec![code_plane];

    let (unit_output, _) = deblocked_f32(&weights, &unit_planes, 24, 16, None);
    let code_task = Deblocked {
        weights: &weights,
        planes: &code_planes,
        width: 24,
        height: 16,
        output_width: 24,
        strength: STRENGTH,
        peak: 255.0,
        path: body_path(precision),
        precision,
        tile: None,
    };
    let (code_output, _) = backend().build(code_task).expect("the frame deblocks");

    let scaled: Vec<Vec<f64>> = unit_output
        .iter()
        .map(|values| values.iter().map(|value| f64::from(*value) * 255.0).collect())
        .collect();
    let error = largest_plane_error(&code_output, &scaled);
    println!("peak 255 against peak 1 times 255: largest difference {error:e}, bound {PEAK_BOUND:e}");
    assert!(error <= PEAK_BOUND, "difference {error:e} passes {PEAK_BOUND:e}");
}

#[test]
fn tiled_runs_write_every_sample_once() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(70, 46)];
    let padded = Padded::of(70, 46);
    let tile_count = tiles(padded.width, padded.height, 16, OVERLAP).len();

    let (whole, whole_runs) = deblocked_f32(&weights, &planes, 70, 46, None);
    let (tiled, tiled_runs) = deblocked_f32(&weights, &planes, 70, 46, Some(16));

    for output in [&whole, &tiled] {
        let untouched = output[0].contains(&UNTOUCHED);
        assert!(!untouched, "a sample was never written");
    }

    assert_eq!(whole_runs.len(), RESBLOCK_CONVOLUTIONS);
    assert_eq!(tiled_runs.len(), tile_count * RESBLOCK_CONVOLUTIONS);

    let whole_values = widened(&whole);
    let difference = largest_plane_error(&tiled, &whole_values);
    println!("tiles of 16 against one tile: largest difference {difference:e}, bound {TILED_BOUND:e}");
    assert!(
        difference <= TILED_BOUND,
        "difference {difference:e} passes {TILED_BOUND:e}"
    );
}

#[test]
#[should_panic(expected = "the output plane is")]
fn deblock_refuses_mismatched_planes() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let planes = vec![textured(24, 16)];
    let precision = Precision::F32;
    // The output is eight samples wider than the input.
    let task = Deblocked {
        weights: &weights,
        planes: &planes,
        width: 24,
        height: 16,
        output_width: 32,
        strength: STRENGTH,
        peak: 1.0,
        path: body_path(precision),
        precision,
        tile: None,
    };

    let ran = backend().build(task);

    println!("wider output: {ran:?}");
}

#[test]
fn warm_up_runs_every_tile() {
    let weights = synthetic_weights(Variant::Gray, 1);
    let precision = Precision::F16;
    let padded = Padded::of(40, 30);
    let tile_count = tiles(padded.width, padded.height, 16, OVERLAP).len();
    let task = WarmUp {
        weights: &weights,
        width: 40,
        height: 30,
        tile: 16,
        path: body_path(precision),
        precision,
    };

    let runs = backend().build(task).expect("the warm-up runs");

    assert_eq!(runs.len(), tile_count * RESBLOCK_CONVOLUTIONS);
}
