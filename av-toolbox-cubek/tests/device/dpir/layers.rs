use av_toolbox_cubek::cnn::{
    BodyPath,
    PADDED_CHANNELS,
    PaddedLastLayer,
    Precision,
    Region,
    run_padded_layer,
};
use av_toolbox_cubek::dpir::{
    HeadLayer,
    LayerKind,
    Placement,
    WeightLayer,
    add_in_place,
    gather,
    head,
    matrix_product,
    scatter,
    tail,
    write_out,
};
use av_toolbox_cubek::mirror;
use av_toolbox_cubek::testing::dpir::{
    reference_conv3x3,
    reference_down,
    reference_head,
    reference_mirror,
    reference_up,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::artcnn::{
    BlueprintRuns,
    as_stored,
    finite_error,
    forced_eligible,
    read_maps,
    signed_values,
    supports,
    synced,
    textured,
    upload_maps,
};
use crate::support::{Sizing, backend, empty, plane, read_f32, read_u32, upload_f32};

const PRECISIONS: [Precision; 2] = [Precision::F32, Precision::F16];

/// The relative spacing of f16 values, `2^-10`.
const F16_EPSILON: f64 = 1.0 / 1024.0;

/// The relative error allowed on f32 maps.
const F32_BOUND: f64 = 1e-5;

/// The relative error allowed on the tail's output, which sums in f32 and writes f32 whatever the maps' type.
///
/// Rounding the output to 8-bit codes moves it by up to 2e-2, far past this.
const TAIL_BOUND: f64 = 1e-4;

/// The magnitude an error is measured against when the expected value is smaller, a tenth of the values the
/// tests expect.
const FLOOR: f64 = 0.1;

/// The value the output planes are filled with before the write-out.
const UNTOUCHED: f32 = -1.0;

/// The features the head writes and the tail reads.
const FEATURES: u32 = 64;

/// The first index the mirror kernel folds.
const FIRST_INDEX: i32 = -20;

/// How many indexes the mirror kernel folds, from -20 to 20.
const INDEXES: usize = 41;

#[cube(launch)]
fn mirror_kernel(output: &mut [u32], first: i32, length: i32) {
    if ABSOLUTE_POS < output.len() {
        let index = first + ABSOLUTE_POS as i32;
        let folded = mirror(index, length);

        output[ABSOLUTE_POS] = folded as u32;
    }
}

struct Mirror {
    length: i32,
}

impl BuildTask for Mirror {
    type Output = Vec<u32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let output = empty(client, INDEXES * size_of::<u32>());

        // Safety: the output holds one u32 an index.
        let output_array = unsafe { BufferArg::from_raw_parts(output.clone(), INDEXES) };

        mirror_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            output_array,
            FIRST_INDEX,
            self.length,
        );
        synced(client);

        read_u32(client, &output)
    }
}

#[test]
fn mirror_matches_its_reference() {
    for length in [1, 2, 3, 5] {
        let actual = backend().build(Mirror { length });

        let mut expected = Vec::with_capacity(INDEXES);
        for offset in 0..INDEXES as i64 {
            let index = i64::from(FIRST_INDEX) + offset;
            let folded = reference_mirror(index, i64::from(length));
            expected.push(folded as u32);
        }

        assert_eq!(actual, expected, "length {length}");
    }
}

/// A layer of `kind` from `in_channels` to `out_channels`, with values between -`scale` and `scale`.
pub(super) fn weight_layer(
    kind: LayerKind,
    in_channels: u32,
    out_channels: u32,
    seed: u32,
    scale: f32,
) -> WeightLayer {
    let taps = match kind {
        LayerKind::Conv => 9,
        LayerKind::Down | LayerKind::Up => 4,
    };
    let count = (taps * in_channels * out_channels) as usize;
    let values = signed_values(count, seed, scale);

    WeightLayer {
        kind,
        in_channels,
        out_channels,
        values,
    }
}

/// `layer` with its values as the device holds them in `precision`.
pub(super) fn stored_layer(layer: &WeightLayer, precision: Precision) -> WeightLayer {
    let stored = as_stored(&layer.values, precision);
    let values = stored.into_iter().map(|value| value as f32).collect();

    WeightLayer {
        values,
        ..layer.clone()
    }
}

/// The relative error a layer summing `depth` products may reach in `precision`.
pub(super) fn bound(precision: Precision, depth: u32) -> f64 {
    match precision {
        Precision::F32 => F32_BOUND,
        Precision::F16 => F16_EPSILON * f64::from(depth).sqrt(),
    }
}

/// Asserts `actual` lies within `allowed` relative error of `expected`, measuring each error against the
/// expected value's magnitude, or [FLOOR] where that is smaller.
pub(super) fn assert_matches(
    name: &str,
    precision: Precision,
    allowed: f64,
    actual: &[f32],
    expected: &[f64],
) {
    assert_eq!(actual.len(), expected.len(), "{name}: the value counts differ");

    let mut largest_error = 0.0f64;
    let mut largest_relative = 0.0f64;
    for (got, want) in actual.iter().zip(expected) {
        let error = finite_error(*got, *want);
        let relative = error / want.abs().max(FLOOR);
        largest_error = largest_error.max(error);
        largest_relative = largest_relative.max(relative);
    }

    println!(
        "{name} {precision:?}: largest error {largest_error:e}, largest relative error {largest_relative:e}, \
         bound {allowed:e}"
    );
    assert!(
        largest_relative <= allowed,
        "{name} {precision:?}: relative error {largest_relative:e} passes {allowed:e}"
    );
}

struct Head {
    layer: WeightLayer,
    planes: Vec<Vec<f32>>,
    width: u32,
    height: u32,
    peak: f32,
    strength: f32,
    placement: Placement,
    precision: Precision,
}

impl BuildTask for Head {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let layer = HeadLayer::upload(client, &self.layer, self.precision);
        let map_count = (self.placement.region.samples() * FEATURES) as usize;

        let output = empty(client, map_count * self.precision.bytes());
        let handles: Vec<_> = self
            .planes
            .iter()
            .map(|values| upload_f32(client, values))
            .collect();
        let planes: Vec<_> = handles
            .iter()
            .map(|handle| plane(handle, self.width, self.height))
            .collect();

        head(
            client,
            &sizing,
            &layer,
            &planes,
            self.peak,
            self.strength,
            self.placement,
            &output,
        );
        synced(client);

        read_maps(client, &output, self.precision)
    }
}

/// A head over `planes`, each `width` by `height`, compared with the reference in both precisions.
struct HeadCase {
    name: &'static str,
    planes: Vec<Vec<f32>>,
    width: u32,
    height: u32,
    peak: f32,
    strength: f32,
    placement: Placement,
}

impl HeadCase {
    fn check(self) {
        let image_channels = self.planes.len() as u32;
        let in_channels = image_channels + 1;
        let layer = weight_layer(LayerKind::Conv, in_channels, FEATURES, 31, 0.3);
        let widened: Vec<Vec<f64>> = self
            .planes
            .iter()
            .map(|values| values.iter().map(|value| f64::from(*value)).collect())
            .collect();
        let region = self.placement.region;
        let placement = (
            region.x,
            region.y,
            region.width,
            region.height,
            self.placement.left,
            self.placement.top,
        );

        for precision in PRECISIONS {
            let stored = stored_layer(&layer, precision);
            let expected = reference_head(
                &widened,
                self.width,
                self.height,
                f64::from(self.peak),
                f64::from(self.strength),
                &stored,
                placement,
            );

            let task = Head {
                layer: layer.clone(),
                planes: self.planes.clone(),
                width: self.width,
                height: self.height,
                peak: self.peak,
                strength: self.strength,
                placement: self.placement,
                precision,
            };
            let actual = backend().build(task);

            let allowed = bound(precision, 9 * in_channels);
            assert_matches(self.name, precision, allowed, &actual, &expected);
        }
    }
}

/// `width` by `height` textured code values at `peak`.
fn code_plane(width: u32, height: u32, peak: f32) -> Vec<f32> {
    textured(width, height)
        .into_iter()
        .map(|value| value * peak)
        .collect()
}

#[test]
fn the_head_matches_the_reference_inside_the_plane() {
    let placement = Placement {
        region: Region {
            x: 0,
            y: 0,
            width: 16,
            height: 8,
        },
        left: 0,
        top: 0,
    };
    let case = HeadCase {
        name: "head inside the plane",
        planes: vec![code_plane(16, 8, 255.0)],
        width: 16,
        height: 8,
        peak: 255.0,
        strength: 10.0,
        placement,
    };

    case.check();
}

/// The 13x7 plane pads to 16x8, one column before and two after, and one row after.
#[test]
fn the_head_mirrors_past_the_plane() {
    let placement = Placement {
        region: Region {
            x: 0,
            y: 0,
            width: 16,
            height: 8,
        },
        left: 1,
        top: 0,
    };
    let case = HeadCase {
        name: "head past the plane",
        planes: vec![code_plane(13, 7, 255.0)],
        width: 13,
        height: 7,
        peak: 255.0,
        strength: 10.0,
        placement,
    };

    case.check();
}

/// The region sits inside the padded frame and touches its right edge.
#[test]
fn the_head_reads_three_planes_and_the_strength() {
    let width = 13;
    let height = 7;
    let red = textured(width, height);
    let green: Vec<f32> = red.iter().rev().copied().collect();
    let blue: Vec<f32> = red.iter().map(|value| 1.0 - value * 0.5).collect();
    let placement = Placement {
        region: Region {
            x: 4,
            y: 1,
            width: 12,
            height: 6,
        },
        left: 1,
        top: 0,
    };
    let case = HeadCase {
        name: "head in colour",
        planes: vec![red, green, blue],
        width,
        height,
        peak: 1.0,
        strength: 37.0,
        placement,
    };

    case.check();
}

/// The gather then the matrix product of a 2x2 stride 2 convolution.
pub(super) struct Down {
    pub(super) layer: WeightLayer,
    pub(super) maps: Vec<f32>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) precision: Precision,
}

impl BuildTask for Down {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let precision = self.precision;
        let in_channels = self.layer.in_channels;
        let out_channels = self.layer.out_channels as usize;
        let rows = (self.width / 2 * self.height / 2) as usize;
        let depth = 4 * in_channels as usize;

        let input = upload_maps(client, &self.maps, precision);
        let matrix = upload_maps(client, &self.layer.values, precision);
        let gathered = empty(client, self.maps.len() * precision.bytes());
        let output = empty(client, rows * out_channels * precision.bytes());

        gather(
            client,
            &sizing,
            &input,
            self.width,
            self.height,
            in_channels,
            precision,
            &gathered,
        );
        let product = matrix_product(
            client,
            &gathered,
            rows,
            depth,
            &matrix,
            out_channels,
            precision,
            &output,
        );
        synced(client);
        product.expect("the matrix product runs");

        read_maps(client, &output, precision)
    }
}

#[test]
fn the_gather_and_product_match_the_strided_convolution() {
    let width = 6;
    let height = 4;
    let channels = 16;
    let layer = weight_layer(LayerKind::Down, channels, 32, 41, 0.25);
    let maps = signed_values((width * height * channels) as usize, 42, 1.0);

    for precision in PRECISIONS {
        let stored_maps = as_stored(&maps, precision);
        let stored = stored_layer(&layer, precision);
        let expected = reference_down(&stored_maps, width, height, &stored);

        let task = Down {
            layer: layer.clone(),
            maps: maps.clone(),
            width,
            height,
            precision,
        };
        let actual = backend().build(task);

        let allowed = bound(precision, 4 * channels);
        assert_matches("down", precision, allowed, &actual, &expected);
    }
}

/// The matrix product then the scatter of a 2x2 stride 2 transposed convolution.
pub(super) struct Up {
    pub(super) layer: WeightLayer,
    pub(super) maps: Vec<f32>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) precision: Precision,
}

impl BuildTask for Up {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let precision = self.precision;
        let out_channels = self.layer.out_channels;
        let rows = (self.width * self.height) as usize;
        let depth = self.layer.in_channels as usize;
        let columns = 4 * out_channels as usize;
        let product_bytes = rows * columns * precision.bytes();

        let input = upload_maps(client, &self.maps, precision);
        let matrix = upload_maps(client, &self.layer.values, precision);
        let product = empty(client, product_bytes);
        let output = empty(client, product_bytes);

        let multiplied = matrix_product(client, &input, rows, depth, &matrix, columns, precision, &product);
        scatter(
            client,
            &sizing,
            &product,
            self.width,
            self.height,
            out_channels,
            precision,
            &output,
        );
        synced(client);
        multiplied.expect("the matrix product runs");

        read_maps(client, &output, precision)
    }
}

#[test]
fn the_product_and_scatter_match_the_transposed_convolution() {
    let width = 3;
    let height = 2;
    let channels = 32;
    let layer = weight_layer(LayerKind::Up, channels, 16, 51, 0.25);
    let maps = signed_values((width * height * channels) as usize, 52, 1.0);

    for precision in PRECISIONS {
        let stored_maps = as_stored(&maps, precision);
        let stored = stored_layer(&layer, precision);
        let expected = reference_up(&stored_maps, width, height, &stored);

        let task = Up {
            layer: layer.clone(),
            maps: maps.clone(),
            width,
            height,
            precision,
        };
        let actual = backend().build(task);

        let allowed = bound(precision, channels);
        assert_matches("up", precision, allowed, &actual, &expected);
    }
}

struct Add {
    values: Vec<f32>,
    addend: Vec<f32>,
    precision: Precision,
}

impl BuildTask for Add {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let precision = self.precision;

        let values = upload_maps(client, &self.values, precision);
        let addend = upload_maps(client, &self.addend, precision);

        add_in_place(client, &sizing, &values, &addend, self.values.len(), precision);
        synced(client);

        read_maps(client, &values, precision)
    }
}

#[test]
fn add_in_place_adds() {
    let count = 1000;
    let values = signed_values(count, 61, 2.0);
    let addend = signed_values(count, 62, 2.0);

    for precision in PRECISIONS {
        let stored_values = as_stored(&values, precision);
        let stored_addend = as_stored(&addend, precision);
        let expected: Vec<f64> = stored_values
            .iter()
            .zip(&stored_addend)
            .map(|(value, added)| value + added)
            .collect();

        let task = Add {
            values: values.clone(),
            addend: addend.clone(),
            precision,
        };
        let actual = backend().build(task);

        let allowed = bound(precision, 1);
        assert_matches("add in place", precision, allowed, &actual, &expected);
    }
}

/// The tail over a region's maps along `path`, then the write-out of its core into planes filled with
/// [UNTOUCHED].
struct Tail {
    layer: WeightLayer,
    maps: Vec<f32>,
    plane_width: u32,
    plane_height: u32,
    placement: Placement,
    core: Region,
    peak: f32,
    precision: Precision,
    path: BodyPath,
}

struct TailOutputs {
    scratch: Vec<f32>,
    planes: Vec<Vec<f32>>,
    blueprints: Option<BlueprintRuns>,
}

impl BuildTask for Tail {
    type Output = TailOutputs;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let precision = self.precision;
        let channels = self.layer.out_channels;
        let region = self.placement.region;
        let layer = PaddedLastLayer::upload(client, &self.layer.values, None, FEATURES, channels, precision);
        let scratch_bytes = region.samples() as usize * PADDED_CHANNELS * size_of::<f32>();
        let untouched = vec![UNTOUCHED; (self.plane_width * self.plane_height) as usize];

        let input = upload_maps(client, &self.maps, precision);
        let scratch = empty(client, scratch_bytes);
        let handles: Vec<_> = (0..channels).map(|_| upload_f32(client, &untouched)).collect();
        let planes: Vec<_> = handles
            .iter()
            .map(|handle| plane(handle, self.plane_width, self.plane_height))
            .collect();

        let forced_ran = match self.path {
            BodyPath::Im2col => {
                tail(client, &sizing, &layer, self.path, &input, region, &scratch).map(|()| false)
            },
            BodyPath::Convolution | BodyPath::UnfusedConvolution => {
                run_padded_layer(client, &layer, &input, region, &scratch, true)
            },
        };
        write_out(
            client,
            &sizing,
            &scratch,
            self.placement,
            self.core,
            channels,
            self.peak,
            &planes,
        );
        synced(client);

        let forced_ran = forced_ran.expect("the tail runs");
        let compares_blueprints = precision == Precision::F16 && self.path != BodyPath::Im2col;
        let blueprints =
            compares_blueprints.then(|| rerun_inferred(client, &layer, &input, region, &scratch, forced_ran));
        let scratch_values = read_f32(client, &scratch);
        let planes = handles.iter().map(|handle| read_f32(client, handle)).collect();

        TailOutputs {
            scratch: scratch_values,
            planes,
            blueprints,
        }
    }
}

/// Runs `tail` again with the inferred blueprint, beside the run already in `forced`.
fn rerun_inferred(
    client: &Client,
    tail: &PaddedLastLayer,
    input: &Handle,
    region: Region,
    forced: &Handle,
    forced_ran: bool,
) -> BlueprintRuns {
    let scratch_bytes = region.samples() as usize * PADDED_CHANNELS * size_of::<f32>();
    let inferred = empty(client, scratch_bytes);

    let inferred_ran = run_padded_layer(client, tail, input, region, &inferred, false);
    synced(client);

    let inferred_forced = inferred_ran.expect("the inferred padded tail runs");
    assert!(!inferred_forced, "the inferred run took the forced blueprint");

    let forced_values = read_f32(client, forced);
    let inferred_values = read_f32(client, &inferred);
    let eligible = forced_eligible(client, 16);

    BlueprintRuns {
        forced: forced_values,
        inferred: inferred_values,
        forced_ran,
        eligible,
    }
}

/// How many of the core's reference values fall below 0, inside `0..=1` and above 1.
#[derive(Debug, Default)]
struct ClampCounts {
    below: usize,
    inside: usize,
    above: usize,
}

/// The 125x61 plane the tail tests write, padded to the 128x64 region one sample before and two after on each
/// axis.
const TAIL_PLANE_WIDTH: u32 = 125;
const TAIL_PLANE_HEIGHT: u32 = 61;

const TAIL_REGION: Region = Region {
    x: 0,
    y: 0,
    width: 128,
    height: 64,
};

/// The tail's 64 features over [TAIL_REGION].
fn tail_maps() -> Vec<f32> {
    let count = (TAIL_REGION.samples() * FEATURES) as usize;

    signed_values(count, 71, 1.0)
}

/// A tail from 64 features to `channels`.
fn tail_layer(channels: u32) -> WeightLayer {
    weight_layer(LayerKind::Conv, FEATURES, channels, 72 + channels, 0.1)
}

/// `values` of `channels` a sample, followed by zeros up to 16 channels a sample.
fn padded_channels(values: &[f64], channels: u32) -> Vec<f64> {
    let mut padded = Vec::with_capacity(values.len() / channels as usize * PADDED_CHANNELS);

    for sample in values.chunks(channels as usize) {
        padded.extend_from_slice(sample);
        padded.resize(padded.len() + PADDED_CHANNELS - sample.len(), 0.0);
    }

    padded
}

/// Runs the tail along `path` and the write-out, at 1 and 3 channels in each of `precisions`, and compares
/// the whole scratch and the planes with the reference. The core starts inside the region and runs past the
/// plane's right and bottom edges.
fn check_tail(path: BodyPath, precisions: &[Precision]) {
    let plane_width = TAIL_PLANE_WIDTH;
    let plane_height = TAIL_PLANE_HEIGHT;
    let peak = 255.0f32;
    let region = TAIL_REGION;
    let placement = Placement {
        region,
        left: 1,
        top: 1,
    };
    let core = Region {
        x: 4,
        y: 2,
        width: 124,
        height: 62,
    };
    let maps = tail_maps();

    for channels in [1, 3] {
        let layer = tail_layer(channels);

        for precision in precisions.iter().copied() {
            let stored_maps = as_stored(&maps, precision);
            let stored = stored_layer(&layer, precision);
            let reference = reference_conv3x3(&stored_maps, region.width, region.height, &stored);

            let task = Tail {
                layer: layer.clone(),
                maps: maps.clone(),
                plane_width,
                plane_height,
                placement,
                core,
                peak,
                precision,
                path,
            };
            let outputs = backend().build(task);

            let name = format!("{path:?} tail to {channels} channels");
            let scratch_name = format!("{name}, the whole scratch");
            let padded_reference = padded_channels(&reference, channels);
            assert_matches(
                &scratch_name,
                precision,
                TAIL_BOUND,
                &outputs.scratch,
                &padded_reference,
            );

            let mut actual = Vec::new();
            let mut expected = Vec::new();
            let mut counts = ClampCounts::default();
            for (channel, values) in outputs.planes.iter().enumerate() {
                for row in 0..plane_height {
                    for column in 0..plane_width {
                        let got = values[(row * plane_width + column) as usize];
                        let padded_column = column + placement.left;
                        let padded_row = row + placement.top;
                        let inside_core = padded_column >= core.x
                            && padded_row >= core.y
                            && padded_column < core.x + core.width
                            && padded_row < core.y + core.height;

                        if !inside_core {
                            assert_eq!(
                                got, UNTOUCHED,
                                "({column}, {row}) lies outside the core but was written"
                            );
                            continue;
                        }

                        let local = (padded_row - region.y) * region.width + padded_column - region.x;
                        let want = reference[local as usize * channels as usize + channel];
                        if want < 0.0 {
                            counts.below += 1;
                        } else if want > 1.0 {
                            counts.above += 1;
                        } else {
                            counts.inside += 1;
                        }

                        actual.push(got / peak);
                        expected.push(want.clamp(0.0, 1.0));
                    }
                }
            }

            println!("{name} {precision:?}: {counts:?}");
            assert!(
                counts.below > 0 && counts.inside > 0 && counts.above > 0,
                "{counts:?}"
            );

            assert_matches(&name, precision, TAIL_BOUND, &actual, &expected);

            if let Some(blueprints) = outputs.blueprints {
                blueprints.assert_forced_matches_inferred(&name);
            }
        }
    }
}

#[test]
fn the_padded_tail_and_write_out_match_the_reference() {
    let mut precisions = Vec::new();
    for precision in PRECISIONS {
        if supports(BodyPath::Convolution, precision) {
            precisions.push(precision);
        } else {
            println!("padded tail {precision:?}: no cooperative matrix, so there is nothing to run");
        }
    }

    check_tail(BodyPath::Convolution, &precisions);
}

#[test]
fn the_direct_tail_and_write_out_match_the_reference() {
    check_tail(BodyPath::Im2col, &PRECISIONS);
}

/// The tail of the same maps along the direct and the padded-layer paths, each into its own scratch.
struct TailPaths {
    layer: WeightLayer,
    maps: Vec<f32>,
    precision: Precision,
}

struct TailScratches {
    direct: Vec<f32>,
    padded: Vec<f32>,
}

impl BuildTask for TailPaths {
    type Output = TailScratches;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let precision = self.precision;
        let channels = self.layer.out_channels;
        let region = TAIL_REGION;
        let layer = PaddedLastLayer::upload(client, &self.layer.values, None, FEATURES, channels, precision);
        let scratch_bytes = region.samples() as usize * PADDED_CHANNELS * size_of::<f32>();

        let input = upload_maps(client, &self.maps, precision);
        let direct = empty(client, scratch_bytes);
        let padded = empty(client, scratch_bytes);

        let direct_ran = tail(client, &sizing, &layer, BodyPath::Im2col, &input, region, &direct);
        let padded_ran = tail(
            client,
            &sizing,
            &layer,
            BodyPath::Convolution,
            &input,
            region,
            &padded,
        );
        synced(client);
        direct_ran.expect("the direct tail runs");
        padded_ran.expect("the padded tail runs");

        let direct_values = read_f32(client, &direct);
        let padded_values = read_f32(client, &padded);

        TailScratches {
            direct: direct_values,
            padded: padded_values,
        }
    }
}

/// Every one of the scratch's 16 channels agrees, the zeros past the real channels included.
#[test]
fn the_direct_and_padded_tails_agree_in_f16() {
    if !supports(BodyPath::Convolution, Precision::F16) {
        println!("tail paths: no f16 cooperative matrix, so there is nothing to compare");
        return;
    }

    let maps = tail_maps();

    for channels in [1, 3] {
        let task = TailPaths {
            layer: tail_layer(channels),
            maps: maps.clone(),
            precision: Precision::F16,
        };

        let scratches = backend().build(task);

        let padded: Vec<f64> = scratches.padded.iter().map(|value| f64::from(*value)).collect();
        let name = format!("direct against padded tail to {channels} channels");
        assert_matches(&name, Precision::F16, TAIL_BOUND, &scratches.direct, &padded);
    }
}
