// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::matrix::MatrixSamples;
use crate::plane::assert_holds;
use crate::{DevicePlane, Dispatch, mirror};

/// The largest index of the threshold tables.
pub const MAX_INDEX: i32 = 60;

/// The step across an edge below which a line is filtered, by alpha index.
///
/// Entries up to 51 are H.264's `alpha'`, and the rest extend it for stronger deblocking.
pub const ALPHA_TABLE: [u8; 61] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28,
    32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255,
];

/// How flat each side of an edge must be for a line to be filtered, by beta index.
///
/// Entries up to 51 are H.264's `beta'`, and the rest extend it for stronger deblocking.
pub const BETA_TABLE: [u8; 61] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
];

/// How far the samples next to an edge may move, by alpha index.
///
/// Entries up to 51 are H.264's `tC0'` for a boundary strength of 2, and the rest extend it for stronger
/// deblocking.
pub const TC0_TABLE: [u8; 61] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,
];

/// Frames are padded on the right and bottom to a multiple of this before deblocking.
const PAD_MULTIPLE: u32 = 8;

/// What every line of a plane is filtered against, in the plane's own code values.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Thresholds {
    /// A line is filtered only when the step across its edge is below this.
    pub alpha: f32,
    /// A line is filtered only when the step next to its edge on each side is below this.
    pub beta: f32,
    /// The furthest the second sample from the edge may move.
    pub c0: f32,
    /// How much further the samples next to the edge may move for each flat side.
    pub c1: f32,
    /// The largest code value. Float samples are never clamped, and take 1.
    pub peak: f32,
}

/// The thresholds `quant`, `aoffset` and `boffset` give a plane of `samples`.
///
/// The alpha index is `quant + aoffset` and the beta index `quant + boffset`, each kept between 0 and 60.
/// Integer thresholds scale with the depth, and float thresholds are divided by 255.
pub fn thresholds(quant: i32, aoffset: i32, boffset: i32, samples: MatrixSamples) -> Thresholds {
    let alpha_index = table_index(quant, aoffset);
    let beta_index = table_index(quant, boffset);
    let alpha = f32::from(ALPHA_TABLE[alpha_index]);
    let beta = f32::from(BETA_TABLE[beta_index]);
    let c0 = f32::from(TC0_TABLE[alpha_index]);

    let Some(bits) = samples.bits() else {
        return Thresholds {
            alpha: alpha / 255.0,
            beta: beta / 255.0,
            c0: c0 / 255.0,
            c1: 1.0 / 255.0,
            peak: 1.0,
        };
    };

    let scale = (1u32 << (bits - 8)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;

    Thresholds {
        alpha: alpha * scale,
        beta: beta * scale,
        c0: c0 * scale,
        c1: scale,
        peak,
    }
}

fn table_index(quant: i32, offset: i32) -> usize {
    let index = quant.saturating_add(offset);
    let clamped = index.clamp(0, MAX_INDEX);

    clamped as usize
}

/// The size a `width` by `height` frame is padded to before deblocking, each side rounded up to a
/// multiple of 8.
pub fn padded_frame(width: u32, height: u32) -> (u32, u32) {
    let padded_width = width.next_multiple_of(PAD_MULTIPLE);
    let padded_height = height.next_multiple_of(PAD_MULTIPLE);

    (padded_width, padded_height)
}

/// How many values describe one plane in the uploaded layout, its offset, width and height.
const REGION_VALUES: usize = 3;

/// Where one padded plane sits in the scratch, in samples, and its padded size.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct PlaneRegion {
    offset: u32,
    width: u32,
    height: u32,
}

impl PlaneRegion {
    fn samples(self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Regions of the `padded` sizes, back to back from the start of the scratch.
fn regions_of(padded: &[(u32, u32)]) -> Vec<PlaneRegion> {
    let mut regions = Vec::with_capacity(padded.len());
    let mut offset = 0u32;

    for (width, height) in padded {
        regions.push(PlaneRegion {
            offset,
            width: *width,
            height: *height,
        });
        offset += width * height;
    }

    regions
}

/// How many bytes the scratch keeps for each sample, whatever the plane's sample type.
const WORD_BYTES: usize = size_of::<u32>();

/// Where every padded plane of a frame sits in one scratch buffer, back to back, kept on the host and the
/// device.
///
/// The scratch keeps each sample as an atomic 32-bit word, the integer value or the float's bits.
#[derive(Debug)]
pub struct DeblockLayout {
    regions: Vec<PlaneRegion>,
    values: Handle,
}

impl DeblockLayout {
    /// How many samples the scratch holds, every padded plane together.
    pub fn samples(&self) -> usize {
        self.regions.iter().map(|region| region.samples()).sum()
    }

    /// How many bytes the scratch needs, one 32-bit word to a sample.
    pub fn bytes(&self) -> usize {
        self.samples() * WORD_BYTES
    }

    fn values<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.values.clone();
        let count = self.regions.len() * REGION_VALUES;

        // Safety: the buffer was made from exactly `REGION_VALUES` u32 values for each region.
        unsafe { ArrayArg::from_raw_parts(handle, count) }
    }
}

/// Lays planes of the `padded` sizes out back to back in one scratch, and uploads where each one sits.
///
/// Every padded side must be a whole number of 4 sample blocks.
pub fn upload_layout<R: Runtime>(client: &ComputeClient<R>, padded: &[(u32, u32)]) -> DeblockLayout {
    assert!(!padded.is_empty(), "a deblocking layout needs at least one plane");

    for (width, height) in padded {
        let whole_blocks = *width > 0 && *height > 0 && width % 4 == 0 && height % 4 == 0;

        assert!(
            whole_blocks,
            "a padded plane must be whole 4 by 4 blocks, not {width}x{height}"
        );
    }

    let regions = regions_of(padded);
    let flattened: Vec<u32> = regions
        .iter()
        .flat_map(|region| [region.offset, region.width, region.height])
        .collect();
    let bytes = u32::as_bytes(&flattened);
    let values = client.create_from_slice(bytes);

    DeblockLayout { regions, values }
}

/// The planes one deblock reads and the planes it writes, in the order of the layout's regions.
#[derive(Debug, Clone, Copy)]
pub struct DeblockPlanes<'a> {
    pub inputs: &'a [DevicePlane<'a>],
    pub outputs: &'a [DevicePlane<'a>],
}

/// What a line of integer samples is filtered against.
#[derive(CubeType)]
struct IntegerLimits {
    alpha: i32,
    beta: i32,
    c0: i32,
    c1: i32,
    peak: i32,
}

/// What a line of float samples is filtered against.
#[derive(CubeType)]
struct FloatLimits {
    alpha: f32,
    beta: f32,
    c0: f32,
    c1: f32,
}

#[cube]
fn load_integer(samples: &Array<Atomic<u32>>, index: usize) -> i32 {
    let word = samples[index].load();

    i32::cast_from(word)
}

#[cube]
fn store_integer(samples: &Array<Atomic<u32>>, index: usize, value: i32) {
    let word = u32::cast_from(value);

    samples[index].store(word);
}

#[cube]
fn load_float(samples: &Array<Atomic<u32>>, index: usize) -> f32 {
    let word = samples[index].load();

    f32::reinterpret(word)
}

#[cube]
fn store_float(samples: &Array<Atomic<u32>>, index: usize, value: f32) {
    let word = u32::reinterpret(value);

    samples[index].store(word);
}

/// The scratch word for `value`, its bits for a float and its value for an integer.
#[cube]
fn to_word<E: CubePrimitive>(value: E, #[comptime] float: bool) -> u32 {
    if float {
        u32::reinterpret(value)
    } else {
        u32::cast_from(value)
    }
}

/// The sample `word` holds, the inverse of [to_word].
#[cube]
fn from_word<E: CubePrimitive>(word: u32, #[comptime] float: bool) -> E {
    if float {
        E::reinterpret(word)
    } else {
        E::cast_from(word)
    }
}

/// Filters the line of integer samples whose first sample past the edge is `first_after`, its taps `stride`
/// apart.
#[cube]
fn filter_integer_line(samples: &Array<Atomic<u32>>, first_after: u32, stride: u32, limits: &IntegerLimits) {
    let p0_index = (first_after - stride) as usize;
    let p1_index = (first_after - 2 * stride) as usize;
    let p2_index = (first_after - 3 * stride) as usize;
    let q0_index = first_after as usize;
    let q1_index = (first_after + stride) as usize;
    let q2_index = (first_after + 2 * stride) as usize;

    let p0 = load_integer(samples, p0_index);
    let p1 = load_integer(samples, p1_index);
    let p2 = load_integer(samples, p2_index);
    let q0 = load_integer(samples, q0_index);
    let q1 = load_integer(samples, q1_index);
    let q2 = load_integer(samples, q2_index);

    let edge_passes = i32::abs(p0 - q0) < limits.alpha;
    let sides_pass = i32::abs(p1 - p0) < limits.beta && i32::abs(q1 - q0) < limits.beta;
    if edge_passes && sides_pass {
        let p_flat = i32::abs(p2 - p0) < limits.beta;
        let q_flat = i32::abs(q2 - q0) < limits.beta;
        let mut limit = limits.c0;
        if p_flat {
            limit += limits.c1;
        }

        if q_flat {
            limit += limits.c1;
        }

        let average = (p0 + q0 + 1i32) >> 1i32;
        let raw_delta = (((q0 - p0) << 2i32) + p1 - q1 + 4i32) >> 3i32;
        let delta = i32::clamp(raw_delta, -limit, limit);
        let new_p0 = i32::clamp(p0 + delta, 0, limits.peak);
        let new_q0 = i32::clamp(q0 - delta, 0, limits.peak);
        store_integer(samples, p0_index, new_p0);
        store_integer(samples, q0_index, new_q0);

        if p_flat {
            let raw_p1 = (p2 + average - (p1 << 1i32)) >> 1i32;
            let delta_p1 = i32::clamp(raw_p1, -limits.c0, limits.c0);
            let new_p1 = i32::clamp(p1 + delta_p1, 0, limits.peak);
            store_integer(samples, p1_index, new_p1);
        }

        if q_flat {
            let raw_q1 = (q2 + average - (q1 << 1i32)) >> 1i32;
            let delta_q1 = i32::clamp(raw_q1, -limits.c0, limits.c0);
            let new_q1 = i32::clamp(q1 + delta_q1, 0, limits.peak);
            store_integer(samples, q1_index, new_q1);
        }
    }
}

/// Filters the line of float samples whose first sample past the edge is `first_after`, its taps `stride`
/// apart.
///
/// Nothing is rounded or clamped to a range.
#[cube]
fn filter_float_line(samples: &Array<Atomic<u32>>, first_after: u32, stride: u32, limits: &FloatLimits) {
    let p0_index = (first_after - stride) as usize;
    let p1_index = (first_after - 2 * stride) as usize;
    let p2_index = (first_after - 3 * stride) as usize;
    let q0_index = first_after as usize;
    let q1_index = (first_after + stride) as usize;
    let q2_index = (first_after + 2 * stride) as usize;

    let p0 = load_float(samples, p0_index);
    let p1 = load_float(samples, p1_index);
    let p2 = load_float(samples, p2_index);
    let q0 = load_float(samples, q0_index);
    let q1 = load_float(samples, q1_index);
    let q2 = load_float(samples, q2_index);

    let edge_passes = f32::abs(p0 - q0) < limits.alpha;
    let sides_pass = f32::abs(p1 - p0) < limits.beta && f32::abs(q1 - q0) < limits.beta;
    if edge_passes && sides_pass {
        let p_flat = f32::abs(p2 - p0) < limits.beta;
        let q_flat = f32::abs(q2 - q0) < limits.beta;
        let mut limit = limits.c0;
        if p_flat {
            limit += limits.c1;
        }

        if q_flat {
            limit += limits.c1;
        }

        let average = (p0 + q0) * 0.5;
        let raw_delta = ((q0 - p0) * 4.0 + p1 - q1) * 0.125;
        let delta = f32::clamp(raw_delta, -limit, limit);
        store_float(samples, p0_index, p0 + delta);
        store_float(samples, q0_index, q0 - delta);

        if p_flat {
            let raw_p1 = (p2 + average - p1 * 2.0) * 0.5;
            let delta_p1 = f32::clamp(raw_p1, -limits.c0, limits.c0);
            store_float(samples, p1_index, p1 + delta_p1);
        }

        if q_flat {
            let raw_q1 = (q2 + average - q1 * 2.0) * 0.5;
            let delta_q1 = f32::clamp(raw_q1, -limits.c0, limits.c0);
            store_float(samples, q1_index, q1 + delta_q1);
        }
    }
}

#[cube]
fn filter_line(
    samples: &Array<Atomic<u32>>,
    first_after: u32,
    stride: u32,
    integer_limits: &IntegerLimits,
    float_limits: &FloatLimits,
    #[comptime] float: bool,
) {
    if float {
        filter_float_line(samples, first_after, stride, float_limits);
    } else {
        filter_integer_line(samples, first_after, stride, integer_limits);
    }
}

/// Filters every plane of the layout in place, one cube to a plane.
///
/// Block `(i, j)` runs at step `i + 2j`. Each step filters its blocks' top edges, then their left edges, and
/// no two edges of one phase touch the same sample, so the cube's units share each phase's lines. Every unit
/// meets the others after each phase, which gives the same result as filtering the blocks in raster order.
///
/// The waves of a cube pass samples to each other through the scratch, so every access to it is atomic. A
/// barrier makes atomic writes visible to every wave of a cube, but not plain ones.
#[cube(launch_unchecked)]
fn wavefront_kernel(
    samples: &mut Array<Atomic<u32>>,
    layout: &Array<u32>,
    alpha: f32,
    beta: f32,
    c0: f32,
    c1: f32,
    peak: f32,
    #[comptime] float: bool,
) {
    let entry = CUBE_POS * REGION_VALUES;
    let offset = layout[entry];
    let width = layout[entry + 1];
    let height = layout[entry + 2];
    let across = (width / 4) as i32;
    let down = (height / 4) as i32;
    let steps = (across + 2 * down - 2) as u32;
    let units = CUBE_DIM as i32;

    let integer_limits = IntegerLimits {
        alpha: i32::cast_from(alpha),
        beta: i32::cast_from(beta),
        c0: i32::cast_from(c0),
        c1: i32::cast_from(c1),
        peak: i32::cast_from(peak),
    };
    let float_limits = FloatLimits { alpha, beta, c0, c1 };

    for step_index in 0..steps {
        let step = step_index as i32;
        let first_row = i32::max((step - across + 2) / 2, 0);
        let last_row = i32::min(step / 2, down - 1);
        let lines = (last_row - first_row + 1) * 4;

        let mut top_line = UNIT_POS as i32;
        while top_line < lines {
            let row = first_row + top_line / 4;
            let column = step - 2 * row;

            if row > 0 {
                let x = (column * 4 + top_line % 4) as u32;
                let y = (row * 4) as u32;
                let first_after = offset + y * width + x;
                filter_line(samples, first_after, width, &integer_limits, &float_limits, float);
            }

            top_line += units;
        }

        sync_storage();

        let mut left_line = UNIT_POS as i32;
        while left_line < lines {
            let row = first_row + left_line / 4;
            let column = step - 2 * row;

            if column > 0 {
                let x = (column * 4) as u32;
                let y = (row * 4 + left_line % 4) as u32;
                let first_after = offset + y * width + x;
                filter_line(samples, first_after, 1u32, &integer_limits, &float_limits, float);
            }

            left_line += units;
        }

        sync_storage();
    }
}

#[cube(launch_unchecked)]
fn pad_kernel<E: CubePrimitive>(
    input: &Array<E>,
    samples: &mut Array<Atomic<u32>>,
    input_width: u32,
    input_height: u32,
    offset: u32,
    padded_width: u32,
    padded_samples: u32,
    #[comptime] float: bool,
) {
    if ABSOLUTE_POS >= padded_samples as usize {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = (position % padded_width) as i32;
    let row = (position / padded_width) as i32;
    let source_column = mirror(column, input_width as i32);
    let source_row = mirror(row, input_height as i32);
    let source = source_row as u32 * input_width + source_column as u32;

    let value = input[source as usize];
    let word = to_word::<E>(value, float);
    samples[(offset + position) as usize].store(word);
}

#[cube(launch_unchecked)]
fn crop_kernel<E: CubePrimitive>(
    samples: &Array<Atomic<u32>>,
    output: &mut Array<E>,
    offset: u32,
    padded_width: u32,
    output_width: u32,
    #[comptime] float: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % output_width;
    let row = position / output_width;
    let source = offset + row * padded_width + column;

    let word = samples[source as usize].load();
    output[ABSOLUTE_POS] = from_word::<E>(word, float);
}

fn mismatched_planes(count: usize, inputs: usize, outputs: usize) -> String {
    format!("a deblock of {count} planes was given {inputs} inputs and {outputs} outputs")
}

/// Asserts there is one input and one output per region, each pair one size, inside its region and not empty.
fn assert_fits(planes: DeblockPlanes<'_>, layout: &DeblockLayout) {
    let count = layout.regions.len();
    let inputs = planes.inputs.len();
    let outputs = planes.outputs.len();
    assert!(
        inputs == count && outputs == count,
        "{}",
        mismatched_planes(count, inputs, outputs)
    );

    let paired = planes.inputs.iter().zip(planes.outputs).zip(&layout.regions);
    for ((input, output), region) in paired {
        let width = input.width();
        let height = input.height();

        assert!(width > 0 && height > 0, "a deblocked plane must not be empty");
        assert!(
            output.width() == width && output.height() == height,
            "a deblocked plane must be {width}x{height} like its input, not {}x{}",
            output.width(),
            output.height(),
        );
        assert!(
            width <= region.width && height <= region.height,
            "a {width}x{height} plane does not fit its {}x{} region",
            region.width,
            region.height,
        );
    }
}

/// Deblocks every plane of `planes` into its output with H.264's in-loop filter, in one wavefront launch.
///
/// Each input is padded with its mirror into its region of `scratch`, filtered there, and cropped into its
/// output. One cube runs each plane, so `dispatch`'s cube size sets how many units share its lines.
pub fn deblock<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: DeblockPlanes<'_>,
    scratch: &Handle,
    layout: &DeblockLayout,
    samples: MatrixSamples,
    thresholds: Thresholds,
) {
    assert_fits(planes, layout);

    let bytes = layout.bytes();
    assert_holds(scratch, bytes, "deblocking scratch");

    match samples {
        MatrixSamples::U8 => launch::<u8, R>(client, dispatch, planes, scratch, layout, thresholds, false),
        MatrixSamples::U16 { .. } => {
            launch::<u16, R>(client, dispatch, planes, scratch, layout, thresholds, false)
        },
        MatrixSamples::F32 => launch::<f32, R>(client, dispatch, planes, scratch, layout, thresholds, true),
    }
}

fn launch<E: CubePrimitive, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: DeblockPlanes<'_>,
    scratch: &Handle,
    layout: &DeblockLayout,
    thresholds: Thresholds,
    float: bool,
) {
    let total = layout.samples();

    let inputs = planes.inputs.iter().zip(&layout.regions);
    for (input, region) in inputs {
        launch_pad::<E, R>(client, dispatch, *input, scratch, total, *region, float);
    }

    launch_wavefront::<R>(client, dispatch, scratch, layout, thresholds, float);

    let outputs = planes.outputs.iter().zip(&layout.regions);
    for (output, region) in outputs {
        launch_crop::<E, R>(client, dispatch, scratch, total, *region, *output, float);
    }
}

fn launch_pad<E: CubePrimitive, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    scratch: &Handle,
    total: usize,
    region: PlaneRegion,
    float: bool,
) {
    let input_array = input.array::<R>();
    let handle = scratch.clone();

    // Safety: `deblock` checked the scratch holds `total` words.
    let scratch_array = unsafe { ArrayArg::from_raw_parts(handle, total) };
    let samples = region.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every write lands in the region, which the layout placed inside the scratch, and every read is
    // mirrored into the input, which holds `width * height` samples of `E` as `DevicePlane::new` requires.
    unsafe {
        pad_kernel::launch_unchecked::<E, R>(
            client,
            cubes,
            unit,
            input_array,
            scratch_array,
            input.width(),
            input.height(),
            region.offset,
            region.width,
            samples as u32,
            float,
        );
    }
}

fn launch_wavefront<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    scratch: &Handle,
    layout: &DeblockLayout,
    thresholds: Thresholds,
    float: bool,
) {
    let total = layout.samples();
    let handle = scratch.clone();

    // Safety: `deblock` checked the scratch holds `total` words.
    let samples = unsafe { ArrayArg::from_raw_parts(handle, total) };
    let regions = layout.values::<R>();
    let planes = layout.regions.len() as u32;
    let cubes = CubeCount::Static(planes, 1, 1);
    let unit = dispatch.cube_dim();

    // Safety: cube `k` reads region `k`, and every tap of every line it filters lies inside that region,
    // since a region is whole 4 by 4 blocks and only edges with three samples on each side are filtered.
    unsafe {
        wavefront_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            samples,
            regions,
            thresholds.alpha,
            thresholds.beta,
            thresholds.c0,
            thresholds.c1,
            thresholds.peak,
            float,
        );
    }
}

fn launch_crop<E: CubePrimitive, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    scratch: &Handle,
    total: usize,
    region: PlaneRegion,
    output: DevicePlane<'_>,
    float: bool,
) {
    let handle = scratch.clone();

    // Safety: `deblock` checked the scratch holds `total` words.
    let scratch_array = unsafe { ArrayArg::from_raw_parts(handle, total) };
    let output_array = output.array::<R>();
    let samples = output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `deblock` checked the output fits inside its region, so every read lands in the region, and the
    // output holds `width * height` samples of `E` as `DevicePlane::new` requires.
    unsafe {
        crop_kernel::launch_unchecked::<E, R>(
            client,
            cubes,
            unit,
            scratch_array,
            output_array,
            region.offset,
            region.width,
            output.width(),
            float,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eight_bit(quant: i32, aoffset: i32, boffset: i32) -> Thresholds {
        thresholds(quant, aoffset, boffset, MatrixSamples::U8)
    }

    #[test]
    fn the_tables_hold_h264s_values_at_known_indices() {
        let alphas = [15, 16, 25, 51, 60].map(|index| ALPHA_TABLE[index]);
        let betas = [15, 16, 25, 51, 52, 60].map(|index| BETA_TABLE[index]);
        let limits = [20, 21, 25, 51, 52, 60].map(|index| TC0_TABLE[index]);

        assert_eq!(alphas, [0, 4, 13, 255, 255]);
        assert_eq!(betas, [0, 2, 4, 18, 19, 27]);
        assert_eq!(limits, [0, 1, 1, 17, 19, 35]);
    }

    #[test]
    fn quant_25_at_eight_bits_gives_alpha_13_beta_4_and_c0_1() {
        let found = eight_bit(25, 0, 0);

        let expected = Thresholds {
            alpha: 13.0,
            beta: 4.0,
            c0: 1.0,
            c1: 1.0,
            peak: 255.0,
        };
        assert_eq!(found, expected);
    }

    #[test]
    fn ten_bit_thresholds_scale_by_four() {
        let found = thresholds(25, 0, 0, MatrixSamples::U16 { bits: 10 });

        let expected = Thresholds {
            alpha: 52.0,
            beta: 16.0,
            c0: 4.0,
            c1: 4.0,
            peak: 1023.0,
        };
        assert_eq!(found, expected);
    }

    #[test]
    fn c0_follows_the_alpha_index() {
        let found = eight_bit(30, 10, -10);

        assert_eq!((found.alpha, found.c0, found.beta), (80.0, 5.0, 3.0));
    }

    #[test]
    fn offsets_clamp_the_indices_into_the_tables() {
        let high = eight_bit(40, 30, -100);
        let extreme = eight_bit(60, i32::MAX, i32::MIN);

        assert_eq!((high.alpha, high.c0, high.beta), (255.0, 35.0, 0.0));
        assert_eq!((extreme.alpha, extreme.c0, extreme.beta), (255.0, 35.0, 0.0));
    }

    #[test]
    fn float_thresholds_are_divided_by_255() {
        let found = thresholds(51, 0, 0, MatrixSamples::F32);

        let expected = Thresholds {
            alpha: 1.0,
            beta: 18.0 / 255.0,
            c0: 17.0 / 255.0,
            c1: 1.0 / 255.0,
            peak: 1.0,
        };
        assert_eq!(found, expected);
    }

    #[test]
    fn padded_frames_round_each_side_up_to_a_multiple_of_eight() {
        assert_eq!(padded_frame(64, 48), (64, 48));
        assert_eq!(padded_frame(70, 46), (72, 48));
        assert_eq!(padded_frame(1920, 1080), (1920, 1080));
        assert_eq!(padded_frame(1, 1), (8, 8));
    }

    #[test]
    fn regions_sit_back_to_back() {
        let regions = regions_of(&[(72, 48), (36, 24), (36, 24)]);

        let offsets: Vec<u32> = regions.iter().map(|region| region.offset).collect();
        assert_eq!(offsets, vec![0, 3456, 4320]);
        assert_eq!(regions[2].samples(), 864);
    }
}
