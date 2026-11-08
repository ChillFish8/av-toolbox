// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::h264::peak_of;
use super::rounding::{OPAQUE_ZERO, rounded_product};
use crate::matrix::MatrixSamples;
use crate::{DevicePlane, Dispatch};

/// The f32 nearest 1.01, which scales how far the strong pass moved each border sample.
pub const CHANGE_SCALE: f32 = 1.01;

/// The weights `cos((2k + 1) π / 16)` for `k` from 0 to 7, as f32.
///
/// With a constant, they pick out the four lowest DCT frequencies of an 8 by 8 block, which its low-pass
/// keeps. Weight `7 - k` is weight `k` negated.
pub const COSINES: [f32; 8] = [
    0.98078525,
    0.8314696,
    0.55557024,
    0.19509032,
    -0.19509032,
    -0.55557024,
    -0.8314696,
    -0.98078525,
];

/// Frames are extended on the right and bottom to a multiple of this before the low-pass.
const LOWPASS_MULTIPLE: u32 = 16;

/// The size a `width` by `height` frame is extended to before the low-pass, each side rounded up to a
/// multiple of 16.
pub fn lowpass_frame(width: u32, height: u32) -> (u32, u32) {
    let lowpass_width = width.next_multiple_of(LOWPASS_MULTIPLE);
    let lowpass_height = height.next_multiple_of(LOWPASS_MULTIPLE);

    (lowpass_width, lowpass_height)
}

/// The change map's value where the strong pass moved nothing, on every plane.
///
/// It is half the code range for integer samples and 0 for float samples.
pub fn neutral_of(samples: MatrixSamples) -> f32 {
    match samples.bits() {
        Some(bits) => (1u32 << (bits - 1)) as f32,
        None => 0.0,
    }
}

/// The change map's value on a border sample whose source sits `difference` above the strong pass.
///
/// Integer samples take `difference * 1.01 + neutral` as one fused multiply-add, rounded half to even and
/// kept between 0 and the peak. Float samples take `difference * 1.01`, neither rounded nor clamped.
pub fn change_value(difference: f32, samples: MatrixSamples) -> f32 {
    if samples == MatrixSamples::F32 {
        return difference * CHANGE_SCALE;
    }

    let neutral = neutral_of(samples);
    let peak = peak_of(samples);
    let fused = difference.mul_add(CHANGE_SCALE, neutral);

    stored_change(fused, peak)
}

/// `fused` rounded half to even, then kept between 0 and `peak`.
fn stored_change(fused: f32, peak: f32) -> f32 {
    let rounded = fused.round_ties_even();

    rounded.clamp(0.0, peak)
}

/// The change map's value for every difference of a plane's samples from the strong pass.
///
/// Integer samples hold one entry for each difference from minus the peak to the peak. Float samples need no
/// table, and hold one unused entry.
#[derive(Debug)]
pub struct ChangeTable {
    values: Handle,
    entries: usize,
    samples: MatrixSamples,
}

impl ChangeTable {
    fn values(&self) -> BufferArg {
        let handle = self.values.clone();

        // Safety: the buffer was made from exactly `entries` f32 values.
        unsafe { BufferArg::from_raw_parts(handle, self.entries) }
    }
}

/// The change map's value for each difference from minus the peak to the peak, or one unused entry for float
/// samples.
fn change_entries(samples: MatrixSamples) -> Vec<f32> {
    let Some(bits) = samples.bits() else {
        return vec![0.0];
    };

    let peak = (1i32 << bits) - 1;

    (-peak..=peak)
        .map(|difference| change_value(difference as f32, samples))
        .collect()
}

/// Builds the change table for planes of `samples` and uploads it.
pub fn upload_change_table(client: &Client, samples: MatrixSamples) -> ChangeTable {
    let entries = change_entries(samples);
    let count = entries.len();
    let bytes = f32::as_bytes(&entries);
    let values = client.create_from_slice(bytes);

    ChangeTable {
        values,
        entries: count,
        samples,
    }
}

/// The planes one QED launch reads and writes, all the same size.
#[derive(Debug, Clone, Copy)]
pub struct QedPlanes<'a> {
    pub source: DevicePlane<'a>,
    /// The strong pass of `source`.
    pub strong: DevicePlane<'a>,
    /// The normal pass of `source`, whose interior samples the launch overwrites.
    pub output: DevicePlane<'a>,
}

/// Where the low-pass reads position `index` of a side of `length` samples.
///
/// Positions inside read themselves. The next `length` positions mirror back with the edge sample repeated,
/// and every later one reads 0.
#[cube]
fn lowpass_mirror(index: u32, length: u32) -> u32 {
    let mut source = 0u32;

    if index < length {
        source = index;
    } else if index < 2 * length {
        source = 2 * length - 1 - index;
    }

    source
}

/// Whether the sample at `column`, `row` lies on the border of its 8 by 8 block.
#[cube]
fn is_border(column: u32, row: u32) -> bool {
    let column_place = column % 8;
    let row_place = row % 8;

    column_place == 0 || column_place == 7 || row_place == 0 || row_place == 7
}

/// What the change map is built against.
#[derive(CubeType)]
struct ChangeLimits {
    /// The largest difference the change table holds, the peak code value.
    largest: i32,
    neutral: f32,
    opaque_zero: u32,
}

/// The change map's value at `column`, `row` of a `width` wide plane.
///
/// Integer samples read the change table at their difference, kept between minus and plus the peak.
#[cube]
fn change_at<E: Scalar>(
    source: &[E],
    strong: &[E],
    changes: &[f32],
    column: u32,
    row: u32,
    width: u32,
    limits: &ChangeLimits,
    #[comptime] float: bool,
) -> f32 {
    let index = (row * width + column) as usize;
    let mut change = limits.neutral;

    if is_border(column, row) {
        if float {
            let difference = f32::cast_from(source[index]) - f32::cast_from(strong[index]);
            change = rounded_product(difference, CHANGE_SCALE, limits.opaque_zero);
        } else {
            let difference = i32::cast_from(source[index]) - i32::cast_from(strong[index]);
            let kept = i32::clamp(difference, -limits.largest, limits.largest);
            let entry = (kept + limits.largest) as u32;
            change = changes[entry as usize];
        }
    }

    change
}

/// The four sums a block's low-pass is built from.
#[derive(CubeType)]
struct BlockSums {
    total: f32,
    row_weighted: f32,
    column_weighted: f32,
    cross: f32,
}

/// Sums the 64 changes of a block, in raster order in `values`.
///
/// Rows run left to right and columns top to bottom, and the total adds the rows top to bottom. Each weighted
/// sum pairs line `k` with line `7 - k`, whose weights differ only in sign, and adds the four pairs in order.
#[cube]
fn block_sums(values: &[f32], opaque_zero: u32) -> BlockSums {
    let mut rows = Array::<f32>::new(8usize);
    let mut columns = Array::<f32>::new(8usize);

    #[unroll]
    for line in 0..8usize {
        let mut row_total = 0.0f32;
        let mut column_total = 0.0f32;

        #[unroll]
        for place in 0..8usize {
            let along_row = comptime!(line * 8 + place);
            let along_column = comptime!(place * 8 + line);

            row_total += values[along_row];
            column_total += values[along_column];
        }

        rows[line] = row_total;
        columns[line] = column_total;
    }

    let mut total = 0.0f32;

    #[unroll]
    for line in 0..8usize {
        total += rows[line];
    }

    let mut row_weighted = 0.0f32;
    let mut column_weighted = 0.0f32;
    let mut cross = 0.0f32;

    #[unroll]
    for outer in 0..4usize {
        let mirrored = comptime!(7 - outer);
        let outer_cosine = comptime!(f64::from(COSINES[outer]));
        let outer_weight = f32::new(outer_cosine);
        let row_difference = rows[outer] - rows[mirrored];
        let column_difference = columns[outer] - columns[mirrored];
        let mut part = 0.0f32;

        row_weighted += rounded_product(outer_weight, row_difference, opaque_zero);
        column_weighted += rounded_product(outer_weight, column_difference, opaque_zero);

        #[unroll]
        #[expect(
            clippy::needless_range_loop,
            reason = "the unrolled loops read the cosines when the kernel compiles"
        )]
        for inner in 0..4usize {
            let inner_cosine = comptime!(f64::from(COSINES[inner]));
            let inner_weight = f32::new(inner_cosine);
            let near_first = comptime!(outer * 8 + inner);
            let near_second = comptime!(outer * 8 + 7 - inner);
            let far_first = comptime!(mirrored * 8 + inner);
            let far_second = comptime!(mirrored * 8 + 7 - inner);
            let near = values[near_first] - values[near_second];
            let far = values[far_first] - values[far_second];

            part += rounded_product(inner_weight, near - far, opaque_zero);
        }

        cross += rounded_product(outer_weight, part, opaque_zero);
    }

    BlockSums {
        total,
        row_weighted,
        column_weighted,
        cross,
    }
}

/// `value` rounded to a whole number, a half rounding away from zero.
#[cube]
fn round_half_away(value: f32) -> f32 {
    let lower = f32::floor(value);
    let upper = f32::ceil(value);
    let mut rounded = upper;

    if value >= 0.0 {
        rounded = lower;

        if value - lower >= 0.5 {
            rounded = lower + 1.0;
        }
    } else if upper - value >= 0.5 {
        rounded = upper - 1.0;
    }

    rounded
}

/// An interior sample's output from its `source` and its `lowpass`.
///
/// Float samples take `source - lowpass`. Integer samples round the low-pass half away from zero into the
/// sample range, then move `source` by its distance from `neutral`, kept in the sample range.
#[cube]
fn interior_value<E: Scalar>(source: E, lowpass: f32, peak: f32, neutral: f32, #[comptime] float: bool) -> E {
    let original = f32::cast_from(source);

    if float {
        let value = original - lowpass;
        E::cast_from(value)
    } else {
        let rounded = round_half_away(lowpass);
        let stored = f32::clamp(rounded, 0.0, peak);
        let moved = original - (stored - neutral);
        let value = f32::clamp(moved, 0.0, peak);
        E::cast_from(value)
    }
}

/// Overwrites the interior samples of `output` with the source moved by the low-pass of the change toward the
/// strong pass, one unit to an 8 by 8 block of the plane extended to its low-pass size.
///
/// Blocks are independent, so no unit waits for another. A block reads past the plane's right and bottom
/// edges through [lowpass_mirror], and writes only its interior samples that lie inside the plane. A block
/// with no interior sample inside the plane writes nothing.
#[cube(launch_unchecked)]
fn qed_kernel<E: Scalar>(
    source: &[E],
    strong: &[E],
    output: &mut [E],
    changes: &[f32],
    width: u32,
    height: u32,
    blocks_across: u32,
    blocks: u32,
    peak: f32,
    neutral: f32,
    opaque_zero: u32,
    #[comptime] float: bool,
) {
    if ABSOLUTE_POS >= blocks as usize {
        terminate!();
    }

    let block = ABSOLUTE_POS as u32;
    let left = (block % blocks_across) * 8;
    let top = (block / blocks_across) * 8;

    if left + 1 >= width || top + 1 >= height {
        terminate!();
    }

    let limits = ChangeLimits {
        largest: i32::cast_from(peak),
        neutral,
        opaque_zero,
    };
    let mut values = Array::<f32>::new(64usize);

    #[unroll]
    for row in 0..8usize {
        let source_row = lowpass_mirror(top + row as u32, height);

        #[unroll]
        for column in 0..8usize {
            let source_column = lowpass_mirror(left + column as u32, width);
            let place = comptime!(row * 8 + column);

            values[place] = change_at::<E>(
                source,
                strong,
                changes,
                source_column,
                source_row,
                width,
                &limits,
                float,
            );
        }
    }

    let sums = block_sums(values.as_slice(), opaque_zero);
    let mean = sums.total * 0.015625;
    let row_part = sums.row_weighted * 0.03125;
    let column_part = sums.column_weighted * 0.03125;
    let cross_part = sums.cross * 0.0625;

    #[unroll]
    #[expect(
        clippy::needless_range_loop,
        reason = "the unrolled loops read the cosines when the kernel compiles"
    )]
    for row in 1..7usize {
        let y = top + row as u32;

        if y < height {
            let row_cosine = comptime!(f64::from(COSINES[row]));
            let row_weight = f32::new(row_cosine);
            let row_value = mean + rounded_product(row_weight, row_part, opaque_zero);

            #[unroll]
            #[expect(
                clippy::needless_range_loop,
                reason = "the unrolled loops read the cosines when the kernel compiles"
            )]
            for column in 1..7usize {
                let x = left + column as u32;

                if x < width {
                    let column_cosine = comptime!(f64::from(COSINES[column]));
                    let cross_cosine = comptime!(f64::from(COSINES[row] * COSINES[column]));
                    let column_weight = f32::new(column_cosine);
                    let cross_weight = f32::new(cross_cosine);
                    let with_column = row_value + rounded_product(column_weight, column_part, opaque_zero);
                    let lowpass = with_column + rounded_product(cross_weight, cross_part, opaque_zero);
                    let index = (y * width + x) as usize;

                    output[index] = interior_value::<E>(source[index], lowpass, peak, neutral, float);
                }
            }
        }
    }
}

/// Panics unless `plane` is `width` by `height`, naming it `name`.
fn assert_size(plane: DevicePlane<'_>, width: u32, height: u32, name: &str) {
    assert!(
        plane.width() == width && plane.height() == height,
        "the {name} plane is {}x{}, expected {width}x{height}",
        plane.width(),
        plane.height(),
    );
}

/// Overwrites every interior sample of `planes.output` with the QED output, in one launch.
///
/// The change map of `planes.source` against `planes.strong` is extended through its mirror to
/// `lowpass_size`, each 8 by 8 block is low-passed, and each interior sample of the plane moves by its
/// block's low-pass. Border samples are left as they are, so the output starts as the normal pass.
///
/// # Panics
///
/// Panics when the planes differ in size or are empty, when `lowpass_size` is not whole 8 by 8 blocks
/// covering the plane, or when `table` was made for other samples.
pub fn qed_interiors(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: QedPlanes<'_>,
    lowpass_size: (u32, u32),
    table: &ChangeTable,
    samples: MatrixSamples,
) {
    let width = planes.source.width();
    let height = planes.source.height();
    assert!(width > 0 && height > 0, "a QED plane must not be empty");
    assert_size(planes.strong, width, height, "strong");
    assert_size(planes.output, width, height, "output");

    let (lowpass_width, lowpass_height) = lowpass_size;
    let covers = lowpass_width >= width && lowpass_height >= height;
    let whole_blocks = lowpass_width % 8 == 0 && lowpass_height % 8 == 0;
    assert!(
        covers && whole_blocks,
        "a {width}x{height} plane does not fit a {lowpass_width}x{lowpass_height} low-pass"
    );
    assert!(
        table.samples == samples,
        "the change table was made for {:?} samples, not {samples:?}",
        table.samples
    );

    match samples {
        MatrixSamples::U8 => launch::<u8>(client, dispatch, planes, lowpass_size, table, samples, false),
        MatrixSamples::U16 { .. } => {
            launch::<u16>(client, dispatch, planes, lowpass_size, table, samples, false)
        },
        MatrixSamples::F32 => launch::<f32>(client, dispatch, planes, lowpass_size, table, samples, true),
    }
}

fn launch<E: Scalar>(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: QedPlanes<'_>,
    lowpass_size: (u32, u32),
    table: &ChangeTable,
    samples: MatrixSamples,
    float: bool,
) {
    let (lowpass_width, lowpass_height) = lowpass_size;
    let blocks_across = lowpass_width / 8;
    let blocks = blocks_across * (lowpass_height / 8);
    let cubes = dispatch.cube_count(blocks as usize);
    let unit = dispatch.cube_dim();
    let width = planes.source.width();
    let height = planes.source.height();
    let peak = peak_of(samples);
    let neutral = neutral_of(samples);
    let source = planes.source.array();
    let strong = planes.strong.array();
    let output = planes.output.array();
    let changes = table.values();

    // Safety: every read is mirrored into the plane, which `source` and `strong` hold as `width * height`
    // samples of `E`, and every write is an interior sample inside the plane, which `output` holds the same
    // way, as `DevicePlane::new` requires. The change table is read only for integer samples, between 0 and
    // twice the peak, which a table made for `samples` holds, as `qed_interiors` checked.
    unsafe {
        qed_kernel::launch_unchecked::<E>(
            client,
            cubes,
            unit,
            source,
            strong,
            output,
            changes,
            width,
            height,
            blocks_across,
            blocks,
            peak,
            neutral,
            OPAQUE_ZERO,
            float,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cosines_are_the_dct_weights_and_mirror_each_other() {
        for (index, cosine) in COSINES.iter().enumerate() {
            let angle = (2 * index + 1) as f64 * std::f64::consts::PI / 16.0;
            let exact = angle.cos() as f32;
            let mirrored = COSINES[7 - index];

            assert_eq!(*cosine, exact, "weight {index}");
            assert_eq!(mirrored, -*cosine, "weight {index} against its mirror");
        }
    }

    #[test]
    fn the_change_map_rounds_a_half_to_even() {
        // 50 * 1.01 + 128 rounds once to exactly 178.5 in f32, and -50 * 1.01 + 128 to 77.5.
        assert_eq!(change_value(50.0, MatrixSamples::U8), 178.0);
        assert_eq!(change_value(-50.0, MatrixSamples::U8), 78.0);
    }

    #[test]
    fn the_change_maps_multiply_add_rounds_once() {
        let samples = MatrixSamples::U16 { bits: 16 };

        let found = change_value(-32350.0, samples);

        // Rounded after the multiply as well, the sum would be exactly 94.5 and give 94.
        assert_eq!(found, 95.0);
    }

    #[test]
    fn the_change_map_stays_between_0_and_the_peak() {
        assert_eq!(change_value(-255.0, MatrixSamples::U8), 0.0);
        assert_eq!(change_value(255.0, MatrixSamples::U8), 255.0);
        assert_eq!(change_value(0.0, MatrixSamples::U16 { bits: 10 }), 512.0);
    }

    #[test]
    fn float_changes_are_scaled_without_rounding() {
        let found = change_value(0.25, MatrixSamples::F32);
        let below = change_value(-0.125, MatrixSamples::F32);

        assert_eq!(found, 0.25 * CHANGE_SCALE);
        assert_eq!(below, -0.125 * CHANGE_SCALE);
    }

    #[test]
    fn neutral_is_half_the_code_range_or_0_for_float() {
        let neutrals = [
            neutral_of(MatrixSamples::U8),
            neutral_of(MatrixSamples::U16 { bits: 10 }),
            neutral_of(MatrixSamples::U16 { bits: 16 }),
            neutral_of(MatrixSamples::F32),
        ];

        assert_eq!(neutrals, [128.0, 512.0, 32768.0, 0.0]);
    }

    #[test]
    fn lowpass_frames_round_each_side_up_to_16() {
        assert_eq!(lowpass_frame(64, 48), (64, 48));
        assert_eq!(lowpass_frame(70, 46), (80, 48));
        assert_eq!(lowpass_frame(1920, 1080), (1920, 1088));
        assert_eq!(lowpass_frame(6, 4), (16, 16));
    }

    #[test]
    fn the_change_table_covers_every_difference_from_minus_the_peak_to_the_peak() {
        let eight_bit = change_entries(MatrixSamples::U8);
        let float = change_entries(MatrixSamples::F32);

        assert_eq!(eight_bit.len(), 511);
        assert_eq!(
            (eight_bit[0], eight_bit[255], eight_bit[510]),
            (0.0, 128.0, 255.0)
        );
        assert_eq!(eight_bit[255 + 50], change_value(50.0, MatrixSamples::U8));
        assert_eq!(float, vec![0.0]);
    }
}
