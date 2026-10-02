mod check;
mod exact;
mod interpolate;
mod mask;
mod path;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

#[cfg(any(test, feature = "test-support"))]
pub use self::exact::eedi3_mul_exact;
use crate::plane::{assert_holds, assert_plane};
use crate::{DevicePlane, Dispatch};

/// The widest `mdis`, which sizes the path kernel's shared memory.
const EEDI3_MAX_MDIS: u32 = 40;

/// The backpointers packed into one `u32`, 2 bits each.
const BACKPOINTERS_PER_WORD: u32 = 16;

/// The active columns packed into one `i32`, 1 bit each.
const ACTIVE_PER_WORD: u32 = 32;

/// The backpointer code of a step that comes from direction 0, whatever direction it leads into.
const TO_ZERO_CODE: u32 = 3;

/// The arguments of one EEDI3 field pass, on the 8-bit scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eedi3Arguments {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub nrad: u32,
    pub mdis: u32,
    pub vcheck: u32,
    pub vthresh: [f32; 3],
}

impl Default for Eedi3Arguments {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            beta: 0.25,
            gamma: 20.0,
            nrad: 2,
            mdis: 20,
            vcheck: 2,
            vthresh: [32.0, 64.0, 4.0],
        }
    }
}

/// The f32 weights of the costs and the reliability check, scaled to the plane's bit depth.
///
/// Each is one rounded f32 operation on the arguments, in the order the plugin computes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eedi3Weights {
    /// `alpha / 3`, the weight of each of the three band sums.
    pub alpha3: f32,
    pub beta: f32,
    pub gamma: f32,
    /// `(1 - alpha) - beta`, the weight of the vertical difference.
    pub vertical: f32,
    pub reciprocal0: f32,
    pub reciprocal1: f32,
    pub reciprocal2: f32,
    pub vthresh2: f32,
}

/// The settings of one EEDI3 field pass, with its weights computed for the plane's bit depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eedi3Settings {
    weights: Eedi3Weights,
    nrad: u32,
    mdis: u32,
    vcheck: u32,
    peak: u32,
}

/// Whether `value` is 0 or between `1e-30` and 1.
fn is_weight(value: f32) -> bool {
    value == 0.0 || (1e-30..=1.0).contains(&value)
}

/// Asserts that `arguments` lie inside the limits that keep every weight and cost finite and normal.
fn check_arguments(arguments: &Eedi3Arguments, bits: u32) {
    assert!(
        (8..=16).contains(&bits),
        "EEDI3 takes 8 to 16 bits per sample, not {bits}"
    );
    assert!(
        is_weight(arguments.alpha),
        "EEDI3's alpha is 0 or between 1e-30 and 1, not {}",
        arguments.alpha
    );
    assert!(
        is_weight(arguments.beta),
        "EEDI3's beta is 0 or between 1e-30 and 1, not {}",
        arguments.beta
    );

    let weight_sum = arguments.alpha + arguments.beta;
    assert!(
        weight_sum <= 1.0,
        "EEDI3's alpha and beta add up to at most 1, not {weight_sum}"
    );
    assert!(
        (0.0..=1e30).contains(&arguments.gamma),
        "EEDI3's gamma is between 0 and 1e30, not {}",
        arguments.gamma
    );
    assert!(
        arguments.nrad <= 3,
        "EEDI3's nrad is at most 3, not {}",
        arguments.nrad
    );
    assert!(
        (1..=EEDI3_MAX_MDIS).contains(&arguments.mdis),
        "EEDI3's mdis is between 1 and {EEDI3_MAX_MDIS}, not {}",
        arguments.mdis
    );
    assert!(
        arguments.vcheck <= 3,
        "EEDI3's vcheck is at most 3, not {}",
        arguments.vcheck
    );

    for threshold in arguments.vthresh {
        assert!(
            threshold.is_finite() && threshold >= 1e-30,
            "EEDI3's vthresh values are finite and at least 1e-30, not {threshold}"
        );
    }
}

impl Eedi3Settings {
    /// Scales `arguments` to a plane of `bits` per sample.
    ///
    /// `beta`, `gamma`, `vthresh[0]` and `vthresh[1]` scale by `2^(bits - 8)`. `alpha` and `vthresh[2]` do
    /// not. The divisions run here on the host, so the device only multiplies.
    pub fn new(arguments: &Eedi3Arguments, bits: u32) -> Self {
        check_arguments(arguments, bits);

        let unit = (1u32 << (bits - 8)) as f32;
        let without_alpha = 1.0 - arguments.alpha;
        let threshold0 = arguments.vthresh[0] * unit;
        let threshold1 = arguments.vthresh[1] * unit;

        let weights = Eedi3Weights {
            alpha3: arguments.alpha / 3.0,
            beta: arguments.beta * unit,
            gamma: arguments.gamma * unit,
            vertical: without_alpha - arguments.beta,
            reciprocal0: 1.0 / threshold0,
            reciprocal1: 1.0 / threshold1,
            reciprocal2: 1.0 / arguments.vthresh[2],
            vthresh2: arguments.vthresh[2],
        };

        Self {
            weights,
            nrad: arguments.nrad,
            mdis: arguments.mdis,
            vcheck: arguments.vcheck,
            peak: (1u32 << bits) - 1,
        }
    }

    pub fn weights(&self) -> &Eedi3Weights {
        &self.weights
    }

    pub fn mdis(&self) -> u32 {
        self.mdis
    }
}

/// The scratch buffers of one EEDI3 field pass, one row per rebuilt row.
///
/// - `rebuilt` holds the rebuilt rows before the reliability check, as f32 values.
/// - `directions` holds the direction each rebuilt sample was interpolated along, as `i32` values.
/// - `backpointers` holds the path's backpointers, 2 bits each, 16 columns to a `u32`. Each rebuilt row
///   holds one run of words per direction, each run covering the columns in order.
/// - `lenders` holds `i32` values used under a mask. First comes one per rebuilt row, whose bits say
///   whether its column 0 and column 1 are active. Then come two per rebuilt row, the rebuilt rows that
///   lend it the costs of column 0 and column 1. After them come the active columns of each rebuilt row,
///   one bit a column, 32 to an `i32`.
///
/// The four buffers must be distinct. [eedi3_scratch](crate::aa::eedi3_scratch) gives the size of each.
#[derive(Debug, Clone, Copy)]
pub struct Eedi3Scratch<'a> {
    pub rebuilt: &'a Handle,
    pub directions: &'a Handle,
    pub backpointers: &'a Handle,
    pub lenders: &'a Handle,
}

/// The byte size of each buffer of an [Eedi3Scratch](crate::aa::Eedi3Scratch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eedi3ScratchSizes {
    pub rebuilt: usize,
    pub directions: usize,
    pub backpointers: usize,
    pub lenders: usize,
}

/// The scratch sizes a `width` by `height` field pass searching up to `mdis` columns needs.
pub fn eedi3_scratch(width: u32, height: u32, mdis: u32) -> Eedi3ScratchSizes {
    let rows = height as usize / 2;
    let rebuilt_samples = rows * width as usize;
    let directions = 2 * mdis as usize + 1;
    let groups = width.div_ceil(BACKPOINTERS_PER_WORD) as usize;
    let words = rows * directions * groups;
    let active_words = width.div_ceil(ACTIVE_PER_WORD) as usize;
    let lenders = 3 * rows + rows * active_words;

    Eedi3ScratchSizes {
        rebuilt: rebuilt_samples * size_of::<f32>(),
        directions: rebuilt_samples * size_of::<i32>(),
        backpointers: words * size_of::<u32>(),
        lenders: lenders * size_of::<i32>(),
    }
}

/// The optional planes a field pass reads beside its input, each at the input's size.
///
/// `sclip` replaces the vertical cubic the reliability check falls back to. `mclip` holds 8-bit mask
/// values, and a column is searched only where a non-zero value lies within `mdis` columns of it on the
/// same row.
#[derive(Debug, Clone, Copy, Default)]
pub struct Eedi3DeviceClips<'a> {
    pub sclip: Option<DevicePlane<'a>>,
    pub mclip: Option<DevicePlane<'a>>,
}

/// The plane row of the rebuilt row `field_row`, as the kernels count them.
#[cube]
pub(super) fn plane_row(field_row: u32, first_rebuilt: u32) -> u32 {
    2 * field_row + first_rebuilt
}

/// `index` reflected into `0..length` without repeating the edge, then clamped into it.
///
/// A single reflection covers every read of a plane of at least 4 by 4. The clamp only keeps the index in
/// bounds.
#[cube]
pub(super) fn reflect(index: i32, length: u32) -> u32 {
    let last = length as i32 - 1;
    let mut reflected = index;
    if index < 0 {
        reflected = -index;
    } else if index > last {
        reflected = 2 * last - index;
    }

    let inside = i32::clamp(reflected, 0, last);
    inside as u32
}

/// The sample at `column` of the plane row starting at `row_start`, reflecting the column once past the
/// plane's side.
#[cube]
pub(super) fn sample_at(input: &Array<f32>, row_start: u32, column: i32, width: u32) -> i32 {
    let inside = reflect(column, width);

    input[(row_start + inside) as usize] as i32
}

/// The start of plane row `row` in a `width` wide plane `height` rows tall, reflecting the row once past
/// the plane's top or bottom.
#[cube]
pub(super) fn row_start(row: i32, width: u32, height: u32) -> u32 {
    let inside = reflect(row, height);

    inside * width
}

/// One field pass over planes and a scratch whose sizes have been checked.
///
/// The pass rebuilds `rows` rows of `width` samples, rebuilt row `k` being plane row `2k + first_rebuilt`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldPass<'a> {
    input: DevicePlane<'a>,
    output: DevicePlane<'a>,
    sclip: Option<DevicePlane<'a>>,
    mclip: Option<DevicePlane<'a>>,
    scratch: Eedi3Scratch<'a>,
    settings: Eedi3Settings,
    offset: u32,
    first_rebuilt: u32,
    width: u32,
    height: u32,
    rows: u32,
}

impl<'a> FieldPass<'a> {
    /// Checks the planes, the offset and every scratch buffer's size.
    fn new(
        input: DevicePlane<'a>,
        output: DevicePlane<'a>,
        scratch: Eedi3Scratch<'a>,
        offset: u32,
        settings: &Eedi3Settings,
        clips: Eedi3DeviceClips<'a>,
    ) -> Self {
        let width = input.width();
        let height = input.height();

        assert!(offset <= 1, "an EEDI3 field offset is 0 or 1, not {offset}");
        assert!(
            width >= 4 && height >= 4 && height.is_multiple_of(2),
            "an EEDI3 plane needs a width and an even height of at least 4, not {width}x{height}"
        );

        let sizes = eedi3_scratch(width, height, settings.mdis);
        let words = sizes.backpointers / size_of::<u32>();
        assert!(
            input.samples() <= u32::MAX as usize && words <= u32::MAX as usize,
            "an EEDI3 plane of {width}x{height} has too many samples to index"
        );

        assert_plane(&input, width, height, "input");
        assert_plane(&output, width, height, "output");
        if let Some(sclip) = clips.sclip {
            assert_plane(&sclip, width, height, "sclip");
        }

        if let Some(mclip) = clips.mclip {
            assert_plane(&mclip, width, height, "mclip");
        }

        assert_holds(scratch.rebuilt, sizes.rebuilt, "rebuilt scratch");
        assert_holds(scratch.directions, sizes.directions, "directions scratch");
        assert_holds(scratch.backpointers, sizes.backpointers, "backpointers scratch");
        assert_holds(scratch.lenders, sizes.lenders, "lenders scratch");

        Self {
            input,
            output,
            sclip: clips.sclip,
            mclip: clips.mclip,
            scratch,
            settings: *settings,
            offset,
            first_rebuilt: 1 - offset,
            width,
            height,
            rows: height / 2,
        }
    }

    fn rebuilt_samples(&self) -> usize {
        self.rows as usize * self.width as usize
    }

    fn span(&self) -> u32 {
        2 * self.settings.mdis + 1
    }

    fn groups(&self) -> u32 {
        self.width.div_ceil(BACKPOINTERS_PER_WORD)
    }

    fn active_words(&self) -> u32 {
        self.width.div_ceil(ACTIVE_PER_WORD)
    }

    /// The rebuilt rows scratch as a kernel argument.
    fn rebuilt_array<R: Runtime>(&self) -> ArrayArg<R> {
        let owned = self.scratch.rebuilt.clone();
        let samples = self.rebuilt_samples();

        // Safety: `new` checked that `rebuilt` holds an f32 value per rebuilt sample.
        unsafe { ArrayArg::from_raw_parts(owned, samples) }
    }

    /// The directions scratch as a kernel argument.
    fn directions_array<R: Runtime>(&self) -> ArrayArg<R> {
        let owned = self.scratch.directions.clone();
        let samples = self.rebuilt_samples();

        // Safety: `new` checked that `directions` holds an `i32` per rebuilt sample.
        unsafe { ArrayArg::from_raw_parts(owned, samples) }
    }

    /// The backpointers scratch as a kernel argument.
    fn backpointers_array<R: Runtime>(&self) -> ArrayArg<R> {
        let owned = self.scratch.backpointers.clone();
        let words = self.rows as usize * self.span() as usize * self.groups() as usize;

        // Safety: `new` checked that `backpointers` holds a `u32` per word of every rebuilt row.
        unsafe { ArrayArg::from_raw_parts(owned, words) }
    }

    /// The lenders scratch as a kernel argument.
    fn lenders_array<R: Runtime>(&self) -> ArrayArg<R> {
        let owned = self.scratch.lenders.clone();
        let rows = self.rows as usize;
        let values = 3 * rows + rows * self.active_words() as usize;

        // Safety: `new` checked that `lenders` holds three `i32` values per rebuilt row, then a word of
        // active bits per 32 columns of every rebuilt row.
        unsafe { ArrayArg::from_raw_parts(owned, values) }
    }

    /// The mask plane, or the input standing in for it when the pass has none.
    fn mask_plane(&self) -> DevicePlane<'a> {
        self.mclip.unwrap_or(self.input)
    }
}

/// Which reliability check a field pass ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eedi3Check {
    /// One launch walked every row, holding the previous row in shared memory.
    Persistent,
    /// Each row was its own launch, for a plane wider than the shared memory holds.
    PerRow,
    /// `vcheck` was 0, so nothing was checked.
    Skipped,
}

/// The widest plane the persistent check runs on, an eighth of the device's shared memory in samples.
pub fn eedi3_check_capacity<R: Runtime>(client: &ComputeClient<R>) -> u32 {
    let shared_bytes = client.properties().hardware.max_shared_memory_size;
    let samples = shared_bytes / (2 * size_of::<f32>());

    u32::try_from(samples).unwrap_or(u32::MAX)
}

/// One EEDI3 field pass, keeping the rows of parity `offset` and rebuilding the others into `output`.
///
/// The planes hold integer code values and must be distinct buffers of the same size, at least 4 by 4
/// with an even height. Each rebuilt row follows the cheapest path of directions along the row, and the
/// reliability check then blends unreliable samples towards `sclip` or a vertical cubic. Under `mclip`,
/// a column with no set mask sample within `mdis` columns skips its search and takes direction 0.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn eedi3_field<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    scratch: Eedi3Scratch<'_>,
    offset: u32,
    settings: &Eedi3Settings,
    clips: Eedi3DeviceClips<'_>,
) -> Eedi3Check {
    let pass = FieldPass::new(input, output, scratch, offset, settings, clips);
    let capacity = eedi3_check_capacity(client);

    run_field(client, dispatch, &pass, capacity)
}

/// The same as [eedi3_field](crate::aa::eedi3_field), with the persistent check running only on planes
/// at most `capacity` wide.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn eedi3_field_with_check_capacity<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    scratch: Eedi3Scratch<'_>,
    offset: u32,
    settings: &Eedi3Settings,
    clips: Eedi3DeviceClips<'_>,
    capacity: u32,
) -> Eedi3Check {
    let pass = FieldPass::new(input, output, scratch, offset, settings, clips);

    run_field(client, dispatch, &pass, capacity)
}

/// Runs every step of `pass`, checking it in one launch when it is at most `capacity` wide.
fn run_field<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    capacity: u32,
) -> Eedi3Check {
    if pass.mclip.is_some() {
        mask::mark_active_edges(client, pass);
        mask::find_lenders(client, pass);
    }

    path::search_paths(client, pass);
    interpolate::backtrack(client, pass);
    interpolate::interpolate(client, dispatch, pass);

    if pass.settings.vcheck == 0 {
        check::assemble(client, dispatch, pass);
        return Eedi3Check::Skipped;
    }

    if pass.width <= capacity {
        check::persistent_check(client, pass, capacity);
        return Eedi3Check::Persistent;
    }

    check::assemble(client, dispatch, pass);
    check::reliability_check(client, dispatch, pass);

    Eedi3Check::PerRow
}
