mod directions;
mod masks;
mod votes;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::{DevicePlane, Dispatch};

/// The arguments of the EEDI2 stage, before they are scaled to a plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eedi2Arguments {
    pub mthresh: u32,
    pub lthresh: u32,
    pub vthresh: u32,
    pub estr: u32,
    pub dstr: u32,
    pub maxd: u32,
    pub nt: u32,
    pub pp: u32,
}

impl Default for Eedi2Arguments {
    fn default() -> Self {
        Self {
            mthresh: 10,
            lthresh: 20,
            vthresh: 20,
            estr: 2,
            dstr: 4,
            maxd: 24,
            nt: 50,
            pp: 1,
        }
    }
}

/// The thresholds of one EEDI2 field pass, scaled to the plane's bit depth and subsampling.
///
/// Every threshold is an integer code value held in f32. The reduce factors are powers of two.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eedi2Settings {
    /// `2^(8 - bits)`, which turns a floor multiplication into a right shift by `bits - 8`.
    reduce: f32,
    /// `2^(6 - bits)`, which turns a floor multiplication into a right shift by `bits - 6`.
    reduce_quarter: f32,
    unit: f32,
    peak: f32,
    neutral: f32,
    ten: f32,
    twelve: f32,
    mthresh_squared: f32,
    lthresh: f32,
    vthresh_81: f32,
    estr: f32,
    dstr: f32,
    maxd: u32,
    nt13: f32,
    nt19: f32,
}

impl Eedi2Settings {
    /// Scales `arguments` to a plane of `bits` per sample.
    ///
    /// `ssw` is the log2 horizontal subsampling of the plane in the direction the pass runs, and shifts
    /// `maxd` right.
    pub fn new(arguments: &Eedi2Arguments, bits: u32, ssw: u32) -> Self {
        assert!(
            (8..=16).contains(&bits),
            "EEDI2 takes 8 to 16 bits per sample, not {bits}"
        );
        assert!(ssw <= 2, "EEDI2's subsampling shift is at most 2, not {ssw}");
        assert!(arguments.pp <= 1, "EEDI2's pp is 0 or 1, not {}", arguments.pp);
        assert!(
            (1..=29).contains(&arguments.maxd),
            "EEDI2's maxd is between 1 and 29, not {}",
            arguments.maxd
        );

        let shift = bits - 8;
        let unit = (1u32 << shift) as f32;
        let nt = arguments.nt as f32 * unit;
        let mthresh = arguments.mthresh as f32;
        let reduce = 1.0 / unit;

        Self {
            reduce,
            reduce_quarter: reduce * 0.25,
            unit,
            peak: ((1u32 << bits) - 1) as f32,
            neutral: (1u32 << (bits - 1)) as f32,
            ten: 10.0 * unit,
            twelve: 12.0 * unit,
            mthresh_squared: mthresh * mthresh,
            lthresh: arguments.lthresh as f32,
            vthresh_81: arguments.vthresh as f32 * 81.0,
            estr: arguments.estr as f32,
            dstr: arguments.dstr as f32,
            maxd: arguments.maxd >> ssw,
            nt13: 13.0 * nt,
            nt19: 19.0 * nt,
        }
    }
}

/// The scratch buffers of one EEDI2 field pass, each holding f32 values.
///
/// Passes reuse the buffers once their contents are no longer read:
///
/// - `edges` holds the raw edge mask, then the dilated mask, and from pass 5 on the final edge mask.
/// - `working` holds the eroded masks, then the raw and expanded directions, then the gap directions and
///   the expanded gap directions.
/// - `directions` holds the filtered directions, and from pass 9 on the cleaned directions.
/// - `spare` holds the filtered gap directions.
///
/// The four buffers must be distinct. [eedi2_scratch](crate::aa::eedi2_scratch) gives the size of each.
#[derive(Debug, Clone, Copy)]
pub struct Eedi2Scratch<'a> {
    pub edges: &'a Handle,
    pub working: &'a Handle,
    pub directions: &'a Handle,
    pub spare: &'a Handle,
}

/// The byte size of each buffer of an [Eedi2Scratch](crate::aa::Eedi2Scratch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eedi2ScratchSizes {
    pub edges: usize,
    pub working: usize,
    pub directions: usize,
    pub spare: usize,
}

/// The scratch sizes a `width` by `height` field pass needs.
///
/// A field buffer holds one row per kept row, `height / 2` rows in all, and `spare` one row fewer.
pub fn eedi2_scratch(width: u32, height: u32) -> Eedi2ScratchSizes {
    let field_rows = height as usize / 2;
    let gap_rows = field_rows.saturating_sub(1);
    let field_bytes = field_rows * width as usize * size_of::<f32>();
    let gap_bytes = gap_rows * width as usize * size_of::<f32>();

    Eedi2ScratchSizes {
        edges: field_bytes,
        working: field_bytes,
        directions: field_bytes,
        spare: gap_bytes,
    }
}

/// Asserts that `handle` holds at least `bytes`, naming the scratch buffer `name`.
fn assert_scratch_size(handle: &Handle, bytes: usize, name: &str) {
    let held = handle.size_in_used();

    assert!(
        held >= bytes as u64,
        "the {name} scratch holds {held} bytes, expected {bytes}"
    );
}

/// Field row `row` at `column`, read from plane row `2 * row + offset`.
#[cube]
pub(super) fn field_sample(input: &Array<f32>, width: u32, offset: u32, row: u32, column: u32) -> f32 {
    let plane_row = 2 * row + offset;
    let index = plane_row * width + column;

    input[index as usize]
}

/// Whether the field pixel at `position` lies inside the border.
#[cube]
fn is_interior(position: u32, width: u32, rows: u32) -> bool {
    let row = position / width;
    let column = position % width;

    row >= 1 && row + 1 < rows && column >= 1 && column + 1 < width
}

/// A scratch buffer that holds a whole field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldBuffer {
    Edges,
    Working,
    Directions,
}

/// A scratch buffer that holds a gap map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GapBuffer {
    Working,
    Spare,
}

/// One field pass over a plane and a scratch whose sizes have been checked.
///
/// The field has `rows` rows of `width` samples, row `r` being plane row `2r + offset`. The gap maps have
/// one row fewer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldPass<'a> {
    input: DevicePlane<'a>,
    scratch: Eedi2Scratch<'a>,
    offset: u32,
    width: u32,
    rows: u32,
}

impl<'a> FieldPass<'a> {
    /// Checks the plane, the offset and every scratch buffer's size.
    pub(crate) fn new(input: DevicePlane<'a>, scratch: Eedi2Scratch<'a>, offset: u32) -> Self {
        let width = input.width();
        let height = input.height();

        assert!(offset <= 1, "an EEDI2 field offset is 0 or 1, not {offset}");
        assert!(
            width >= 1 && height >= 4 && height.is_multiple_of(2),
            "an EEDI2 plane needs an even height of at least 4 and a width, not {width}x{height}"
        );

        let sizes = eedi2_scratch(width, height);
        assert_scratch_size(scratch.edges, sizes.edges, "edges");
        assert_scratch_size(scratch.working, sizes.working, "working");
        assert_scratch_size(scratch.directions, sizes.directions, "directions");
        assert_scratch_size(scratch.spare, sizes.spare, "spare");

        Self {
            input,
            scratch,
            offset,
            width,
            rows: height / 2,
        }
    }

    fn field_samples(&self) -> usize {
        self.rows as usize * self.width as usize
    }

    fn gap_samples(&self) -> usize {
        (self.rows as usize - 1) * self.width as usize
    }

    /// The field buffer `buffer` as a kernel argument.
    fn field_array<R: Runtime>(&self, buffer: FieldBuffer) -> ArrayArg<R> {
        let handle = match buffer {
            FieldBuffer::Edges => self.scratch.edges,
            FieldBuffer::Working => self.scratch.working,
            FieldBuffer::Directions => self.scratch.directions,
        };

        let owned = handle.clone();
        let samples = self.field_samples();

        // Safety: `new` checked that `edges`, `working` and `directions` each hold a field.
        unsafe { ArrayArg::from_raw_parts(owned, samples) }
    }

    /// The gap buffer `buffer` as a kernel argument.
    fn gap_array<R: Runtime>(&self, buffer: GapBuffer) -> ArrayArg<R> {
        let handle = match buffer {
            GapBuffer::Working => self.scratch.working,
            GapBuffer::Spare => self.scratch.spare,
        };

        let owned = handle.clone();
        let samples = self.gap_samples();

        // Safety: `new` checked that `working` and `spare` each hold at least a gap map.
        unsafe { ArrayArg::from_raw_parts(owned, samples) }
    }
}

/// Launches EEDI2's pass `number` over `pass`.
fn run_pass<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
    number: u32,
) {
    match number {
        1 => masks::edge_mask(client, dispatch, pass, settings),
        2 => masks::erode(client, dispatch, pass, settings),
        3 => masks::dilate(client, dispatch, pass, settings),
        4 => masks::erode(client, dispatch, pass, settings),
        5 => masks::remove_small_gaps(client, dispatch, pass),
        6 => directions::raw_directions(client, dispatch, pass, settings),
        7 => directions::filter_directions(client, dispatch, pass, settings),
        8 => directions::expand_directions(client, dispatch, pass, settings),
        9 => directions::clean_directions(client, dispatch, pass, settings),
        10 => directions::gap_directions(client, dispatch, pass, settings),
        11 => directions::filter_gaps(client, dispatch, pass, settings),
        12 => directions::expand_gaps(client, dispatch, pass, settings),
        _ => panic!("EEDI2's mask and direction passes are 1 to 12, not {number}"),
    }
}

/// Runs EEDI2's passes 1 to `last_pass` over `pass`, in order.
pub(crate) fn run_passes<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
    last_pass: u32,
) {
    for number in 1..=last_pass {
        run_pass(client, dispatch, pass, settings, number);
    }
}

/// An intermediate of EEDI2's first 12 passes that a test can read.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eedi2Stage {
    /// `E0`, the raw edge mask of pass 1.
    RawEdges,
    /// `E`, the final edge mask of pass 5.
    Edges,
    /// `D0`, the raw directions of pass 6.
    RawDirections,
    /// `D3`, the cleaned directions of pass 9.
    Directions,
    /// `X0`, the gap directions of pass 10.
    GapDirections,
    /// `X2`, the expanded gap directions of pass 12.
    ExpandedGaps,
}

/// Runs EEDI2's passes up to the one writing `stage`, and gives back the scratch buffer holding it.
///
/// The field keeps the rows of parity `offset`. A field stage fills `height / 2` rows of the buffer and a
/// gap stage one row fewer, each `width` values wide.
#[cfg(any(test, feature = "test-support"))]
pub fn eedi2_passes_through<'a, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'a>,
    scratch: Eedi2Scratch<'a>,
    offset: u32,
    settings: &Eedi2Settings,
    stage: Eedi2Stage,
) -> &'a Handle {
    let pass = FieldPass::new(input, scratch, offset);
    let (last_pass, buffer) = match stage {
        Eedi2Stage::RawEdges => (1, scratch.edges),
        Eedi2Stage::Edges => (5, scratch.edges),
        Eedi2Stage::RawDirections => (6, scratch.working),
        Eedi2Stage::Directions => (9, scratch.directions),
        Eedi2Stage::GapDirections => (10, scratch.working),
        Eedi2Stage::ExpandedGaps => (12, scratch.working),
    };

    run_passes(client, dispatch, &pass, settings, last_pass);

    buffer
}
