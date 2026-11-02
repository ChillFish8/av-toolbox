// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::frame::format::{ChromaLocation, ColourInfo, FrameFormat, MaskChroma, MaskFormat};
use crate::frame::gpu::FrameMeta;
use crate::frame::layout::{Channel, FrameLayout, Subsampling};
use crate::frame::sample::SampleFormat;
use crate::memory::{RingGeometry, RingSpec};
use crate::{Backend, BuildContext, BuildTask, test_device};

/// A base no index into these buffers can reach, so a sample above it was never written again.
const BASELINE: u32 = 1_000_000;

/// Writes `base` plus each element's own index into `output`.
///
/// A wrong offset and a wrong length fail differently. An offset error moves where the run of
/// indices lands, and a length error leaves a tail holding whatever was in it before.
#[cube(launch_unchecked)]
pub(crate) fn write_index(output: &mut [u32], base: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = base + ABSOLUTE_POS as u32;
}

fn format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

/// How many `u32` values `bytes` of a buffer hold.
pub(crate) const fn elements(bytes: usize) -> usize {
    bytes / size_of::<u32>()
}

fn read_back(client: &Client, handle: &Handle) -> Vec<u32> {
    let bytes = client
        .read_one(handle.clone())
        .expect("the device should hand back what it was asked for");

    u32::from_bytes(&bytes).to_vec()
}

/// Runs the kernel over `output`, sized from the context so the launch covers every element.
pub(crate) fn launch(context: &BuildContext, output: BufferArg, count: usize, base: u32) {
    let frames = context.frame_context();
    let cubes = frames.cube_count(count);
    let shape = frames.cube_dim();

    // Safety: `output` names no more elements than the buffer behind it holds, and the kernel
    // writes nothing past the length it is given.
    unsafe {
        write_index::launch_unchecked(context.client(), cubes, shape, output, base);
    }
}

/// Writes indices into a fresh frame's luma plane and hands its samples back.
struct WriteLuma;

impl BuildTask for WriteLuma {
    type Output = Vec<u32>;

    fn run(self, context: &BuildContext) -> Self::Output {
        let frames = context.frame_context();
        let frame = frames.alloc_frame(&format(), FrameMeta::at(0));
        let luma = frame.plane(Channel::Y).expect("a yuv frame carries luma");

        let count = elements(luma.buffer().bytes());
        let binding = luma.array::<u32>();
        launch(context, binding, count, 0);

        read_back(context.client(), luma.buffer().handle())
    }
}

/// The width of the plane [WriteOversizedPlane] writes.
const OVERSIZED_WIDTH: usize = 8192;

/// The most memory the oversized plane may take, so a device with a huge axis limit fails the
/// test rather than the allocation.
const OVERSIZED_MAX_BYTES: usize = 1 << 30;

/// A plane with more cubes in it than a device takes on one axis of a launch.
///
/// Its height is grown until its `u32` elements need one cube more than `axis_limit`, counted in
/// cubes of `units_per_cube`, since the cube size and the limit both vary by device.
fn oversized_format(units_per_cube: usize, axis_limit: usize) -> MaskFormat {
    let elements_needed = (axis_limit + 1) * units_per_cube;
    let bytes_needed = elements_needed * size_of::<u32>();

    assert!(
        bytes_needed <= OVERSIZED_MAX_BYTES,
        "this device takes {axis_limit} cubes of {units_per_cube} units on one axis, so a plane \
         that overruns it needs {bytes_needed} bytes, more than this test allows itself",
    );

    let height = bytes_needed.div_ceil(OVERSIZED_WIDTH);

    MaskFormat {
        frame_rate: None,
        width: OVERSIZED_WIDTH as u32,
        height: height as u32,
        sample: SampleFormat::U8,
        chroma: None,
    }
}

/// Writes indices across a plane too large for a single-axis launch.
///
/// It hands back how many elements the plane holds and the samples that came back.
struct WriteOversizedPlane;

impl BuildTask for WriteOversizedPlane {
    type Output = (usize, Vec<u32>);

    fn run(self, context: &BuildContext) -> Self::Output {
        let frames = context.frame_context();
        let units_per_cube = frames.cube_dim().num_elems() as usize;
        let limit = frames.capabilities().max_cube_count.0 as usize;
        let format = oversized_format(units_per_cube, limit);

        let mask = frames.alloc_mask(&format, FrameMeta::at(0));
        let plane = mask.plane();
        let count = elements(plane.buffer().bytes());
        let cubes = count.div_ceil(units_per_cube);

        assert!(
            cubes > limit,
            "this plane is {cubes} cubes of {units_per_cube} units, which fits the first axis \
             limit of {limit}, so the launch below never spreads and this test pins nothing",
        );

        let binding = plane.array::<u32>();
        launch(context, binding, count, 0);

        let samples = read_back(context.client(), plane.buffer().handle());

        (count, samples)
    }
}

/// Fills a whole ring, rewrites one slot's luma through that slot's own binding, then hands every
/// slot's luma back.
struct WriteMiddleSlot;

impl BuildTask for WriteMiddleSlot {
    type Output = (RingGeometry, Vec<Vec<u32>>);

    fn run(self, context: &BuildContext) -> Self::Output {
        let frames = context.frame_context();
        let ring = frames.ring(RingSpec {
            slots: 3,
            format: format(),
        });

        let total = elements(ring.geometry().total_bytes());
        let handle = ring.whole().clone();

        // Safety: the length is the ring's own byte count divided by the element size.
        let whole = unsafe { BufferArg::from_raw_parts(handle, total) };
        launch(context, whole, total, BASELINE);

        let middle = ring.slot_plane(1, Channel::Y).expect("the ring carries luma");
        let count = elements(middle.buffer().bytes());
        let binding = middle.array::<u32>();
        launch(context, binding, count, 0);

        let mut slots = Vec::with_capacity(ring.slots());
        for slot in 0..ring.slots() {
            let luma = ring.slot_plane(slot, Channel::Y).expect("the ring carries luma");

            slots.push(read_back(context.client(), luma.buffer().handle()));
        }

        (ring.geometry().clone(), slots)
    }
}

/// Writes a mask's full plane, adds a chroma plane and writes that, then hands both back.
struct WriteAddedChroma;

impl BuildTask for WriteAddedChroma {
    type Output = (Vec<u32>, Vec<u32>);

    fn run(self, context: &BuildContext) -> Self::Output {
        let frames = context.frame_context();
        let format = MaskFormat {
            frame_rate: None,
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: None,
        };
        let chroma = MaskChroma {
            subsampling: Subsampling::Yuv420,
            location: ChromaLocation::Left,
        };

        let meta = FrameMeta::at(0);

        let mask = frames.alloc_mask(&format, meta);
        let full_bytes = mask.plane().buffer().bytes();
        let full_count = elements(full_bytes);
        let full_binding = mask.plane().array::<u32>();
        launch(context, full_binding, full_count, BASELINE);

        let mask = frames.add_chroma(mask, chroma);
        let chroma_plane = mask.chroma_plane().expect("chroma was added");
        let chroma_bytes = chroma_plane.buffer().bytes();
        let chroma_count = elements(chroma_bytes);
        let chroma_binding = chroma_plane.array::<u32>();
        launch(context, chroma_binding, chroma_count, 0);

        let client = context.client();
        let full_handle = mask.plane().buffer().handle();
        let chroma_handle = chroma_plane.buffer().handle();
        let full = read_back(client, full_handle);
        let chroma = read_back(client, chroma_handle);

        (full, chroma)
    }
}

fn backend() -> Backend {
    Backend::open_default(test_device()).expect("a backend should open")
}

#[test]
fn a_kernel_writes_every_sample_a_plane_binding_names() {
    let samples = backend().build(WriteLuma);
    assert_eq!(samples.len(), elements(64 * 64));

    let expected: Vec<u32> = (0..samples.len() as u32).collect();
    assert_eq!(samples, expected, "the launch did not cover the whole plane");
}

/// The case the spread across cube count axes exists for. A launch piled onto one axis is refused
/// by the device rather than producing wrong samples, so this fails as a panic from the launch.
#[test]
fn a_kernel_covers_a_plane_too_large_for_a_single_axis_launch() {
    let (count, samples) = backend().build(WriteOversizedPlane);

    let wrong = samples
        .iter()
        .enumerate()
        .find(|(index, sample)| **sample != *index as u32);

    assert_eq!(samples.len(), count);
    assert_eq!(wrong, None, "the launch did not cover the whole plane");
}

/// The case the ring's offset arithmetic exists for. A binding covering the whole allocation would
/// overwrite its neighbours, and one carrying the wrong offset would land on the wrong slot.
#[test]
fn a_kernel_writing_one_ring_slot_leaves_the_others_alone() {
    let (geometry, slots) = backend().build(WriteMiddleSlot);

    let written: Vec<u32> = (0..slots[1].len() as u32).collect();
    assert_eq!(slots[1], written, "the slot's own samples were not written");

    for slot in [0, 2] {
        let offset = geometry.plane_offset(slot, 0).expect("the plane exists");
        let base = BASELINE + elements(offset) as u32;
        let untouched: Vec<u32> = (base..base + slots[slot].len() as u32).collect();

        assert_eq!(slots[slot], untouched, "slot {slot} was written through");
    }
}

/// A chroma plane added to a mask is memory of its own, so writing it leaves the full plane alone.
#[test]
fn a_chroma_plane_added_to_a_mask_binds_apart_from_its_full_plane() {
    let (full, chroma) = backend().build(WriteAddedChroma);

    let untouched: Vec<u32> = (BASELINE..BASELINE + full.len() as u32).collect();
    let written: Vec<u32> = (0..chroma.len() as u32).collect();

    assert_eq!(full.len(), elements(64 * 64));
    assert_eq!(chroma.len(), elements(32 * 32));
    assert_eq!(full, untouched, "the full plane was written through");
    assert_eq!(chroma, written, "the chroma plane's samples were not written");
}
