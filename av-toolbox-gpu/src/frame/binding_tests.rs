// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::frame::format::{ColourInfo, FrameFormat, MaskFormat};
use crate::frame::gpu::FrameMeta;
use crate::frame::layout::{Channel, FrameLayout, Subsampling};
use crate::frame::sample::SampleFormat;
use crate::memory::{RingGeometry, RingSpec};
use crate::{Backend, BuildContext, BuildTask, Device};

/// A base no index into these buffers can reach, so a sample above it was never written again.
const BASELINE: u32 = 1_000_000;

/// Writes `base` plus each element's own index into `output`.
///
/// A wrong offset and a wrong length fail differently. An offset error moves where the run of
/// indices lands, and a length error leaves a tail holding whatever was in it before.
#[cube(launch_unchecked)]
fn write_index(output: &mut Array<u32>, base: u32) {
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
    }
}

/// How many `u32` values `bytes` of a buffer hold.
const fn elements(bytes: usize) -> usize {
    bytes / size_of::<u32>()
}

fn read_back<R: Runtime>(client: &ComputeClient<R>, handle: &Handle) -> Vec<u32> {
    let bytes = client
        .read_one(handle.clone())
        .expect("the device should hand back what it was asked for");

    u32::from_bytes(&bytes).to_vec()
}

/// Runs the kernel over `output`, sized from the context so the launch covers every element.
fn launch<R: Runtime>(context: &BuildContext<R>, output: ArrayArg<R>, count: usize, base: u32) {
    let frames = context.frame_context();
    let cubes = frames.cube_count(count);
    let shape = frames.cube_dim();

    // Safety: `output` names no more elements than the buffer behind it holds, and the kernel
    // writes nothing past the length it is given.
    unsafe {
        write_index::launch_unchecked::<R>(context.client(), cubes, shape, output, base);
    }
}

/// Writes indices into a fresh frame's luma plane and hands its samples back.
struct WriteLuma;

impl BuildTask for WriteLuma {
    type Output = Vec<u32>;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        let frames = context.frame_context();
        let frame = frames.alloc_frame(&format(), FrameMeta::at(0));
        let luma = frame.plane(Channel::Y).expect("a yuv frame carries luma");

        let count = elements(luma.buffer().bytes());
        let binding = luma.array::<u32, R>();
        launch(context, binding, count, 0);

        read_back(context.client(), luma.buffer().handle())
    }
}

/// A plane with more cubes in it than a device takes on one axis of a launch.
///
/// At 8192 square it is 64 MiB of `u32` elements, which is 65536 cubes of 256 units on a device
/// whose cube is 16 by 16. A device with a wider cube covers the same plane in fewer, so
/// [WriteOversizedPlane] checks the count against what this device reports rather than assuming it.
fn oversized_format() -> MaskFormat {
    MaskFormat {
        width: 8192,
        height: 8192,
        sample: SampleFormat::U8,
    }
}

/// Writes indices across a plane too large for a single-axis launch and hands its samples back.
struct WriteOversizedPlane;

impl BuildTask for WriteOversizedPlane {
    type Output = Vec<u32>;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        let frames = context.frame_context();
        let mask = frames.alloc_mask(&oversized_format(), FrameMeta::at(0));
        let plane = mask.plane();

        let count = elements(plane.buffer().bytes());

        let units_per_cube = frames.cube_dim().num_elems() as usize;
        let cubes = count.div_ceil(units_per_cube);
        let limit = frames.capabilities().max_cube_count.0 as usize;

        assert!(
            cubes > limit,
            "this plane is {cubes} cubes of {units_per_cube} units, which fits the first axis \
             limit of {limit}, so the launch below never spreads and this test pins nothing. \
             Grow the plane until it does not fit rather than deleting the test.",
        );

        let binding = plane.array::<u32, R>();
        launch(context, binding, count, 0);

        read_back(context.client(), plane.buffer().handle())
    }
}

/// Fills a whole ring, rewrites one slot's luma through that slot's own binding, then hands every
/// slot's luma back.
struct WriteMiddleSlot;

impl BuildTask for WriteMiddleSlot {
    type Output = (RingGeometry, Vec<Vec<u32>>);

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        let frames = context.frame_context();
        let ring = frames.ring(RingSpec {
            slots: 3,
            format: format(),
        });

        let total = elements(ring.geometry().total_bytes());
        let handle = ring.whole().clone();

        // Safety: the length is the ring's own byte count divided by the element size.
        let whole = unsafe { ArrayArg::from_raw_parts(handle, total) };
        launch(context, whole, total, BASELINE);

        let middle = ring.slot_plane(1, Channel::Y).expect("the ring carries luma");
        let count = elements(middle.buffer().bytes());
        let binding = middle.array::<u32, R>();
        launch(context, binding, count, 0);

        let mut slots = Vec::with_capacity(ring.slots());
        for slot in 0..ring.slots() {
            let luma = ring.slot_plane(slot, Channel::Y).expect("the ring carries luma");

            slots.push(read_back(context.client(), luma.buffer().handle()));
        }

        (ring.geometry().clone(), slots)
    }
}

fn backend() -> Backend {
    Backend::open_default(Device::Default).expect("a backend should open")
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
    let samples = backend().build(WriteOversizedPlane);

    let wrong = samples
        .iter()
        .enumerate()
        .find(|(index, sample)| **sample != *index as u32);

    assert_eq!(samples.len(), elements(8192 * 8192));
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
