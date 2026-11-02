use std::sync::mpsc;
use std::thread;

use cubecl::prelude::*;

use crate::frame::tests::binding::{elements, launch};
use crate::probe::open_client;
use crate::testing::TEST_ACCELERATOR;
use crate::{Backend, BuildContext, FrameMeta, Mask, MaskFormat, SampleFormat, StreamPolicy, test_device};

/// How many hand-offs each policy is put through.
const HAND_OFFS: usize = 40;

/// The base the first thread writes, before the buffer goes back to the pool.
const FIRST_WRITE: u32 = 0;

/// The base the second thread writes, after taking the buffer back out of the pool.
const SECOND_WRITE: u32 = 1_000_000;

/// Copies `input` into `output`, element by element, as far as `output` reaches.
#[cube(launch_unchecked)]
fn copy_values(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
}

fn mask_format() -> MaskFormat {
    MaskFormat {
        frame_rate: None,
        width: 256,
        height: 64,
        sample: SampleFormat::U8,
        chroma: None,
    }
}

/// A smaller mask than [mask_format], so the pool never hands one out in place of the other.
fn copy_format() -> MaskFormat {
    MaskFormat {
        frame_rate: None,
        width: 128,
        height: 64,
        sample: SampleFormat::U8,
        chroma: None,
    }
}

/// Writes `base` plus each element's index across `mask`'s plane.
fn write_mask(context: &BuildContext, mask: &Mask, base: u32) {
    let plane = mask.plane();
    let bytes = plane.buffer().bytes();
    let count = elements(bytes);
    let binding = plane.array::<u32>();

    launch(context, binding, count, base);
}

fn read_mask(context: &BuildContext, mask: &Mask) -> Vec<u32> {
    let handle = mask.plane().buffer().handle().clone();
    let bytes = context
        .client()
        .read_one(handle)
        .expect("the device should hand back what it was asked for");

    u32::from_bytes(&bytes).to_vec()
}

/// Writes a mask, confirms the write landed, and releases it back to the pool.
fn write_and_release(context: &BuildContext) {
    let frames = context.frame_context();
    let format = mask_format();
    let meta = FrameMeta::at(0);
    let mask = frames.alloc_mask(&format, meta);
    write_mask(context, &mask, FIRST_WRITE);

    let written = read_mask(context, &mask);
    assert_eq!(written[1], FIRST_WRITE + 1);
}

/// Takes a mask back out of the pool, writes it again, and hands it on.
fn rewrite_recycled(context: &BuildContext) -> Mask {
    let frames = context.frame_context();
    let format = mask_format();
    let meta = FrameMeta::at(1);
    let mask = frames.alloc_mask(&format, meta);
    write_mask(context, &mask, SECOND_WRITE);

    frames.flush();

    mask
}

/// Writes a fresh mask and submits the write.
fn write_fresh(context: &BuildContext) -> Mask {
    let frames = context.frame_context();
    let format = mask_format();
    let meta = FrameMeta::at(0);
    let mask = frames.alloc_mask(&format, meta);
    write_mask(context, &mask, FIRST_WRITE);

    frames.flush();

    mask
}

/// Copies the start of `source` into a fresh mask of [copy_format].
fn copy_mask(context: &BuildContext, source: &Mask) -> Mask {
    let frames = context.frame_context();
    let format = copy_format();
    let meta = FrameMeta::at(0);
    let copy = frames.alloc_mask(&format, meta);

    let bytes = copy.plane().buffer().bytes();
    let count = elements(bytes);
    let cubes = frames.cube_count(count);
    let shape = frames.cube_dim();
    let input = source.plane().array::<u32>();
    let output = copy.plane().array::<u32>();

    // Safety: `output` is the smaller of the two buffers, and the kernel reads and writes nothing
    // past its length.
    unsafe {
        copy_values::launch_unchecked(context.client(), cubes, shape, input, output);
    }

    copy
}

/// Copies `shared`, lets go of it, and reads the copy back once `reused` says the buffer behind
/// `shared` was written again.
///
/// With `flushes` set, this thread's queued copy is flushed before `shared` is let go.
fn read_then_release(
    context: &BuildContext,
    shared: Mask,
    flushes: bool,
    released: mpsc::Sender<()>,
    reused: mpsc::Receiver<()>,
) -> Vec<u32> {
    let frames = context.frame_context();
    let copy = copy_mask(context, &shared);
    if flushes {
        frames.flush();
    }

    drop(shared);
    released.send(()).ok();

    reused
        .recv()
        .expect("the buffer should be written again before the copy is read");

    read_mask(context, &copy)
}

/// Fans one buffer out to two threads, lets both go of it, reuses it on a third, and answers
/// whether the first reader's copy still holds the first write.
///
/// The reader lets go of a clone that is not the last, so the pool only learns the buffer is free
/// when the holding thread drops its own clone.
fn reader_copies_the_first_write(context: &BuildContext, reader_flushes: bool) -> bool {
    thread::scope(|scope| {
        let written = scope
            .spawn(|| write_fresh(context))
            .join()
            .expect("the writer should not panic");
        let held = written.clone();

        let (released_sender, released_receiver) = mpsc::channel();
        let (reused_sender, reused_receiver) = mpsc::channel();
        let reader = scope.spawn(move || {
            read_then_release(context, written, reader_flushes, released_sender, reused_receiver)
        });

        released_receiver
            .recv()
            .expect("the reader should let go of its clone");

        scope
            .spawn(move || drop(held))
            .join()
            .expect("the holder should not panic");

        scope
            .spawn(|| rewrite_recycled(context))
            .join()
            .expect("the second writer should not panic");
        reused_sender.send(()).ok();

        let values = reader.join().expect("the reader should not panic");

        values
            .iter()
            .enumerate()
            .all(|(index, value)| *value == FIRST_WRITE + index as u32)
    })
}

/// Passes one buffer through three threads and answers whether the last saw the second write.
///
/// The first thread writes and releases the buffer, the second takes it from the pool and writes
/// it again, and the third reads it.
fn reader_sees_the_second_write(context: &BuildContext) -> bool {
    thread::scope(|scope| {
        scope
            .spawn(|| write_and_release(context))
            .join()
            .expect("the first writer should not panic");

        let (mask_sender, mask_receiver) = mpsc::channel();
        scope.spawn(move || {
            let mask = rewrite_recycled(context);
            mask_sender.send(mask).ok();
        });

        let reader = scope.spawn(move || {
            let mask = mask_receiver
                .recv()
                .expect("the second writer should hand its mask on");

            read_mask(context, &mask)
        });

        let values = reader.join().expect("the reader should not panic");

        values
            .iter()
            .enumerate()
            .all(|(index, value)| *value == SECOND_WRITE + index as u32)
    })
}

/// How the context under test keeps work across threads in order.
#[derive(Copy, Clone)]
enum WorkOrder {
    Unordered,
    Policy(StreamPolicy),
}

/// Which cross-thread hazard each hand-off is put through.
#[derive(Copy, Clone)]
enum Hazard {
    /// A buffer written again after coming back through the pool, then read on another thread.
    WriteThenRead,
    /// A buffer read on one thread and let go there while another thread still holds it, then
    /// reused. `reader_flushes` says whether the reader submits its read before letting go.
    ReadThenReuse { reader_flushes: bool },
}

fn reads_fresh_data(context: &BuildContext, hazard: Hazard) -> bool {
    match hazard {
        Hazard::WriteThenRead => reader_sees_the_second_write(context),
        Hazard::ReadThenReuse { reader_flushes } => reader_copies_the_first_write(context, reader_flushes),
    }
}

/// Counts how many of [HAND_OFFS] hand-offs through `hazard` read stale data under `ordering`.
fn stale_reads(ordering: WorkOrder, hazard: Hazard) -> usize {
    let selector = test_device();
    let converted = selector.to_cubecl(TEST_ACCELERATOR);
    let cubecl_device = converted.expect("the test device names a device on the test backend");
    let opened = open_client(TEST_ACCELERATOR, selector, &cubecl_device);
    let client = opened.expect("the backend should open");

    let context = match ordering {
        WorkOrder::Unordered => BuildContext::unordered(TEST_ACCELERATOR, selector, cubecl_device, client),
        WorkOrder::Policy(policy) => {
            BuildContext::with_policy(TEST_ACCELERATOR, selector, cubecl_device, client, policy)
        },
    };

    (0..HAND_OFFS)
        .filter(|_| !reads_fresh_data(&context, hazard))
        .count()
}

#[test]
fn without_a_policy_a_handed_off_buffer_is_read_stale() {
    let stale = stale_reads(WorkOrder::Unordered, Hazard::WriteThenRead);

    assert!(
        stale > 0,
        "every one of {HAND_OFFS} hand-offs read the second write"
    );
}

#[test]
fn a_shared_stream_reads_a_handed_off_buffer_after_its_write() {
    let ordering = WorkOrder::Policy(StreamPolicy::SharedStream);
    let stale = stale_reads(ordering, Hazard::WriteThenRead);

    assert_eq!(stale, 0, "{stale} of {HAND_OFFS} hand-offs read stale data");
}

#[test]
fn flushing_at_hand_off_reads_a_handed_off_buffer_after_its_write() {
    let ordering = WorkOrder::Policy(StreamPolicy::FlushAtHandOff);
    let stale = stale_reads(ordering, Hazard::WriteThenRead);

    assert_eq!(stale, 0, "{stale} of {HAND_OFFS} hand-offs read stale data");
}

#[test]
fn without_a_step_flush_a_fanned_out_buffer_is_reused_before_its_read() {
    let ordering = WorkOrder::Policy(StreamPolicy::FlushAtHandOff);
    let hazard = Hazard::ReadThenReuse {
        reader_flushes: false,
    };

    let stale = stale_reads(ordering, hazard);

    assert!(stale > 0, "every one of {HAND_OFFS} copies held the first write");
}

#[test]
fn flushing_at_hand_off_reads_a_fanned_out_buffer_before_it_is_reused() {
    let ordering = WorkOrder::Policy(StreamPolicy::FlushAtHandOff);
    let hazard = Hazard::ReadThenReuse { reader_flushes: true };
    let stale = stale_reads(ordering, hazard);

    assert_eq!(stale, 0, "{stale} of {HAND_OFFS} copies read reused data");
}

#[test]
fn a_shared_stream_reads_a_fanned_out_buffer_before_it_is_reused() {
    let ordering = WorkOrder::Policy(StreamPolicy::SharedStream);
    let hazard = Hazard::ReadThenReuse {
        reader_flushes: false,
    };

    let stale = stale_reads(ordering, hazard);

    assert_eq!(stale, 0, "{stale} of {HAND_OFFS} copies read reused data");
}

#[test]
fn a_backend_keeps_its_accelerators_policy() {
    let backend = Backend::open(TEST_ACCELERATOR, test_device()).expect("should open");
    let expected = StreamPolicy::for_accelerator(TEST_ACCELERATOR);

    assert_eq!(backend.stream_policy(), expected);
    assert_eq!(backend.frame_context().stream_policy(), expected);
}
