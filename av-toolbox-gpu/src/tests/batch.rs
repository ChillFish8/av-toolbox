use std::time::{Duration, Instant};

use cubecl::prelude::Runtime;

use crate::frame::tests::binding::{elements, launch};
use crate::{
    Attachments,
    Backend,
    BuildContext,
    BuildTask,
    Channel,
    ChromaLocation,
    ColourInfo,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostItem,
    HostPlanes,
    HostStreamItem,
    LoadItem,
    MaskChroma,
    MaskFormat,
    PendingBatch,
    SampleFormat,
    SceneScore,
    StreamFormat,
    StreamItem,
    Subsampling,
    TryWait,
    test_device,
};

fn backend() -> Backend {
    Backend::open_default(test_device()).expect("a backend should open")
}

fn frame_format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

fn frames_format() -> StreamFormat {
    StreamFormat::Frames(frame_format())
}

fn mask_format(width: u32, height: u32) -> StreamFormat {
    StreamFormat::Mask(MaskFormat {
        width,
        height,
        sample: SampleFormat::U8,
        chroma: None,
    })
}

/// Fills `plane` so every byte depends on both `seed` and where it sits.
///
/// The run repeats every 251 bytes, and no plane size or row length here is a multiple of that, so
/// a plane that came back shifted by whole rows still holds the wrong bytes.
fn fill_pattern(plane: &mut [u8], seed: u8) {
    for (index, byte) in plane.iter_mut().enumerate() {
        *byte = seed.wrapping_add((index % 251) as u8);
    }
}

/// The channels that name each of `format`'s planes once.
///
/// A mask's chroma plane answers to both U and V, so a mask lists only U for it.
fn distinct_channels(format: &StreamFormat) -> &'static [Channel] {
    match format {
        StreamFormat::Frames(_) => &[Channel::Y, Channel::U, Channel::V],
        StreamFormat::Mask(_) => &[Channel::Y, Channel::U],
    }
}

/// Planes for `format`, each one filled with a pattern of its own.
///
/// U and V are the same size at 4:2:0, so their patterns are what tells a plane that came back
/// from its neighbour.
fn filled(format: &StreamFormat, seed: u8) -> HostPlanes {
    let mut planes = HostPlanes::zeroed(format);
    let channels = distinct_channels(format);

    for (offset, channel) in channels.iter().enumerate() {
        let Some(plane) = planes.plane_mut(*channel) else {
            continue;
        };

        fill_pattern(plane, seed.wrapping_add(offset as u8 * 37));
    }

    planes
}

fn assert_plane_matches(actual: &[u8], expected: &[u8], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} came back the wrong length");

    let wrong = actual.iter().zip(expected).position(|(got, want)| got != want);
    assert_eq!(wrong, None, "{what} came back holding other bytes at that index");
}

/// Checks every plane of `actual` against the planes that were uploaded.
fn assert_planes_match(actual: &HostItem, uploaded: &HostPlanes, what: &str) {
    assert_eq!(actual.len(), uploaded.len(), "{what} came back with other planes");

    for channel in [Channel::Y, Channel::U, Channel::V] {
        let Some(expected) = uploaded.plane(channel) else {
            continue;
        };
        let plane = actual.plane(channel).expect("both sets carry the same channels");

        assert_plane_matches(plane, expected, &format!("the {channel} plane of {what}"));
    }
}

/// Starts reading back one item of `format`, uploaded from `seed`.
fn start_read(frames: &FrameContext, format: &StreamFormat, seed: u8) -> PendingBatch {
    let planes = filled(format, seed);
    let item = LoadItem::new(&planes, FrameMeta::at(0));
    let loaded = frames.load_one(&item).expect("the planes match their format");

    frames.unload_one(loaded)
}

#[test]
fn a_frame_comes_back_from_the_device_byte_for_byte() {
    let frames = backend().frame_context();
    let format = frames_format();

    let items = start_read(&frames, &format, 1)
        .wait()
        .expect("the device should hand back what it was given");

    assert_eq!(items.len(), 1);
    assert_planes_match(&items[0], &filled(&format, 1), "the frame");
}

/// A frame and a mask carry different plane counts, which is what catches a map that assumes every
/// item has the same. The mask is sized to match the frame's chroma, so three of the four planes
/// read back are the same size and only their contents tell them apart.
#[test]
fn a_frame_and_a_mask_in_one_batch_each_come_back_whole() {
    let frames = backend().frame_context();
    let frame_format = frames_format();
    let mask = mask_format(32, 32);

    let frame_planes = filled(&frame_format, 1);
    let mask_planes = filled(&mask, 200);
    let items = vec![
        LoadItem::new(&frame_planes, FrameMeta::at(0)),
        LoadItem::new(&mask_planes, FrameMeta::at(1)),
    ];
    let loaded = frames.load(&items).expect("the planes match their formats");
    let returned = frames
        .unload(loaded)
        .wait()
        .expect("the device should hand back what it was given");

    assert_eq!(returned.len(), 2);
    assert_eq!(returned[0].len(), 3, "a yuv frame has three planes");
    assert_eq!(returned[1].len(), 1, "a mask has one");
    assert_planes_match(&returned[0], &filled(&frame_format, 1), "the frame");
    assert_planes_match(&returned[1], &filled(&mask, 200), "the mask");
}

#[test]
fn a_host_frame_keeps_its_bytes_and_place_through_an_upload_and_unload() {
    let frames = backend().frame_context();
    let format = frame_format();
    let meta = FrameMeta {
        index: 12,
        chunk: 3,
        timestamp: Some(Duration::from_millis(480)),
        scene_score: Some(SceneScore::new(2.5)),
        attachments: Attachments::new(),
    };

    let mut host_frame = frames.alloc_host_frame(&format, meta.clone());
    for (offset, channel) in [Channel::Y, Channel::U, Channel::V].iter().enumerate() {
        let plane = host_frame
            .plane_mut(*channel)
            .expect("a yuv frame carries every channel");
        fill_pattern(plane, 1 + offset as u8 * 37);
    }

    let frozen = host_frame.freeze();
    let uploaded = HostStreamItem::from(frozen);

    let loaded = frames.load_host(&uploaded).expect("the frame should upload");
    let landed = frames.unload_host(loaded).expect("the frame should come back");

    assert_eq!(landed.meta(), &meta);

    let original = uploaded.into_frame().expect("a frame went up");
    let returned = landed.into_frame().expect("a frame came back");
    for channel in [Channel::Y, Channel::U, Channel::V] {
        let expected = original
            .plane(channel)
            .expect("a yuv frame carries every channel");
        let actual = returned
            .plane(channel)
            .expect("a yuv frame carries every channel");
        let what = format!("the {channel} plane");

        assert_plane_matches(actual.bytes(), expected.bytes(), &what);
    }
}

fn chroma_mask_format() -> MaskFormat {
    MaskFormat {
        width: 37,
        height: 23,
        sample: SampleFormat::U8,
        chroma: Some(MaskChroma {
            subsampling: Subsampling::Yuv420,
            location: ChromaLocation::Left,
        }),
    }
}

/// The two masks have the same shape, so only their contents tell one item's planes from the
/// other's.
#[test]
fn a_batch_of_two_chroma_masks_comes_back_with_both_planes_of_each() {
    let frames = backend().frame_context();
    let mask_format = chroma_mask_format();
    let format = StreamFormat::Mask(mask_format);

    let first = filled(&format, 3);
    let second = filled(&format, 150);
    let first_meta = FrameMeta::at(0);
    let second_meta = FrameMeta::at(1);
    let items = vec![
        LoadItem::new(&first, first_meta),
        LoadItem::new(&second, second_meta),
    ];
    let loaded = frames.load(&items).expect("the planes match their formats");
    let returned = frames
        .unload(loaded)
        .wait()
        .expect("the device should hand back what it was given");

    assert_eq!(returned.len(), 2);
    assert_eq!(returned[0].len(), 2, "a mask with chroma has two planes");
    assert_eq!(returned[1].len(), 2, "a mask with chroma has two planes");
    assert_planes_match(&returned[0], &first, "the first mask");
    assert_planes_match(&returned[1], &second, "the second mask");
}

#[test]
fn a_host_mask_with_chroma_keeps_both_planes_through_an_upload_and_unload() {
    let frames = backend().frame_context();
    let format = chroma_mask_format();
    let meta = FrameMeta::at(4);

    let mut host_mask = frames.alloc_host_mask(&format, meta);
    let full = host_mask.plane_mut();
    fill_pattern(full, 9);
    let chroma = host_mask.chroma_plane_mut().expect("the format carries chroma");
    fill_pattern(chroma, 120);

    let frozen = host_mask.freeze();
    let uploaded = HostStreamItem::from(frozen);

    let loaded = frames.load_host(&uploaded).expect("the mask should upload");
    let landed = frames.unload_host(loaded).expect("the mask should come back");

    assert_eq!(landed.meta().index, 4);

    let original = uploaded.into_mask().expect("a mask went up");
    let returned = landed.into_mask().expect("a mask came back");
    let original_full = original.plane().bytes();
    let returned_full = returned.plane().bytes();
    let original_chroma = original.chroma_plane().expect("the format carries chroma");
    let returned_chroma = returned.chroma_plane().expect("the format carries chroma");
    let original_chroma_bytes = original_chroma.bytes();
    let returned_chroma_bytes = returned_chroma.bytes();

    assert_eq!(returned.format(), &format);
    assert_plane_matches(returned_full, original_full, "the full plane");
    assert_plane_matches(returned_chroma_bytes, original_chroma_bytes, "the chroma plane");
}

/// Loads a frame, writes indices across its luma plane, and unloads what the device then holds.
struct WriteLumaOfALoadedFrame {
    planes: HostPlanes,
}

impl BuildTask for WriteLumaOfALoadedFrame {
    type Output = Vec<HostItem>;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        let frames = context.frame_context();
        let item = LoadItem::new(&self.planes, FrameMeta::at(0));
        let loaded = frames.load_one(&item).expect("the planes match their format");

        {
            let frame = loaded.as_frame().expect("a frame went in");
            let luma = frame.plane(Channel::Y).expect("a yuv frame carries luma");

            let count = elements(luma.buffer().bytes());
            let binding = luma.array::<u32, R>();
            launch(context, binding, count, 0);
        }

        frames
            .unload_one(loaded)
            .wait()
            .expect("the device should hand back what the kernel wrote")
    }
}

/// The whole path this phase exists for: samples up, a kernel over them, samples back.
#[test]
fn a_plane_a_kernel_wrote_comes_back_written_and_the_rest_come_back_as_uploaded() {
    let format = frames_format();
    let uploaded = filled(&format, 1);

    let items = backend().build(WriteLumaOfALoadedFrame {
        planes: uploaded.clone(),
    });

    let luma = items[0].plane(Channel::Y).expect("a yuv frame carries luma");
    let (samples, remainder) = luma.as_chunks::<{ size_of::<u32>() }>();
    let written: Vec<u32> = samples.iter().copied().map(u32::from_ne_bytes).collect();
    let expected: Vec<u32> = (0..written.len() as u32).collect();

    assert!(remainder.is_empty(), "the luma plane is whole u32 samples");
    assert_eq!(written.len(), elements(64 * 64));
    assert_eq!(written, expected, "the kernel's samples did not come back");

    for channel in [Channel::U, Channel::V] {
        let plane = items[0].plane(channel).expect("a yuv frame carries chroma");
        let uploaded = uploaded.plane(channel).expect("a yuv frame carries chroma");

        assert_plane_matches(plane, uploaded, &format!("the {channel} plane"));
    }
}

/// The address of every plane of every item, in order.
fn addresses(items: &[HostItem]) -> Vec<*const u8> {
    let mut found = Vec::new();
    for planes in items {
        for channel in [Channel::Y, Channel::U, Channel::V] {
            if let Some(plane) = planes.plane(channel) {
                found.push(plane.as_ptr());
            }
        }
    }

    found
}

/// The seed one item of a batch is filled from, so no two items carry the same bytes.
fn seed_of(seed: u8, item: usize) -> u8 {
    seed.wrapping_add(item as u8 * 11)
}

/// Reads one item of each format back into `items`, uploaded from `seed`.
fn round_trip_into(frames: &FrameContext, formats: &[StreamFormat], seed: u8, items: &mut Vec<HostItem>) {
    let uploaded: Vec<HostPlanes> = formats
        .iter()
        .enumerate()
        .map(|(index, format)| filled(format, seed_of(seed, index)))
        .collect();
    let to_load: Vec<LoadItem<'_>> = uploaded
        .iter()
        .enumerate()
        .map(|(index, planes)| LoadItem::new(planes, FrameMeta::at(index as u64)))
        .collect();

    let loaded = frames.load(&to_load).expect("the planes match their formats");
    frames
        .unload(loaded)
        .wait_into(items)
        .expect("the device should hand back what it was given");
}

/// Running frame after frame allocates nothing per frame, which only holds if a real readback
/// copies into the planes it was handed rather than replacing them with buffers of its own.
///
/// Planes that were freed and asked for again often come back at the addresses they had, so what
/// tells a reused set from a rebuilt one is that no plane here is alone in its size. Both luma
/// planes are one size and the five chroma and mask planes another, so an allocator handing the
/// same blocks back in any other order moves them.
#[test]
fn waiting_into_the_same_planes_twice_keeps_their_allocations() {
    let frames = backend().frame_context();
    let formats = [frames_format(), frames_format(), mask_format(32, 32)];

    let mut items = Vec::new();
    round_trip_into(&frames, &formats, 1, &mut items);
    let before = addresses(&items);

    round_trip_into(&frames, &formats, 90, &mut items);
    let after = addresses(&items);

    assert_eq!(items.len(), 3);
    assert_eq!(before.len(), 7, "two yuv frames and a mask are seven planes");
    assert_eq!(before, after, "a plane's allocation was replaced");

    // The second read has to have landed in those same allocations, or matching addresses would
    // only mean nothing was written at all.
    for (index, (planes, format)) in items.iter().zip(&formats).enumerate() {
        let uploaded = filled(format, seed_of(90, index));

        assert_planes_match(planes, &uploaded, &format!("item {index} of the second batch"));
    }
}

/// A read large enough that no device lands it on the first poll, so the batch comes back polled
/// and unlanded, which is the state the drop rule is about.
fn unlanded_format() -> StreamFormat {
    mask_format(4096, 4096)
}

/// How many reads [polled_read] starts before deciding this device lands every one on its first
/// poll.
const POLL_ATTEMPTS: usize = 10;

/// Starts a read and polls it once, handing it back still in flight.
///
/// A read can land on its first poll now and then when the device is idle and the thread is slow to
/// check, so another is started until one does not.
fn polled_read(frames: &FrameContext, format: &StreamFormat, seed: u8) -> PendingBatch {
    for _ in 0..POLL_ATTEMPTS {
        let batch = start_read(frames, format, seed);
        let polled = batch.try_wait().expect("a pending read is not a failure");

        if let TryWait::NotReady(batch) = polled {
            return batch;
        }
    }

    panic!(
        "this device landed {POLL_ATTEMPTS} 16 MiB reads on their first poll, so nothing here is \
         ever polled and unlanded and these tests pin nothing. Grow the read until it does not \
         land rather than deleting them."
    );
}

/// The regression the drop rule exists for.
///
/// The first poll reserves a staging buffer, records the copy into it and maps it, so a batch
/// abandoned there hands a still-mapped buffer back to the staging pool. The reads after it ask
/// for that same size, which is what takes the abandoned buffer out again.
#[test]
fn abandoning_a_polled_batch_leaves_the_device_usable() {
    let frames = backend().frame_context();
    let format = unlanded_format();

    let batch = polled_read(&frames, &format, 0);
    drop(batch);

    for round in 1..=3u8 {
        let items = start_read(&frames, &format, round)
            .wait()
            .expect("the device should still answer after a batch was abandoned");

        assert_planes_match(&items[0], &filled(&format, round), &format!("read {round}"));
    }
}

/// The invariant the drop rule's early return rests on.
///
/// An unpolled read is abandoned rather than waited out, which is only safe while starting one has
/// reserved nothing on the device. CubeCL reserves and maps as soon as a read is started, so it is
/// the deferral in `ClientAllocator::read` that makes that true, and an eager read would leave a
/// mapped staging buffer behind here for the reads below to fail on.
#[test]
fn starting_a_read_reserves_nothing_until_it_is_polled() {
    let frames = backend().frame_context();
    let format = unlanded_format();

    drop(start_read(&frames, &format, 0));

    for round in 1..=3u8 {
        let items = start_read(&frames, &format, round)
            .wait()
            .expect("the device should still answer after an unpolled batch was dropped");

        assert_planes_match(&items[0], &filled(&format, round), &format!("read {round}"));
    }
}

/// How long dropping `batch` takes.
fn time_of_drop(batch: PendingBatch) -> Duration {
    let started = Instant::now();
    drop(batch);

    started.elapsed()
}

/// How many reads the landing time is the fastest of.
///
/// Other tests share the device, so one read can land slowly while it is busy. The fastest of
/// several is what a read costs on its own, which load cannot inflate.
const LANDING_SAMPLES: usize = 5;

/// The fastest a read that has already been polled once lands, out of [LANDING_SAMPLES] reads.
fn fastest_landing(frames: &FrameContext, format: &StreamFormat) -> Duration {
    (0..LANDING_SAMPLES)
        .map(|_| time_to_land_after_a_poll(frames, format))
        .min()
        .expect("at least one read is timed")
}

/// How long a read that has already been polled once takes to land.
///
/// That is what dropping a polled batch has left to wait for, so it is what such a drop is
/// measured against.
fn time_to_land_after_a_poll(frames: &FrameContext, format: &StreamFormat) -> Duration {
    let batch = polled_read(frames, format, 0);

    let started = Instant::now();
    batch.wait().expect("the read should land");

    started.elapsed()
}

/// How many drops of each kind are timed.
const DROP_SAMPLES: usize = 5;

/// The slowest drop of a polled read, out of [DROP_SAMPLES] reads.
///
/// A polled read can land before it is dropped when the thread is slow to get there, which makes
/// that one drop free. The slowest drop is the one that really waited.
fn slowest_polled_drop(frames: &FrameContext, format: &StreamFormat) -> Duration {
    (0..DROP_SAMPLES)
        .map(|_| {
            let batch = polled_read(frames, format, 0);
            time_of_drop(batch)
        })
        .max()
        .expect("at least one drop is timed")
}

/// The fastest drop of an unpolled read, out of [DROP_SAMPLES] reads.
///
/// The thread can be preempted mid-drop on a busy machine, which only ever adds time.
fn fastest_unpolled_drop(frames: &FrameContext, format: &StreamFormat) -> Duration {
    (0..DROP_SAMPLES)
        .map(|_| {
            let batch = start_read(frames, format, 0);
            time_of_drop(batch)
        })
        .min()
        .expect("at least one drop is timed")
}

/// The drop rule in both directions, measured against what waiting for this read actually costs.
///
/// A drop that waits costs about what a device wait costs, and one that does not costs orders less,
/// so the read's own landing time is what tells them apart on any device.
#[test]
fn dropping_a_polled_batch_waits_for_it_and_dropping_an_unpolled_one_does_not() {
    let frames = backend().frame_context();
    let format = unlanded_format();

    // The first read of a size grows the staging pool, which is not what is being timed here.
    time_to_land_after_a_poll(&frames, &format);
    let landing = fastest_landing(&frames, &format);

    let polled = slowest_polled_drop(&frames, &format);
    let unpolled = fastest_unpolled_drop(&frames, &format);

    assert!(
        polled * 10 > landing,
        "this read takes {landing:?} to land and dropping a polled one took {polled:?}, \
         which is far too little to have waited for it"
    );
    assert!(
        unpolled * 10 < landing,
        "this read takes {landing:?} to land and dropping an unpolled one took {unpolled:?}, \
         so it waited on a read that had asked the device for nothing"
    );
}

/// The base one frame's luma is stamped with, spaced so no two frames' samples can overlap.
fn stamp(index: u32) -> u32 {
    (index + 1) * 1_000_000
}

/// The `u32` samples of one plane that came back.
fn samples_of(plane: &[u8]) -> Vec<u32> {
    let (whole, remainder) = plane.as_chunks::<{ size_of::<u32>() }>();
    assert!(remainder.is_empty(), "the plane is whole u32 samples");

    whole.iter().copied().map(u32::from_ne_bytes).collect()
}

/// Stamps a run of frames, keeping each one's readback in flight while the next is written.
///
/// This is how a sink uses the API. The batch started for one frame is only waited on after the
/// next frame has been allocated and written, so a frame whose memory is released by the call that
/// starts its readback is handed straight to the frame after it.
struct PipelinedStamps {
    frames: u32,
}

impl BuildTask for PipelinedStamps {
    type Output = Vec<Vec<HostItem>>;

    fn run<R: Runtime>(self, context: &BuildContext<R>) -> Self::Output {
        let frames = context.frame_context();
        let format = frame_format();

        let mut landed = Vec::new();
        let mut in_flight: Option<PendingBatch> = None;

        for index in 0..self.frames {
            let frame = frames.alloc_frame(&format, FrameMeta::at(u64::from(index)));

            {
                let luma = frame.plane(Channel::Y).expect("a yuv frame carries luma");
                let count = elements(luma.buffer().bytes());
                let binding = luma.array::<u32, R>();

                launch(context, binding, count, stamp(index));
            }

            let batch = frames.unload_one(StreamItem::Frame(frame));

            if let Some(previous) = in_flight.replace(batch) {
                landed.push(previous.wait().expect("the device should hand the frame back"));
            }
        }

        if let Some(last) = in_flight {
            landed.push(last.wait().expect("the device should hand the frame back"));
        }

        landed
    }
}

/// A readback in flight has to keep the memory it is going to read.
///
/// The frames here are allocated from the pool, so a frame released while its own readback is
/// still pending is the next frame's memory, and that frame's kernel overwrites it before the
/// read has recorded anything.
#[test]
fn a_batch_in_flight_keeps_its_frame_while_the_next_frame_is_written() {
    let count = 4;

    let landed = backend().build(PipelinedStamps { frames: count });

    assert_eq!(landed.len(), count as usize);

    for (index, items) in landed.iter().enumerate() {
        let luma = items[0].plane(Channel::Y).expect("a yuv frame carries luma");
        let written = samples_of(luma);
        let base = stamp(index as u32);

        assert_eq!(
            written.first().copied(),
            Some(base),
            "frame {index} came back holding another frame's samples"
        );

        let wrong = written
            .iter()
            .enumerate()
            .position(|(at, sample)| *sample != base + at as u32);
        assert_eq!(wrong, None, "frame {index} came back holding other samples there");
    }
}
