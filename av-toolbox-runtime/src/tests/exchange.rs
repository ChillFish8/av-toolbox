use std::collections::VecDeque;
use std::time::Duration;

use av_toolbox_gpu::{
    AnyItem,
    ColourInfo,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    SampleFormat,
    StreamItem,
    Subsampling,
    testing_context,
};

use crate::{Aggregate, Dispatch, Disperse, RoundRobin, Routed, WorkItem, WorkResult};

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
    }
}

/// Builds one item, timestamped by `position`, the order it was built in.
///
/// The aggregate renumbers `index` as it emits, so `index` cannot identify a frame once it comes
/// back out. `timestamp` is never touched, so it is what a test can use to tell frames apart.
fn item(index: u64, chunk: u64, position: u64) -> AnyItem {
    let context = testing_context();
    let mut meta = FrameMeta::at(index).in_chunk(chunk);
    meta.timestamp = Some(Duration::from_secs(position));
    StreamItem::Frame(context.alloc_frame(&format(), meta)).into()
}

fn stream(pairs: &[(u64, u64)]) -> Vec<AnyItem> {
    pairs
        .iter()
        .enumerate()
        .map(|(position, &(index, chunk))| item(index, chunk, position as u64))
        .collect()
}

fn indices(output: &[AnyItem]) -> Vec<u64> {
    output.iter().map(|item| item.meta().index).collect()
}

fn chunks(output: &[AnyItem]) -> Vec<u64> {
    output.iter().map(|item| item.meta().chunk).collect()
}

/// The build-order timestamp of every item, in the order they came out.
///
/// Unlike `index`, this survives the aggregate untouched, so it is a true identity check on which
/// physical frame came out where, not just on how many frames came out.
fn timestamps(output: &[AnyItem]) -> Vec<u64> {
    output
        .iter()
        .map(|item| {
            item.meta()
                .timestamp
                .expect("every test item carries a timestamp")
                .as_secs()
        })
        .collect()
}

fn identity(item: AnyItem) -> Vec<AnyItem> {
    vec![item]
}

/// `AnyItem` derives `Clone`, and so do both sides it wraps, so duplicating one is a plain clone
/// rather than something that needs rebuilding from its index and chunk.
fn copy(item: &AnyItem) -> AnyItem {
    item.clone()
}

/// Hands out workers in a fixed, pre-chosen order, unrelated to chunk id or ascending worker id.
struct Scripted {
    order: VecDeque<usize>,
    workers: usize,
}

impl Scripted {
    fn new(order: &[usize], workers: usize) -> Self {
        Self {
            order: order.iter().copied().collect(),
            workers,
        }
    }
}

impl Dispatch for Scripted {
    fn choose(&mut self, _chunk: u64) -> Option<usize> {
        let worker = self
            .order
            .pop_front()
            .expect("the script names a worker for every chunk dispatched");
        Some(worker)
    }

    fn workers(&self) -> usize {
        self.workers
    }
}

fn run<F>(input: Vec<AnyItem>, workers: usize, transform: F) -> Vec<AnyItem>
where
    F: Fn(AnyItem) -> Vec<AnyItem>,
{
    drive(input, RoundRobin::new(workers), transform)
}

/// Runs one stream through a disperse, a worker played by `transform`, and an aggregate.
///
/// Every step is sequential. The disperse routes every input item into one queue per worker,
/// each worker's queue is turned into a queue of results, and the aggregate drains those result
/// queues in the order its assignments name.
fn drive<D, F>(input: Vec<AnyItem>, dispatch: D, transform: F) -> Vec<AnyItem>
where
    D: Dispatch,
    F: Fn(AnyItem) -> Vec<AnyItem>,
{
    let workers = dispatch.workers();
    let mut disperse = Disperse::new(&[true], 0, dispatch);
    let mut work_queues = new_queues(workers);

    for stream_item in input {
        let routed = disperse
            .offer(0, stream_item)
            .expect("a single-slot offer always routes");
        enqueue(&mut work_queues, routed);
        place_held_bundle(&mut disperse, &mut work_queues);
    }

    let finished = disperse.finish();
    enqueue(&mut work_queues, finished.routed);
    assert!(
        finished.dropped.is_none(),
        "every test stream feeds its one slot all the way to the end"
    );

    let mut result_queues = new_queues(workers);
    for worker in 0..workers {
        let queue = std::mem::take(&mut work_queues[worker]);
        result_queues[worker] = play_worker(queue, &transform);
    }

    let mut aggregate = Aggregate::new();
    for assignment in disperse.assignments() {
        aggregate.assign(*assignment);
    }
    aggregate.finish();

    drain_aggregate(aggregate, result_queues)
}

/// Places a held bundle, if one is waiting, and enqueues the worker it lands on.
///
/// Every dispatch used in these tests always has a free worker, so a boundary resolves on its
/// first try.
fn place_held_bundle<D: Dispatch>(disperse: &mut Disperse<D>, work_queues: &mut [VecDeque<WorkItem>]) {
    if !disperse.is_holding() {
        return;
    }

    let placed = disperse
        .place()
        .expect("every dispatch in these tests always has a free worker");
    enqueue(work_queues, vec![placed.routed]);
}

fn new_queues<T>(workers: usize) -> Vec<VecDeque<T>> {
    (0..workers).map(|_| VecDeque::new()).collect()
}

fn enqueue(queues: &mut [VecDeque<WorkItem>], routed: Vec<Routed>) {
    for entry in routed {
        queues[entry.worker].push_back(entry.item);
    }
}

/// Turns one worker's queue of `WorkItem`s into a queue of `WorkResult`s.
///
/// A bundle becomes whatever `transform` produces for it. A unit does the same, then closes
/// itself immediately, since frame routing never sends a separate close. `WorkItem::EndOfChunk`
/// becomes `WorkResult::EndOfUnit`, and `EndOfStream` is carried straight through, since a worker
/// never changes chunk or stream boundaries.
fn play_worker<F>(queue: VecDeque<WorkItem>, transform: &F) -> VecDeque<WorkResult>
where
    F: Fn(AnyItem) -> Vec<AnyItem>,
{
    let mut results = VecDeque::new();

    for work_item in queue {
        match work_item {
            WorkItem::Bundle(mut bundle) => {
                let bundled_item = bundle.remove(0).expect("the only slot is always filled");

                for produced in transform(bundled_item) {
                    results.push_back(WorkResult::Item(produced));
                }
            },
            WorkItem::Unit(unit, mut bundle) => {
                let bundled_item = bundle.remove(0).expect("the only slot is always filled");

                for produced in transform(bundled_item) {
                    results.push_back(WorkResult::Item(produced));
                }

                results.push_back(WorkResult::EndOfUnit(unit));
            },
            WorkItem::EndOfChunk(chunk) => results.push_back(WorkResult::EndOfUnit(chunk)),
            WorkItem::EndOfStream => results.push_back(WorkResult::EndOfStream),
        }
    }

    results
}

fn drain_aggregate(mut aggregate: Aggregate, mut result_queues: Vec<VecDeque<WorkResult>>) -> Vec<AnyItem> {
    let mut output = Vec::new();

    while let Some(worker) = aggregate.next_worker() {
        let result = result_queues[worker]
            .pop_front()
            .expect("the worker holding the current chunk has its next message queued");
        let emitted = aggregate
            .accept(worker, result)
            .expect("the driver only ever feeds the worker the aggregate asked for");
        output.extend(emitted);
    }

    output
}

#[test]
fn a_stream_comes_back_in_the_order_it_went_in() {
    let input = stream(&[(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]);

    let output = run(input, 3, identity);

    assert_eq!(timestamps(&output), vec![0, 1, 2, 3, 4]);
}

#[test]
fn the_same_stream_gives_the_same_output_at_any_width() {
    let input = stream(&[(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3)]);

    let narrow = run(input.clone(), 1, identity);
    let wide = run(input, 4, identity);

    assert_eq!(timestamps(&narrow), timestamps(&wide));
}

/// A worker that emits three frames for one, then none, then one.
#[test]
fn a_worker_changing_how_many_frames_it_emits_still_comes_back_dense() {
    let input = stream(&[(0, 0), (1, 1), (2, 2)]);

    let output = run(input, 3, |item| match item.meta().chunk {
        0 => vec![copy(&item), copy(&item), item],
        1 => vec![],
        _ => vec![item],
    });

    assert_eq!(indices(&output), vec![0, 1, 2, 3]);
}

#[test]
fn a_chunk_that_produces_nothing_does_not_stall_the_stream() {
    let input = stream(&[(0, 0), (1, 1), (2, 2)]);

    let output = run(input, 3, |item| match item.meta().chunk {
        1 => vec![],
        _ => vec![item],
    });

    assert_eq!(output.len(), 2);
}

/// Chunk 0 goes to worker 2, chunk 1 to worker 0 and chunk 2 to worker 1, so ascending worker id
/// and assignment order genuinely disagree. Draining workers 0, 1, 2 in that order would read
/// chunks 1, 2, 0. The aggregate must follow the assignments instead, and read chunks 0, 1, 2.
#[test]
fn chunks_come_back_in_assignment_order_not_ascending_worker_id() {
    let input = stream(&[(0, 0), (1, 1), (2, 2)]);
    let dispatch = Scripted::new(&[2, 0, 1], 3);

    let output = drive(input, dispatch, identity);

    assert_eq!(chunks(&output), vec![0, 1, 2]);
}
