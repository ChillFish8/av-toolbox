use av_toolbox_gpu::{
    AnyItem,
    ColourInfo,
    FrameFormat,
    FrameLayout,
    MetaRecord,
    SampleFormat,
    StreamItem,
    Subsampling,
    testing_context,
};

use crate::tests::routes::{meta_in, route};
use crate::{Disperse, RoundRobin, Routed, WorkItem};

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

fn item(index: u64, chunk: u64) -> AnyItem {
    let context = testing_context();
    let meta = meta_in(index, 0, chunk);
    StreamItem::Frame(context.alloc_frame(&format(), meta)).into()
}

fn chunked_disperse(workers: usize) -> Disperse<RoundRobin> {
    let dispatch = RoundRobin::new(workers);

    Disperse::new(&[true], 0, dispatch).with_route(Some(route(0)))
}

fn framed_disperse(workers: usize) -> Disperse<RoundRobin> {
    let dispatch = RoundRobin::new(workers);

    Disperse::by_frame(&[true], dispatch)
}

fn records(kinds: &[&str]) -> Vec<MetaRecord> {
    kinds.iter().map(|kind| MetaRecord::new(kind)).collect()
}

/// Each message as its worker and a short description, naming records by their kinds.
fn described(routed: &[Routed]) -> Vec<(usize, String)> {
    routed
        .iter()
        .map(|entry| (entry.worker, described_item(&entry.item)))
        .collect()
}

fn described_item(item: &WorkItem) -> String {
    match item {
        WorkItem::Bundle(_) => String::from("bundle"),
        WorkItem::Unit(unit, _) => format!("unit {unit}"),
        WorkItem::Records(records) => {
            let kinds: Vec<&str> = records.iter().map(MetaRecord::kind).collect();
            format!("records {}", kinds.join(" "))
        },
        WorkItem::EndOfChunk(chunk) => format!("end of chunk {chunk}"),
        WorkItem::EndOfStream => String::from("end of stream"),
    }
}

fn entry(worker: usize, description: &str) -> (usize, String) {
    (worker, String::from(description))
}

#[test]
fn pending_records_go_ahead_of_the_next_bundle_to_its_worker() {
    let mut disperse = chunked_disperse(2);
    disperse.offer(0, item(0, 7)).expect("should accept");
    let placed = disperse.place().expect("round robin always has a free worker");
    let worker = placed.routed.worker;

    disperse.offer_records(records(&["first", "second"]));
    let routed = disperse.offer(0, item(1, 7)).expect("should accept");

    assert_eq!(
        described(&routed),
        vec![entry(worker, "records first second"), entry(worker, "bundle")]
    );
}

#[test]
fn a_bundle_with_no_pending_records_goes_alone() {
    let mut disperse = chunked_disperse(2);
    disperse.offer(0, item(0, 7)).expect("should accept");
    let placed = disperse.place().expect("round robin always has a free worker");
    let worker = placed.routed.worker;

    let routed = disperse.offer(0, item(1, 7)).expect("should accept");

    assert!(placed.records.is_none());
    assert_eq!(described(&routed), vec![entry(worker, "bundle")]);
}

#[test]
fn a_held_bundle_carries_its_records_to_the_worker_it_is_placed_on() {
    let mut disperse = framed_disperse(2);
    disperse.offer_records(records(&["mark"]));
    disperse.offer(0, item(0, 0)).expect("should accept");

    let placed = disperse.place().expect("round robin always has a free worker");
    let worker = placed.routed.worker;
    let ordered = placed.in_order();

    assert_eq!(
        described(&ordered),
        vec![entry(worker, "records mark"), entry(worker, "unit 0")]
    );
}

/// The records reached the disperse after chunk 7's last frame and before chunk 8's first, so they
/// open chunk 8 on its own worker rather than following chunk 7's close.
#[test]
fn records_before_a_new_chunk_go_with_its_held_bundle() {
    let mut disperse = chunked_disperse(2);
    disperse.offer(0, item(0, 7)).expect("should accept");
    disperse.place().expect("round robin always has a free worker");

    disperse.offer_records(records(&["chunk"]));
    let closing = disperse.offer(0, item(1, 8)).expect("should accept");
    let placed = disperse.place().expect("round robin always has a free worker");
    let ordered = placed.in_order();

    assert_eq!(described(&closing), vec![entry(0, "end of chunk 7")]);
    assert_eq!(
        described(&ordered),
        vec![entry(1, "records chunk"), entry(1, "bundle")]
    );
}

#[test]
fn leftover_records_go_to_the_chunk_in_flight_ahead_of_its_close() {
    let mut disperse = chunked_disperse(2);
    disperse.offer(0, item(0, 7)).expect("should accept");
    disperse.place().expect("round robin always has a free worker");
    disperse.offer_records(records(&["mark", "chunk"]));

    let finished = disperse.finish();

    assert_eq!(
        described(&finished.routed),
        vec![
            entry(0, "records mark chunk"),
            entry(0, "end of chunk 7"),
            entry(0, "end of stream"),
            entry(1, "end of stream"),
        ]
    );
}

#[test]
fn leftover_records_go_to_the_first_closing_unit_under_frame_routing() {
    let mut disperse = framed_disperse(2);
    disperse.offer(0, item(0, 0)).expect("should accept");
    disperse.place().expect("round robin always has a free worker");
    disperse.offer_records(records(&["mark", "chunk"]));

    let finished = disperse.finish();

    assert_eq!(
        described(&finished.routed),
        vec![
            entry(0, "records mark chunk"),
            entry(0, "end of chunk 1"),
            entry(1, "end of chunk 2"),
            entry(0, "end of stream"),
            entry(1, "end of stream"),
        ]
    );
}

#[test]
fn leftover_records_with_no_worker_ever_chosen_are_dropped() {
    let mut disperse = chunked_disperse(2);
    disperse.offer_records(records(&["chunk"]));

    let finished = disperse.finish();

    assert_eq!(
        described(&finished.routed),
        vec![entry(0, "end of stream"), entry(1, "end of stream")]
    );
    assert!(finished.dropped.is_none());
}
