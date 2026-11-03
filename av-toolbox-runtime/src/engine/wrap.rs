use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use av_toolbox_core::{
    Artefact,
    Inputs,
    MetaInputs,
    MetaSink,
    Sink,
    Source,
    StageError,
    StageInstance,
    WantedArtefacts,
};
use av_toolbox_gpu::{FrameContext, MetaRecord, Side};
use av_toolbox_graph::{NodeId, StageBox};

use crate::Progress;

/// Wraps a source so the engine can tell, from outside, when it has produced its last frame, and counts
/// the frames it reads.
///
/// `Source::next` only ever returns `None` once, per its own contract, so storing that once is
/// enough to know the source will never be called again.
pub(super) struct WatchedSource<S: Side> {
    inner: Box<dyn Source<S>>,
    exhausted: Arc<AtomicBool>,
    limit: Option<u64>,
    produced: u64,
    read: Arc<AtomicU64>,
}

impl<S: Side> StageInstance for WatchedSource<S> {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }

    fn wants_artefacts(&mut self, wanted: &WantedArtefacts) -> Result<(), StageError> {
        self.inner.wants_artefacts(wanted)
    }

    fn take_artefacts(&mut self, context: &mut FrameContext) -> Result<Vec<Artefact>, StageError> {
        self.inner.take_artefacts(context)
    }
}

impl<S: Side> Source<S> for WatchedSource<S> {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<S::Frame>, StageError> {
        let reached = self.limit.is_some_and(|limit| self.produced >= limit);
        if reached {
            self.exhausted.store(true, Ordering::SeqCst);
            return Ok(None);
        }

        let frame = self.inner.next(context)?;
        match frame {
            Some(_) => {
                self.produced += 1;
                self.read.fetch_add(1, Ordering::Relaxed);
            },
            None => self.exhausted.store(true, Ordering::SeqCst),
        }

        Ok(frame)
    }
}

/// Tells [Progress](crate::Progress) when one sink first consumes a frame.
struct OutputReport {
    progress: Progress,
    reported: bool,
}

impl OutputReport {
    fn new(progress: &Progress) -> Self {
        Self {
            progress: progress.clone(),
            reported: false,
        }
    }

    /// Notes a sink step that consumed `seen` items.
    fn consumed(&mut self, seen: u64) {
        if self.reported || seen == 0 {
            return;
        }

        self.reported = true;
        self.progress.output_reached();
    }
}

/// What reached the sinks, and when the counted part of the run began.
pub(super) struct SinkTally {
    frames: AtomicU64,
    records: AtomicU64,
    warmup: u64,
    counted_from: OnceLock<Instant>,
}

impl SinkTally {
    pub(super) fn new(warmup: u64) -> Self {
        Self {
            frames: AtomicU64::new(0),
            records: AtomicU64::new(0),
            warmup,
            counted_from: OnceLock::new(),
        }
    }

    /// Adds `seen` items, marking the moment the warmup is passed.
    fn add(&self, seen: u64) {
        let before = self.frames.fetch_add(seen, Ordering::SeqCst);
        let after = before + seen;

        let crossed = before < self.warmup && after >= self.warmup;
        if crossed {
            let _ = self.counted_from.set(Instant::now());
        }
    }

    pub(super) fn frames(&self) -> u64 {
        self.frames.load(Ordering::SeqCst)
    }

    fn add_record(&self) {
        self.records.fetch_add(1, Ordering::SeqCst);
    }

    pub(super) fn records(&self) -> u64 {
        self.records.load(Ordering::SeqCst)
    }

    /// How long the run took after its warmup, given when it started and ended.
    pub(super) fn counted_wall(&self, started: Instant, ended: Instant) -> Duration {
        if self.warmup == 0 {
            return ended.duration_since(started);
        }

        match self.counted_from.get() {
            Some(from) => ended.duration_since(*from),
            None => Duration::ZERO,
        }
    }
}

/// Wraps a sink so the engine can count what reached it, for
/// [RunReport::frames](crate::engine::RunReport::frames), and counts its steps for
/// [Progress](crate::Progress).
struct CountingSink<S: Side> {
    inner: Box<dyn Sink<S>>,
    tally: Arc<SinkTally>,
    steps: Arc<AtomicU64>,
    output: OutputReport,
}

impl<S: Side> StageInstance for CountingSink<S> {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }

    fn wants_artefacts(&mut self, wanted: &WantedArtefacts) -> Result<(), StageError> {
        self.inner.wants_artefacts(wanted)
    }

    fn take_artefacts(&mut self, context: &mut FrameContext) -> Result<Vec<Artefact>, StageError> {
        self.inner.take_artefacts(context)
    }
}

impl<S: Side> Sink<S> for CountingSink<S> {
    fn consume(&mut self, context: &mut FrameContext, mut inputs: Inputs<S>) -> Result<(), StageError> {
        let mut slots = Vec::with_capacity(inputs.len());
        let mut seen = 0u64;

        for slot in 0..inputs.len() {
            let item = inputs.take(slot);
            if item.is_some() {
                seen += 1;
            }
            slots.push(item);
        }

        self.tally.add(seen);
        if seen > 0 {
            self.steps.fetch_add(1, Ordering::Relaxed);
        }

        self.inner.consume(context, Inputs::from_slots(slots))?;
        self.output.consumed(seen);

        Ok(())
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.end_of_stream(context)
    }
}

/// Wraps a metadata-only sink so the engine can count what reached it, and counts its steps for
/// [Progress](crate::Progress).
struct CountingMetaSink {
    inner: Box<dyn MetaSink>,
    tally: Arc<SinkTally>,
    steps: Arc<AtomicU64>,
    output: OutputReport,
}

impl StageInstance for CountingMetaSink {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }

    fn wants_artefacts(&mut self, wanted: &WantedArtefacts) -> Result<(), StageError> {
        self.inner.wants_artefacts(wanted)
    }

    fn take_artefacts(&mut self, context: &mut FrameContext) -> Result<Vec<Artefact>, StageError> {
        self.inner.take_artefacts(context)
    }
}

impl MetaSink for CountingMetaSink {
    fn consume(&mut self, context: &mut FrameContext, inputs: MetaInputs) -> Result<(), StageError> {
        let seen = inputs.filled() as u64;
        self.tally.add(seen);
        if seen > 0 {
            self.steps.fetch_add(1, Ordering::Relaxed);
        }

        self.inner.consume(context, inputs)?;
        self.output.consumed(seen);

        Ok(())
    }

    fn consume_record(&mut self, context: &mut FrameContext, record: &MetaRecord) -> Result<(), StageError> {
        self.inner.consume_record(context, record)?;
        self.tally.add_record();

        Ok(())
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.end_of_stream(context)
    }
}

/// Replaces the region head's source with a watched one, leaving every other stage alone.
pub(super) fn wrap_head_source(
    nodes: Vec<(NodeId, StageBox)>,
    exhausted: Arc<AtomicBool>,
    limit: Option<u64>,
    read: Arc<AtomicU64>,
) -> Vec<(NodeId, StageBox)> {
    nodes
        .into_iter()
        .enumerate()
        .map(|(index, (id, stage))| {
            if index == 0 {
                match stage {
                    StageBox::Source(inner) => {
                        let exhausted = exhausted.clone();
                        return (
                            id,
                            StageBox::Source(Box::new(WatchedSource {
                                inner,
                                exhausted,
                                limit,
                                produced: 0,
                                read: read.clone(),
                            })),
                        );
                    },
                    StageBox::HostSource(inner) => {
                        let exhausted = exhausted.clone();
                        return (
                            id,
                            StageBox::HostSource(Box::new(WatchedSource {
                                inner,
                                exhausted,
                                limit,
                                produced: 0,
                                read: read.clone(),
                            })),
                        );
                    },
                    other => return (id, other),
                }
            }

            (id, stage)
        })
        .collect()
}

/// Replaces every sink in a region's stages with a counting one.
pub(super) fn wrap_sinks(
    nodes: Vec<(NodeId, StageBox)>,
    tally: Arc<SinkTally>,
    progress: &Progress,
) -> Vec<(NodeId, StageBox)> {
    nodes
        .into_iter()
        .map(|(id, stage)| match stage {
            StageBox::Sink(inner) => {
                let steps = progress.sink_counter(id);
                let sink = CountingSink {
                    inner,
                    tally: tally.clone(),
                    steps,
                    output: OutputReport::new(progress),
                };

                (id, StageBox::Sink(Box::new(sink)))
            },
            StageBox::HostSink(inner) => {
                let steps = progress.sink_counter(id);
                let sink = CountingSink {
                    inner,
                    tally: tally.clone(),
                    steps,
                    output: OutputReport::new(progress),
                };

                (id, StageBox::HostSink(Box::new(sink)))
            },
            StageBox::MetaSink(inner) => {
                let steps = progress.sink_counter(id);
                let sink = CountingMetaSink {
                    inner,
                    tally: tally.clone(),
                    steps,
                    output: OutputReport::new(progress),
                };

                (id, StageBox::MetaSink(Box::new(sink)))
            },
            other => (id, other),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use av_toolbox_core::ArtefactData;
    use av_toolbox_gpu::{Frame, testing_context};

    use super::*;

    /// A stage of every wrapped kind, recording whether it was told `frames` is wanted and handing over one
    /// artefact.
    struct Handing {
        wanted: Arc<Mutex<Option<bool>>>,
    }

    impl StageInstance for Handing {
        fn wants_artefacts(&mut self, wanted: &WantedArtefacts) -> Result<(), StageError> {
            let mut seen = self.wanted.lock().expect("unpoisoned");
            *seen = Some(wanted.contains("frames"));

            Ok(())
        }

        fn take_artefacts(&mut self, _context: &mut FrameContext) -> Result<Vec<Artefact>, StageError> {
            let artefact = Artefact {
                name: "frames",
                data: ArtefactData::Opaque(vec![7]),
            };

            Ok(vec![artefact])
        }
    }

    impl Source for Handing {
        fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
            Ok(None)
        }
    }

    impl Sink for Handing {
        fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
            Ok(())
        }
    }

    impl MetaSink for Handing {
        fn consume(&mut self, _context: &mut FrameContext, _inputs: MetaInputs) -> Result<(), StageError> {
            Ok(())
        }
    }

    fn handing() -> (Handing, Arc<Mutex<Option<bool>>>) {
        let wanted = Arc::new(Mutex::new(None));
        let stage = Handing {
            wanted: wanted.clone(),
        };

        (stage, wanted)
    }

    /// Wraps `stage` as the engine would, as the only node of a region.
    fn wrapped(stage: StageBox) -> StageBox {
        let id = NodeId::new(0);
        let progress = Progress::new([id]);
        let tally = Arc::new(SinkTally::new(0));
        let exhausted = Arc::new(AtomicBool::new(false));
        let read = Arc::new(AtomicU64::new(0));

        let watched = wrap_head_source(vec![(id, stage)], exhausted, None, read);
        let mut counted = wrap_sinks(watched, tally, &progress);
        let (_, stage) = counted.pop().expect("one node");

        stage
    }

    fn assert_forwards_artefacts(stage: StageBox, wanted: &Mutex<Option<bool>>) {
        let mut stage = wrapped(stage);
        let instance = stage.instance_mut().expect("a stage instance");
        let mut context = testing_context();

        let wanting = WantedArtefacts::new(vec!["frames"]);
        instance.wants_artefacts(&wanting).expect("accepts");
        let taken = instance.take_artefacts(&mut context).expect("takes");

        assert_eq!(*wanted.lock().expect("unpoisoned"), Some(true));
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].name, "frames");
    }

    #[test]
    fn a_wrapped_source_forwards_artefacts() {
        let (stage, wanted) = handing();

        assert_forwards_artefacts(StageBox::Source(Box::new(stage)), &wanted);
    }

    #[test]
    fn a_wrapped_sink_forwards_artefacts() {
        let (stage, wanted) = handing();

        assert_forwards_artefacts(StageBox::Sink(Box::new(stage)), &wanted);
    }

    #[test]
    fn a_wrapped_meta_sink_forwards_artefacts() {
        let (stage, wanted) = handing();

        assert_forwards_artefacts(StageBox::MetaSink(Box::new(stage)), &wanted);
    }

    struct RecordingMeta {
        kinds: Arc<Mutex<Vec<String>>>,
    }

    impl StageInstance for RecordingMeta {}

    impl MetaSink for RecordingMeta {
        fn consume(&mut self, _context: &mut FrameContext, _inputs: MetaInputs) -> Result<(), StageError> {
            Ok(())
        }

        fn consume_record(
            &mut self,
            _context: &mut FrameContext,
            record: &MetaRecord,
        ) -> Result<(), StageError> {
            self.kinds
                .lock()
                .expect("unpoisoned")
                .push(record.kind().to_string());

            Ok(())
        }
    }

    #[test]
    fn a_wrapped_meta_sink_forwards_records_and_counts_them_apart_from_frames() {
        let kinds = Arc::new(Mutex::new(Vec::new()));
        let inner = RecordingMeta { kinds: kinds.clone() };
        let id = NodeId::new(0);
        let progress = Progress::new([id]);
        let tally = Arc::new(SinkTally::new(0));
        let stage = StageBox::MetaSink(Box::new(inner));
        let mut counted = wrap_sinks(vec![(id, stage)], tally.clone(), &progress);
        let (_, stage) = counted.pop().expect("one node");
        let StageBox::MetaSink(mut sink) = stage else {
            panic!("a meta sink");
        };
        let mut context = testing_context();
        let record = MetaRecord::new("chunk");

        sink.consume_record(&mut context, &record).expect("consumes");

        assert_eq!(*kinds.lock().expect("unpoisoned"), vec![String::from("chunk")]);
        assert_eq!(tally.frames(), 0);
        assert_eq!(tally.records(), 1);
    }
}
