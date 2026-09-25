use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, mpsc};

use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    MaskFormat,
    SampleFormat,
    Subsampling,
    default_accelerators,
};

pub(crate) fn format() -> FrameFormat {
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

/// A device-free set of capabilities, exactly as
/// [testing_context](av_toolbox_gpu::testing_context) builds its own, so `solve` never has to
/// touch a real backend either.
pub(crate) fn testing_capabilities() -> Capabilities {
    Capabilities::builder(default_accelerators()[0])
        .plane_size(16, 16)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

pub(crate) fn mask_format() -> MaskFormat {
    MaskFormat {
        width: 512,
        height: 512,
        sample: SampleFormat::U8,
        chroma: None,
    }
}

// Every static below is process wide, not per test, so it is only safe because the `Justfile`
// runs `cargo nextest run`, which gives each test its own process. Running this file's tests
// through plain `cargo test` instead puts them all in one process and one set of statics, and
// they will race each other.

/// How many fixture stages were built, and how many have since been dropped.
///
/// Every stage belongs to exactly one thread of a run and is dropped as that thread's own frame
/// goes away, so equal counts once a run has returned mean no thread is still holding one.
pub(crate) static STAGES_BUILT: AtomicU32 = AtomicU32::new(0);
pub(crate) static STAGES_DROPPED: AtomicU32 = AtomicU32::new(0);

/// Counts one fixture stage from the moment it is built to the moment it is dropped.
pub(crate) struct StageLife;

impl StageLife {
    pub(crate) fn new() -> Self {
        STAGES_BUILT.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for StageLife {
    fn drop(&mut self) {
        STAGES_DROPPED.fetch_add(1, Ordering::SeqCst);
    }
}

/// The id the next fixture instance takes.
///
/// A worker owns private instances of every node in its region, so an instance's own id stands in
/// for the worker running it without anything having to plumb a worker index through the engine.
static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(0);

/// Every frame a recording fixture processed, as `(instance_id, timestamp)` pairs, in the order
/// they were processed.
pub(crate) static ASSIGNED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Tells one fixture instance apart from every other one built in this process.
pub(crate) struct InstanceId(u64);

impl InstanceId {
    pub(crate) fn new() -> Self {
        Self(NEXT_INSTANCE_ID.fetch_add(1, Ordering::SeqCst))
    }

    pub(crate) fn processed(&self, timestamp: u64) {
        ASSIGNED.lock().expect("not poisoned").push((self.0, timestamp));
    }
}

/// Taken by the gated fixture's worker of the matching index, which waits there on its first
/// call before it has recorded anything. `None` means run unblocked, which is every worker of
/// every test that never sets one.
///
/// One set is shared by every gated fixture, since no test gates two stages at once. A second
/// gated stage in one graph would find the gates already taken and run open, so [GATE_TAKEN]
/// turns that into a panic rather than a quietly wrong count.
pub(crate) static GATE_HOLDS: [Mutex<Option<mpsc::Receiver<()>>>; 4] = [
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
];

/// Signalled by a gated worker the moment before it waits on its gate.
pub(crate) static GATE_REACHED: [Mutex<Option<mpsc::Sender<()>>>; 4] = [
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
];

/// Signalled by any gated worker each time it finishes a frame, so a test can wait for one worker
/// to produce while another is still held.
pub(crate) static GATE_PROCESSED: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

/// Whether each worker's gate has been claimed by a stage already.
///
/// Only one gated stage per graph can be held, since the gates are one shared set. A second one
/// would take nothing and run open, and its arrivals would be counted against the first, which
/// is a wrong answer rather than a failure. Claiming twice panics instead.
static GATE_TAKEN: [AtomicBool; 4] = [
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];

/// Holds one fixture instance inside its own `process` call until a test releases it.
///
/// A held worker is never marked free, so a dispatch policy choosing the least loaded worker has
/// to look elsewhere. That is what turns a test about spreading from a hope into a certainty.
pub(crate) struct WorkerGate {
    reached: Option<mpsc::Sender<()>>,
    hold: Option<mpsc::Receiver<()>>,
}

impl WorkerGate {
    /// A gate nothing holds back, for the registry build path no test drives.
    pub(crate) fn open() -> Self {
        Self {
            reached: None,
            hold: None,
        }
    }

    /// The gate for `worker`, holding whichever signal and release that worker was given.
    pub(crate) fn for_worker(worker: usize) -> Self {
        let reached = GATE_REACHED
            .get(worker)
            .and_then(|reached| reached.lock().expect("not poisoned").take());
        let hold = GATE_HOLDS
            .get(worker)
            .and_then(|hold| hold.lock().expect("not poisoned").take());

        if hold.is_some() {
            GATE_TAKEN[worker].store(true, Ordering::SeqCst);
        } else if worker < GATE_TAKEN.len() {
            assert!(
                !GATE_TAKEN[worker].load(Ordering::SeqCst),
                "worker {worker}'s gate was already taken by another stage, so this one would run open and its arrivals would be counted against the wrong stage"
            );
        }

        Self { reached, hold }
    }

    /// Waits here on the first call, when a test gave this worker a gate.
    pub(crate) fn wait(&mut self) {
        if let Some(reached) = self.reached.take() {
            reached.send(()).ok();
        }

        if let Some(hold) = self.hold.take() {
            hold.recv().ok();
        }
    }

    /// Announces that a frame came all the way through a gated stage.
    pub(crate) fn processed() {
        if let Some(sender) = GATE_PROCESSED.lock().expect("not poisoned").as_ref() {
            sender.send(()).ok();
        }
    }
}

/// The second a test frame's timestamp names, which is how every fixture tells frames apart.
pub(crate) fn frame_second(meta: &FrameMeta) -> u64 {
    meta.timestamp
        .expect("every test frame carries a timestamp")
        .as_secs()
}
