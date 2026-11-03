use std::fs::File;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use av_toolbox_gpu::{COMPILATION_CACHE_ENV, compilation_cache_dir, finish_compilation_cache};
use cubecl_environment::persistence::{Access, Database, Origin};
use tempfile::TempDir;

use crate::backend::install_cache;
use crate::build_hash;

const NAMESPACE: &str = "av-toolbox/test/kernels";
const KEY: &str = "compiled";
const SIGNAL_DIR_ENV: &str = "AVTB_TEST_LIFECYCLE_DIR";
const EXIT_CHILD_TEST: &str = "tests::lifecycle::exit_child";
const FINISH_CHILD_TEST: &str = "tests::lifecycle::finish_child";
const READY_FILE: &str = "ready";
const GO_FILE: &str = "go";
const FINISHED_FILE: &str = "finished";

/// How long a child may take to exit once the parent says go, which is well past the exit's lock wait.
const EXIT_DEADLINE: Duration = Duration::from_secs(10);

/// How long a child may take to install its cache and say it is ready.
const READY_DEADLINE: Duration = Duration::from_secs(60);

/// A child process that is killed and reaped when dropped, so a failing test leaves none behind.
struct ChildGuard(Child);

impl Deref for ChildGuard {
    type Target = Child;

    fn deref(&self) -> &Child {
        &self.0
    }
}

impl DerefMut for ChildGuard {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Starts a copy of the test binary that runs only the ignored test `test_name`.
///
/// The copy caches its kernels under `cache_root` and signals through files in `signal_dir`.
fn spawn_child_test(test_name: &str, cache_root: &Path, signal_dir: &Path) -> ChildGuard {
    let binary = std::env::current_exe().expect("the test binary path");

    let child = Command::new(binary)
        .args([test_name, "--exact", "--ignored", "--nocapture"])
        .env(COMPILATION_CACHE_ENV, cache_root)
        .env(SIGNAL_DIR_ENV, signal_dir)
        .spawn()
        .expect("spawn the child test");

    ChildGuard(child)
}

fn wait_for_file(path: &Path, deadline: Duration) {
    let start = Instant::now();
    while !path.exists() {
        assert!(start.elapsed() < deadline, "{path:?} never appeared");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The private database of the only process caching under `build_dir`.
///
/// Its name is the stem of the liveness lock the process holds next to it.
fn private_database(build_dir: &Path) -> PathBuf {
    let private_dir = build_dir.join("private");
    let entries = std::fs::read_dir(&private_dir).expect("read the private dir");

    let lock_name = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .find(|name| name.ends_with(".lock"))
        .expect("a liveness lock");
    let stem = lock_name.trim_end_matches(".lock");

    private_dir.join(format!("{stem}.db"))
}

fn private_databases(build_dir: &Path) -> usize {
    let private_dir = build_dir.join("private");
    let entries = std::fs::read_dir(private_dir).expect("read the private dir");

    entries
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".db"))
        .count()
}

fn insert_local_entry(database: &Path) {
    let opened = Database::open_at(database, Access::ReadWrite).expect("open the database");
    opened.insert(NAMESPACE, KEY.as_bytes(), b"kernel", Origin::Local);
}

/// The shared database directly under `build_dir`, once something has written it.
fn shared_database(build_dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(build_dir).expect("read the build dir");

    entries
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_file() && path.extension().is_some_and(|extension| extension == "db"))
}

fn shared_holds_entry(build_dir: &Path) -> bool {
    let shared = shared_database(build_dir);
    shared.is_some_and(|path| {
        let opened = Database::open_at(&path, Access::ReadOnly).expect("open the shared database");
        opened.get(NAMESPACE, KEY.as_bytes()).is_some()
    })
}

/// Installs the cache through the library, inserts a local entry and waits for the parent to say go.
///
/// Returns `None` when not run as a child of another test.
fn prepare_child() -> Option<PathBuf> {
    let signal_dir = std::env::var_os(SIGNAL_DIR_ENV)?;
    let signal_dir = PathBuf::from(signal_dir);

    install_cache();
    let build_dir = compilation_cache_dir().expect("the cache is installed");
    let private = private_database(build_dir);
    insert_local_entry(&private);

    File::create(signal_dir.join(READY_FILE)).expect("write the ready file");
    wait_for_file(&signal_dir.join(GO_FILE), READY_DEADLINE);

    Some(signal_dir)
}

/// Runs inside the child processes of the exit tests below, and does nothing when run on its own.
#[test]
#[ignore = "run as a child process by another test"]
fn exit_child() {
    prepare_child();
}

/// Runs inside the child process of the finish test below, and does nothing when run on its own.
#[test]
#[ignore = "run as a child process by another test"]
fn finish_child() {
    let Some(signal_dir) = prepare_child() else {
        return;
    };

    finish_compilation_cache();
    File::create(signal_dir.join(FINISHED_FILE)).expect("write the finished file");
}

/// A child test run to its exit, with the directories it worked in.
struct ExitedChild {
    _cache_root: TempDir,
    signal_dir: TempDir,
    build_dir: PathBuf,
}

/// Runs `test_name` in a child process, holding the shared cache's lock from just before the child
/// goes on when `hold_lock` is set, and asserts the child exits successfully within [EXIT_DEADLINE].
fn run_child_to_exit(test_name: &str, hold_lock: bool) -> ExitedChild {
    let cache_root = tempfile::tempdir().expect("a cache root");
    let signal_dir = tempfile::tempdir().expect("a signal dir");
    let mut child = spawn_child_test(test_name, cache_root.path(), signal_dir.path());
    wait_for_file(&signal_dir.path().join(READY_FILE), READY_DEADLINE);

    let build_dir = cache_root.path().join(build_hash());
    let merge_lock = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(build_dir.join("merge.lock"))
        .expect("open the merge lock");
    if hold_lock {
        merge_lock.lock().expect("hold the merge lock");
    }

    File::create(signal_dir.path().join(GO_FILE)).expect("write the go file");
    let start = Instant::now();
    let status = loop {
        let exited = child.try_wait().expect("poll the child");
        if let Some(status) = exited {
            break status;
        }

        assert!(start.elapsed() < EXIT_DEADLINE, "the child did not exit in time");
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(status.success());

    ExitedChild {
        _cache_root: cache_root,
        signal_dir,
        build_dir,
    }
}

#[test]
fn the_exit_finish_gives_up_on_a_held_lock() {
    let exited = run_child_to_exit(EXIT_CHILD_TEST, true);

    let kept = private_databases(&exited.build_dir);
    assert_eq!(kept, 1, "the private file is left for pruning");

    let merged = shared_holds_entry(&exited.build_dir);
    assert!(!merged, "the shared database stays without the entry");
}

#[test]
fn the_exit_finish_merges_and_deletes_without_a_held_lock() {
    let exited = run_child_to_exit(EXIT_CHILD_TEST, false);

    let kept = private_databases(&exited.build_dir);
    assert_eq!(kept, 0, "the private file is deleted once merged");

    let merged = shared_holds_entry(&exited.build_dir);
    assert!(merged, "the shared database gains the entry");
}

#[test]
fn finishing_gives_up_on_a_held_lock() {
    let exited = run_child_to_exit(FINISH_CHILD_TEST, true);

    let finished = exited.signal_dir.path().join(FINISHED_FILE);
    assert!(finished.exists(), "the finish returned before the child exited");

    let kept = private_databases(&exited.build_dir);
    assert_eq!(kept, 1, "the private file is left for pruning");

    let merged = shared_holds_entry(&exited.build_dir);
    assert!(!merged, "the shared database stays without the entry");
}

#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod sessions {
    use std::path::{Path, PathBuf};
    use std::str::FromStr;

    use av_toolbox_gpu::{
        ColourInfo,
        Device,
        FrameFormat,
        FrameLayout,
        FrameRate,
        SampleFormat,
        Subsampling,
        compilation_cache_dir,
    };

    use super::{SIGNAL_DIR_ENV, insert_local_entry, private_database, shared_holds_entry, spawn_child_test};
    use crate::{BackendOptions, PipelineInput, PipelineSource, Session, open_backend};

    const SESSIONS_CHILD_TEST: &str = "tests::lifecycle::sessions::sessions_child";
    const AFTER_FIRST_FILE: &str = "after-first";
    const AFTER_SECOND_FILE: &str = "after-second";

    const PASSTHROUGH: &str = r#"
[source.input]
type = "channel/frames"

[sink.output]
type = "channel/frames"
inputs = ["input"]
"#;

    fn format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: FrameRate::new(24, 1),
            field_order: None,
        }
    }

    fn device() -> Device {
        let value = std::env::var("AVTB_DEVICE").unwrap_or_default();
        if value.is_empty() {
            return Device::Default;
        }

        Device::from_str(&value).expect("AVTB_DEVICE names a device")
    }

    fn open_session() -> Session {
        let options = BackendOptions {
            device: device(),
            ..BackendOptions::default()
        };
        let backend = open_backend(&options).expect("the backend opens");

        let input = PipelineInput::Text {
            name: "passthrough.toml".to_owned(),
            text: PASSTHROUGH.to_owned(),
        };
        let source = PipelineSource::read(input, &[]).expect("the pipeline reads");
        let channels = [("input".to_owned(), format())];

        Session::new(source, backend, &channels).expect("the session opens")
    }

    /// Writes whether the shared database holds the test entry to `report`.
    fn report_shared(build_dir: &Path, report: &Path) {
        let holds_entry = shared_holds_entry(build_dir);
        std::fs::write(report, holds_entry.to_string()).expect("write the report");
    }

    fn read_report(path: &Path) -> String {
        std::fs::read_to_string(path).expect("the child wrote its report")
    }

    /// Runs inside the child process of the test below, and does nothing when run on its own.
    #[test]
    #[ignore = "run as a child process by another test"]
    fn sessions_child() {
        let Some(signal_dir) = std::env::var_os(SIGNAL_DIR_ENV) else {
            return;
        };
        let signal_dir = PathBuf::from(signal_dir);

        let first = open_session();
        let second = open_session();
        let build_dir = compilation_cache_dir().expect("the cache is installed");
        let private = private_database(build_dir);
        insert_local_entry(&private);

        drop(first);
        report_shared(build_dir, &signal_dir.join(AFTER_FIRST_FILE));

        drop(second);
        report_shared(build_dir, &signal_dir.join(AFTER_SECOND_FILE));
    }

    #[test]
    fn dropping_the_last_session_merges() {
        let cache_root = tempfile::tempdir().expect("a cache root");
        let signal_dir = tempfile::tempdir().expect("a signal dir");
        let mut child = spawn_child_test(SESSIONS_CHILD_TEST, cache_root.path(), signal_dir.path());

        let status = child.wait().expect("wait for the child");
        assert!(status.success());

        let after_first = read_report(&signal_dir.path().join(AFTER_FIRST_FILE));
        let after_second = read_report(&signal_dir.path().join(AFTER_SECOND_FILE));
        assert_eq!(after_first, "false", "one session is still open");
        assert_eq!(after_second, "true", "the last session merges");
    }
}
