use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::cache::{compilation_cache_dir, merge_compilation_cache, refill_compilation_cache};

/// How long a process waits for the one ahead of it before compiling for itself.
///
/// Compiling a full kernel set takes around ten seconds on a quiet machine, and a machine running
/// an encode is not quiet, so the limit is generous. Waiting longer is worse than duplicating the
/// work, because whatever is downstream has nothing to do until a frame arrives.
const WAIT_LIMIT: Duration = Duration::from_secs(180);

/// How often a waiting process retries the lock.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The folder inside the build's cache directory that holds the locks and stamps.
const WARMUP_FOLDER: &str = "warmup";

/// Keys this process already holds a place for.
static CLAIMED: LazyLock<Mutex<HashSet<u128>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// How a warm-up shares its kernels through the kernel cache.
#[derive(Debug, Clone, Copy)]
struct CacheHooks {
    merge: fn(),
    refill: fn(),
}

const SHARED_CACHE: CacheHooks = CacheHooks {
    merge: merge_compilation_cache,
    refill: refill_compilation_cache,
};

/// A claim on compiling one kernel set, held while this process compiles it.
#[derive(Debug)]
pub struct WarmUp {
    key: u128,
    directory: PathBuf,
    hooks: CacheHooks,
    #[expect(dead_code, reason = "held only so dropping it releases the OS lock")]
    lock: File,
}

/// How a wait for another process's warm-up ended.
#[derive(Debug)]
enum Waited {
    Locked(File),
    Warm,
    GaveUp,
}

impl WarmUp {
    /// Claims the right to compile the kernel set named by `key`, waiting for whoever holds it.
    ///
    /// Returns `None` when there is nothing to coordinate: no cache directory, the set is already
    /// stamped as compiled, this process already holds the claim, the lock file could not be
    /// opened, or the wait ran out. In every one of those cases the caller should go ahead and
    /// compile. A stamp found at once or after waiting first copies the kernels other processes
    /// shared into this one's cache.
    ///
    /// The wait blocks the calling thread for up to 180 seconds, and cancelling a run does not
    /// interrupt it.
    pub fn begin(key: u128) -> Option<Self> {
        let cache = compilation_cache_dir()?;
        let directory = cache.join(WARMUP_FOLDER);

        Self::begin_in(&directory, key, WAIT_LIMIT, SHARED_CACHE)
    }

    /// The body of [WarmUp::begin], against a directory, wait limit and cache the caller names.
    fn begin_in(directory: &Path, key: u128, wait_limit: Duration, hooks: CacheHooks) -> Option<Self> {
        // The build's cache directory may be fresh, so the warmup folder may not exist yet.
        if let Err(error) = std::fs::create_dir_all(directory) {
            tracing::debug!("could not make the warmup directory: {error}");
            return None;
        }

        if is_warm(directory, key) {
            (hooks.refill)();
            return None;
        }

        let mut claimed = CLAIMED.lock().unwrap_or_else(|error| error.into_inner());
        if !claimed.insert(key) {
            return None;
        }
        drop(claimed);

        let waited = wait_for_lock(directory, key, wait_limit);
        let lock = match waited {
            Waited::Locked(lock) => lock,
            Waited::Warm => {
                release(key);
                (hooks.refill)();
                return None;
            },
            Waited::GaveUp => {
                release(key);
                return None;
            },
        };

        Some(Self {
            key,
            directory: directory.to_path_buf(),
            hooks,
            lock,
        })
    }

    /// Shares the compiled kernels with other processes, then records that this kernel set is compiled.
    ///
    /// A `WarmUp` dropped without this records nothing, so an interrupted compile is never
    /// reported as warm. The next run pays the full cost again.
    pub fn finish(self) {
        (self.hooks.merge)();

        if let Err(error) = std::fs::write(stamp_path(&self.directory, self.key), []) {
            tracing::debug!("could not stamp the warmed kernel set: {error}");
        }
    }
}

impl Drop for WarmUp {
    /// Releases the claim and the lock together, however the compile ended.
    fn drop(&mut self) {
        release(self.key);
    }
}

fn release(key: u128) {
    let mut claimed = CLAIMED.lock().unwrap_or_else(|error| error.into_inner());
    claimed.remove(&key);
}

/// Takes the lock for `key`, waiting up to `wait_limit` for whoever holds it.
fn wait_for_lock(directory: &Path, key: u128, wait_limit: Duration) -> Waited {
    let started = Instant::now();
    let mut announced = false;

    loop {
        let opened = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path(directory, key));

        match opened {
            Ok(file) => match file.try_lock() {
                Ok(()) => {
                    if is_warm(directory, key) {
                        return Waited::Warm;
                    }

                    return Waited::Locked(file);
                },
                Err(std::fs::TryLockError::WouldBlock) => {},
                Err(std::fs::TryLockError::Error(error)) => {
                    tracing::debug!("could not lock the warmup file: {error}");
                    return Waited::GaveUp;
                },
            },
            Err(error) => {
                tracing::debug!("could not open the warmup lock: {error}");
                return Waited::GaveUp;
            },
        }

        if is_warm(directory, key) {
            return Waited::Warm;
        }

        if !announced {
            tracing::info!("waiting for another process to compile this pipeline's kernels");
            announced = true;
        }

        if started.elapsed() >= wait_limit {
            tracing::debug!("gave up waiting for another process to compile, compiling here");
            return Waited::GaveUp;
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Whether a previous run already compiled this kernel set.
fn is_warm(directory: &Path, key: u128) -> bool {
    stamp_path(directory, key).exists()
}

fn lock_path(directory: &Path, key: u128) -> PathBuf {
    directory.join(format!("warmup-{key:032x}.lock"))
}

fn stamp_path(directory: &Path, key: u128) -> PathBuf {
    directory.join(format!("warmup-{key:032x}.stamp"))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;

    /// Short enough that a contended lock fails a test quickly rather than after three minutes.
    const BRIEFLY: Duration = Duration::from_millis(50);

    thread_local! {
        /// The stamp the probing merge looks for.
        static PROBED_STAMP: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
        /// Whether that stamp existed when the merge ran, once it has run.
        static STAMPED_AT_MERGE: Cell<Option<bool>> = const { Cell::new(None) };
        static REFILLS: Cell<usize> = const { Cell::new(0) };
    }

    fn probe_merge() {
        let stamp = PROBED_STAMP.with_borrow(Clone::clone);
        let stamped = stamp.is_some_and(|path| path.exists());

        STAMPED_AT_MERGE.set(Some(stamped));
    }

    fn count_refill() {
        let refills = REFILLS.get();
        REFILLS.set(refills + 1);
    }

    const PROBED: CacheHooks = CacheHooks {
        merge: probe_merge,
        refill: count_refill,
    };

    fn held_warm_up(directory: &Path, key: u128) -> WarmUp {
        let Waited::Locked(lock) = wait_for_lock(directory, key, BRIEFLY) else {
            panic!("should take a free lock");
        };

        CLAIMED
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(key);

        WarmUp {
            key,
            directory: directory.to_path_buf(),
            hooks: PROBED,
            lock,
        }
    }

    fn is_claimed(key: u128) -> bool {
        CLAIMED
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains(&key)
    }

    #[test]
    fn a_key_names_its_own_lock_and_stamp() {
        let directory = Path::new("/cache/warmup");

        let lock = lock_path(directory, 42);
        let stamp = stamp_path(directory, 42);

        assert_ne!(lock, stamp);
        assert!(lock.starts_with(directory), "{lock:?}");
        assert!(stamp.starts_with(directory), "{stamp:?}");
    }

    #[test]
    fn different_keys_name_different_files() {
        let directory = Path::new("/cache/warmup");

        assert_ne!(lock_path(directory, 1), lock_path(directory, 2));
        assert_ne!(stamp_path(directory, 1), stamp_path(directory, 2));
    }

    #[test]
    fn a_stamped_key_is_already_warm() {
        let directory = tempfile::tempdir().expect("should make a directory");

        assert!(!is_warm(directory.path(), 7));
        std::fs::write(stamp_path(directory.path(), 7), []).expect("should write the stamp");
        assert!(is_warm(directory.path(), 7));
    }

    #[test]
    fn beginning_without_a_cache_directory_does_not_block() {
        assert!(WarmUp::begin(1).is_none());
    }

    #[test]
    fn a_free_lock_is_taken_and_released_when_the_file_drops() {
        let directory = tempfile::tempdir().expect("should make a directory");

        let first = wait_for_lock(directory.path(), 3, BRIEFLY);
        assert!(matches!(first, Waited::Locked(_)), "{first:?}");
        drop(first);

        let second = wait_for_lock(directory.path(), 3, BRIEFLY);
        assert!(matches!(second, Waited::Locked(_)), "{second:?}");
    }

    #[test]
    fn a_lock_held_elsewhere_runs_out_the_wait() {
        let directory = tempfile::tempdir().expect("should make a directory");
        let elsewhere = File::create(lock_path(directory.path(), 8)).expect("should make the lock");
        elsewhere.lock().expect("should lock");

        let waited = wait_for_lock(directory.path(), 8, BRIEFLY);
        assert!(matches!(waited, Waited::GaveUp), "{waited:?}");
    }

    #[test]
    fn taking_the_lock_on_a_stamped_key_reports_it_warm() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::write(stamp_path(directory.path(), 4), []).expect("should write the stamp");

        let waited = wait_for_lock(directory.path(), 4, BRIEFLY);
        assert!(matches!(waited, Waited::Warm), "{waited:?}");
    }

    #[test]
    fn finishing_merges_then_stamps() {
        let directory = tempfile::tempdir().expect("should make a directory");
        let stamp = stamp_path(directory.path(), 5);
        PROBED_STAMP.set(Some(stamp));

        let warm_up = held_warm_up(directory.path(), 5);
        warm_up.finish();

        assert_eq!(
            STAMPED_AT_MERGE.get(),
            Some(false),
            "the merge ran after the stamp, or not at all"
        );
        assert!(is_warm(directory.path(), 5));
        assert!(!is_claimed(5), "finishing left the key claimed");
    }

    #[test]
    fn dropping_without_finishing_releases_the_claim_and_stamps_nothing() {
        let directory = tempfile::tempdir().expect("should make a directory");

        drop(held_warm_up(directory.path(), 6));

        assert!(
            !is_warm(directory.path(), 6),
            "an unfinished compile was stamped warm"
        );
        assert_eq!(STAMPED_AT_MERGE.get(), None, "an unfinished compile was merged");
        assert!(!is_claimed(6), "dropping left the key claimed");
    }

    #[test]
    fn beginning_makes_the_directory_and_claims_the_key() {
        let directory = tempfile::tempdir().expect("should make a directory");
        let root = directory.path().join("not-made-yet").join(WARMUP_FOLDER);

        let warm_up = WarmUp::begin_in(&root, 10, BRIEFLY, PROBED).expect("should claim a cold key");
        assert!(root.is_dir(), "the warmup directory was not made");
        assert!(is_claimed(10));

        warm_up.finish();
        assert!(is_warm(&root, 10));
    }

    #[test]
    fn beginning_twice_in_one_process_claims_once() {
        let directory = tempfile::tempdir().expect("should make a directory");

        let first = WarmUp::begin_in(directory.path(), 11, BRIEFLY, PROBED).expect("should claim a cold key");
        let second = WarmUp::begin_in(directory.path(), 11, BRIEFLY, PROBED);
        assert!(second.is_none(), "the same process claimed one key twice");

        drop(first);
    }

    #[test]
    fn beginning_on_a_stamped_key_refills_and_reports_nothing_to_do() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::write(stamp_path(directory.path(), 12), []).expect("should write the stamp");

        let began = WarmUp::begin_in(directory.path(), 12, BRIEFLY, PROBED);

        assert!(began.is_none());
        assert_eq!(REFILLS.get(), 1);
    }

    #[test]
    fn a_stamp_found_while_waiting_triggers_a_refill() {
        let directory = tempfile::tempdir().expect("should make a directory");
        let lock = lock_path(directory.path(), 13);
        let stamp = stamp_path(directory.path(), 13);
        let elsewhere = File::create(lock).expect("should make the lock");
        elsewhere.lock().expect("should lock");

        let finisher = std::thread::spawn(move || {
            std::thread::sleep(POLL_INTERVAL * 3);
            std::fs::write(stamp, []).expect("should write the stamp");
            drop(elsewhere);
        });
        let began = WarmUp::begin_in(directory.path(), 13, Duration::from_secs(30), PROBED);
        finisher.join().expect("the finisher should not panic");

        assert!(
            began.is_none(),
            "a waiter compiled kernels the holder already stamped"
        );
        assert_eq!(REFILLS.get(), 1);
        assert!(!is_claimed(13), "the waiter kept its claim");
    }

    #[test]
    fn running_out_the_wait_releases_the_claim_without_a_refill() {
        let directory = tempfile::tempdir().expect("should make a directory");
        let elsewhere = File::create(lock_path(directory.path(), 14)).expect("should make the lock");
        elsewhere.lock().expect("should lock");

        let began = WarmUp::begin_in(directory.path(), 14, BRIEFLY, PROBED);

        assert!(began.is_none());
        assert_eq!(REFILLS.get(), 0);
        assert!(!is_claimed(14), "giving up kept the claim");
    }
}
