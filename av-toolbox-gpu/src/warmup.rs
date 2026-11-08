use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::cache::compilation_cache_dir;

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

/// A claim on compiling one kernel set, held while this process compiles it.
#[derive(Debug)]
pub struct WarmUp {
    key: u128,
    directory: PathBuf,
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
    /// compile.
    ///
    /// The wait blocks the calling thread for up to 180 seconds, and cancelling a run does not
    /// interrupt it.
    pub fn begin(key: u128) -> Option<Self> {
        let cache = compilation_cache_dir()?;
        let directory = cache.join(WARMUP_FOLDER);

        Self::begin_in(&directory, key, WAIT_LIMIT)
    }

    /// The body of [WarmUp::begin], against a directory and wait limit the caller names.
    fn begin_in(directory: &Path, key: u128, wait_limit: Duration) -> Option<Self> {
        // The build's cache directory may be fresh, so the warmup folder may not exist yet.
        if let Err(error) = std::fs::create_dir_all(directory) {
            tracing::debug!("could not make the warmup directory: {error}");
            return None;
        }

        if is_warm(directory, key) {
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
            Waited::Warm | Waited::GaveUp => {
                release(key);
                return None;
            },
        };

        Some(Self {
            key,
            directory: directory.to_path_buf(),
            lock,
        })
    }

    /// Records that this kernel set is compiled, so other processes load it rather than compile it.
    ///
    /// A `WarmUp` dropped without this records nothing, so an interrupted compile is never
    /// reported as warm. The next run pays the full cost again.
    pub fn finish(self) {
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
    use super::*;

    /// Short enough that a contended lock fails a test quickly rather than after three minutes.
    const BRIEFLY: Duration = Duration::from_millis(50);

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
    fn finishing_stamps_and_releases_the_claim() {
        let directory = tempfile::tempdir().expect("should make a directory");

        let warm_up = held_warm_up(directory.path(), 5);
        warm_up.finish();

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
        assert!(!is_claimed(6), "dropping left the key claimed");
    }

    #[test]
    fn beginning_makes_the_directory_and_claims_the_key() {
        let directory = tempfile::tempdir().expect("should make a directory");
        let root = directory.path().join("not-made-yet").join(WARMUP_FOLDER);

        let warm_up = WarmUp::begin_in(&root, 10, BRIEFLY).expect("should claim a cold key");
        assert!(root.is_dir(), "the warmup directory was not made");
        assert!(is_claimed(10));

        warm_up.finish();
        assert!(is_warm(&root, 10));
    }

    #[test]
    fn beginning_twice_in_one_process_claims_once() {
        let directory = tempfile::tempdir().expect("should make a directory");

        let first = WarmUp::begin_in(directory.path(), 11, BRIEFLY).expect("should claim a cold key");
        let second = WarmUp::begin_in(directory.path(), 11, BRIEFLY);
        assert!(second.is_none(), "the same process claimed one key twice");

        drop(first);
    }

    #[test]
    fn beginning_on_a_stamped_key_reports_nothing_to_do() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::write(stamp_path(directory.path(), 12), []).expect("should write the stamp");

        let began = WarmUp::begin_in(directory.path(), 12, BRIEFLY);

        assert!(began.is_none());
    }

    #[test]
    fn a_stamp_found_while_waiting_reports_nothing_to_do() {
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
        let began = WarmUp::begin_in(directory.path(), 13, Duration::from_secs(30));
        finisher.join().expect("the finisher should not panic");

        assert!(
            began.is_none(),
            "a waiter compiled kernels the holder already stamped"
        );
        assert!(!is_claimed(13), "the waiter kept its claim");
    }

    #[test]
    fn running_out_the_wait_releases_the_claim() {
        let directory = tempfile::tempdir().expect("should make a directory");
        let elsewhere = File::create(lock_path(directory.path(), 14)).expect("should make the lock");
        elsewhere.lock().expect("should lock");

        let began = WarmUp::begin_in(directory.path(), 14, BRIEFLY);

        assert!(began.is_none());
        assert!(!is_claimed(14), "giving up kept the claim");
    }
}
