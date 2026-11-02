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

/// Keys this process already holds a place for.
static CLAIMED: LazyLock<Mutex<HashSet<u128>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// The kernel cache generation these stamps describe.
///
/// CubeCL keeps its compiled kernels in its environment database, which a dependency bump can
/// leave cold. A stamp from an older generation must not be read as covering a newer
/// one, so bump this whenever the CubeCL dependency changes.
const STAMP_GENERATION: &str = "cubecl-0.11";

/// A claim on compiling one kernel set, held while this process compiles it.
pub struct WarmUp {
    key: u128,
    directory: PathBuf,
    #[expect(dead_code, reason = "held only so dropping it releases the OS lock")]
    lock: File,
}

impl WarmUp {
    /// Claims the right to compile the kernel set named by `key`, waiting for whoever holds it.
    ///
    /// Returns `None` when there is nothing to coordinate: no cache directory, the set is already
    /// stamped as compiled, this process already holds the claim, the lock file could not be
    /// opened, or the wait ran out. In every one of those cases the caller should go ahead and
    /// compile.
    pub fn begin(key: u128) -> Option<Self> {
        let directory = compilation_cache_dir()?;

        Self::begin_in(directory, key)
    }

    /// The body of [WarmUp::begin], against a directory the caller names.
    fn begin_in(directory: &Path, key: u128) -> Option<Self> {
        // CubeCL makes the cache directory lazily, when it first writes a kernel, so on a cold
        // cache it does not exist yet and the lock file cannot be opened without this.
        let generation = directory.join(STAMP_GENERATION);
        if let Err(error) = std::fs::create_dir_all(&generation) {
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

        let Some(lock) = wait_for_lock(directory, key) else {
            let mut claimed = CLAIMED.lock().unwrap_or_else(|error| error.into_inner());
            claimed.remove(&key);

            return None;
        };

        Some(Self {
            key,
            directory: directory.to_path_buf(),
            lock,
        })
    }

    /// Records that this kernel set is compiled, so later processes skip the lock entirely.
    ///
    /// A `WarmUp` dropped without this records nothing, which is deliberate: an interrupted compile
    /// must not be reported as warm. The next run pays the full cost again.
    pub fn finish(self) {
        if let Err(error) = std::fs::write(stamp_path(&self.directory, self.key), []) {
            tracing::debug!("could not stamp the warmed kernel set: {error}");
        }
    }
}

impl Drop for WarmUp {
    /// Releases the claim and the lock together, however the compile ended.
    fn drop(&mut self) {
        let mut claimed = CLAIMED.lock().unwrap_or_else(|error| error.into_inner());
        claimed.remove(&self.key);
    }
}

/// Takes the lock for `key`, waiting up to [WAIT_LIMIT] for whoever holds it.
///
/// Returns `None` when the wait ran out, or when the set was stamped while waiting.
fn wait_for_lock(directory: &Path, key: u128) -> Option<File> {
    let started = Instant::now();
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
                        return None;
                    }

                    return Some(file);
                },
                Err(std::fs::TryLockError::WouldBlock) => {},
                Err(std::fs::TryLockError::Error(error)) => {
                    tracing::debug!("could not lock the warmup file: {error}");
                    return None;
                },
            },
            Err(error) => {
                tracing::debug!("could not open the warmup lock: {error}");
                return None;
            },
        }

        if is_warm(directory, key) {
            return None;
        }

        if started.elapsed() >= WAIT_LIMIT {
            tracing::debug!("gave up waiting for another process to compile, compiling here");
            return None;
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Whether a previous run already compiled this kernel set.
fn is_warm(directory: &Path, key: u128) -> bool {
    stamp_path(directory, key).exists()
}

fn lock_path(directory: &Path, key: u128) -> PathBuf {
    directory
        .join(STAMP_GENERATION)
        .join(format!("warmup-{key:032x}.lock"))
}

fn stamp_path(directory: &Path, key: u128) -> PathBuf {
    directory
        .join(STAMP_GENERATION)
        .join(format!("warmup-{key:032x}.stamp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_names_its_own_lock_and_stamp() {
        let directory = Path::new("/cache");

        let lock = lock_path(directory, 42);
        let stamp = stamp_path(directory, 42);

        assert_ne!(lock, stamp);
        assert!(lock.starts_with(directory), "{lock:?}");
        assert!(stamp.starts_with(directory), "{stamp:?}");
    }

    #[test]
    fn different_keys_name_different_files() {
        let directory = Path::new("/cache");

        assert_ne!(lock_path(directory, 1), lock_path(directory, 2));
        assert_ne!(stamp_path(directory, 1), stamp_path(directory, 2));
    }

    #[test]
    fn a_stamped_key_is_already_warm() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::create_dir_all(directory.path().join(STAMP_GENERATION)).expect("should make it");

        assert!(!is_warm(directory.path(), 7));
        std::fs::write(stamp_path(directory.path(), 7), []).expect("should write the stamp");
        assert!(is_warm(directory.path(), 7));
    }

    /// Without a cache directory there is nothing to coordinate around, so a caller should compile
    /// rather than wait for a lock that can never be taken.
    #[test]
    fn beginning_without_a_cache_directory_does_not_block() {
        assert!(WarmUp::begin(1).is_none());
    }

    #[test]
    fn a_free_lock_is_taken_and_released_when_the_file_drops() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::create_dir_all(directory.path().join(STAMP_GENERATION)).expect("should make it");

        let first = wait_for_lock(directory.path(), 3).expect("should take a free lock");
        drop(first);

        let second = wait_for_lock(directory.path(), 3);
        assert!(second.is_some(), "the lock was not released when dropped");
    }

    /// The holder stamps before releasing, so a waiter that takes the lock afterwards is told there
    /// is nothing left to compile.
    #[test]
    fn taking_the_lock_on_a_stamped_key_reports_nothing_to_do() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::create_dir_all(directory.path().join(STAMP_GENERATION)).expect("should make it");
        std::fs::write(stamp_path(directory.path(), 4), []).expect("should write the stamp");

        assert!(wait_for_lock(directory.path(), 4).is_none());
    }

    #[test]
    fn finishing_stamps_the_key_and_releases_the_claim() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::create_dir_all(directory.path().join(STAMP_GENERATION)).expect("should make it");
        let lock = wait_for_lock(directory.path(), 5).expect("should take a free lock");

        CLAIMED
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(5);

        let warm_up = WarmUp {
            key: 5,
            directory: directory.path().to_path_buf(),
            lock,
        };
        warm_up.finish();

        assert!(is_warm(directory.path(), 5));
        assert!(
            !CLAIMED
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .contains(&5),
            "finishing left the key claimed",
        );
    }

    #[test]
    fn dropping_without_finishing_releases_the_claim_and_stamps_nothing() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::create_dir_all(directory.path().join(STAMP_GENERATION)).expect("should make it");
        let lock = wait_for_lock(directory.path(), 6).expect("should take a free lock");

        CLAIMED
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(6);

        drop(WarmUp {
            key: 6,
            directory: directory.path().to_path_buf(),
            lock,
        });

        assert!(
            !is_warm(directory.path(), 6),
            "an unfinished compile was stamped warm"
        );
        assert!(
            !CLAIMED
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .contains(&6),
            "dropping left the key claimed",
        );
    }

    #[test]
    fn beginning_makes_the_directory_and_claims_the_key() {
        let directory = tempfile::tempdir().expect("should make a directory");
        let root = directory.path().join("not-made-yet");

        let warm_up = WarmUp::begin_in(&root, 10).expect("should claim a cold key");
        assert!(
            root.join(STAMP_GENERATION).is_dir(),
            "the warmup directory was not made"
        );

        warm_up.finish();
        assert!(is_warm(&root, 10));
    }

    #[test]
    fn beginning_twice_in_one_process_claims_once() {
        let directory = tempfile::tempdir().expect("should make a directory");

        let first = WarmUp::begin_in(directory.path(), 11).expect("should claim a cold key");
        assert!(
            WarmUp::begin_in(directory.path(), 11).is_none(),
            "the same process claimed one key twice",
        );

        drop(first);
    }

    #[test]
    fn beginning_on_a_stamped_key_reports_nothing_to_do() {
        let directory = tempfile::tempdir().expect("should make a directory");
        std::fs::create_dir_all(directory.path().join(STAMP_GENERATION)).expect("should make it");
        std::fs::write(stamp_path(directory.path(), 12), []).expect("should write the stamp");

        assert!(WarmUp::begin_in(directory.path(), 12).is_none());
    }
}
