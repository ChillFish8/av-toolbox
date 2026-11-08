use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;
use std::time::{Duration, Instant};

use cubecl_environment::persistence::{Access, Database};

/// The lock file beside the kernel cache that keeps its first open to one process at a time.
pub(crate) const FIRST_OPEN_LOCK: &str = "first-open.lock";

/// How long a process waits for another to finish the first open before carrying on without it.
const FIRST_OPEN_WAIT: Duration = Duration::from_secs(30);

/// How often a waiting process checks whether the lock came free.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Opens the shared kernel cache once, one process at a time, before CubeCL does.
///
/// A new cache file is switched to write-ahead logging by whichever process opens it first. Two
/// processes doing that together can make the loser give up on the cache for its whole lifetime.
/// Opening it here under a lock leaves the file ready, so later opens do not race.
///
/// This is best effort. Any failure is logged and the caller carries on.
pub(crate) fn prepare_kernel_cache(database: &Path) {
    let Some(directory) = database.parent() else {
        return;
    };

    let lock_path = directory.join(FIRST_OPEN_LOCK);
    let Some(lock) = lock_within(&lock_path, FIRST_OPEN_WAIT) else {
        return;
    };

    let opened = Database::open_at(database, Access::ReadWrite);
    if let Err(error) = opened {
        tracing::warn!(path = ?database, %error, "cannot open the kernel cache");
    }

    drop(lock);
}

/// Takes the lock at `path`, waiting at most `limit` for another process to let it go.
///
/// The lock is released when the returned file is dropped.
pub(crate) fn lock_within(path: &Path, limit: Duration) -> Option<File> {
    let opened = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path);
    let file = match opened {
        Ok(file) => file,
        Err(error) => {
            tracing::debug!(?path, %error, "cannot open the kernel cache lock");
            return None;
        },
    };

    let started = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Some(file),
            Err(TryLockError::WouldBlock) => {},
            Err(TryLockError::Error(error)) => {
                tracing::debug!(?path, %error, "cannot take the kernel cache lock");
                return None;
            },
        }

        if started.elapsed() >= limit {
            tracing::debug!(?path, "timed out waiting for the kernel cache lock");
            return None;
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}
