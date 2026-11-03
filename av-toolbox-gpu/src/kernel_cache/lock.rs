use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How often a waiting process checks whether a lock came free.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

static LOCK_WARNED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy)]
pub(crate) enum LockKind {
    Shared,
    Exclusive,
}

/// The shared cache's lock, released when dropped.
pub(crate) struct MergeLock {
    file: File,
}

impl Drop for MergeLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Opens the lock file, creating it when this is the first process to ask.
///
/// A file that cannot be opened is logged as a warning and gives `None`.
pub(crate) fn open_lock_file(path: &Path) -> Option<File> {
    let opened = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path);

    match opened {
        Ok(file) => Some(file),
        Err(error) => {
            tracing::warn!(?path, %error, "cannot open the lock file, carrying on without it");
            None
        },
    }
}

/// Takes the lock that marks a private file as owned by a running process.
///
/// The operating system releases it when the process exits, however it exits.
pub(crate) fn hold_liveness_lock(path: &Path) -> Option<File> {
    let file = open_lock_file(path)?;
    match file.lock() {
        Ok(()) => Some(file),
        Err(error) => {
            tracing::warn!(?path, %error, "cannot lock the private kernel cache");
            None
        },
    }
}

/// Blocks until this process holds the shared cache's lock in `kind`.
///
/// Returns `None` when the lock cannot be taken, for example on a read-only or network file system,
/// and the caller then leaves the shared file alone.
pub(crate) fn lock_merge(path: &Path, kind: LockKind) -> Option<MergeLock> {
    let file = match open_merge_lock(path) {
        Ok(file) => file,
        Err(error) => {
            warn_lock_failure(path, &error);
            return None;
        },
    };

    let locked = match kind {
        LockKind::Shared => file.lock_shared(),
        LockKind::Exclusive => file.lock(),
    };
    if let Err(error) = locked {
        warn_lock_failure(path, &error);
        return None;
    }

    Some(MergeLock { file })
}

/// Waits up to `limit` for the shared cache's lock in `kind`.
///
/// Returns `None` when the lock stays busy for the whole limit or cannot be taken at all.
pub(crate) fn lock_merge_within(path: &Path, kind: LockKind, limit: Duration) -> Option<MergeLock> {
    let file = match open_merge_lock(path) {
        Ok(file) => file,
        Err(error) => {
            warn_lock_failure(path, &error);
            return None;
        },
    };

    let start = Instant::now();
    loop {
        let attempt = match kind {
            LockKind::Shared => file.try_lock_shared(),
            LockKind::Exclusive => file.try_lock(),
        };
        match attempt {
            Ok(()) => return Some(MergeLock { file }),
            Err(TryLockError::WouldBlock) => {},
            Err(TryLockError::Error(error)) => {
                warn_lock_failure(path, &error);
                return None;
            },
        }

        if start.elapsed() >= limit {
            tracing::debug!(?path, "timed out waiting for the kernel cache lock");
            return None;
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Opens the lock file, read-only when it already exists in a directory this process cannot write.
fn open_merge_lock(path: &Path) -> std::io::Result<File> {
    let writable = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path);

    match writable {
        Ok(file) => Ok(file),
        Err(_) => File::open(path),
    }
}

/// Logs the first failure to take the shared cache's lock as a warning, and later ones quietly.
fn warn_lock_failure(path: &Path, error: &std::io::Error) {
    let warned = LOCK_WARNED.swap(true, Ordering::Relaxed);
    if warned {
        tracing::debug!(?path, %error, "cannot take the kernel cache lock");
        return;
    }

    tracing::warn!(?path, %error, "cannot take the kernel cache lock, the shared cache is left alone");
}
