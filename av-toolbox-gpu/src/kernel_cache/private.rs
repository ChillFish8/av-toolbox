use std::collections::HashSet;
use std::fs::File;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, Once, PoisonError, TryLockError};
use std::time::{Duration, Instant, SystemTime};

use cubecl::config::CubeClRuntimeConfig;
use cubecl::config::cache::CacheConfig;
use cubecl_environment::persistence::{Access, Database};

use super::database::{
    WriteState,
    copy_shared_entries,
    fill_private_file,
    modified,
    new_entries,
    write_shared,
    write_state,
};
use super::lock::{LockKind, MergeLock, hold_liveness_lock, lock_merge, lock_merge_within};
use super::prune::{lock_file_name, private_name, prune_stale_private_files, remove_database_files};

/// How long a process exiting waits for the shared cache's lock before leaving its private file.
pub(crate) const EXIT_LOCK_WAIT: Duration = Duration::from_secs(5);

/// How often an exiting process checks whether another thread let go of the slot.
const SLOT_POLL_INTERVAL: Duration = Duration::from_millis(20);

/// The lock file that guards the shared cache file.
pub(super) const MERGE_LOCK_NAME: &str = "merge.lock";

/// The directory next to the shared cache file that holds each process's private file.
pub(super) const PRIVATE_DIR_NAME: &str = "private";

/// The private file this process works on, once [PrivateKernelCache::start] has filled it.
pub(super) static ACTIVE: Mutex<Option<ActiveCache>> = Mutex::new(None);

type EntryKey = (String, Vec<u8>);

/// A private copy of a shared kernel cache, used by one process only.
///
/// The database engine CubeCL caches kernels in crashes and corrupts its file when several processes
/// write to one file at once. So each process points CubeCL at a private file of its own. The
/// private file starts as a copy of the shared file, and the kernels the process compiles are merged
/// back into the shared file at [merge_kernel_cache] and [finish_kernel_cache].
///
/// A lock file next to the shared file guards it. Copies take it shared, so processes can start in
/// parallel. Merges, refills and checkpoints take it exclusively. Each closes the shared file before
/// it lets the lock go. When the lock cannot be taken, the shared file is left alone, and the process
/// starts cold or keeps its kernels private.
///
/// Starting a process copies the whole shared file. File systems that clone files, such as XFS,
/// btrfs and APFS, make that copy almost free. Elsewhere it reads and writes every byte, so it takes
/// time in proportion to the size of the cache. A merge that has new kernels reads the whole private
/// file, which also takes time in proportion to the size of the cache.
#[derive(Debug)]
pub(crate) struct PrivateKernelCache {
    shared: PathBuf,
    private_dir: PathBuf,
    name: String,
}

impl PrivateKernelCache {
    /// Points `config` at a new private file for this process, next to the shared cache in `dir`.
    ///
    /// The shared file keeps the environment name `config` already had. Call [start](Self::start)
    /// once `config` is installed, before any CubeCL store or client opens the private file.
    pub(crate) fn configure(dir: &Path, config: &mut CubeClRuntimeConfig) -> Self {
        let shared_file_name = cubecl::environment::file_name(&config.environment.name);
        let shared = dir.join(shared_file_name);
        let private_dir = dir.join(PRIVATE_DIR_NAME);
        let name = private_name();

        config.environment.path = CacheConfig::Directory(private_dir.clone());
        config.environment.name = name.clone();

        Self {
            shared,
            private_dir,
            name,
        }
    }

    pub(super) fn private_file(&self) -> PathBuf {
        let file_name = cubecl::environment::file_name(&self.name);
        self.private_dir.join(file_name)
    }

    /// Fills the private file from the shared cache and makes it the one later merges read.
    ///
    /// It must run before any CubeCL store or client opens the private file, because the fill writes
    /// the file in place. It also deletes private files that exited processes left behind. Every
    /// failure is logged and the process carries on, with a cold private file at worst.
    pub(crate) fn start(self) {
        if let Err(error) = std::fs::create_dir_all(&self.private_dir) {
            tracing::warn!(dir = ?self.private_dir, %error, "cannot create the private kernel cache directory");
        }

        let liveness_path = self.private_dir.join(lock_file_name(&self.name));
        let liveness = hold_liveness_lock(&liveness_path);
        prune_stale_private_files(&self.private_dir, &self.name, SystemTime::now());

        let private = self.private_file();
        let merge_lock = merge_lock_path(&self.shared);
        let filled = guarded(|| fill_private_file(&self.shared, &private, &merge_lock));
        let shared_seen = filled.flatten();

        let written = write_state(&private);
        let active = ActiveCache {
            shared: self.shared,
            private,
            merge_lock,
            liveness_path,
            liveness,
            shared_seen,
            written,
            merged: HashSet::new(),
        };

        let mut slot = active_slot();
        *slot = Some(active);
    }
}

/// Copies the kernels this process compiled since the last merge into the shared cache.
///
/// Does nothing when no private file is in use or nothing was written since the last merge. A failed
/// merge is logged and the process carries on, because the cache only saves compile time.
pub(crate) fn merge_kernel_cache() {
    let mut slot = active_slot();
    if let Some(active) = slot.as_mut() {
        guarded(|| active.merge(None));
    }
}

/// Copies into the private file what other processes merged into the shared cache since this process
/// last read it.
///
/// Does nothing when the shared file has not changed, or when no private file is in use.
pub(crate) fn refill_kernel_cache() {
    let mut slot = active_slot();
    if let Some(active) = slot.as_mut() {
        guarded(|| active.refill_if_changed());
    }
}

/// Merges a last time and deletes this process's private file.
///
/// Later merges do nothing. Finishing is part of exiting, so it waits at most [EXIT_LOCK_WAIT] for
/// the shared cache's lock, and leaves the private file for a later process to delete when the lock
/// stays busy. A private file that cannot be deleted while it is open is also left for a later
/// process to delete once this one has exited.
pub(crate) fn finish_kernel_cache() {
    finish_waiting(EXIT_LOCK_WAIT);
}

/// Runs [finish_kernel_cache] when the process exits normally.
///
/// It suits processes without a single place to finish from, such as test binaries.
pub(crate) fn finish_kernel_cache_at_exit() {
    static REGISTER: Once = Once::new();

    REGISTER.call_once(|| {
        // SAFETY: `finish_at_exit` takes no arguments, never unwinds, and stays valid for the life
        // of the process.
        let failed = unsafe { atexit(finish_at_exit) };
        if failed != 0 {
            tracing::warn!("cannot finish the kernel cache at exit, its kernels stay private");
        }
    });
}

unsafe extern "C" {
    pub(crate) fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
}

/// Finishes the kernel cache on a new thread.
///
/// The exiting thread may already have dropped its thread locals, which the database engine needs.
extern "C" fn finish_at_exit() {
    let name = String::from("kernel-cache-exit");
    let spawned = std::thread::Builder::new().name(name).spawn(finish_kernel_cache);

    if let Ok(handle) = spawned {
        let _ = handle.join();
    }
}

/// Merges a last time, waiting at most `limit` for the shared cache's lock, and deletes the private
/// file once its kernels are safe.
///
/// The private file is kept when the merge does not land, so that nothing it holds is lost. The
/// whole call is bounded by `limit`, including the time spent getting the slot.
///
/// A process that never held its liveness lock deletes the file anyway, because no later process
/// could prove it stale and prune it.
pub(super) fn finish_waiting(limit: Duration) {
    let start = Instant::now();
    let slot = active_slot_within(limit);
    let taken = slot.and_then(|mut slot| slot.take());
    let Some(mut active) = taken else {
        return;
    };

    let wait = limit.saturating_sub(start.elapsed());
    let merged = guarded(|| active.merge(Some(wait))).unwrap_or(false);
    if !merged && active.liveness.is_some() {
        return;
    }

    active.delete();
}

/// The slot holding the private file in use.
///
/// A panic is never let out while the slot is held, so a poisoned slot still holds a usable value.
pub(super) fn active_slot() -> MutexGuard<'static, Option<ActiveCache>> {
    ACTIVE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Takes the slot, giving up with `None` when another thread still holds it after `limit`.
fn active_slot_within(limit: Duration) -> Option<MutexGuard<'static, Option<ActiveCache>>> {
    let start = Instant::now();
    loop {
        match ACTIVE.try_lock() {
            Ok(slot) => return Some(slot),
            Err(TryLockError::Poisoned(poisoned)) => return Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => {},
        }

        if start.elapsed() >= limit {
            return None;
        }

        std::thread::sleep(SLOT_POLL_INTERVAL);
    }
}

/// Runs cache work, turning a panic into a logged failure.
///
/// The database engine panics on some corrupt files, and a cache that cannot be read is only a
/// slower start, never a reason to stop the process.
pub(super) fn guarded<T>(work: impl FnOnce() -> T) -> Option<T> {
    let caught = std::panic::catch_unwind(AssertUnwindSafe(work));
    if caught.is_err() {
        tracing::warn!("the kernel cache work panicked, carrying on without it");
    }

    caught.ok()
}

pub(super) fn merge_lock_path(shared: &Path) -> PathBuf {
    let current_dir = Path::new(".");
    let dir = shared.parent().unwrap_or(current_dir);
    dir.join(MERGE_LOCK_NAME)
}

/// Takes the shared cache's lock exclusively, waiting at most `wait` or without limit when `None`.
fn lock_exclusive(path: &Path, wait: Option<Duration>) -> Option<MergeLock> {
    match wait {
        Some(limit) => lock_merge_within(path, LockKind::Exclusive, limit),
        None => lock_merge(path, LockKind::Exclusive),
    }
}

/// The private file in use and what is known about its contents.
pub(super) struct ActiveCache {
    pub(super) shared: PathBuf,
    pub(super) private: PathBuf,
    pub(super) merge_lock: PathBuf,
    pub(super) liveness_path: PathBuf,
    pub(super) liveness: Option<File>,
    /// The shared file's modification time when this process last read or wrote it.
    pub(super) shared_seen: Option<SystemTime>,
    /// The private file as of the last fill or merge.
    pub(super) written: WriteState,
    /// Entries already merged, so a later merge skips them.
    pub(super) merged: HashSet<EntryKey>,
}

impl ActiveCache {
    /// Merges the new entries, waiting at most `wait` for the lock or without limit when `None`.
    ///
    /// Reports whether the private file's kernels are now safe, which is also true when there was
    /// nothing new to merge.
    pub(super) fn merge(&mut self, wait: Option<Duration>) -> bool {
        let written = write_state(&self.private);
        if written == self.written {
            return true;
        }

        let opened = Database::open_at(&self.private, Access::ReadWrite);
        let private = match opened {
            Ok(private) => private,
            Err(error) => {
                let path = &self.private;
                tracing::warn!(?path, %error, "cannot open the private kernel cache to merge it");
                return false;
            },
        };

        let entries = new_entries(&private, &self.merged);
        drop(private);

        if !entries.is_empty() {
            let Some(lock) = lock_exclusive(&self.merge_lock, wait) else {
                return false;
            };

            let merged = write_shared(&self.shared, &entries);
            self.shared_seen = modified(&self.shared);
            drop(lock);

            if !merged {
                return false;
            }

            let keys = entries.into_iter().map(|(namespace, key, _)| (namespace, key));
            self.merged.extend(keys);
        }

        self.written = written;
        true
    }

    pub(super) fn refill_if_changed(&mut self) {
        let current = modified(&self.shared);
        if current.is_none() || current == self.shared_seen {
            return;
        }

        let Some(lock) = lock_merge(&self.merge_lock, LockKind::Exclusive) else {
            return;
        };

        copy_shared_entries(&self.shared, &self.private);
        self.shared_seen = modified(&self.shared);
        drop(lock);

        self.written = write_state(&self.private);
    }

    pub(super) fn delete(self) {
        remove_database_files(&self.private);
        drop(self.liveness);

        let _ = std::fs::remove_file(&self.liveness_path);
    }
}
