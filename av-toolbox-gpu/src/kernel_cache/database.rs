use std::collections::HashSet;
use std::path::Path;
use std::time::SystemTime;

use cubecl_environment::bytes::Bytes;
use cubecl_environment::persistence::{Access, Database, Origin};

use super::lock::{LockKind, lock_merge};
use super::private::guarded;
use super::prune::{remove_database_files, sidecar};

/// Namespaces that describe how one process built its cache, rather than holding cached kernels.
const SKIPPED_PREFIX: &str = "records/";

/// The suffix of the write-ahead log the database engine keeps next to a database file.
const LOG_SUFFIX: &str = "-wal";

/// The size of a write-ahead log header, which the engine writes when it opens a file, before any
/// write.
const LOG_HEADER_BYTES: u64 = 32;

/// A cached kernel as its namespace, key and value.
pub(super) type Entry = (String, Vec<u8>, Vec<u8>);

/// The size and modification time of one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileState {
    len: u64,
    modified: Option<SystemTime>,
}

fn file_state(path: &Path) -> Option<FileState> {
    let metadata = std::fs::metadata(path).ok()?;
    let modified = metadata.modified().ok();

    Some(FileState {
        len: metadata.len(),
        modified,
    })
}

/// A database file and its write-ahead log as they sit on disk, which change on every write.
///
/// A log holding only its header counts as no log, and only the database file's length is kept,
/// because opening a file rewrites its header. A process that only reads therefore never counts as
/// written. New entries either add log frames or, once checkpointed, lengthen the database file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WriteState {
    database_len: Option<u64>,
    log: Option<FileState>,
}

pub(super) fn write_state(database: &Path) -> WriteState {
    let log_path = sidecar(database, LOG_SUFFIX);
    let log = file_state(&log_path).filter(|state| state.len > LOG_HEADER_BYTES);
    let database = file_state(database);
    let database_len = database.map(|state| state.len);

    WriteState { database_len, log }
}

pub(super) fn has_log(database: &Path) -> bool {
    let state = write_state(database);
    state.log.is_some()
}

pub(super) fn modified(path: &Path) -> Option<SystemTime> {
    let metadata = std::fs::metadata(path).ok()?;
    metadata.modified().ok()
}

/// Copies the shared file over the private one and returns the shared file's modification time.
///
/// The copy runs under the shared lock, so other processes can copy at the same time while no merge
/// or checkpoint can. A shared file with a write-ahead log is first checkpointed under the exclusive
/// lock, so the copy holds every entry without the log. Nothing is copied when the shared file does
/// not exist yet or the lock cannot be taken, and the private file then starts empty. A copy that
/// cannot be opened is deleted, and the private file then starts empty too.
pub(super) fn fill_private_file(shared: &Path, private: &Path, merge_lock: &Path) -> Option<SystemTime> {
    let lock = lock_merge(merge_lock, LockKind::Shared)?;
    if !shared.exists() {
        return None;
    }

    if !has_log(shared) {
        let seen = copy_shared_file(shared, private);
        drop(lock);
        return seen;
    }

    // A file lock cannot be raised from shared to exclusive in one step, so the log is checked again
    // once the exclusive lock is held.
    drop(lock);
    let lock = lock_merge(merge_lock, LockKind::Exclusive)?;
    if has_log(shared) && !make_standalone(shared) {
        return None;
    }

    let seen = copy_shared_file(shared, private);
    drop(lock);
    seen
}

fn copy_shared_file(shared: &Path, private: &Path) -> Option<SystemTime> {
    let seen = modified(shared);
    if let Err(error) = std::fs::copy(shared, private) {
        tracing::warn!(?shared, ?private, %error, "cannot copy the shared kernel cache, starting cold");
        remove_database_files(private);
        return None;
    }

    let usable = guarded(|| Database::open_at(private, Access::ReadWrite).is_ok());
    if usable != Some(true) {
        tracing::warn!(?shared, "the shared kernel cache is unreadable, starting cold");
        remove_database_files(private);
        return None;
    }

    seen
}

/// Folds the shared file's write-ahead log into it and closes it, reporting whether that worked.
fn make_standalone(shared: &Path) -> bool {
    let database = match Database::open_at(shared, Access::ReadWrite) {
        Ok(database) => database,
        Err(error) => {
            tracing::warn!(?shared, %error, "cannot open the shared kernel cache");
            return false;
        },
    };

    match database.into_standalone() {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!(?shared, %error, "cannot checkpoint the shared kernel cache");
            false
        },
    }
}

/// The entries this process wrote to `private` that are not in `merged` yet.
///
/// Entries copied from the shared file are marked imported, so only local ones are new.
pub(super) fn new_entries(private: &Database, merged: &HashSet<(String, Vec<u8>)>) -> Vec<Entry> {
    let mut entries = Vec::new();

    for namespace in private.namespaces() {
        if namespace.starts_with(SKIPPED_PREFIX) {
            continue;
        }

        private.scan_with_origin(&namespace, &mut |key, value, origin| {
            if origin != Origin::Local {
                return;
            }

            let entry_key = (namespace.clone(), key.to_vec());
            if !merged.contains(&entry_key) {
                entries.push((namespace.clone(), key.to_vec(), value.to_vec()));
            }
        });
    }

    entries
}

/// Inserts `entries` into the shared file and closes it, reporting whether the merge landed.
///
/// They go in as imported entries, so a later fill marks them as not this process's own.
pub(super) fn write_shared(shared: &Path, entries: &[Entry]) -> bool {
    let database = match Database::open_at(shared, Access::ReadWrite) {
        Ok(database) => database,
        Err(error) => {
            tracing::warn!(?shared, %error, "cannot open the shared kernel cache to merge into it");
            return false;
        },
    };

    let mut failed = 0;
    let namespaces: HashSet<&String> = entries.iter().map(|(namespace, _, _)| namespace).collect();
    for namespace in namespaces {
        let mut batch = entries
            .iter()
            .filter(|(entry_namespace, _, _)| entry_namespace == namespace)
            .map(|(_, key, value)| to_bytes_pair(key, value));
        let summary = database.insert_many(namespace, &mut batch, Origin::Imported);
        failed += summary.failed;
    }

    if failed > 0 {
        tracing::warn!(
            ?shared,
            failed,
            "some kernels could not be merged into the shared cache"
        );
    }

    if let Err(error) = database.into_standalone() {
        tracing::warn!(?shared, %error, "cannot checkpoint the shared kernel cache after a merge");
    }

    failed == 0
}

/// Inserts the shared file's entries that `private` lacks, without overwriting any.
pub(super) fn copy_shared_entries(shared: &Path, private: &Path) {
    if !shared.exists() {
        return;
    }

    let source = match Database::open_at(shared, Access::ReadWrite) {
        Ok(source) => source,
        Err(error) => {
            tracing::warn!(?shared, %error, "cannot open the shared kernel cache to refill from it");
            return;
        },
    };

    let target = match Database::open_at(private, Access::ReadWrite) {
        Ok(target) => target,
        Err(error) => {
            tracing::warn!(?private, %error, "cannot open the private kernel cache to refill it");
            return;
        },
    };

    for namespace in source.namespaces() {
        if namespace.starts_with(SKIPPED_PREFIX) {
            continue;
        }

        let mut entries = Vec::new();
        source.scan(&namespace, &mut |key, value| {
            let pair = to_bytes_pair(key, value);
            entries.push(pair);
        });

        let mut batch = entries.into_iter();
        target.insert_many(&namespace, &mut batch, Origin::Imported);
    }

    drop(target);
    if let Err(error) = source.into_standalone() {
        tracing::warn!(?shared, %error, "cannot checkpoint the shared kernel cache after a refill");
    }
}

fn to_bytes_pair(key: &[u8], value: &[u8]) -> (Bytes, Bytes) {
    let key = Bytes::from_bytes_vec(key.to_vec());
    let value = Bytes::from_bytes_vec(value.to_vec());
    (key, value)
}
