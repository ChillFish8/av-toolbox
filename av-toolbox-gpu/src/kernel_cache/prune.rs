use std::collections::HashSet;
use std::fs::OpenOptions;
use std::hash::{BuildHasher, RandomState};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Sidecar files SQLite-style databases keep beside the database file.
const SIDECARS: [&str; 3] = ["-wal", "-tshm", "-shm"];

/// How long a private file must go unused before it can be deleted.
pub(crate) const STALE_PRIVATE_AGE: Duration = Duration::from_secs(5 * 60);

/// A process id plus a random suffix, so no two live processes share a name.
pub(crate) fn private_name() -> String {
    let process = std::process::id();
    let now = SystemTime::now();
    let random = RandomState::new().hash_one(now);

    format!("{process}-{random:016x}")
}

pub(crate) fn lock_file_name(name: &str) -> String {
    format!("{name}.lock")
}

/// Whether `name` looks like a [private_name], which is a process id, a dash and sixteen hex characters.
pub(crate) fn is_private_name(name: &str) -> bool {
    let Some((process, random)) = name.split_once('-') else {
        return false;
    };

    let is_process = !process.is_empty() && process.bytes().all(|byte| byte.is_ascii_digit());
    let is_random = random.len() == 16
        && random
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));

    is_process && is_random
}

pub(crate) fn sidecar(database: &Path, suffix: &str) -> PathBuf {
    let mut path = database.as_os_str().to_owned();
    path.push(suffix);
    PathBuf::from(path)
}

/// Deletes a database file and its sidecars, ignoring files that are missing or still open.
pub(crate) fn remove_database_files(database: &Path) {
    let _ = std::fs::remove_file(database);
    for suffix in SIDECARS {
        let path = sidecar(database, suffix);
        let _ = std::fs::remove_file(path);
    }
}

/// Deletes the private files in `dir` whose process has exited and that have gone unused for longer
/// than [STALE_PRIVATE_AGE].
///
/// A process holds a lock on its own private file for as long as it runs, so a file whose lock can
/// be taken has no running owner. Only files named like a private file are candidates.
pub(crate) fn prune_stale_private_files(dir: &Path, own_name: &str, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    let mut names = HashSet::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };

        let stem = file_name.split('.').next().unwrap_or_default();
        if stem != own_name && is_private_name(stem) {
            names.insert(stem.to_string());
        }
    }

    for name in names {
        prune_if_stale(dir, &name, now);
    }
}

fn prune_if_stale(dir: &Path, name: &str, now: SystemTime) {
    let lock_path = dir.join(lock_file_name(name));
    let lock = OpenOptions::new().read(true).write(true).open(&lock_path);
    let lock = match lock {
        Ok(lock) => Some(lock),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => return,
    };

    let owned = lock.as_ref().is_some_and(|lock| lock.try_lock().is_err());
    if owned {
        return;
    }

    let file_name = cubecl::environment::file_name(name);
    let database = dir.join(file_name);
    let last_used = last_modified(&database, &lock_path);
    let age = now.duration_since(last_used).unwrap_or_default();
    if age <= STALE_PRIVATE_AGE {
        return;
    }

    remove_database_files(&database);
    drop(lock);

    let _ = std::fs::remove_file(lock_path);
}

/// The latest modification time among a private file, its sidecars and its lock file.
fn last_modified(database: &Path, lock_path: &Path) -> SystemTime {
    let sidecars = SIDECARS.iter().map(|suffix| sidecar(database, suffix));
    let paths = [database.to_path_buf(), lock_path.to_path_buf()]
        .into_iter()
        .chain(sidecars);

    paths
        .filter_map(|path| std::fs::metadata(path).ok())
        .filter_map(|metadata| metadata.modified().ok())
        .max()
        .unwrap_or(SystemTime::UNIX_EPOCH)
}
