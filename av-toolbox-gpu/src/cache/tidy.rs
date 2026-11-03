use std::fs::File;
#[cfg(windows)]
use std::fs::OpenOptions;
use std::io;
#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::STALE_BUILD_AGE;

pub(crate) fn build_cache_dir(root: &Path, build: &str) -> PathBuf {
    root.join(build)
}

/// Marks the build directory of `build` as used and removes stale directories of other builds.
///
/// Both steps are best effort, because a cache that is tidied late still works.
pub(crate) fn tidy_cache_root(root: &Path, build: &str) {
    let now = SystemTime::now();
    let build_dir = build_cache_dir(root, build);

    if let Err(error) = touch_directory(&build_dir, now) {
        tracing::debug!(?build_dir, %error, "cannot mark the kernel cache as recently used");
    }

    prune_stale_builds(root, build, now);
}

/// Removes directories of builds other than `current` that have gone unused for longer than
/// [STALE_BUILD_AGE].
///
/// Only real directories named like a build hash are candidates, so symlinks and anything else in the root
/// are left alone. Every error is ignored.
pub(crate) fn prune_stale_builds(root: &Path, current: &str, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };

        if name == current || !is_build_hash(name) {
            continue;
        }

        // `DirEntry::metadata` does not follow symlinks, so a link never reads as a directory.
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };

        let age = now.duration_since(modified).unwrap_or_default();
        if metadata.is_dir() && age > STALE_BUILD_AGE {
            let stale_path = entry.path();
            let _ = std::fs::remove_dir_all(stale_path);
        }
    }
}

pub(crate) fn is_build_hash(name: &str) -> bool {
    name.len() == 16 && name.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn touch_directory(path: &Path, now: SystemTime) -> io::Result<()> {
    let handle = open_directory(path)?;
    handle.set_modified(now)
}

#[cfg(not(windows))]
fn open_directory(path: &Path) -> io::Result<File> {
    File::open(path)
}

/// Opens a directory so its times can be set.
///
/// Windows only opens a directory handle with `FILE_FLAG_BACKUP_SEMANTICS`, and setting times needs
/// `FILE_WRITE_ATTRIBUTES` access.
#[cfg(windows)]
fn open_directory(path: &Path) -> io::Result<File> {
    const FILE_WRITE_ATTRIBUTES: u32 = 0x0000_0100;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

    OpenOptions::new()
        .access_mode(FILE_WRITE_ATTRIBUTES)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
}
